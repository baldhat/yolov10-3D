import argparse
from collections import OrderedDict
import json
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
import logging
import os
from detectron2.structures import BoxMode
from typing import List
import numpy as np
import torch
from fvcore.common.timer import Timer
from detectron2.utils.file_io import PathManager
import contextlib
import io
from pycocotools.coco import COCO

from omni3d_evaluation import Omni3DEvaluationHelper
from omni3d import is_ignore

logger = logging.getLogger("default")


class Evaluator:
    def __init__(self, dataset_names: List[str], gt_ann_files: List[str], pred_ann_files: List[str]):
        self.dataset_names = dataset_names
        self.gt_ann_files = gt_ann_files
        self.pred_ann_files = pred_ann_files

        self.target_mapping = self.check_and_setup_categories()
        self.filter_settings = {
            'category_names': list(self.target_mapping.values()),
            'ignore_names': ['dontcare', 'void', 'ignore'],
            'truncation_thres': 0.33333333,
            'visibility_thres': 0.33333333,
            'min_height_thres': 0.0625,
            'modal_2D_boxes': False,
            'trunc_2D_boxes': True,
            'max_depth': 100000000.0,
            'max_height_thres': 1.5
        }
        self.register_datasets()

    def register_datasets(self):
        for dataset_name, gt_ann_file in zip(self.dataset_names, self.gt_ann_files):
            self._simple_register(dataset_name, gt_ann_file, self.filter_settings)

    def check_and_setup_categories(self):
        target_mapping, target_mapping_inv = None, None
        for gt_ann_file in self.gt_ann_files:
            data = json.load(open(gt_ann_file, 'r'))
            new_mapping = {cat['id']: cat['name'] for cat in data['categories']}
            new_mapping_inv = {cat['name']: cat['id'] for cat in data['categories']}
            if target_mapping is None:
                target_mapping = new_mapping
                target_mapping_inv = new_mapping_inv
            else:
                assert target_mapping == new_mapping
                assert target_mapping_inv == new_mapping_inv

        for pred_ann_file in self.pred_ann_files:
            data = torch.load(pred_ann_file)
            unique_category_ids = set(np.unique([i['category_id'] for ins in data for i in ins['instances']]))
            assert len(set(list(target_mapping.keys())).difference(unique_category_ids)) == 0

        logger.info("Categories are valid")

        target_mapping = OrderedDict(sorted(target_mapping.items(), key=lambda t: t[0]))
        MetadataCatalog.get('omni3d_model').thing_classes = list(target_mapping.values())
        MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id = {t: t for t in target_mapping.keys()}
        return target_mapping

    def _load_omni3d_json(self, json_file, image_root, dataset_name, filter_settings, filter_empty=False):
        # read in the dataset
        timer = Timer()
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

        # the global meta information for the full dataset
        meta_model = MetadataCatalog.get('omni3d_model')

        # load the meta information
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds(filter_settings['category_names']))
        cats = coco_api.loadCats(cat_ids)
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # the id mapping must be based on the model!
        id_map = meta_model.thing_dataset_id_to_contiguous_id
        meta.thing_dataset_id_to_contiguous_id = id_map

        # sort indices for reproducible results
        img_ids = sorted(coco_api.imgs.keys())
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        total_num_valid_anns = sum([len(x) for x in anns])
        total_num_anns = len(coco_api.anns)
        if total_num_valid_anns < total_num_anns:
            logger.info(f"{json_file} contains {total_num_anns} annotations, but only "
                        f"{total_num_valid_anns} of them match to images in the file.")

        imgs_anns = list(zip(imgs, anns))
        logger.info("Loaded {} images in Omni3D format from {}".format(len(imgs_anns), json_file))

        dataset_dicts = []

        # annotation keys to pass along
        ann_keys = [
            "bbox",
            "bbox3D_cam",
            "bbox2D_proj",
            "bbox2D_trunc",
            "bbox2D_tight",
            "center_cam",
            "dimensions",
            "pose",
            "R_cam",
            "category_id",
        ]

        # optional per image keys to pass if exists
        # this property is unique to KITTI.
        img_keys_optional = ['p2']

        invalid_count = 0

        for (img_dict, anno_dict_list) in imgs_anns:

            has_valid_annotation = False

            record = {}
            record["file_name"] = os.path.join(image_root, img_dict["file_path"])
            record["dataset_id"] = img_dict["dataset_id"]
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            record["K"] = img_dict["K"]

            # store optional keys when available
            for img_key in img_keys_optional:
                if img_key in img_dict:
                    record[img_key] = img_dict[img_key]

            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                assert anno["image_id"] == image_id

                obj = {key: anno[key] for key in ann_keys if key in anno}

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                annotation_category_id = obj["category_id"]

                # category is not part of ids and is not in the ignore category?
                if not (annotation_category_id in id_map) and not (anno['category_name']
                                                                   in filter_settings['ignore_names']):
                    continue

                ignore = is_ignore(anno, filter_settings, img_dict["height"])

                obj['iscrowd'] = False
                obj['ignore'] = ignore

                if filter_settings['modal_2D_boxes'] and 'bbox2D_tight' in anno and anno['bbox2D_tight'][0] != -1:
                    obj['bbox'] = BoxMode.convert(anno['bbox2D_tight'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

                elif filter_settings['trunc_2D_boxes'] and 'bbox2D_trunc' in anno and not np.all(
                    [val == -1 for val in anno['bbox2D_trunc']]):
                    obj['bbox'] = BoxMode.convert(anno['bbox2D_trunc'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

                elif 'bbox2D_proj' in anno:
                    obj['bbox'] = BoxMode.convert(anno['bbox2D_proj'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

                else:
                    continue

                obj['pose'] = anno['R_cam']

                # store category as -1 for ignores!
                obj["category_id"] = -1 if ignore else id_map[annotation_category_id]

                objs.append(obj)

                has_valid_annotation |= (not ignore)

            if has_valid_annotation or (not filter_empty):
                record["annotations"] = objs
                dataset_dicts.append(record)

            else:
                invalid_count += 1

        logger.info("Filtered out {}/{} images without valid annotations".format(invalid_count, len(imgs_anns)))

        return dataset_dicts

    def _simple_register(self, dataset_name, path_to_json, filter_settings):
        # DatasetCatalog.register(dataset_name, lambda: self._load_omni3d_json(
        #     path_to_json, "",
        #     dataset_name, filter_settings, filter_empty=False
        # ))

        self._load_omni3d_json(path_to_json, "", dataset_name, filter_settings, filter_empty=False)
        MetadataCatalog.get(dataset_name).set(json_file=path_to_json, image_root="", evaluator_type="coco")

    def evaluate(self, dataset_names: List[str], pred_ann_files: List[str]):
        eval_helper = Omni3DEvaluationHelper(
            self.dataset_names,
            self.filter_settings,
            output_folder="",
            iter_label=0,
            only_2d=False,
        )

        for dataset_name, pred_file in zip(dataset_names, pred_ann_files):
            results_json = torch.load(pred_file)

            eval_helper.add_predictions(dataset_name, results_json)
            eval_helper.evaluate(dataset_name)

        eval_helper.summarize_all()


def to_indiv_files(paths):
    assert "[" in paths and "]" in paths
    return paths.replace("[", "").replace("]", "").split(",")


def evaluate(dataset_names, gt_ann_files: str, pred_ann_files: str, log_dir: str) -> None:
    setup_logger(output=log_dir, name="default")

    dataset_names = to_indiv_files(dataset_names)
    gt_ann_files, pred_ann_files = to_indiv_files(gt_ann_files), to_indiv_files(pred_ann_files)
    assert len(dataset_names) == len(gt_ann_files) == len(pred_ann_files)

    eval = Evaluator(dataset_names, gt_ann_files, pred_ann_files)
    eval.evaluate(dataset_names, pred_ann_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_names', type=str, required=True)
    parser.add_argument('--gt_ann_files', type=str, required=True)
    parser.add_argument('--pred_ann_files', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(**vars(args))
