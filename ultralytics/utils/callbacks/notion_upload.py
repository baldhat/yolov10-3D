import os
from notion_client import Client
import pandas as pd
import yaml
import numpy as np
import numbers
import sys
from pathlib import Path
import matplotlib.pyplot as plt


class Run:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.ignore_list = ['dfl', 'mode', 'agnostic_nms', 'warmup_momentum', 'close_mosaic', 'save_conf', 'cfg',
                            'half', 'workspace', 'crop_fraction', 'rect', 'copy_paste', 'multi_scale', 'profile',
                            'line_width', 'classes', 'show_boxes', 'embed', 'format', 'save_txt', 'tracker', 'verbose',
                            'int8', 'erasing', 'retina_masks', 'save_frames', 'dynamic', 'save_json', 'mosaic',
                            'save_period', 'simplify', 'kobj', 'stream_buffer', 'dnn', 'pose', 'label_smoothing',
                            'degrees', 'keras', 'source', 'save', 'freeze', 'save_hybrid', 'device', 'vid_stride',
                            'amp', 'exist_ok', 'show_conf', 'nbs', 'cache', 'overlap_mask', 'visualize', 'save_crop',
                            'show_labels', 'opset', 'show', 'augment', "auto_augment", "flipud"]
        self.args = self.get_args()
        self.results = self.get_results()


    def toProperties(self, current_properties):
        dict = {}
        for key, value in self.args.items():
            if (isinstance(value, numbers.Number) and not isinstance(value, bool)
                and not current_properties.get(key, {}).get("type", "") == "rich_text"):
                dict.update({key: self.to_number_property(value)})
            else:
                dict.update({key: self.to_text_property(value)})
        dict.update({
            "Name": {"title": [{"text": {"content": self.args["name"]}}]},
            "AP@0.7": self.get_best_property()
        })
        return dict

    def getChildren(self):
        '''        return [
        {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": 2,
            "has_column_header": False,
            "has_row_header": False,
            "children": self.create_data_table()
        }
        }
        ]'''
        return []



    def create_data_table(self):
        rows = []
        for epoch, ap in zip(self.results["epoch"], self.results["metrics/Car3D@0.7"]):
            rows.append( {"type": "table_row",
                         "table_row": {
                             "cells": [
                                 [
                                     {
                                        "type": "text",
                                        "text": {
                                        "content": str(epoch),
                                        "link": None
                                        },
                                        "annotations": {
                                        "bold": False,
                                        "italic": False,
                                        "strikethrough": False,
                                        "underline": False,
                                        "code": False,
                                        "color": "default"
                                        },
                                        "plain_text": "column 1 content",
                                        "href": "null",
                                     }
                                 ],
                                 [
                                     {
                                         "type": "text",
                                         "text": {
                                             "content": str(ap),
                                             "link": None
                                         },
                                         "annotations": {
                                             "bold": False,
                                             "italic": False,
                                             "strikethrough": False,
                                             "underline": False,
                                             "code": False,
                                             "color": "default"
                                         },
                                         "plain_text": "column 1 content",
                                         "href": "null",
                                     }
                                 ],
                             ]
                         }
                         })
        return rows

    def get_name_property(self):
        return {"title": self.to_text_property(self.args.get("name", "Unknown"))}

    def to_text_property(self, value):
        return {
            "rich_text": [
                {
                    "text": {
                        "content": str(value)
                    }
                }
            ]
        }

    def to_number_property(self, value):
        return {
            "type": "number",
            "number": value
        }

    def to_text_property_def(self, value):
        return {
            "rich_text": {}
        }

    def to_number_property_def(self, value):
        return {
            "number": {}
        }

    def get_best_property(self):
        return {
            "number": float(np.max(self.results["metrics/Car3D@0.7"]))
        }

    def get_args(self):
        with open(os.path.join(self.run_dir, "args.yaml")) as file:
            args = yaml.safe_load(file)
        args = self.filter_args(args)
        return args

    def filter_args(self, args):
        new_args = {}
        for key, value in args.items():
            if key not in self.ignore_list:
                new_args[key] = value
        return new_args


    def get_results(self):
        df = pd.read_csv(os.path.join(self.run_dir, "results.csv"), header=0)
        df = df.rename(columns=lambda x: x.strip())
        return df

    def create_property_defs(self, current_properties):
        dic = {}
        for key, value in self.args.items():
            if key not in ["Name"]:
                if (isinstance(value, numbers.Number) and not isinstance(value, bool)
                        and not current_properties.get(key, {}).get("type", "") == "rich_text"):
                    dic.update({key: self.to_number_property_def(value)})
                else:
                    dic.update({key: self.to_text_property_def(value)})
        return dic

    def annot_max(self, x, y, ax):
        xmax = x[np.argmax(y)]
        ymax = y.max()
        text = f"{xmax}: {ymax}".format(xmax, ymax)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data', textcoords="data",
                  arrowprops=arrowprops, ha="left", va="top")
        ax.annotate(text, xy=(xmax, ymax), xytext=(xmax + 1, ymax + 1), **kw)

    def get_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        x = self.results["epoch"]
        y = self.results["metrics/Car3D@0.7"]
        ax.plot(x, y)
        self.annot_max(x, y, ax)
        fig.canvas.draw()
        img_plot = np.array(fig.canvas.renderer.buffer_rgba())
        return img_plot

def upload_to_notion(run_dir):
    file_path = os.path.join(os.path.expanduser("~"), ".integrations/run_result_uploader")
    content = open(file_path, "r").readlines()
    page = content[0].strip()
    secret = content[1].strip()
    run = Run(run_dir)

    notion = Client(auth=secret)

    database = notion.databases.retrieve(page)
    current_properties = database["properties"]
    props = run.toProperties(current_properties)

    notion.databases.update(page, properties=run.create_property_defs(current_properties))

    filter = {"and": [{"property": "Name", "title": {"equals": run.args["name"]}}]}
    runs = notion.databases.query(page, filter=filter)["results"]
    if len(runs) > 0:
        props = {"AP@0.7": run.get_best_property()}
        notion.pages.update(runs[0]["id"], properties=props)
    else:
        notion.pages.create(
            parent={"database_id": page},
            properties=props,
            children=run.getChildren()
        )


if __name__ == '__main__':
    args = sys.argv[1:]
    for arg in args:
        if os.path.exists(arg):
            upload_to_notion(arg)
        else:
            path = os.path.join(Path.home(),"experiments/results",arg)
            if os.path.exists(path):
                upload_to_notion(path)
            else:
                print(f"Unknown path {path} or {arg}!")