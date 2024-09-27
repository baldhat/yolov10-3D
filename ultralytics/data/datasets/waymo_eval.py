from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow.compat.v1 as tf
from typing import Dict, List
import json
print()
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2

from google.protobuf import text_format

ERROR = 1e-6


class WaymoEvaluation(tf.test.TestCase):
    # Source: https://github.com/abhi1kumar/DEVIANT/blob/main/data/waymo/waymo_eval.py
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  #

        if iou == 0.7:
            self._config_text = """
num_desired_score_cutoffs: 11
breakdown_generator_ids: OBJECT_TYPE
breakdown_generator_ids: RANGE
difficulties {
levels: 1
levels: 2
}
difficulties {
levels: 1
levels: 2
}
matcher_type: TYPE_HUNGARIAN
iou_thresholds: 0.0
iou_thresholds: 0.7
iou_thresholds: 0.5
iou_thresholds: 0.5
iou_thresholds: 0.5
box_type: TYPE_3D
"""
        elif np.isclose(iou, 0.5):
            self._config_text = """
num_desired_score_cutoffs: 11
breakdown_generator_ids: OBJECT_TYPE
breakdown_generator_ids: RANGE
difficulties {
levels: 1
levels: 2
}
difficulties {
levels: 1
levels: 2
}
matcher_type: TYPE_HUNGARIAN
iou_thresholds: 0.0
iou_thresholds: 0.7
iou_thresholds: 0.3
iou_thresholds: 0.3
iou_thresholds: 0.3
box_type: TYPE_3D
"""
        else:
            raise NotImplementedError()

    @staticmethod
    def format_annos(annos: Dict[str, List]) -> Dict[str, np.ndarray]:
        for k, v in annos.items():
            if k in ('bbox', 'score', 'difficulty'):
                annos[k] = np.asarray(v, dtype=np.float32)
            elif k == 'type':
                annos[k] = np.asarray(v, dtype=np.uint8)
            elif k == 'frame_id':
                annos[k] = np.asarray(v, dtype=np.int64)

    def _BuildConfig(self):
        config = metrics_pb2.Config()
        # pdb.set_trace()
        text_format.Merge(self._config_text, config)
        return config

    def _BuildGraph(self, graph):
        with graph.as_default():
            self._pd_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._pd_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._pd_score = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._gt_difficulty = tf.compat.v1.placeholder(dtype=tf.uint8)
            metrics = detection_metrics.get_detection_metric_ops(
                config=self._BuildConfig(),
                prediction_frame_id=self._pd_frame_id,
                prediction_bbox=self._pd_bbox,
                prediction_type=self._pd_type,
                prediction_score=self._pd_score,
                prediction_overlap_nlz=tf.zeros_like(self._pd_frame_id, dtype=tf.bool),
                ground_truth_bbox=self._gt_bbox,
                ground_truth_type=self._gt_type,
                ground_truth_frame_id=self._gt_frame_id,
                ground_truth_difficulty=self._gt_difficulty,
                recall_at_precision=0.95,
            )
            return metrics

    def _EvalUpdateOps(
        self,
        sess,
        graph,
        metrics,
        prediction_frame_id,
        prediction_bbox,
        prediction_type,
        prediction_score,
        ground_truth_frame_id,
        ground_truth_bbox,
        ground_truth_type,
        ground_truth_difficulty,
    ):
        sess.run(
            [tf.group([value[1] for value in metrics.values()])], feed_dict={
                self._pd_bbox: prediction_bbox,
                self._pd_frame_id: prediction_frame_id,
                self._pd_type: prediction_type,
                self._pd_score: prediction_score,
                self._gt_bbox: ground_truth_bbox,
                self._gt_type: ground_truth_type,
                self._gt_frame_id: ground_truth_frame_id,
                self._gt_difficulty: ground_truth_difficulty,
            })

    def _EvalValueOps(self, sess, graph, metrics):
        ddd = {}
        for item in metrics.items():
            ddd[item[0]] = sess.run([item[1][0]])
        return ddd

    def test(self):  # pred_annos: Dict[str, List], gt_annos: Dict[str, List]
        self.format_annos(data['gt'])
        self.format_annos(data['pred'])

        graph = tf.Graph()
        metrics = self._BuildGraph(graph)
        with self.test_session(graph=graph) as sess:
            sess.run(tf.compat.v1.initializers.local_variables())
            self._EvalUpdateOps(sess, graph, metrics, data['pred']['frame_id'], data['pred']['bbox'],
                                data['pred']['type'], data['pred']['score'], data['gt']['frame_id'], data['gt']['bbox'],
                                data['gt']['type'], data['gt']['diff'])

            aps = self._EvalValueOps(sess, graph, metrics)
            category_list = ["VEHICLE", "CYCLIST", "PEDESTRIAN", "SIGN"]
            level_list = [1, 2]
            metric_list = ["AP", "APH", "Recall@0.95"]

            eval_text = "--------------------------------------------------------------------------------------------\n"
            eval_text += "Class      | L |         {:11s}     |         {:11s}     |     {:11s}     \n".format(
                "AP_3D", "APH_3D", "Recall@0.95")
            eval_text += "--------------------------------------------------------------------------------------------\n"

            for category in category_list:
                for level in level_list:
                    text = "{:10s} | {} ".format(category, level)
                    key_list = ["OBJECT_TYPE_TYPE_{}_LEVEL_{}".format(category, level), \
                                "RANGE_TYPE_{}_[0, 30)_LEVEL_{}".format(category, level),\
                                "RANGE_TYPE_{}_[30, 50)_LEVEL_{}".format(category, level),\
                                "RANGE_TYPE_{}_[50, +inf)_LEVEL_{}".format(category, level)]
                    for metric in metric_list:
                        key0 = os.path.join(key_list[0], metric)
                        key1 = os.path.join(key_list[1], metric)
                        key2 = os.path.join(key_list[2], metric)
                        key3 = os.path.join(key_list[3], metric)
                        # Report in percentage.
                        multiplier = 100.0
                        text += "| {:5.2f} {:5.2f} {:5.2f} {:5.2f} ".format(multiplier * aps[key0][0],
                                                                            multiplier * aps[key1][0],
                                                                            multiplier * aps[key2][0],
                                                                            multiplier * aps[key3][0])
                    eval_text += text + "\n"

        print(eval_text)


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('iou', 0.7, 'IoU threshold')
flags.DEFINE_string('pred', '', 'Predictions and ground truth')

if __name__ == '__main__':

    data = json.load(open(FLAGS.pred, 'r'))

    assert FLAGS.iou in [0.5, 0.7]
    iou = FLAGS.iou

    tf.compat.v1.disable_eager_execution()
    tf.test.main()
