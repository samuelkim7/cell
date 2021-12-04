#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""


import os
import logging
import numpy as np
from pathlib import Path
import pycocotools.mask as mask_util
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import verify_results
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T


def setup(args):
    """
    Create configs and perform basic setups.
    """
    dataDir = '/home/samuelkim/.kaggle/data/sartorious/'
    dataPath = Path(dataDir)

    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT='bitmask'
    # register_coco_instances('sartorius_train',{}, dataDir + 'annotations_train.json', dataPath)
    register_coco_instances('sartorius_train',{}, dataDir + 'annotations_all.json', dataPath)
    register_coco_instances('sartorius_val',{}, dataDir + 'annotations_val.json', dataPath)
    metadata = MetadataCatalog.get('sartorius_train')
    train_ds = DatasetCatalog.get('sartorius_train')

    cfg.MODEL_NAME = 'mask_rcnn_R_50_FPN_1x_no_Aug_all'
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("sartorius_train",)
    cfg.DATASETS.TEST = ("sartorius_val",)
    cfg.DATALOADER.NUM_WORKERS = 12
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 20000    
    cfg.SOLVER.STEPS = [10000]        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
    # cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get('sartorius_train')) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.OUTPUT_DIR = '/home/samuelkim/workspace/detectron2/tools/outputs/' + cfg.MODEL_NAME + '/'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)


def score(pred, targ):
    pred_masks = pred['instances'].pred_masks.cpu().numpy()
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)


class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in dataset_dicts}
            
    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)    
            else:
                targ = self.annotations_cache[inp['image_id']]
                self.scores.append(score(out, targ))

    def evaluate(self):
        return {"MaP IoU": np.mean(self.scores)}


class Trainer(DefaultTrainer):
    # @classmethod
    # def build_train_loader(cls, cfg):
    #     return build_detection_train_loader(cfg,
    #                mapper=DatasetMapper(cfg, is_train=True, augmentations=[
    #                   T.RandomBrightness(0.9, 1.1),
    #                   T.RandomContrast(0.9, 1.1),
    #                   T.RandomSaturation(0.9, 1.1),
    #                   T.RandomLighting(0.9),
    #                   T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
    #                   T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    #                ]))
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)


def sc_logger(name, log_file_path) -> logging.Logger:
    """
    Supports stream and file handler
    Print level is set to debug as default
    :param name (str): logger name
    :param log_file_path (str): path for saving logs to file
    return (logging.Logger) logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Printing formatter
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    # Add handlers: stream and file
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if not os.path.isdir(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def main(args):
    cfg = setup(args)

    logging_path = ''
    logger = sc_logger('cell_segmentation', f'{cfg.OUTPUT_DIR}/{cfg.MODEL_NAME}.log')

    msg = ''
    for k, v in cfg.items():
        msg += k + ': ' + str(v) + '\n'
    logger.info(f'::: Training Configurations ::: \n{msg}')

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
