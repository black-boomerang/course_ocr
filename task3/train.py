import os

import mmcv
from mmcv import Config
from mmseg.apis import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor


def train(config_path: str):
    config = Config.fromfile(config_path)
    mmcv.mkdir_or_exist(os.path.abspath(config.work_dir))
    model = build_segmentor(config.model)
    dataset = build_dataset(config.data.train)
    model.CLASSES = dataset.CLASSES
    model.PALETTE = dataset.PALETTE
    train_segmentor(model, dataset, config, distributed=False, validate=True)
