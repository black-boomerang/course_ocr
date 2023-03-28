import os

import mmcv
from mmcv import Config
from mmpose.apis import train_model
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet


def train(config_path: str):
    config = Config.fromfile(config_path)
    mmcv.mkdir_or_exist(os.path.abspath(config.work_dir))
    model = build_posenet(config.model)
    datasets = [build_dataset(config.data.train)]
    train_model(model, datasets, config, distributed=False, validate=True)
