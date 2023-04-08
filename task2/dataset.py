from typing import Any, Union, List

import albumentations as A
import cv2
from torch.utils.data import Dataset, DataLoader

import config
from data_reader import HWDBDatasetHelper, LMDBReader


class HWDBDataset(Dataset):
    def __init__(self, helper: HWDBDatasetHelper, transforms: Any):
        self.helper = helper
        self.transforms = transforms

    def __len__(self):
        return self.helper.size()

    def __getitem__(self, idx):
        image, label = self.helper.get_item(idx)
        image = self.transforms(image=image)['image']
        return image[None, :, :], label


def init_dataloaders(path: str, split_set: bool = True) -> Union[DataLoader, List[DataLoader]]:
    reader = LMDBReader(path).open()

    if split_set:
        helper = HWDBDatasetHelper(reader)
        helpers = helper.train_val_split()
    else:
        helper = HWDBDatasetHelper(reader, prefix='Test')
        helpers = [helper]

    transforms = A.Compose([
        A.LongestMaxSize(max_size=config.image_side),
        A.PadIfNeeded(min_height=config.image_side, min_width=config.image_side,
                      border_mode=cv2.BORDER_CONSTANT, value=255),
        A.Normalize(mean=0.5, std=1.0)
    ])

    dataloaders = []
    shuffle = False
    for helper in helpers[::-1]:
        dataset = HWDBDataset(helper, transforms)
        dataloaders.append(DataLoader(dataset, batch_size=config.batch_size, num_workers=0, shuffle=shuffle,
                                      pin_memory=True))
        shuffle = True

    if split_set:
        return dataloaders[::-1]
    return dataloaders[0]
