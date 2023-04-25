from mmseg.datasets import DATASETS
from mmseg.datasets import CustomDataset


@DATASETS.register_module(force=True)
class BarcodesDataset(CustomDataset):
    CLASSES = ('background', 'barcode')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self, **kwargs):
        super(BarcodesDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
