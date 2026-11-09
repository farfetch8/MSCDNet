import importlib
from os import path as osp

from basicsr.utils import scandir

data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
_dataset_modules = [importlib.import_module(f'data.{file_name}') for file_name in dataset_filenames]
