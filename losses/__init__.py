import importlib
from os import path as osp

from basicsr.utils import scandir

loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(loss_folder) if v.endswith('_loss.py')]
_model_modules = [importlib.import_module(f'losses.{file_name}') for file_name in loss_filenames]
