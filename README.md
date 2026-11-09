# MSCDNet

MSCDNet 用于图像曝光校正。本仓库提供推理和训练源码，以及 MSEC、LCDP 两套预训练权重。

## 曝光校正

使用 Python 3.12 安装推理依赖。

```bash
python -m pip install -r requirements.txt
```

使用 MSEC 权重处理单张图片或一个目录。输出为 PNG，保持输入尺寸。

```bash
python inference.py --input "images" --output "results" --device cpu
```

使用 LCDP 权重时添加 `--model lcdp`。

```bash
python inference.py --input "image.jpg" --output "results/lcdp" --model lcdp --device cpu
```

`--recursive` 处理输入目录的子目录。程序默认自动选择可用的 CUDA 或 CPU，并优先加载检查点中的 `params_ema`。

| 模型选项 | 权重 |
| --- | --- |
| `msec` | `experiments/msec_net.pth` |
| `lcdp` | `experiments/lcdp_net.pth` |

## 训练与测试

`train.py` 和 `test.py` 使用 BasicSR，需另行配置兼容的 BasicSR、Torchvision、pytorch-msssim 和 PyYAML 环境。使用自己的数据集，并在相应配置中设置 `dataroot_lq`、`dataroot_gt` 和 `num_gpu`。训练配置默认从头训练。

```bash
python train.py -opt options/train/train_msec.yml
python test.py -opt options/test/test_msec.yml
```

LCDP 使用对应的 `train_lcdp.yml` 和 `test_lcdp.yml` 配置。
