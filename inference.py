"""Correct image exposure with the bundled MSCDNet pretrained checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from mscdnet import MSCDNet


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_model(checkpoint_path: Path, device: torch.device, state_key: str):
    """Load only tensor weights and reject incomplete or incompatible states."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Checkpoint must contain a state dictionary.')
    if state_key == 'auto':
        selected_key = next((key for key in ('params_ema', 'params') if key in checkpoint), None)
        state = checkpoint[selected_key] if selected_key else checkpoint
    else:
        if state_key not in checkpoint:
            raise ValueError(f'Checkpoint does not contain {state_key!r}.')
        selected_key = state_key
        state = checkpoint[state_key]
    if not isinstance(state, Mapping) or not state:
        raise ValueError('Checkpoint state dictionary is empty or invalid.')
    normalized = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError('State dictionary must map parameter names to tensors.')
        name = key.removeprefix('module.')
        if name in normalized:
            raise ValueError(f'Duplicate parameter name after normalization: {name}')
        normalized[name] = value
    model = MSCDNet(input_resolution=256, n_vertices_3d=17)
    model.load_state_dict(normalized, strict=True)
    model.to(device).eval()
    return model, selected_key or 'raw state dictionary'


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def correct_image(model, bgr: np.ndarray, device: torch.device) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).div_(255.0)
    with torch.inference_mode():
        output = model(tensor)
    if output.shape != tensor.shape:
        raise ValueError(f'Output shape {tuple(output.shape)} differs from input {tuple(tensor.shape)}.')
    if not torch.isfinite(output).all().item():
        raise ValueError('Model output contains NaN or infinity.')
    result = output[0].clamp(0, 1).mul(255).round().to(torch.uint8)
    result = result.permute(1, 2, 0).contiguous().cpu().numpy()
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def collect_jobs(input_path: Path, output_dir: Path, recursive: bool):
    if input_path.is_file():
        paths = [input_path] if input_path.suffix.lower() in IMAGE_SUFFIXES else []
        input_root = input_path.parent
    elif input_path.is_dir():
        input_root = input_path
        if output_dir == input_root or input_root in output_dir.parents:
            raise ValueError('Output directory must be outside the input directory.')
        candidates = input_path.rglob('*') if recursive else input_path.iterdir()
        paths = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise FileNotFoundError(f'Input does not exist: {input_path}')
    if not paths:
        raise ValueError(f'No supported images found: {input_path}')
    jobs = [(p, output_dir / p.relative_to(input_root).with_suffix('.png')) for p in paths]
    destinations = set()
    for source, destination in jobs:
        key = str(destination).casefold()
        if key in destinations:
            raise ValueError(f'Output filename collision: {destination}. Rename inputs with the same stem.')
        if destination.resolve() == source.resolve():
            raise ValueError(f'Output would overwrite its input: {source}')
        destinations.add(key)
    return jobs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='MSCDNet image exposure correction')
    parser.add_argument('--input', type=Path, required=True, help='Image or directory')
    parser.add_argument('--output', type=Path, default=ROOT / 'results', help='Destination directory')
    parser.add_argument('--model', choices=('msec', 'lcdp'), default='msec', help='Bundled pretrained model')
    parser.add_argument('--checkpoint', type=Path, help='Custom checkpoint, overrides --model')
    parser.add_argument('--state-key', choices=('auto', 'params_ema', 'params'), default='auto')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--recursive', action='store_true', help='Process subdirectories')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing result files')
    parser.add_argument('--sharpen', action='store_true', help='Optional sharpening, disabled by default')
    args = parser.parse_args(argv)

    try:
        device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        device = torch.device(device_name)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('Supported devices are cpu and cuda.')
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable. Use --device cpu or install a CUDA-enabled PyTorch build.')
        checkpoint = args.checkpoint or ROOT / 'experiments' / f'{args.model}_net.pth'
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
        jobs = collect_jobs(args.input.expanduser().resolve(), args.output.expanduser().resolve(), args.recursive)
        if not args.overwrite:
            existing = next((destination for _, destination in jobs if destination.exists()), None)
            if existing:
                raise FileExistsError(f'Result already exists: {existing}. Use --overwrite or another output directory.')
        model, state_key = load_model(checkpoint, device, args.state_key)
        print(f'Device: {device}; checkpoint: {checkpoint.name}; state: {state_key}', flush=True)
        print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}; images: {len(jobs)}', flush=True)
        failed = 0
        for index, (source, destination) in enumerate(jobs, 1):
            try:
                image = read_image(source)
                result = correct_image(model, image, device)
                if args.sharpen:
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                    result = cv2.filter2D(result, -1, kernel)
                ok, encoded = cv2.imencode('.png', result)
                if not ok:
                    raise OSError(f'Cannot encode output: {destination}')
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open('wb' if args.overwrite else 'xb') as stream:
                    stream.write(encoded.tobytes())
                print(f'[{index}/{len(jobs)}] Saved {destination} ({result.shape[1]}x{result.shape[0]})', flush=True)
            except (OSError, ValueError, RuntimeError, cv2.error) as error:
                failed += 1
                print(f'[{index}/{len(jobs)}] Failed {source}: {error}', file=sys.stderr, flush=True)
        print(f'Finished: {len(jobs) - failed} succeeded, {failed} failed.', flush=True)
        return 1 if failed else 0
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        print(f'Error: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
