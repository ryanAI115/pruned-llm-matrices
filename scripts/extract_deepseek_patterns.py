"""Extract the fixed published DeepSeek patterns; no GPU, pruning or SpMV."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path,
                        help='Frozen C4/1.5B/sparseXX/model.safetensors file')
    parser.add_argument('--sparsity', required=True, type=int, choices=(70, 80, 90))
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/extended_manifest.csv')
    parser.add_argument('--layers', nargs='+', type=int,
                        help='Optional small subset of layers; omitted means all 28')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    import numpy as np
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    with args.manifest.open(encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['cohort'] == f'deepseek_c4_s{args.sparsity}']
    if len(rows) != 140:
        raise RuntimeError('The source manifest must contain exactly 140 tensors per rate')
    if args.layers is not None:
        if not set(args.layers).issubset(range(28)):
            raise ValueError('Layer IDs must be in 0..27')
        rows = [r for r in rows if int(r['source_tensor'].split('.')[2]) in args.layers]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = [args.output_dir / (r['matrix'] + '.mtx') for r in rows]
    if any(p.exists() for p in targets):
        raise FileExistsError('Use a new output directory; existing matrix files are not overwritten')
    output = []
    with safe_open(str(args.checkpoint), framework='pt', device='cpu') as reader:
        available = set(reader.keys())
        for row, target in zip(rows, targets):
            key = row['source_tensor']
            if key not in available:
                raise KeyError(f'Missing source tensor: {key}')
            weight = reader.get_tensor(key).contiguous()
            shape = (int(row['rows']), int(row['cols']))
            if tuple(weight.shape) != shape:
                raise ValueError(f'Source tensor shape differs from frozen index: {key}')
            digest = hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
            if digest != row['source_tensor_sha256']:
                raise ValueError(f'Source tensor bytes differ from frozen checkpoint: {key}')
            mask = (weight != 0).numpy()
            nnz = int(np.count_nonzero(mask))
            if nnz != int(row['nnz']):
                raise ValueError(f'Source tensor NNZ differs from frozen index: {key}')
            temp = target.with_suffix('.mtx.partial')
            with temp.open('x', encoding='ascii', newline='\n') as f:
                f.write('%%MatrixMarket matrix coordinate pattern general\n')
                f.write('% Source tensor listed in data/extended_manifest.csv; values implicit one\n')
                f.write(f'{shape[0]} {shape[1]} {nnz}\n')
                for start in range(0, shape[0], 64):
                    ri, ci = np.nonzero(mask[start:start + 64])
                    f.writelines(f'{int(i) + start + 1} {int(j) + 1}\n' for i, j in zip(ri, ci))
            temp.rename(target)
            record = dict(row, output_file=target.name,
                          packed_mask_sha256=hashlib.sha256(np.packbits(mask, axis=None).tobytes()).hexdigest())
            output.append(record)
            print(f'EXPORTED {len(output)}/{len(rows)} {target.name}', flush=True)
            del weight, mask
    index = args.output_dir / 'extraction_receipt.json'
    with index.open('x', encoding='utf-8') as f:
        json.dump(dict(matrix_count=len(output), source_revision=rows[0]['source_revision'],
                       sparsity=args.sparsity, entries=output), f, indent=2)


if __name__ == '__main__':
    main()
