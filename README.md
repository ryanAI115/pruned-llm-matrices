# All-Layer Pruned LLM Matrix Collection

This anonymous artifact contains code and data for reproducing a collection of
1,584 sparse attention-projection matrices derived from four open-weight LLMs.
The released matrices cover every transformer block, both square attention
projections used by the study, two pruning scores, and three sparsity levels.

The Matrix Market files contain **binary sparsity patterns only**. They do not
contain the surviving weight values, prompts, model outputs, or checkpoints.
They are intended for sparse-kernel, storage-format, feature-extraction, and
matrix-structure research. They do not establish post-pruning language-model
quality.

## Dataset at a glance

| Model | Hugging Face ID | Layers | Projections | Matrices |
|---|---|---:|---|---:|
| Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` | 36 | `q_proj`, `o_proj` | 432 |
| OPT-2.7B | `facebook/opt-2.7b` | 32 | `q_proj`, `out_proj` | 384 |
| Llama-2-7B-hf | `meta-llama/Llama-2-7b-hf` | 32 | `q_proj`, `o_proj` | 384 |
| Mistral-7B-v0.3 | `mistralai/Mistral-7B-v0.3` | 32 | `q_proj`, `o_proj` | 384 |

Each of the 264 source weights is pruned with:

- matrix-global magnitude pruning at 70%, 80%, and 90% sparsity; and
- matrix-global WANDA-score pruning at 70%, 80%, and 90% sparsity.

The WANDA score is `abs(weight) * input_channel_activation_RMS`. Activation RMS
is measured with the fixed 16-prompt calibration list in
[`scripts/collect_activation_scales.py`](scripts/collect_activation_scales.py).

## Download the matrices

The data are hosted outside GitHub because the uncompressed collection is about
31 GiB. Download links and checksums are recorded in
[`data/archives.json`](data/archives.json). To download and verify all parts:

```bash
python scripts/download_dataset.py --output-dir dataset
```

To download one model only:

```bash
python scripts/download_dataset.py --model qwen25_3b --output-dir dataset
```

The Drive files are transport-sized parts. The downloader joins each model's
parts into one archive, verifies both part and archive SHA-256 values, and then
extracts the `.mtx` files under `dataset/matrices/`.

For a manual download, concatenate a model's parts in numeric order first.
Extraction then requires `zstd` and `tar`:

```bash
cat pruned_llm_qwen25_3b.tar.zst.chunk-* > pruned_llm_qwen25_3b.tar.zst
tar --use-compress-program=unzstd -xf pruned_llm_qwen25_3b.tar.zst
```

## Use the matrices

Install the lightweight data-use dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Load a pattern and run a sample SpMV:

```bash
python examples/use_matrix.py \
  dataset/matrices/qwen25_3b_l00_q_proj_wanda_s80.mtx
```

In Python:

```python
from scipy.io import mmread

A = mmread("dataset/matrices/qwen25_3b_l00_q_proj_wanda_s80.mtx").tocsr()
y = A @ x
```

Matrix Market `pattern` entries have implicit value 1. If a benchmark needs
numeric values, assign a supported dtype after loading, for example
`A = A.astype("float32")`. These files reproduce sparsity layouts, not pruned
model inference.

Validate coverage against the release manifest:

```bash
python scripts/verify_dataset.py dataset/matrices
python scripts/verify_dataset.py dataset/matrices --deep
```

The default check verifies names and coverage. `--deep` also loads every matrix,
checks shape/NNZ, and recomputes the packed-mask digest; it is much slower.

## Reproduce the collection from model checkpoints

### 1. Install generation dependencies

```bash
pip install -r requirements-generation.txt
```

Generation was performed one model at a time. A CUDA GPU with enough memory is
recommended for activation collection. Llama 2 is gated on Hugging Face; accept
its license and authenticate with `huggingface-cli login` before downloading.

### 2. Download model snapshots

```bash
python scripts/download_models.py --output-dir models
```

Use `--model qwen25_3b` to download only one model. The downloader delegates to
`huggingface_hub` and therefore honors `HF_TOKEN` and the normal Hugging Face
cache configuration.

### 3. Collect activation RMS

```bash
python scripts/collect_activation_scales.py \
  --model models/qwen25_3b \
  --model-key qwen25_3b \
  --output activations/qwen25_3b.npz
```

Repeat for the other three keys. This runs the fixed 16 prompts once with eager
attention and records one RMS vector per selected projection.

### 4. Extract weights, prune, and write Matrix Market files

```bash
python scripts/generate_matrices.py \
  --model models/qwen25_3b \
  --model-key qwen25_3b \
  --model-label Qwen2.5-3B-Instruct \
  --activation-scales activations/qwen25_3b.npz \
  --output-dir reproduced/qwen25_3b
```

With no `--layers` argument, the script discovers and requires every layer. Use
`--layers 0 1 2` only for a small subset. It writes `matrices/*.mtx`,
`manifest.csv`, and `metadata.json`. Existing completed output is not
overwritten.

### 5. Verify a reproduced directory

```bash
python scripts/verify_dataset.py reproduced/qwen25_3b/matrices \
  --manifest reproduced/qwen25_3b/manifest.csv --deep
```

## File naming and manifest

Names follow:

```text
<model_key>_l<layer>_<projection>_<score>_s<sparsity>.mtx
```

Example: `opt_2p7b_l21_q_proj_wanda_s70.mtx`.

[`data/manifest.csv`](data/manifest.csv) records the source model, source tensor,
layer, projection, shape, pruning method, target and realized sparsity, NNZ, and
a 16-hex packed-mask digest for every matrix. The manifest is the authoritative
definition of the 1,584-file release.

## Reproducibility notes

- The pruning threshold is matrix-global, including for WANDA scores.
- The fixed prompts calibrate sparsity scores only; they are not training data.
- Model weights are converted to float32 before scoring.
- Exactly `round(numel * (1 - sparsity))` entries are retained per matrix.
- Ties at the pruning boundary follow NumPy `argpartition`; use the pinned
  dependency versions for byte-identical regeneration.
- The matrices contain structure only, so original model licenses still govern
  checkpoint acquisition while the released files do not redistribute weights.

## License

Code and artifact metadata are released under the MIT License. Upstream model
checkpoints remain subject to their respective licenses and access terms.
