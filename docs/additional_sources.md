# Additional pruned LLM matrix sources

This page indexes the source matrices used alongside the original release.
It documents source checkpoints and retrieval steps. It does not redistribute
model checkpoints or publish kernel performance results.

| Group | Sparsity | Matrices | Retrieval |
|---|---|---:|---|
| Original paper: Magnitude and WANDA, four LLMs | 80%, 90% | 1,056 | Existing matrix downloader |
| Original 70% support: the same four LLMs and two methods | 70% | 528 | Already included in the same download |
| Wanda++: same four LLMs, controlled q/o subset adaptation | 80%, 90% | 528 | Exact frozen masks in GitHub Release |
| Published DeepSeek-R1-Distill-Qwen-1.5B, C4 pruning checkpoints | 70% | 140 | Download upstream checkpoint and extract patterns |
| Same DeepSeek source | 80% | 140 | Same |
| Same DeepSeek source | 90% | 140 | Same |
| **All indexed matrices** | | **2,532** | |

The authoritative addition is [`data/extended_manifest.csv`](../data/extended_manifest.csv).
Every row records its source model, exact source tensor, dimensions, NNZ,
nominal/realized sparsity, retrieval route, and available provenance hashes.
The original [`data/manifest.csv`](../data/manifest.csv) remains the authoritative
definition of the existing 1,584-file release. The two files must not be added
together: the 1,584 original rows are already present in the extended index.

## 1. Existing original collection: 1,584 patterns

The original paper subset is **1,056**, consisting of 264 source weights times
two methods times two sparsities. The existing release also includes **528**
70% support patterns, giving **1,584** downloadable matrices in total.

| Model | Hugging Face source | Source weights | Existing patterns | New Wanda++ patterns |
|---|---|---:|---:|---:|
| Llama-2-7B-hf | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 64 | 384 | 128 |
| Mistral-7B-v0.3 | [mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) | 64 | 384 | 128 |
| OPT-2.7B | [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b) | 64 | 384 | 128 |
| Qwen2.5-3B-Instruct | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 72 | 432 | 144 |

Download the existing binary patterns using the repository's current script:

```bash
pip install -r requirements.txt
python scripts/download_dataset.py --output-dir dataset
```

This command retrieves the existing **1,584 patterns**, not the added Wanda++
or DeepSeek groups. The current [`data/archives.json`](../data/archives.json)
contains the Google Drive transport parts and their SHA-256 values.
For one model, add `--model qwen25_3b` (or another key from `configs/models.json`).

Magnitude and WANDA both use matrix-global thresholds in this collection.
WANDA uses `abs(weight) * input_channel_activation_RMS` with the fixed
16-prompt calibration list already included in this repository.

## 2. Wanda++: 528 generated patterns

The method source is [Wanda++: Pruning Large Language Models via Regional
Gradients, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.224/).
Our 528 matrices are an **Eq. 4 / Algorithm 1 adaptation** to the original 264
q/o source weights, at 80% and 90% sparsity. They are not an official
author-provided dataset or an official implementation.

The adaptation uses per-weight unstructured regional gradient scores and
regional optimization, with final repruning. Calibration uses the same 16
original prompts (maximum length 128). Fixed generation settings are alpha
100, RMSprop learning rate `3e-7`, five RGS/RO iterations, four sampled prompts
per iteration, and seed `20261026`. Only the selected q/o tensors are pruned.
The index pins the original model snapshot for every generated matrix.

Download the exact frozen masks from the [Wanda++ data release](https://github.com/ryanAI115/pruned-llm-matrices/releases/tag/wandapp-patterns-20261028).

| Model | Archive | Matrices | Bytes |
|---|---|---:|---:|
| `llama2_7b` | [wandapp_llama2_7b_s80_s90.tar.gz](https://github.com/ryanAI115/pruned-llm-matrices/releases/download/wandapp-patterns-20261028/wandapp_llama2_7b_s80_s90.tar.gz) | 128 | 170621712 |
| `mistral7b_v03` | [wandapp_mistral7b_v03_s80_s90.tar.gz](https://github.com/ryanAI115/pruned-llm-matrices/releases/download/wandapp-patterns-20261028/wandapp_mistral7b_v03_s80_s90.tar.gz) | 128 | 178824941 |
| `opt_2p7b` | [wandapp_opt_2p7b_s80_s90.tar.gz](https://github.com/ryanAI115/pruned-llm-matrices/releases/download/wandapp-patterns-20261028/wandapp_opt_2p7b_s80_s90.tar.gz) | 128 | 66597749 |
| `qwen25_3b` | [wandapp_qwen25_3b_s80_s90.tar.gz](https://github.com/ryanAI115/pruned-llm-matrices/releases/download/wandapp-patterns-20261028/wandapp_qwen25_3b_s80_s90.tar.gz) | 144 | 48458719 |

For example:

```bash
curl -L --fail -o wandapp_llama2_7b_s80_s90.tar.gz \
  https://github.com/ryanAI115/pruned-llm-matrices/releases/download/wandapp-patterns-20261028/wandapp_llama2_7b_s80_s90.tar.gz
tar -xzf wandapp_llama2_7b_s80_s90.tar.gz
```

Each archive contains the binary masks, a source manifest, and a decoding README.
Archive SHA-256 values are in the release asset `SHA256SUMS.txt` and in
[`data/additional_sources.json`](../data/additional_sources.json).
The individual mask hashes are in the complete index and each archive manifest.

These masks contain row-major bits packed with NumPy's default big bit order:

```python
import numpy as np

packed = np.fromfile("MATRIX.mask.bin", dtype=np.uint8)
mask = np.unpackbits(packed, bitorder="big", count=rows * cols).reshape(rows, cols)
```

The SHA-256 of each packed mask is recorded in `packed_mask_sha256`.
The regional optimization's surviving numeric weights are not needed to
recover the unit-valued SpMV pattern and are outside this source index.

## 3. Published DeepSeek pruning checkpoints: 420 patterns

Upstream source: [AnKhanh/Deepseek-Distill-Qwen-prune](https://huggingface.co/AnKhanh/Deepseek-Distill-Qwen-prune).
The base model is [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).
The source card tags the repository as SparseGPT pruning; this collection
extracts its published zero/nonzero layout without additional pruning.

Pin the source revision to:

```text
44b090355bcdad922866a23288c829733799d71e
```

**Use the `C4/` paths below.** The upstream card's abbreviated layout omits
this prefix. Other calibration folders, model sizes, and sparsities in that
repository are outside this index.

| Sparsity | Frozen checkpoint | Matrices |
|---|---|---:|
| 70% | [C4/1.5B/sparse70/model.safetensors](https://huggingface.co/AnKhanh/Deepseek-Distill-Qwen-prune/resolve/44b090355bcdad922866a23288c829733799d71e/C4/1.5B/sparse70/model.safetensors) | 140 |
| 80% | [C4/1.5B/sparse80/model.safetensors](https://huggingface.co/AnKhanh/Deepseek-Distill-Qwen-prune/resolve/44b090355bcdad922866a23288c829733799d71e/C4/1.5B/sparse80/model.safetensors) | 140 |
| 90% | [C4/1.5B/sparse90/model.safetensors](https://huggingface.co/AnKhanh/Deepseek-Distill-Qwen-prune/resolve/44b090355bcdad922866a23288c829733799d71e/C4/1.5B/sparse90/model.safetensors) | 140 |

Each checkpoint is 3,554,214,752 bytes. Frozen checkpoint SHA-256 values and
per-tensor source hashes are recorded in the metadata/index.

Download with the Hugging Face CLI, for example for 70%:

```bash
pip install huggingface_hub
hf download AnKhanh/Deepseek-Distill-Qwen-prune \
  --revision 44b090355bcdad922866a23288c829733799d71e \
  --include 'C4/1.5B/sparse70/model.safetensors' \
  --local-dir upstream/deepseek
```

Repeat with `sparse80` and `sparse90` for the other two checkpoints, or use the
three direct links in the table. The pattern extraction helper runs on CPU:

```bash
pip install -r requirements-extraction.txt
python scripts/extract_deepseek_patterns.py \
  --checkpoint upstream/deepseek/C4/1.5B/sparse70/model.safetensors \
  --sparsity 70 --output-dir dataset/deepseek70
```

The helper reads only tensor IDs listed in the extended manifest, checks
source tensor identity and structure metadata, and writes Matrix Market
`pattern coordinate general` files. It does not run a model, prune more
weights, benchmark a kernel, or compute a numerical SpMV reference.

Each rate contains 28 decoder layers times five projections: `q_proj`,
`o_proj`, `gate_proj`, `up_proj`, and `down_proj`. Both dimensions exceed
1,000. Narrow k/v projections are outside this fixed selected set. Nominal
70/80/90% checkpoints can contain slightly more zeros, so the index preserves
the actual NNZ and realized sparsity.

## Artifact and licensing scope

The matrix artifact describes binary structure with nonzero values equal to
one. It is not a release of fine-tuned model checkpoints or surviving weight
values. Code and metadata follow this repository's MIT license; obtain
upstream checkpoints under their own model licenses and access terms.
No post-pruning quality threshold or performance-outcome filter was used to
choose rows in this public source index.
