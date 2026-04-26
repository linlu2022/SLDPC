# SLDPC

Official codebase for:

**SLDPC: Slide-Level Dual-Prompt Collaboration for Few-Shot Whole Slide Image Classification**

Published in *Computerized Medical Imaging and Graphics* (Elsevier), 2026.
&nbsp;·&nbsp; DOI: [10.1016/j.compmedimag.2026.102768](https://doi.org/10.1016/j.compmedimag.2026.102768)

---

## What this repository provides

A reference implementation of SLDPC on the slide-level VLM backbone, including:

- **Stage-1**: Continuous Prompt Initialization (CPI) trained with cross-entropy.
- **Stage-2**: Dynamic Hard Negative Sampling (DHNO) + symmetric InfoNCE (SICL).
- **Inference**: Weighted fusion (WFM) of base prompt `P` and task-specific prompt `P'`.
- A **zero-shot baseline** integrated into the same pipeline.

---

## Repository layout

```text
SLDPC/
├── sldpc/
│   ├── backbones/
│   │   ├── registry.py            
│   │   └── titan/                  # PromptedTitan, TitanPromptLearner, encode_text
│   ├── core/
│   │   ├── prompt_learner_base.py  # P / P' / fused
│   │   ├── losses.py               # CE + symmetric InfoNCE
│   │   ├── negative_sampler.py     # DHNO
│   │   └── fusion.py               # WFM (omega-weighted ctx mix)
│   ├── data/
│   │   └── slide_feature_dataset.py
│   ├── trainers/
│   │   ├── base_trainer.py
│   │   ├── stage1_trainer.py
│   │   ├── stage2_trainer.py
│   │   └── titan_pipeline.py       # main entry — argparse + Stage1 + Stage2 + ZS
│   └── utils/
│       ├── metrics.py              # ACC / Macro-F1 / AUC
│       ├── seed.py
│       ├── run_logging.py
│       └── zero_shot.py
├── scripts/
│   └── train_titan.py              # thin CLI wrapper
├── configs/
│   ├── backbones/titan.yaml
│   └── datasets/
│       ├── ubc_ocean.yaml
│       ├── tcga_nsclc.yaml
│       ├── tcga_rcc.yaml
│       ├── tcga_ot.yaml
│       └── data_io_config.yaml     # CSV schema for split parsing
├── data/
│   └── datasets/
│       ├── ubc_ocean/              # CSV splits only (no features, no images)
│       ├── tcga_nsclc/
│       ├── tcga_rcc/
│       ├── tcga_ot/
│       └── zero_shot_prompts/      # per-dataset class synonyms (yaml)
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

> **Versions are pinned.** TITAN's `trust_remote_code` modeling files
> were written against `transformers==4.46.0` and `torch==2.0.1+cu117`,
> and the entire pinned stack in `requirements.txt` mirrors the
> environment that produced the paper's numbers. Please install
> exactly those versions — newer `transformers` (≥ 4.50) breaks
> TITAN's class hierarchy at load time, and other PyTorch versions
> may produce subtly different numbers.

`requirements.txt` installs the GPU build of PyTorch from PyTorch's
CUDA 11.7 wheel index. The cu117 wheels are forward-compatible with
newer NVIDIA drivers (CUDA 11.8 / 12.x).

**Recommended: conda + Python 3.9**

```bash
conda create -n sldpc python=3.9 -y
conda activate sldpc

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**Alternative: venv (Linux / macOS)**

```bash
python3.9 -m venv .venv          # use Python 3.9, 3.10, or 3.11
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**Verify the GPU is picked up**

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

**Tested**: Python 3.9.22, PyTorch 2.0.1+cu117, transformers 4.46.0,
timm 1.0.3, einops 0.6.1.

**CPU-only / no NVIDIA GPU**

If you do not have an NVIDIA GPU (e.g. on a Mac, or a CPU-only Linux
box), the cu117 wheels will fail to resolve. Edit `requirements.txt`
and replace the GPU torch lines with their plain CPU equivalents:

```text
# Comment out or remove these:
# --index-url https://download.pytorch.org/whl/cu117
# --extra-index-url https://pypi.org/simple
# torch==2.0.1+cu117
# torchvision==0.15.2+cu117

# Add these instead:
torch==2.0.1
torchvision==0.15.2
```

---

## Authenticate for TITAN

This codebase loads the TITAN backbone directly from the official
Hugging Face model hub at <https://huggingface.co/MahmoodLab/TITAN>.
Two one-time steps:

1. Request access on the Hugging Face model page (the upstream gated
   distribution). Approval is automatic for academic use after agreeing
   to MahmoodLab's terms.
2. Authenticate locally so `transformers` can pull the weights:

   ```bash
   huggingface-cli login
   ```

If you are working offline, point the CLI at a previously downloaded
local snapshot instead:

```bash
python scripts/train_titan.py \
    --hf-model-id /path/to/local_titan_snapshot \
    --hf-local-files-only \
    ...
```

---

## Prepare slide-level features

This codebase consumes **pre-extracted slide-level features**, not raw WSIs.
Each dataset is loaded from a single `.pkl` file with two keys:

| Key          | Type                         | Description                                                  |
|--------------|------------------------------|--------------------------------------------------------------|
| `embeddings` | `np.ndarray (N, 768)`        | Slide-level features from TITAN's slide encoder              |
| `filenames`  | `list[str]` of length `N`    | Slide identifiers, matching the `slide_id` column of the CSV |

There are two paths to obtaining this file, depending on your dataset.

### Option A — TCGA features (use the official MahmoodLab release)

For TCGA-OT, MahmoodLab has already published
the full TCGA TITAN slide features as a single `.pkl` on the TITAN
Hugging Face hub.

```python
from huggingface_hub import hf_hub_download

slide_feature_path = hf_hub_download(
    "MahmoodLab/TITAN",
    filename="TCGA_TITAN_features.pkl",
)
print(slide_feature_path)
# /root/.cache/huggingface/hub/.../TCGA_TITAN_features.pkl
```

### Option B — Extract features yourself 

Follow the standard MahmoodLab feature-extraction pipeline. There are
two equivalent ways, both documented in the
[TITAN README](https://github.com/mahmoodlab/TITAN):

- **End-to-end via TRIDENT** ([`mahmoodlab/Trident`](https://github.com/mahmoodlab/Trident)):
  WSI tiling, CONCHv1.5 patch features, and TITAN slide aggregation in
  a single pipeline. 
  
- **Manual two-step** (CLAM + TITAN):
  use [`mahmoodlab/CLAM`](https://github.com/mahmoodlab/CLAM)'s
  `extract_features_fp.py` with `--model_name conch_v1_5` to produce
  patch features and coordinates, then call TITAN's
  `encode_slide_from_patch_features(features, coords, patch_size_lv0)`
  to obtain one slide embedding per WSI. Set `patch_size_lv0=512` for
  20× tiling (`1024` for 40×). 

Either way, pack the resulting `(slide_id, embedding)` pairs into a
single `.pkl` matching the schema above.

---

## Run the main experiment

A complete TCGA-OT run, 16-shot, 5 random seeds, using the default
20/80 stratified split generated per seed:

```bash
python scripts/train_titan.py \
    --dataset tcga_ot --few-shot-k 16 \
    --features-path /path/to/TCGA_TITAN_features.pkl \
    --n-seeds 5 --topk 4 --omega 0.8 \
    --output-dir runs/main/tcga_ot_k16
```

`--features-path` should point at the `.pkl` you obtained in §3

Without `--fixed-train-csv` / `--fixed-test-csv`, the pipeline reads
the slide universe from `data/datasets/tcga_ot/tcga-ot_all.csv` and
generates a fresh stratified 20/80 train-pool / test split for each
seed. The K-shot training subset is then sampled from the train pool.
Both the splits and the K-shot indices are recorded under
`runs/.../seed_<seed>/splits/` so that any single seed is fully
reproducible from the saved CSVs alone.

Outputs:

- Per seed: `runs/.../seed_<seed>/final_report.json` with `zero_shot`,
  `stage1`, and `stage2` blocks.
- Aggregated: `runs/.../seed_summary.json` with mean ± std across seeds.

For the full list of available CLI flags (loss family, sampling
strategy, evaluation mode, fusion weight, prompt length, etc.), run:

```bash
python scripts/train_titan.py --help
```

---

## License

Code in this repository is released under the [Apache License 2.0](LICENSE).

A few upstream components have their own terms which are worth
knowing about when you build on this work:

- **TITAN weights** (`MahmoodLab/TITAN`) are released under CC-BY-NC-ND 4.0
  on Hugging Face. They are loaded directly from the official model
  hub at runtime (see §2).
- **CONCH weights** (`MahmoodLab/CONCH`) are also released by MahmoodLab
  on Hugging Face and are used by the upstream feature-extraction
  pipeline.
- **TCGA / UBC-OCEAN data** are hosted by the TCGA consortium and the
  UBC-OCEAN Kaggle competition respectively. The CSVs in
  `data/datasets/` carry only the public slide identifiers and class
  labels needed to reproduce the paper's splits.

If you publish features derived from TITAN or CONCH, please carry
forward the upstream attribution and licensing as those projects
request.

---

## Acknowledgements

### Acknowledgements

This project is built on top of several outstanding open-source repositories:

- **TITAN** (Ding et al., 2025): A slide-level multimodal foundation model.
- **CONCH** (Lu et al., 2024): Patch-level vision-language pretraining.
- **TRIDENT** (Zhang et al., 2025): A pathology data processing pipeline.

We thank the authors of these works for their valuable contributions and for making their code and models available to the community.

---

##  Citation

If you find our work useful in your research or if you use parts of this code please consider citing our paper:

```
@article{yuan2026sldpc,
  title={SLDPC: Slide-Level Dual-Prompt Collaboration for few-shot whole slide image classification},
  author={Yuan, Lulin and Zheng, Yifeng and Liu, Weiqiang and Zhao, Hong and Zhang, Wenjie and Wei, Baoya and Chen, Liming},
  journal={Computerized Medical Imaging and Graphics},
  pages={102768},
  year={2026},
  publisher={Elsevier}
}
```

