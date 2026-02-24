# RA-LDL: Random Anchors with Low-rank Decorrelated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc/)

Official implementation of **"Random Anchors with Low-rank Decorrelated Learning: A Minimalist Pipeline for Class-Incremental Medical Image Classification"** accepted at ICLR 2026.

## Overview

Class-incremental learning (CIL) in medical image-guided diagnosis requires models to preserve knowledge of historical disease classes while adapting to emerging categories. Pre-trained models (PTMs) with well-generalized features provide a strong foundation, yet most PTM-based CIL strategies, such as prompt tuning, task-specific adapters and model mixtures, rely on increasingly complex designs. While effective in general-domain benchmarks, these methods falter in medical imaging, where low intra-class variability and high inter-domain shifts (from scanners, protocols and institutions) make CIL particularly prone to representation collapse and domain misalignment. Under such conditions, we find that lightweight representation calibration strategies, often dismissed in general-domain CIL for their modest gains, can be remarkably effective for adapting PTMs in medical settings. To this end, we introduce Random Anchors with Low-rank Decorrelated Learning (RA-LDL), a minimalist representation-based framework that combines (a) PTM-based feature extraction with optional ViT-Adapter tuning, (b) feature calibration via frozen Random Anchor projection and a single-session-trained Low-Rank Projection (LRP), and (c) analytical closed-form decorrelated learning. The entire pipeline requires only one training session and minimal task-specific tuning, making it appealing for efficient deployment. Despite its simplicity, RA-LDL achieves consistent and substantial improvements across both general-domain and medical-specific PTMs, and outperforms recent state-of-the-art methods on four diverse medical imaging datasets. These results highlight that minimalist representation recalibration, rather than complex architectural modifications, can unlock the underexplored potential of PTMs in medical CIL. We hope this work establishes a practical and extensible foundation for future research in class-incremental image-guided diagnosis.

### Key Features

- 🔄 **Class-Incremental Learning**: Supports continuous learning of new classes without forgetting
- 🏥 **Medical Image Focus**: Specifically designed for medical image classification tasks
- ⚡ **Minimalist Pipeline**: Simple yet effective approach with minimal overhead
- 🔬 **Single Training Session**: The entire pipeline is trained only in the first session, then applied analytically to all subsequent tasks
- 🧩 **Flexible PETL Backbones**: Supports multiple parameter-efficient tuning methods (Adapter, VPT)

## Method

RA-LDL consists of three components applied on top of a frozen pre-trained ViT backbone:

1. **PTM Feature Extraction with Optional Adapter Tuning**: A pre-trained ViT (ImageNet-1K or ImageNet-21K) is used as the backbone. In the first task, a lightweight ViT-Adapter is fine-tuned via cross-entropy loss. The backbone is then frozen for all subsequent tasks.

2. **Feature Calibration via Random Anchors + Low-Rank Projection (LRP)**: The raw PTM features are projected through:
   - A fixed random matrix **W_rand** (Random Anchor projection, dimension `M`) that expands the feature space and breaks linear correlations.
   - A trainable Low-Rank Projection (LRP) consisting of a down-projection (PTM feature dim → rank 64) followed by an up-projection (rank 64 → M), trained only during the first session with GELU activations.
   - The two projections are summed: `h = ReLU(f @ W_rand) + LRP(f)`.

3. **Closed-Form Decorrelated Learning**: After feature calibration, a class-weight matrix is solved analytically (no back-propagation) using ridge-regularised least squares:
   ```
   W = (G + λI)⁻¹ Q
   ```
   where `G = HᵀH` and `Q = HᵀY` are accumulated across all tasks incrementally. The optimal ridge parameter `λ` is selected automatically via a held-out validation split.

## Project Structure

```
RA-LDL/
├── main.py                  # Entry point; reads config CSV and launches training
├── trainer.py               # Outer training loop, logging, and results saving
├── RALDL.py                 # Core RA-LDL learner (BaseLearner + Learner classes)
├── inc_net.py               # Network definitions (SimpleVitNet, CosineLinear head)
├── args/                    # Per-dataset hyperparameter configs (CSV format)
│   ├── medmnist.csv
│   ├── skin8.csv
│   ├── blood.csv
│   └── covid.csv
├── petl/                    # Parameter-efficient tuning modules
│   └── vision_transformer_adapter.py   # ViT-Adapter (AdaptFormer) implementation
└── utils/
    ├── data.py              # Dataset classes and data loading
    ├── data_manager.py      # DataManager for incremental task construction
    └── toolkit.py           # Helper utilities (accuracy, tensor ops, etc.)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/CUHK-BMEAI/RA-LDL.git
cd RA-LDL

# Install dependencies
pip install torch torchvision tqdm==4.65.0 timm==0.6.5 pandas easydict
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.4
- tqdm == 4.65.0
- timm == 0.6.5
- pandas
- easydict

A CUDA-capable GPU is strongly recommended. The default configs use GPU device index `2`; adjust the `device` field in the relevant `args/*.csv` file to match your setup (e.g., `0` for a single-GPU system).

## Dataset Preparation

We follow [ACL](https://github.com/GiantJun/CL_Pytorch/tree/main) to use the same data index lists for training. After downloading the datasets, update the paths inside `utils/data.py` to point to your local data directories.

| Dataset | Classes | Download |
|---------|---------|----------|
| MedMNIST (PathMNIST, BloodMNIST, OrganMNIST-Axial, TissueMNIST) | 36 total | [medmnist.com](https://medmnist.com/) |
| Skin8 (ISIC 2019) | 8 | [Kaggle](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification) |
| Blood Cell (PBC) | 8 | [Kaggle](https://www.kaggle.com/datasets/kylewang1999/pbc-dataset) |
| COVID-19 | varies | [Kaggle](https://www.kaggle.com/datasets/mustai/continual-learning-of-covid19/data) |

### Path Configuration

Open `utils/data.py` and update the `download_data()` method of each dataset class to point to your local paths:

```python
# medmnist — set src_dir to the folder containing *.npz files
src_dir = "/your/path/to/medmnist"

# skin8 — set base_dir and the .txt index files
base_dir = "/your/path/to/skin8"

# blood — set base_dir to the PBC dataset folder
base_dir = "/your/path/to/PBC_data"

# covid — set train_dir / test_dir to the ImageFolder-style directories
train_dir = "/your/path/to/covid/train"
test_dir  = "/your/path/to/covid/test"
```

## Training

Run training by specifying the dataset with the `-d` flag:

```bash
python main.py -d skin8
# -d options: 'medmnist', 'skin8', 'blood', 'covid'
```

Training proceeds as follows:
1. **Task 0**: The ViT-Adapter backbone is fine-tuned for `tuned_epoch` epochs using SGD + cosine annealing. The model checkpoint is saved as `model_task0.pth`.
2. **Task 0 (continued)**: The LRP projection is trained for another `tuned_epoch` epochs (backbone frozen).
3. **All tasks**: The classifier weights are solved analytically via ridge regression — no gradient updates are needed.

Logs are written to `logs/<model_name>/<dataset>/<init_cls>/<increment>/` and results (accuracy curves and per-class predictions) are saved to `results/`.

## Configuration

Each dataset has a corresponding CSV in `args/` that controls all hyperparameters. The columns are:

| Parameter | Description |
|-----------|-------------|
| `dataset` | Dataset name (`medmnist`, `skin8`, `blood`, `covid`) |
| `shuffle` | Whether to shuffle class order |
| `init_cls` | Number of classes in the first task |
| `increment` | Number of new classes added per subsequent task |
| `model_name` | PETL method: `adapter` or `vpt` (see Supported Backbones table) |
| `convnet_type` | Backbone identifier (e.g. `pretrained_vit_b16_224_adapter`) |
| `device` | CUDA device index (e.g. `2`) |
| `seed` | Random seed |
| `batch_size` | Training batch size |
| `tuned_epoch` | Number of epochs for first-task PETL tuning and LRP training |
| `body_lr` | Learning rate for backbone / LRP training |
| `weight_decay` | Weight decay for SGD optimiser |
| `min_lr` | Minimum LR for cosine annealing scheduler |
| `use_RP` | Whether to enable Random Anchor + LRP feature calibration |
| `M` | Random Anchor projection dimension (set to `0` to disable) |
| `use_input_norm` | Whether to apply ImageNet normalisation to inputs |

### Supported Backbones and PETL Methods

| `convnet_type` | `model_name` | Description |
|----------------|--------------|-------------|
| `pretrained_vit_b16_224_adapter` | `adapter` | ViT-B/16 (IN-1K) + AdaptFormer adapter |
| `pretrained_vit_b16_224_in21k_adapter` | `adapter` | ViT-B/16 (IN-21K) + AdaptFormer adapter |
| `pretrained_vit_b16_224_vpt` | `vpt` | ViT-B/16 (IN-1K) + Visual Prompt Tuning |
| `pretrained_vit_b16_224_in21k_vpt` | `vpt` | ViT-B/16 (IN-21K) + Visual Prompt Tuning |
| `vit_base_patch32_224_clip_laion2b` | *(any non-PETL value)* | ViT-B/32 CLIP (LAION-2B), no PETL — backbone features used directly |

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@inproceedings{raldl2026,
  title={Random Anchors with Low-rank Decorrelated Learning: A Minimalist Pipeline for Class-Incremental Medical Image Classification},
  author={Wu*, Xinyao and Xu*, Zhe, and Tong, Raymond Kai-yu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
