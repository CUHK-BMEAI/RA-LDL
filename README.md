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

## Installation

```bash
# Clone the repository
git clone https://github.com/CUHK-BMEAI/RA-LDL.git
cd RA-LDL
```

## Requirements

- Python >= 3.8  
- PyTorch >= 1.4 
- tqdm == 4.65.0
- timm == 0.6.5

## Training

```bash
$ python main.py -d skin8
- for -d choose from 'medmnist', 'skin8', 'blood', 'covid'
```

## Dataset

We follow [ACL](https://github.com/GiantJun/CL_Pytorch/tree/main) to use the same data index_list for training. Please modify `utils/data.py` to point to your local data path. To download datasets, links: [skin8](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification), 

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@inproceedings{raldl2026,
  title={Random Anchors with Low-rank Decorrelated Learning: A Minimalist Pipeline for Class-Incremental Medical Image Classification},
  author={Xinyao Wu*, Zhe Xu*, Raymond Kai-yu Tong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
