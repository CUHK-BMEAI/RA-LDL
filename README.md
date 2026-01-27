# RA-LDL: Random Anchors with Low-rank Decorrelated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc/)

Official implementation of **"Random Anchors with Low-rank Decorrelated Learning: A Minimalist Pipeline for Class-Incremental Medical Image Classification"** accepted at ICLR 2026.

## Overview

RA-LDL presents a minimalist yet effective pipeline for class-incremental learning in medical image classification. The method leverages random anchors combined with low-rank decorrelated learning to address the challenge of catastrophic forgetting while maintaining computational efficiency.

### Key Features

- 🎯 **Random Anchors**: Novel anchoring mechanism for stable incremental learning
- 🔧 **Low-rank Decorrelated Learning**: Efficient feature representation with reduced redundancy
- 🏥 **Medical Image Focus**: Specifically designed for medical image classification tasks
- ⚡ **Minimalist Pipeline**: Simple yet effective approach with minimal overhead
- 🔄 **Class-Incremental Learning**: Supports continuous learning of new classes without forgetting

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

### Training

```bash
$ python main.py -d skin8
- for -d choose from 'medmnist', 'skin8', 'blood', 'covid'
```

## Dataset

We follow [ACL](https://github.com/GiantJun/CL_Pytorch/tree/main) to use the same data index_list for training.

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@inproceedings{raldl2026,
  title={Random Anchors with Low-rank Decorrelated Learning: A Minimalist Pipeline for Class-Incremental Medical Image Classification},
  author={To be updated},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and discussions, please contact:
- CUHK Biomedical AI Lab: [https://github.com/CUHK-BMEAI](https://github.com/CUHK-BMEAI)

## Acknowledgments

This work is conducted at the Chinese University of Hong Kong (CUHK) Biomedical AI Lab.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: This repository is under active development. Code and documentation will be updated regularly as we prepare for the official release.
