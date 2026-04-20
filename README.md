# TomatoLeaf-AdvancedNN

> **Advanced Neural Network Architectures for Tomato Leaf Disease Diagnosis in Precision Agriculture**
>
> Hritwik Ghosh, Irfan Sadiq Rahat, Md. Mintajur Rahman Emon, Md. Jisan Mashrafi,
> Mohammed Abdul Al Arafat Tanzin, Sachi Nandan Mohanty, Shashi Kant
>
> ***Discover Sustainability*** (Springer Nature) · Volume 6, Article 312 · 2025
>
> DOI: [10.1007/s43621-025-01149-1](https://doi.org/10.1007/s43621-025-01149-1) · Open Access

[![Paper](https://img.shields.io/badge/Paper-Discover%20Sustainability-blue)](https://doi.org/10.1007/s43621-025-01149-1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()

---

## Abstract

This study develops an advanced deep learning diagnostic system for tomato leaf diseases across **10 classes** (9 diseases + healthy) using novel neural network architectures. Multiple architectures are benchmarked with XAI-based interpretability analysis.

---

## Disease Classes (10)

| # | Class |
|---|-------|
| 0 | Bacterial Spot |
| 1 | Early Blight |
| 2 | Late Blight |
| 3 | Leaf Mold |
| 4 | Septoria Leaf Spot |
| 5 | Spider Mites |
| 6 | Target Spot |
| 7 | Mosaic Virus |
| 8 | Yellow Leaf Curl Virus |
| 9 | Healthy |

---

## Setup

```bash
git clone https://github.com/IrfanSadiqRahat/TomatoLeaf-AdvancedNN.git
cd TomatoLeaf-AdvancedNN
pip install -r requirements.txt
# Dataset: PlantVillage (tomato subset) — available on Kaggle
python train.py --data_dir data/tomato --model efficientnet_b3
python evaluate.py --checkpoint outputs/best_model.pth --xai
```

---

## Citation

```bibtex
@article{ghosh2025tomato,
  title={Advanced neural network architectures for tomato leaf disease diagnosis in precision agriculture},
  author={Ghosh, Hritwik and Rahat, Irfan Sadiq and Emon, Md. Mintajur Rahman and
          Mashrafi, Md. Jisan and Tanzin, Mohammed Abdul Al Arafat and
          Mohanty, Sachi Nandan and Kant, Shashi},
  journal={Discover Sustainability},
  volume={6},
  pages={312},
  year={2025},
  publisher={Springer Nature},
  doi={10.1007/s43621-025-01149-1}
}
```
