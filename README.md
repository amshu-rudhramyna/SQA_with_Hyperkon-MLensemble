# Project Setup Instructions

## Environment
Ensure you have the following dependencies installed:
```bash
pip install torch torchvision xgboost scikit-learn numpy pandas tqdm PyWavelets eotdl
```

## Dataset Access (`eotdl`)
To download the HyperView2 dataset using `eotdl`, you may need to authenticate first if the dataset is not public or requires API access.

1.  **Authenticate**:
    Run `eotdl auth login` in your terminal and follow instructions.

2.  **Download**:
    If automated download fails, please download the dataset manually from [https://www.eotdl.com/datasets/HYPERVIEW2](https://www.eotdl.com/datasets/HYPERVIEW2) and stage it in `data/raw/` (create the folder if it doesn't exist).
