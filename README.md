

```md
# Deep Learning Project (NumPy + PyTorch): Zinetov Alikhan and Yernur Bidollin (IT-2301)

This repository contains a 4-part deep learning project:
- **Section 1 (NumPy):** MLP from scratch (MNIST) + activation comparison + universal approximation demo
- **Section 2 (NumPy):** Optimization from scratch (SGD / RMSprop / Adam), LR schedules, gradient checking, gradient magnitude plots
- **Section 3 (NumPy):** CNN from scratch (MNIST/FashionMNIST) + pooling comparison + receptive field + confusion matrix + filters visualization
- **Section 4 (PyTorch, training loop from scratch):** Residual blocks vs plain net + transfer learning + augmentation study (Oxford-IIIT Pet: cat vs dog)

> **Note:** Training loops and gradient descent steps are implemented manually (no `model.fit()` / no high-level trainers).

---

## Project structure

```bash

dl_project/
src/
section1_mlp_numpy.py
section2_optim_numpy.py
section3_cnn_numpy.py
section4_data_torch.py
section4_train_torch.py
optim_scratch_torch.py
mlp_components.py
utils.py
section2_config.py
configs/
section2.yaml
tests/
conftest.py
test_section1_mlp.py
notebooks/
section1.ipynb
section2.ipynb
section3.ipynb
section4.ipynb
requirements.txt
README.md
```
````

---

## Environment (Windows)

### 1) Create venv
From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
````

### 2) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> If you have CUDA, install the matching CUDA build from the official PyTorch installer page.

---

## Datasets

### Section 1–3 (NumPy)

* MNIST / FashionMNIST are loaded via `torchvision` internally in `src/data.py` (or your `src/data` module).
* No manual downloads required.

### Section 4 (PyTorch)

* Uses **Oxford-IIIT Pet Dataset** (`torchvision.datasets.OxfordIIITPet`) with:

  * `split="trainval"`
  * `target_types="binary-category"` → **cat vs dog**
  * `download=True` → no manual download

Data will be downloaded into `./data/` by default.

---

## How to run (Notebooks)

Open notebooks from the project root:

```bash
jupyter notebook
```

Recommended order:

1. `notebooks/section1.ipynb`
2. `notebooks/section2.ipynb`
3. `notebooks/section3.ipynb`
4. `notebooks/section4.ipynb`

Each notebook:

* runs training,
* produces plots (loss/accuracy, LR schedules, gradient magnitudes, confusion matrices),
* and prints final metrics.

---

## Config files (Section 2)

Section 2 uses a config file:

`configs/section2.yaml`

Example:

```yaml
data:
  dataset: "MNIST"
  batch_size: 128

model:
  input_dim: 784
  hidden_dim: 128
  num_classes: 10
  activation: "relu"

train:
  optimizer: "adam"
  lr0: 0.01
  schedule: "step"
  epochs: 12
  seed: 42
```

Load it in the Section 2 notebook:

```python
from src.section2_config import load_section2_config
cfg = load_section2_config("configs/section2.yaml")
```

---

## Unit tests (pytest)


### Run tests

From the project root:

```bash
pytest -q
```

Current tests:

* `tests/test_section1_mlp.py` checks:

  * softmax correctness (row sums),
  * forward/backward shapes,
  * loss is finite,
  * one update step does not increase loss on a small batch.

---

## Key outputs required by the assignment

* **Loss/accuracy curves:** Sections 1–4
* **Activation comparison + histograms:** Section 1
* **Universal approximation (sin regression):** Section 1
* **Gradient checking:** Section 2
* **LR schedule plots:** Section 2
* **Gradient magnitude plots:** Section 2 (+ Section 4 grad flow)
* **CNN receptive field + pooling comparison:** Section 3
* **Confusion matrices:** Sections 3 (and optional for 1/4)
* **ResNet vs plain + gradient flow/stability:** Section 4
* **Transfer learning (scratch vs frozen vs fine-tune):** Section 4
* **Augmentation pipeline (≥5) + effect:** Section 4

---

## Reproducibility

Use `set_seed(...)` (if available in `src/utils.py`) or fix `numpy` seeds directly.
For PyTorch experiments, seed is stored in the `DataCfg/TrainCfg` configs.

---



### Training is slow on CPU (Section 4)

Use smaller settings:

* `img_size=96`
* `batch_size=16`
* `epochs=3`
* limit training subset if needed (optional)

---



