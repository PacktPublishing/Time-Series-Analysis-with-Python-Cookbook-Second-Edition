# Time Series Analysis with Python Cookbook, Second Edition

---

## Repository Structure

```
├── code/           # Jupyter notebooks for each chapter
│   ├── Ch0/        # Setting up your Python environment (GitHub only)
│   ├── Ch1-Ch14/   # Main book chapters
│   ├── Bonus_Ch15/ # Probabilistic Forecasting (GitHub only)
│   └── Bonus_Ch16/ # Frequency Domain Analysis (GitHub only)
├── datasets/       # Data files used in recipes
└── README.md
```

---

## Getting Started

Each chapter folder contains its own environment configuration. Choose **one** of the following methods to set up your environment.

### Option 1: Using `uv` (Recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package manager that handles both virtual environments and dependencies. Each chapter includes:

- **`pyproject.toml`**: Defines the project metadata and dependencies
- **`uv.lock`**: Lock file ensuring reproducible installs

**Setup steps:**

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to a chapter folder
cd code/Ch1

# Create environment and install all dependencies
uv sync

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Launch JupyterLab
jupyter lab
```

**Adding new packages:**

```bash
uv add pandas matplotlib  # Adds to pyproject.toml and installs
```

---

### Option 2: Using `pip` and `venv`

Each chapter folder includes a `requirements.txt` file for pip-based installation.

```bash
# Navigate to a chapter folder
cd code/Ch1

# Create a virtual environment
python -m venv .venv

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Launch JupyterLab
pip install jupyterlab
jupyter lab
```

---

### Option 3: Using `conda`

You can also use Conda to create an environment from the requirements file.

```bash
# Navigate to a chapter folder
cd code/Ch1

# Create a new conda environment with Python 3.12
conda create -n ch1 python=3.12 -y

# Activate the environment
conda activate ch1

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Launch JupyterLab
conda install -c conda-forge jupyterlab -y
jupyter lab
```

> **Note:** Some chapters also include an `environment.yml` file for direct conda environment creation:
> ```bash
> conda env create -f environment.yml
> conda activate <env-name>
> ```

---

## Chapter Overview

| Chapter | Title | Bonus Recipe (GitHub) |
|---------|-------|-----------------------|
| **Ch0** | Setting Up Your Python Environment *(GitHub only)* | |
| **Ch1** | Reading Time Series Data from Files | 1 Recipe |
| **Ch2** | Reading Time Series Data from Databases | 1 Recipe |
| **Ch3** | Persisting Time Series Data to Files | 1 Recipe |
| **Ch4** | Persisting Time Series Data to Databases | 1 Recipe |
| **Ch5** | Working with Date and Time in Python | |
| **Ch6** | Handling Missing Data | |
| **Ch7** | Outlier Detection Using Statistical Methods | |
| **Ch8** | Exploratory Data Analysis and Diagnosis | 2 Recipes |
| **Ch9** | Building Univariate Time Series Models Using Statistical Methods | |
| **Ch10** | Additional Statistical Modeling Techniques for Time Series | |
| **Ch11** | Forecasting Using Supervised Machine Learning | |
| **Ch12** | Deep Learning for Time Series Forecasting | |
| **Ch13** | Outlier Detection Using Unsupervised Machine Learning | |
| **Ch14** | Advanced Techniques for Complex Time Series | |
| **Bonus Ch15** | Probabilistic Forecasting *(GitHub only)* | |
| **Bonus Ch16** | Analyzing Time Series in the Frequency Domain *(GitHub only)* | |

---

## Requirements

- Python 3.10 or higher (3.12 recommended)
- See individual chapter `requirements.txt` or `pyproject.toml` for specific dependencies

---
