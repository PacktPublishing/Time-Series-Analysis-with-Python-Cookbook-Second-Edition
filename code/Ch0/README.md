# Chapter 0: Setting Up Your Python Environment

> **Note**: This chapter is available online only (GitHub) and is not included in the print edition.

**Getting Started with Time Series Analysis**

A comprehensive guide to all commands and code snippets from Chapter 0, covering environment setup, package management, and JupyterLab installation.

---

## Table of Contents

- [Verifying Conda Installation](#verifying-conda-installation)
- [Conda Environment Management](#conda-environment-management)
  - [Creating Environments](#creating-environments)
  - [Activating and Deactivating](#activating-and-deactivating)
  - [Managing Environments](#managing-environments)
- [Venv Environment Management](#venv-environment-management)
- [Checking Active Python Interpreter](#checking-active-python-interpreter)
- [YAML Environment Configuration](#yaml-environment-configuration)
- [Installing Python Libraries](#installing-python-libraries)
  - [Using Conda](#using-conda-for-packages)
  - [Using Pip](#using-pip-for-packages)
- [Bootstrapping Requirements Files](#bootstrapping-requirements-files)
- [Installing JupyterLab](#installing-jupyterlab)
- [Additional Resources](#additional-resources)

---

## Verifying Conda Installation

Check if Conda is properly installed and view configuration details:

```bash
conda info
```

---

## Conda Environment Management

### Creating Environments

**Update Conda and Anaconda:**

```bash
# Update conda to latest version
conda update conda -y

# Update Anaconda to latest version
conda update anaconda
```

**Create new virtual environment with specific Python version:**

```bash
# Basic creation
conda create -n py310 python=3.10

# Skip confirmation prompt
conda create -n py310 python=3.10 -y

# Create with multiple packages
conda create -n timeseries python=3.12 pandas matplotlib statsmodels -y
```

**Create environment with different Python version:**

```bash
conda create -n py313 python=3.13.0 -y
```

**Create environment with custom path:**

```bash
conda create -p /path/to/yourcustom/env
```

### Activating and Deactivating

**Activate environment:**

```bash
conda activate py310
```

**Activate custom path environment:**

```bash
conda activate /path/to/yourcustom/env
```

**Deactivate environment:**

```bash
conda deactivate
```

### Managing Environments

**List all conda environments:**

```bash
conda info --envs
```

**Clone an environment:**

```bash
conda create --name py310_clone --clone py310
```

**Remove environment:**

```bash
conda env remove -n py310
```

**Search available Python versions:**

```bash
conda search python
```

### Checking PATH Variable

**macOS/Linux:**

```bash
echo $PATH
```

**Windows Command Prompt:**

```cmd
echo %path%
```

**Windows PowerShell:**

```powershell
echo $env:path
```

### Installing Packages with Conda

**Install specific package version:**

```bash
conda install pandas=2.2.3

# Skip confirmation
conda install pandas=2.2.3 -y
```

**Add conda-forge channel:**

```bash
# General
conda config --add channels conda-forge

# macOS ARM specific
conda config --add channels conda-forge/osx-arm64
```

> **Note:** Conda uses single equals sign for version specification: `pandas=2.2.3`

---

## Venv Environment Management

### Creating and Activating Venv

**Create virtual environment:**

```bash
# Navigate to desired directory
cd Desktop

# Create environment
python -m venv py310
```

**Activate environment (macOS/Linux):**

```bash
source py310/bin/activate
```

**Activate environment (Windows PowerShell):**

```powershell
# Short form
.\py310\Scripts\activate

# Explicit form
.\py310\Scripts\Activate.ps1
```

**Check Python version:**

```bash
python --version
```

**Deactivate environment:**

```bash
deactivate
```

### Full Example

**macOS/Linux:**

```bash
python -m venv ~/Desktop/timeseries
source ~/Desktop/timeseries/bin/activate
pip install -r requirements.txt
```

**Windows:**

```powershell
python -m venv C:\Users\<your username>\Desktop\timeseries
C:\Users\<your username>\Desktop\timeseries\Scripts\activate
pip install -r requirements.txt
```

---

## Checking Active Python Interpreter

**macOS/Linux:**

```bash
which python
```

**Windows Command Prompt:**

```cmd
where.exe python
```

**Windows PowerShell:**

```powershell
Get-Command python
```

---

## YAML Environment Configuration

### Creating env.yml File

Create a file named `env.yml`:

```yaml
# A YAML for creating a conda environment
# file: env.yml
name: tscookbook
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  # Data Analysis
  - statsmodels
  - scipy
  - pandas
  - numpy
  - tqdm
  # Plotting
  - matplotlib
  - seaborn
  # Machine learning
  - scikit-learn
  - jupyterlab
```

### Using YAML Files

**Create environment from YAML:**

```bash
conda env create -f env.yml
```

**Activate the environment:**

```bash
conda activate tscookbook
```

**Export environment to YAML:**

```bash
# Three equivalent commands
conda env export -n py310 > env.yml
conda env export -n py310 -f env.yml
conda env export --name py310 --file env.yml
```

**Export only explicitly installed packages:**

```bash
conda env export --from-history > env.yml

# Specify environment name
conda env export -n ch1 --from-history > env.yml
conda env export --name ch1 --from-history -f env.yml
```

**View exported YAML:**

```bash
cat env.yml
```

---

## Installing Python Libraries

### Creating Requirements Files

**requirements.txt example:**

```text
pandas==2.2.3
matplotlib==3.9.2
statsmodels==0.14.4
```

**simple.txt example (no versions):**

```text
pandas
matplotlib
```

### Using Conda for Packages

**Option 1: Create new environment with packages:**

```bash
conda create -n ch1 --file requirements.txt
```

**Option 2: Install into existing environment:**

```bash
conda create -n timeseries -y
conda activate timeseries
conda install --file requirements.txt -y
```

### Using Pip for Packages

**Install from requirements file:**

```bash
pip install -r requirements.txt

# Or from simple.txt
pip install -r simple.txt
```

### Listing Installed Packages

**With pip:**

```bash
pip list
```

**With conda:**

```bash
# List all packages
conda list

# Count packages
conda list | wc -l
```

> **Important:** Conda accepts pip format (`package==version`), but pip cannot read Conda format (`package=version=build`)

---

## Bootstrapping Requirements Files

### Export with Pip

**Export all pip-installed packages:**

```bash
# Activate environment first
source ch1/bin/activate  # or appropriate activation command

# Export to requirements.txt
pip freeze > requirements.txt

# View the file
cat requirements.txt
```

### Export with Conda

**Export all conda packages:**

```bash
# Activate environment
conda activate ch1

# Export full list
conda list -e > conda_requirements.txt

# View the file
cat conda_requirements.txt
```

**Export only explicitly installed packages:**

```bash
conda activate ch1
conda env export --from-history > env.yml
cat env.yml
```

**Export without activating first:**

```bash
# All equivalent
conda env export -n ch1 --from-history > env.yml
conda env export --name ch1 --from-history > env.yml
conda env export -n ch1 --from-history -f env.yml
conda env export --name ch1 --from-history --file env.yml
```

---

## Installing JupyterLab

### Basic Installation

**Create environment with base packages:**

```bash
# With single package
conda create -n timeseries python=3.12 pandas -y

# With multiple packages
conda create -n timeseries python=3.12 pandas matplotlib statsmodels -y
```

**Activate and install JupyterLab:**

```bash
conda activate timeseries
conda install -c conda-forge jupyterlab -y
```

### Launching JupyterLab

**Basic launch:**

```bash
jupyter lab
```

**Launch with specific browser:**

```bash
jupyter lab --browser=chrome
```

**Launch without opening browser:**

```bash
jupyter lab --no-browser
```

**Launch with custom port:**

```bash
jupyter lab --browser=chrome --port 8890
```

### Managing Multiple Kernels

**Install kernel management package:**

```bash
# In base environment
conda install nb_conda_kernels
```

**Install ipykernel in specific environment:**

```bash
conda install --name timeseries ipykernel
```

**Create new environment with ipykernel:**

```bash
conda create -n dev python=3.12 ipykernel -y
```

**List available kernels:**

```bash
python -m nb_conda_kernels list
```

### JupyterLab Configuration

**Generate configuration file:**

```bash
jupyter-lab --generate-config
```

**Register browser (Windows - add to `jupyter_lab_config.py`):**

```python
import webbrowser
webbrowser.register(
    'chrome', None,
    webbrowser.GenericBrowser(
        'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
    )
)
```

### JupyterLab Extensions

**List installed extensions:**

```bash
jupyter labextension list
```

**Install Node.js (required for some extensions):**

```bash
conda install -c conda-forge nodejs
```

---

## Additional Resources

### Anaconda Project

**Install anaconda-project:**

```bash
conda install anaconda-project
```

**View help:**

```bash
anaconda-project --help
```

### Package Version Syntax

| Package Manager | Syntax Example |
|----------------|----------------|
| Conda | `conda install pandas=2.2.3` |
| Pip | `pip install pandas==2.2.3` |

### Configuration File Locations

**jupyter_lab_config.py locations:**

- **Windows:** `C:\Users\<yourusername>\.jupyter\jupyter_lab_config.py`
- **Linux:** `/home/<username>/.jupyter/jupyter_lab_config.py`
- **macOS:** `/Users/<username>/.jupyter/jupyter_lab_config.py`

**Conda environments location:**

- **Anaconda:** `~/opt/anaconda3/envs/` (macOS) or `C:\Users\<user>\anaconda3\envs` (Windows)
- **Miniconda:** Replace `anaconda3` with `miniconda3`

---

## Quick Reference Commands

```bash
# Check conda version
conda info

# Create environment
conda create -n myenv python=3.12 -y

# Activate environment
conda activate myenv

# Install packages
conda install pandas matplotlib -y

# List environments
conda info --envs

# Export environment
conda env export --from-history > environment.yml

# Deactivate
conda deactivate

# Launch JupyterLab
jupyter lab
```

---

## Useful Links

- **Anaconda Documentation:** https://docs.anaconda.com/
- **Conda Documentation:** https://docs.conda.io/
- **PyPI Repository:** https://pypi.org/
- **JupyterLab Documentation:** https://jupyterlab.readthedocs.io/
- **Book GitHub Repository:** https://github.com/PacktPublishing/Time-Series-Analysis-with-Python-Cookbook-Second-Edition/

---

## Notes and Best Practices

*  **Always use virtual environments** for each project to avoid dependency conflicts

*  **Use `-y` flag** to skip confirmation prompts when you're confident

*  **Export environment configurations** regularly for reproducibility

*  **Use conda-forge channel** for access to more packages and latest versions

*  **Install ipykernel** in each environment to make it available as a JupyterLab kernel

*  **Use `--from-history` flag** when exporting to get cleaner YAML files

⚠️ **Conda format vs Pip format:** Remember the syntax differences for version specification

⚠️ **Bootstrap requirements files** before sharing projects to ensure reproducibility

---