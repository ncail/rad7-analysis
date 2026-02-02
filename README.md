# RAD7 Analysis


## Overview 
This repository is in development. Currently included is:
* `rad7_data_processing.ipynb`
    * A notebook for preprocessing `.r7raw` data files output from RAD7 radon monitor
    * Demonstrates basic saving off and loading of the preprocessed data
    * Basic data manipulation such as plotting


## Requirements
- Git installed
- Miniconda or Anaconda installed
- VS Code installed
- VS Code extensions:
  - Python
  - Jupyter


## Setup
To set up the repository and python environment locally:

**1. Clone the repository**
```
git clone https://github.com/ncail/rad7-analysis.git
cd rad7-analysis
```
> **WARNING:** Do not push changes to this repository

**2. Create `conda` environment**

```
conda env create -f environment.yml
conda activate rad7-analysis
```
This requires Miniconda or Anaconda to be installed.

**3. Open the project in VSCode**
```
code .
```

Install extensions for Python and Jupyter if you have not already. Ensure Git is already installed on your system as well (VSCode comes with built-in Git support).

## Updating the project
In VSCode, periodically pull changes made to `main` branch to your local clone of the repository.
> You **do not** need to type any Git commands in VSCode.

1. Click the Source Control icon on the left (looks like a branching graph)

2. If VS Code shows a message like:

    “This branch is behind `origin/main`”

        > click Pull

    VSCode will download the changes and update your files automatically.

3. Merge conflicts

    If you edited a file that was updated remotely, VSCode may show a `merge conflict`. 

    Either:
    * Accept incoming changes (recommended)
    * Or ask me for help


