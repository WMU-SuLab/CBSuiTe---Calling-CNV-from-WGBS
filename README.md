# CBSuiTe---Calling CNV from WGBS
> CBSuiTe is a deep learning based software that performs CNV call predictions on WGBS data using read depth sequences.
> CBSuiTe can predict CNV on both germline and somatic data.

## Contents 

- [Installation](#installation)
- [Run CBSuiTe](#run-cbsuite)
- [Parameters](#parameters)
- [Usage Examples](#usage-examples)
- [Citations](#citations)
- [License](#license)

## Installation

- CBSuiTe is written in python3 and can be run directly after installing the provided environment.

### Requirements

You can directly use ``cbsuite_environment.yml`` file to initialize conda environment with requirements installed:

```shell
$ conda env create --name cbsuite -f cbsuite_environment.yml
$ conda activate cbsuite
```
or install following dependencies:
* Python >= 3.8
* torch >= 1.7.1
* Numpy
* numpy
* pandas
* tqdm
* scikit-learn
* einops
* samtools
  
## Run CBSuiTe
### Step 0: Install Conda and set up your environment
See detail in [Installation](#installation).
### Step 1: Run preprocess script
CBSuiTe need to obtain read depth and methylation level information from ``bam`` files and convert to ``npy`` files.
You can simply run ``preprocess.sh`` by following commands:
```shell
$ source preprocess.sh
```
### Step 2: Run call script
Then you can predict CNV with CBSuiTe pretrained model.
```shell
$ source callCNV.sh
```


## Parameters
