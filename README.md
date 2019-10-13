DEEP Open Catalogue: Massive Online Data Streams (MODS)
==============================
![DEEP-Hybrid-DataCloud logo](https://deep-hybrid-datacloud.eu/wp-content/uploads/sites/2/2018/01/logo.png)

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/mods/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/mods/job/master/)

DEEP Open Catalog entry: [DEEP Open Catalog](https://marketplace.deep-hybrid-datacloud.eu/modules/deep-oc-massive-online-data-streams.html)

**Project:** This work is part of the [DEEP Hybrid-DataCloud](https://deep-hybrid-datacloud.eu/) project that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 777435.

To start using this framework run:

```bash
git clone https://github.com/deephdc/mods
cd mods
pip install -e .
```

**Requirements:**
 
 - This project has been tested in Ubuntu 18.04 with Python 3.6. Further package requirements are described in the `requirements.txt` file.
- (TBD later)


Project Organization
------------

    ├── LICENSE
    ├── README.md   <- The top-level README for developers using this project.
    │
    ├── checkpoints <- Directory for checkpoint storing during train process 
    │
    ├── data        <- Data directory
    │   ├── features       <- datapools
    │   ├── test           <- sample data for test or prediction 
    │   └── train          <- sample data for train
    │
    ├── docs        <- Online documentation http://docs.deep-hybrid-datacloud.eu/en/latest/user/modules/mods.html 
    ├── docker      <- Optional dockerfile(s)
    ├── models      <- Trained models (e.g. defaull model)
    │
    ├── mods        <- Module source code for use in this project.
    │   ├── __init__.py    <- Makes the module a Python module
    │   │
    │   ├── config.py      <- Module configuration file e.g. for hyper-parameter tuning
    │   ├── utils.py       <- Module utilization functions   
    │   │
    │   ├── dataset        <- Scripts to process data at various levels
    │   │   ├── data_utils.py      <- Data utility functions with hybrid data storage (Nextcloud, rclone)
    │   │   └── make_dataset.py    <- Data Preprocessing: sensitive data processing to produce ML/DL data    
    │   │
    │   ├── features       <- Scripts to build ML/DL data
    │   │   ├── build_features.py  <- Data Preprocessing: sensitive data processing to produce ML/DL data 
    │   │   └── select_features.py <- Tests for feature selection
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make prediction
    │   │   ├── api.py             <- Module API to leverage the DEEPaaS API
    │   │   ├── mods_model.py      <- Deep Learning modeling
    │   │   ├── predict.py         <- stand alone script (under revision)
    │   │   ├── test.py            <- stand alone script (under revision)
    │   │   └── train.py           <- stand alone script (under revision)
    │   │
    │   ├── tests          <- Code testing scripts + pylint script
    │   │
    │   └── visualization  <- Visualization oriented scripts
    │       └── visualize.py
    │
    ├── notebooks    <- Jupyter notebooks
    ├── references   <- Explanatory materials such as articles, books, flyers, posters, presentations.
    ├── reports      <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures       <- Generated graphics and figures to be used in reporting
    │
    ├── .rclone.conf <- Default hybrid data storage setting
    ├── Jenkinsfile  <- CI/CD configuration
    │
    ├── requirements.txt   <- The requirements file for reproducing environment, e.g. `pip freeze > requirements.txt`
    │
    ├── setup.cfg    <- Metadata and DEEPaaS entry point definition
    ├── setup.py     <- Makes project pip installable (pip install -e .) so the module can be imported
    │
    └── tox.ini      <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://github.com/indigo-dc/cookiecutter-data-science">DEEP DS template</a>. #cookiecutter #datascience</small></p>

## Workflow

### 1. Data preprocessing

#### 1.1 Prepare the dataset 

#### 1.2 Feature extraction

#### 1.3 Feature selection

### 2. Train and test DL models

#### 2.1 Set the configuration 

#### 2.2 Training

#### 2.3 Model selection

### 3. Prediction throught DEEPaaS API

### 4. DEEP as a Service: [MODS container](https://github.com/deephdc/DEEP-OC-mods)

### 5. Docker Hub: [MODS container image](https://hub.docker.com/r/deephdc/deep-oc-mods) in Docker Hub [`deephdc`](https://hub.docker.com/u/deephdc/) organization

