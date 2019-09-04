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
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── docker             <- Directory for Dockerfile(s)
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials (if many user development),
    │                         and a short `_` delimited description, e.g.
    │                         `1.0-jqp-initial_data_exploration.ipynb`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so mods can be imported
    ├── mods    <- Source code for use in this project.
    │   ├── __init__.py    <- Makes mods a Python module
    │   │
    │   ├── dataset        <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── model.py
    │   │
    │   └── tests          <- Scripts to perfrom code testing + pylint script
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://github.com/indigo-dc/cookiecutter-data-science">DEEP DS template</a>. #cookiecutter #datascience</small></p>

## Workflow

### 1. Data preprocessing

#### 1.1 Prepare the dataset 

#### 1.2 Build features

### 2. Train and test NNs

#### 2.1 Set the configuration 

#### 2.1 Training

### 3. Prediction throught DEEPaaS API

### 4. DEEP as a Service: [MODS container](https://github.com/deephdc/DEEP-OC-mods)

### 5. Docker Hub: [MODS container image](https://hub.docker.com/r/deephdc/deep-oc-mods) in Docker Hub [`deephdc`](https://hub.docker.com/u/deephdc/) organization


