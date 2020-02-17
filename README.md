DEEP Open Catalogue: Massive Online Data Streams (MODS)
==============================
![DEEP-Hybrid-DataCloud logo](https://deep-hybrid-datacloud.eu/wp-content/uploads/sites/2/2018/01/logo.png)

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/mods/test)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/mods/job/test/)

DEEP Open Catalog entry: [DEEP Open Catalog](https://marketplace.deep-hybrid-datacloud.eu/modules/deep-oc-massive-online-data-streams.html)

**Project:** 
This work is part of the [DEEP Hybrid-DataCloud](https://deep-hybrid-datacloud.eu/) project that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 777435.

To start using this framework run:

```bash
git clone https://github.com/deephdc/mods
cd mods
pip install -e .
```

**Requirements:**
 - This project has been tested in Ubuntu 18.04 with Python 3.6. 
 - Further package requirements are described in the `requirements.txt` file.


Project Organization
------------

    ├── LICENSE     
    ├── README.md   <- The top-level README for developers using this project.
    │
    ├── checkpoints <- Directory for checkpoint storing during training process 
    │
    ├── data        <- Data directory
    │   ├── features       <- datapool directory for model training
    │   │   └── datapools
    │   ├── test           <- test data 
    │   │   └── sample-test-w01h-s10m.tsv  <- sample data for prediction that works with the default model
    │   └── train          <- (optional) train data
    │       └── sample train data in the same format as test data
    │
    ├── docs        <- Online documentation http://docs.deep-hybrid-datacloud.eu/en/latest/user/modules/mods.html 
    │
    ├── docker      <- Official docker container https://github.com/deephdc/DEEP-OC-mods
    │
    ├── models      <- Trained models
    │   ├── model_default_cpu.zip          <- default model for CPU    
    │   └── model_default_gpu.zip          <- default model for GPU
    │
    ├── mods        <- Deep learning module source code for use in this project.
    │   ├── __init__.py    <- Makes the module a Python module
    │   │
    │   ├── config.py      <- Configuration file for model training as well as hyper-parameter tuning
    │   ├── utils.py       <- Utilization functions   
    │   │
    │   ├── dataset        <- Scripts to process data at various levels
    │   │   └── make_dataset.py    <- Data Preprocessing (module) to produce ML/DL data    
    │   │
    │   ├── features       <- Scripts to build and select ML/DL data
    │   │   └── select_features.py <- Tests for feature selection
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make prediction
    │   │   ├── api.py             <- Module API to leverage the DEEPaaS API
    │   │   ├── mods_model.py      <- Deep Learning modeling
    │   │   └── auxiliary scripts
    │   │
    │   ├── tests          <- Code testing scripts + pylint script
    │   │
    │   └── visualization  <- Visualization oriented scripts
    │       └── visualize.py
    │
    ├── notebooks    <- Jupyter notebooks
    ├── references   <- Explanatory materials such as articles, flyers, posters, presentations.
    ├── reports      <- Generated analysis
    │   └── figures       <- Generated graphics and figures to be used in reporting
    │
    ├── .rclone.conf <- Configuration file for data transfer (rclone)
    ├── Jenkinsfile  <- CI/CD configuration
    │
    ├── requirements.txt   <- Environment reproducing file, e.g. `pip freeze > requirements.txt`
    │
    ├── setup.cfg    <- Module metadata + DEEPaaS entry point definition
    ├── setup.py     <- Makes project pip installable (pip install -e .) so the module can be imported
    │
    └── tox.ini      <- tox file with settings for running tox; see tox.testrun.org

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/indigo-dc/cookiecutter-data-science">DEEP DS template</a>. #cookiecutter #datascience</small></p>

## Workflow
### Data Preprocessing module
1. Data cleaning and filtering, feature extraction
2. Data transformation, ML/DL datapool creation 
3. Feature selection

### Deep Learning module (MODS)
1. Configuration setting 
2. Model training
3. Model testing

### Prediction and train throught [DEEPaaS API](https://github.com/indigo-dc/DEEPaaS)

### DEEP as a Service
1. [MODS container](https://github.com/deephdc/DEEP-OC-mods) available in [Docker Hub](https://hub.docker.com/r/deephdc/deep-oc-mods) as a part of [`deephdc`](https://hub.docker.com/u/deephdc/) organization
2. [MODS entry](https://marketplace.deep-hybrid-datacloud.eu/modules/deep-oc-massive-online-data-streams.html) in [DEEP Open Catalog](https://marketplace.deep-hybrid-datacloud.eu/) as an [Use Case of DEEP-HybridDataCloud project](https://deep-hybrid-datacloud.eu/use-cases/)
3. [MODS online documentation](http://docs.deep-hybrid-datacloud.eu/en/latest/user/modules/mods.html)

<img src="https://deep-hybrid-datacloud.eu/wp-content/uploads/sites/2/2018/04/datastreams.jpeg" width="600">


