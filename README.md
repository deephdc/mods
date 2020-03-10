DEEP Open Catalogue: Massive Online Data Streams
==============================

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/mods/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/mods/job/master/)

This work challenges [proactive network monitoring](https://doi.org/10.1109/ACCESS.2020.2968718) for security and protection of computing infrastructures. The principle of the anomaly detection for security lies in proactive prediction from time-series using prediction models based on artificial neural networks, concretely deep learning models. These models are capable to predict next step(s) in near future based on given current and past steps. The intelligent module is built as a machine learning application leveraging deep learning modeling in order to enhance functionality of intrusion detection system (network IDS) supervising network traffic flows.

DEEP Open Catalog entry: [DEEP Open Catalog](https://marketplace.deep-hybrid-datacloud.eu/modules/deep-oc-mods.html)

**Project:** 
This work is a part of the [DEEP Hybrid-DataCloud](https://deep-hybrid-datacloud.eu/) project that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 777435.

Documentation
----------

The documentation of the work is hosted on [Read the Docs](https://docs.deep-hybrid-datacloud.eu/en/latest/).

Quickstart
----------

To start using this framework run:

```bash
git clone https://github.com/deephdc/mods
cd mods
pip install -e .
```

**Requirements:**
 - This project has been tested in Ubuntu 18.04 with Python 3.6. 
 - Further package requirements are described in the `requirements.txt` file.


References:
----------

> G. Nguyen, S. Dlugolinsky, V. Tran, A. Lopez Garcia: 
> [Deep learning for proactive network monitoring and security protection](https://doi.org/10.1109/ACCESS.2020.2968718). 
> IEEE Access, Special section on Deep learning: security and forensics research advances and challenges, 
> Volume 8, Issue 1, pp. 19696-19716, ISSN 2169-3536, DOI 10.1109/ACCESS.2020.2968718, 2020. 

> G. Nguyen, S. Dlugolinsky, M. Bobak, V. Tran, A. Lopez Garcia, I. Heredia, P. Malik, L. Hluchy: 
> [Machine Learning and Deep Learning frameworks and libraries for large-scale data mining: a survey](https://doi.org/10.1007/s10462-018-09679-z). 
> Artificial Intelligence Review, Springer Nature, 2019, ISSN 0269-2821, DOI 10.1007/s10462-018-09679-z

**[More references](https://github.com/deephdc/mods/tree/master/references)**


Workflow
----------

### Data Preprocessing module
1. Data cleaning and filtering, feature extraction
2. Data transformation, ML/DL datapool creation 
3. Feature selection

### Deep Learning module (MODS)
1. Configuration setting 
2. Model training
3. Model testing

### Project organization 
1. The package is built based on [DEEP DS template](https://github.com/indigo-dc/cookiecutter-data-science). #cookiecutter #datascience
2. Prediction and train throught [DEEPaaS API](https://github.com/indigo-dc/DEEPaaS)

<img src="https://deep-hybrid-datacloud.eu/wp-content/uploads/sites/2/2018/04/datastreams.jpeg" width="600">
