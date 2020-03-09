DEEP Open Catalogue: Massive Online Data Streams (MODS)
==============================
![DEEP-Hybrid-DataCloud logo](https://deep-hybrid-datacloud.eu/wp-content/uploads/sites/2/2018/01/logo.png)

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/mods/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/mods/job/master/)

DEEP Open Catalog entry: [DEEP Open Catalog](https://marketplace.deep-hybrid-datacloud.eu/modules/deep-oc-mods.html)

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
<p>Project based on the <a target="_blank" href="https://github.com/indigo-dc/cookiecutter-data-science">DEEP DS template</a>. #cookiecutter #datascience</p>

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

<img src="https://deep-hybrid-datacloud.eu/wp-content/uploads/sites/2/2018/04/datastreams.jpeg" width="600">
