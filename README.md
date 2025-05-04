# Modern Bert: An application on Clinical Data

Final project for the course Machine Learning for Natural Language Processing @ ENSAE with Christopher Kermorvant (kermorvant@teklia.com)

![dummy](./figs/bert_doctor.png)

# Presentation

The project is a study and an application of [ModernBert]() on Clinical Trials data. 

# Structure

    ├── README.md
    ├── config.yaml
    ├── main.py
    ├── setup.py
    ├── data/
    │   ├── README.md
    │   ├── data_extraction.py
    │   ├── data_to_bio.py
    │   ├── tag2idx.json
    │   ├── data_bio/
    │   ├── data_chia/
    │   ├── tests/
    │   └── trains/
    ├── dataset/
    │   ├── README.md
    │   ├── color_map.py
    │   ├── get_figs.py
    │   ├── rope.py
    │   └── figs/
    ├── figs/
    ├── logs/
    │   ├── BertBaseValidNW.txt
    │   ├── BioBertValidNW.txt
    │   ├── ClinicalModernBertValidNW.txt
    │   ├── ModernBertValidNW.txt
    │   └── test_logs.txt
    ├── models/
    └── src/
        ├── bert_fcn.py
        └── utils.py


* ``data/`` contains all the guideline to download and convert the data used in the project
* ``dataset/`` contains a quick presentation of the dataset, and a module ``get_figs.py`` to reproduce the figs used in the report
* ``logs/`` contains all the results of the different models presented in the report in txt files
* ``src/`` contains the main functions of the project

# Get Start

First, please instanciate the repo with

```bash
cd ModernBert_ENSAE

pip install -e .
```

Then, to reproduce an experiment, select the parameters in ``config.yaml`` and then run

```bash
python main.py
```
