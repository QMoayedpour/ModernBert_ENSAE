# ModernBert_ENSAE

Final project for the course Machine Learning for Natural Language Processing @ ENSAE with Christopher Kermorvant (kermorvant@teklia.com)

# Presentation

The project is a study and an application of [ModernBert]() on Clinical Trials data. 

# Structure

    ├── README.md
    ├── config.yaml
    ├── main.py
    ├── data/
    │   ├── README.md
    │   ├── __init__.py
    │   ├── data_extraction.py
    │   ├── data_to_bio.py
    │   ├── tag2idx.json
    │   ├── data_bio/
    │   │   └── __init__.py
    │   ├── data_chia/
    │   │   └── __init__.py
    │   ├── tests/
    │   │   └── __init__.py
    │   └── trains/
    │       └── __init__.py
    ├── dataset/
    │   ├── README.md
    │   ├── __init__.py
    │   ├── color_map.py
    │   ├── get_figs.py
    │   ├── rope.py
    │   └── figs/
    │       └── __init__.py
    ├── figs/
    │   └── __init__.py
    ├── logs/
    │   ├── __init__.py
    │   └── test_logs.txt
    ├── models/
    │   └── __init__.py
    ├── ntbk/
    │   └── __init__.py
    └── src/
        ├── __init__.py
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
