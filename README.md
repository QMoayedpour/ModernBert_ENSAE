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

# Usage

```python

from src.utils import prepare_dataset
from src.bert_fcn import BertTrainer

df = pd.read_csv("mydf.csv")

labels, sentences, tag_values, tag2idx = prepare_dataset(df)

model = BertTrainer(sentences=sentences, labels=labels,
                    tag_values=tag_values, tokenizer="google-bert/bert-base-uncased",
                    device="cuda", mode="Bert")

model.preprocess(bs=16) # Batch size

model.train_eval(epochs=10, save_logs="save_my_logs.txt")


# Eval

df_eval = pd.read_csv("inference_df.csv")

model.evaluate_model(df_eval)

# Inference

predictions = model.predict("I need a healthy man")

```

# Contribution

The code was written by Naïl KHELIFA (nail.khelifa@ensae.fr) and Quentin MOAYEDPOUR (quentin.moayedpour@ensae.fr) but **all the commits** were done through Quentin MOAYEDPOUR's account since it had more access to sspcloud than Naïl's account (more GPU available) which was really important for the project (most models should take more than 3 hours for a training on CPU)
