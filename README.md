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

# Results on valid dataset

**Clinical Modern Bert**

| Label           | Precision | Recall | F1-score | Support |
|----------------|-----------|--------|----------|---------|
| I-Person       | 0.00      | 0.00   | 0.00     | 31      |
| I-Mood         | 0.00      | 0.00   | 0.00     | 53      |
| O              | 0.90      | 0.93   | 0.92     | 19618   |
| I-Drug         | 0.45      | 0.81   | 0.58     | 426     |
| I-Observation  | 0.71      | 0.03   | 0.06     | 356     |
| B-Procedure    | 0.68      | 0.46   | 0.55     | 597     |
| I-Procedure    | 0.54      | 0.54   | 0.54     | 538     |
| I-Condition    | 0.81      | 0.73   | 0.76     | 2624    |
| B-Observation  | 0.59      | 0.07   | 0.13     | 217     |
| B-Mood         | 0.60      | 0.41   | 0.49     | 70      |
| B-Person       | 0.83      | 0.72   | 0.77     | 203     |
| B-Condition    | 0.81      | 0.74   | 0.77     | 2623    |
| B-Drug         | 0.67      | 0.88   | 0.76     | 967     |

**Accuracy**: 0.85 (Total: 28323)

| Average Type | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| Macro avg    | 0.58      | 0.49   | 0.49     | 28323   |
| Weighted avg | 0.85      | 0.85   | 0.84     | 28323   |


# Contribution

The code was written by Naïl KHELIFA (nail.khelifa@ensae.fr) and Quentin MOAYEDPOUR (quentin.moayedpour@ensae.fr) but **all the commits** were done through Quentin MOAYEDPOUR's account since it had more access to sspcloud than Naïl's account (more GPU available) which was really important for the project (most models should take more than 3 hours for a training on CPU).
