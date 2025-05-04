from src.bert_fcn import BertTrainer
from src.utils import prepare_dataset, align_class_weights, load_tag2idx
import pandas as pd
import yaml
import warnings
warnings.filterwarnings("ignore")


def main(config):
    df_train = pd.read_csv(config['PATH_DATASET'] + "train.csv")
    df_test = pd.read_csv(config['PATH_DATASET'] + "test.csv")
    df_valid = pd.read_csv(config['PATH_DATASET'] + "dev.csv")

    labels, sentences, tag_values, tag2idx = prepare_dataset(df_train)
    labels_t, sentences_t, tag_values_t, _ = prepare_dataset(df_test)

    tag2idx = load_tag2idx("./data/tag2idx.json")

    trainer = BertTrainer(sentences=sentences, labels=labels, tag_values=tag_values,
                          tokenizer=config['MODEL'], device=config['DEVICE'], mode=config['MODE'],
                          test_sentences=sentences_t, test_labels=labels_t)

    trainer.preprocess(bs=config['BATCH_SIZE'], lr=config['LR'], full_finetuning=config["FULL_FINETUNING"])

    if config['WEIGHTED']:
        weights = align_class_weights(df_train, tag2idx)
    else:
        weights = [1] * len(tag2idx)
    trainer.train_eval(weight=weights, path=config['PATH_MODEL'],
                       epochs=config['EPOCHS'], save_logs=config['PATH_LOGS'],
                       patience=config["PATIENCE"], verbose=False)

    trainer.load_model(config['PATH_MODEL'])

    trainer.evaluate_model(df_valid, config['BATCH_SIZE'], weights,
                           verbose=False, save_logs=config['PATH_VALID_LOGS'])


if __name__ == "__main__":
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)
