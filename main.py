from src.bert_fcn import BertTrainer
from src.utils import prepare_dataset, align_class_weights
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

PATH_MODEL = "./models/ModernBert1"
PATH_LOGS = "./logs/ModernBert1.txt"
PATH_VALID_LOGS = "./logs/ModernBertValid.txt"
MODEL = "answerdotai/ModernBERT-base"
PATH_DATASET = "./data/"
DEVICE = "cuda"
BATCH_SIZE = 16
EPOCHS = 5


def main(config):
    df_train = pd.read_csv(PATH_DATASET + "train.csv")
    df_test = pd.read_csv(PATH_DATASET + "test.csv")
    df_valid = pd.read_csv(PATH_DATASET + "dev.csv")

    labels, sentences, tag_values, tag2idx = prepare_dataset(df_train)
    labels_t, sentences_t, tag_values_t, _ = prepare_dataset(df_test)

    trainer = BertTrainer(sentences=sentences, labels=labels, tag_values=tag_values,
                          tokenizer=MODEL, device=DEVICE, mode="ModernBert",
                          test_sentences=sentences_t, test_labels=labels_t)

    trainer.preprocess(bs=BATCH_SIZE)

    weights = align_class_weights(df_train, tag2idx)

    trainer.train_eval(weight=weights, path=PATH_MODEL,
                       epochs=EPOCHS, save_logs=PATH_LOGS, verbose=False)

    trainer.load_model(PATH_MODEL)

    # ADD EVAL FCN
    trainer.evaluate_model(df_valid, BATCH_SIZE, weights,
                           verbose=False, save_logs=PATH_VALID_LOGS)


if __name__ == "__main__":
    main("")
