import os
import torch


def prepare_dataset(df):
    getter = SentenceGetter(df)

    labels = [[s[1] for s in sentence] for sentence in getter.sentences]
    sentences = [[word[0] for word in sentence] for sentence in getter.sentences]

    tag_values = list(set(df["Label"].values))
    tag_values.append("PAD")
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    return (labels, sentences, tag_values, tag2idx)


def display_results(eval_loss, acc, f1, classification_rep):
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(acc))
    print("Validation F1-Score: {}".format(f1))
    print("Classification Report:\n", classification_rep)
    print()


def align_class_weights(df, tag2idx):
    label_weights_dict = ((df["Label"].value_counts()/len(df))**-1).to_dict()
    num_classes = len(tag2idx)
    weights = torch.zeros(num_classes, dtype=torch.float)
    for label, idx in tag2idx.items():
        weights[idx] = label_weights_dict.get(label, 1.0)
    return weights


def write_file(log_file, eval_loss, acc, f1, classif_report,
               epoch=None, epochs=None, avg_train_loss=None):
    if epoch and epochs:
        log_file.write(f"Epoch {epoch+1}/{epochs}\n")
    if avg_train_loss:
        log_file.write(f"Average train loss: {avg_train_loss}\n")
    log_file.write(f"Validation loss: {eval_loss}\n")
    log_file.write(f"Validation Accuracy: {acc}\n")
    log_file.write(f"Validation F1-Score: {f1}\n")
    log_file.write("Classification Report:\n")
    log_file.write(classif_report + "\n")
    log_file.write("\n" + "="*80 + "\n\n")

    log_file.flush()
    os.fsync(log_file.fileno())


def agg_function(s):
    return [(w, p) for w, p in zip(s["Mot"].values.tolist(), s["Label"].values.tolist())]


def pad_sequences_torch(sequences, maxlen, padding_value=0):

    sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]

    sequences = [seq[:maxlen] if len(seq) > maxlen else seq for seq in sequences]

    padded = torch.full((len(sequences), maxlen), padding_value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        self.grouped = self.data.groupby("sentence").apply(agg_function)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
