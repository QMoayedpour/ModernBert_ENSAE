import os
import torch


def write_file(log_file, epoch, epochs, avg_train_loss, eval_loss, acc, f1, classif_report):
    log_file.write(f"Epoch {epoch+1}/{epochs}\n")
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
