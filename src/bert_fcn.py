import numpy as np
from seqeval.metrics import accuracy_score
import torch.nn as nn
from tqdm import tqdm, trange
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from transformers import BertForTokenClassification, AdamW
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report
from concurrent.futures import ThreadPoolExecutor
from keras_preprocessing.sequence import pad_sequences


def agg_function(s):
    return [(w, p) for w, p in zip(s["Mot"].values.tolist(), s["Label"].values.tolist())]


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


class BertTrainer(object):
    def __init__(self, sentences=[], labels=[], tag_values=[],
                 tokenizer='bert-base-uncased', max_len=512, device="cpu"):
        self.sentences = sentences
        self.tag_values = tag_values
        self.tag2idx = {t: i for i, t in enumerate(self.tag_values)}
        self.tokname = tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer, do_lower_case=True)
        self.labels = labels
        self.MAX_LEN = max_len
        self.Nlabels = None
        self.model = None
        self.device = device

    def getparam(self, MAX_LEN=512):
        self.MAX_LEN = MAX_LEN

    def getTag(self, tag_values):
        self.tag_values = tag_values

    def tokenize_and_preserve_labels(self, sentence, text_labels):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):

            tokenized_word = self.tokenizer.tokenize(str(word))
            n_subwords = len(tokenized_word)

            tokenized_sentence.extend(tokenized_word)

            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels

    def preprocess(self, random_state=100, test_size=0.1, bs=32, FULL_FINETUNING=True,
                   lr=3e-5, eps=1e-8):
        tokenized_texts_and_labels = [self.tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(self.sentences, self.labels) ]
        tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]

        self.Nlabels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

        self.input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                       maxlen=self.MAX_LEN, dtype="long", value=0.0,
                                       truncating="post", padding="post")

        self.tags = pad_sequences([[self.tag2idx.get(l) for l in lab] for lab in self.Nlabels],
                                  maxlen=self.MAX_LEN, value=self.tag2idx["PAD"], padding="post",
                                  dtype="long", truncating="post")

        self.attention_masks = [[float(i != 0.0) for i in ii] for ii in self.input_ids]
        tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(self.input_ids, self.tags,
                                                                    random_state=random_state,
                                                                    test_size=test_size)
        tr_masks, val_masks, _, _ = train_test_split(self.attention_masks, self.input_ids,
                                                     random_state=random_state, test_size=test_size)
        tr_inputs = torch.tensor(tr_inputs)
        val_inputs = torch.tensor(val_inputs)
        tr_tags = torch.tensor(tr_tags)
        val_tags = torch.tensor(val_tags)
        tr_masks = torch.tensor(tr_masks)
        val_masks = torch.tensor(val_masks)
        self.train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        self.train_sampler = RandomSampler(self.train_data)
        self.train_dataloader = DataLoader(self.train_data, sampler=self.train_sampler, batch_size=bs)

        self.valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        self.valid_sampler = SequentialSampler(self.valid_data)
        self.valid_dataloader = DataLoader(self.valid_data, sampler=self.valid_sampler, batch_size=bs)
        self.model = BertForTokenClassification.from_pretrained(
                                                                self.tokname,
                                                                num_labels=len(self.tag2idx),
                                                                output_attentions=False,
                                                                output_hidden_states=False)
        if FULL_FINETUNING:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            eps=eps)
        
    def train_eval(self, epochs=1, max_grad_norm=1.0,
                   weight=[1,1,1], currloss=np.inf, path='./outputsNSD/modeleNSD'):

        loss_values, validation_loss_values = [], []
        self.model.to(self.device)
        total_steps = len(self.train_dataloader) * epochs
        class_weights = torch.tensor(weight, dtype=torch.float).to(self.device)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        for _ in trange(epochs, desc="Epoch"):
            self.model.train()
            total_loss = 0

            for step, batch in enumerate(self.train_dataloader):

                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                self.model.zero_grad()

                outputs = self.model(b_input_ids.long(), token_type_ids=None,
                                     attention_mask=b_input_mask, labels=b_labels.long())

                loss = outputs[0]
                scores = outputs.logits
                targets = b_labels[:, :scores.shape[1]]
                loss_fn = nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fn(scores.view(-1, scores.shape[-1]), targets.view(-1).long())
                loss.backward()

                total_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                               max_norm=max_grad_norm)

                self.optimizer.step()

                scheduler.step()

            avg_train_loss = total_loss / len(self.train_dataloader)
            print("Average train loss: {}".format(avg_train_loss))

            loss_values.append(avg_train_loss)

            self.model.eval()

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predictions , true_labels = [], []
            for batch in self.valid_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():

                    outputs = self.model(b_input_ids.long(), token_type_ids=None,
                                         attention_mask=b_input_mask, labels=b_labels.long())

                logits = outputs[1].detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                eval_loss += outputs[0].mean().item()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(label_ids)

            eval_loss = eval_loss / len(self.valid_dataloader)
            validation_loss_values.append(eval_loss)

            print("Validation loss: {}".format(eval_loss))
            pred_tags = [self.tag_values[p_i] for p, l in zip(predictions, true_labels)
                         for p_i, l_i in zip(p, l) if self.tag_values[l_i] != "PAD"]
            valid_tags = [self.tag_values[l_i] for l in true_labels
                          for l_i in l if self.tag_values[l_i] != "PAD"]
            print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
            print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags, average='weighted')))
            classification_rep = classification_report(valid_tags, pred_tags)
            print("Classification Report:\n", classification_rep)
            print()

            self.loss_values = loss_values
            self.validation_loss_values = validation_loss_values
            if currloss > eval_loss:
                self.save_model(path)
                print("Model saved successfully.")
                currloss = eval_loss

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        self.model = BertForTokenClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model.eval()

    def predict(self, sentence):
        self.model.to(self.device)
        torch.cuda.empty_cache()

        self.model.eval()
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        attention_mask = [[float(i != 0.0) for i in tokens_tensor[0]]]
        attention_mask = torch.tensor(attention_mask).to(self.device)
        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=None, attention_mask=attention_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_indices = np.argmax(logits, axis=2)[0]
        labels = [self.tag_values[label_idx] for label_idx in label_indices]
        return list(zip(tokenized_sentence, labels))

    def batch_predict(self, sentences):
        with ThreadPoolExecutor() as executor:
            predictions = list(tqdm(executor.map(self.predict, sentences)))
        return predictions 

    def plotloss(self):
        sns.set(style='darkgrid')

        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12,6)

        plt.plot(self.loss_values, 'b-o', label="training loss")
        plt.plot(self.validation_loss_values, 'r-o', label="validation loss")

        plt.title("Learning curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.show()
