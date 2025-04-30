import numpy as np
from seqeval.metrics import accuracy_score
import torch.nn as nn
from tqdm import tqdm, trange
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from transformers import BertForTokenClassification, ModernBertForTokenClassification, AutoTokenizer
from torch.optim import AdamW
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report
from concurrent.futures import ThreadPoolExecutor
from src.utils import write_file, pad_sequences_torch, prepare_dataset


class BertTrainer(object):
    def __init__(self, sentences=[], labels=[], tag_values=[],
                 tokenizer='bert-base-uncased', max_len=512, device="cpu", mode="Bert", custom_model=None,
                 test_sentences=None, test_labels=None):
        self.sentences = sentences
        self.tag_values = tag_values
        self.tag2idx = {t: i for i, t in enumerate(self.tag_values)}
        self.tokname = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, do_lower_case=True)
        self.labels = labels
        self.MAX_LEN = max_len
        self.Nlabels = None
        self.model = None
        self.device = device
        self.mode = mode
        self.custom_model = custom_model
        self.batch_loss = []
        self.test_sentences = test_sentences
        self.test_labels = test_labels

    def _load_tokenizer(self, tokenizer="bert-base-uncased", mode="Bert"):
        if mode == "Bert":
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer, do_lower_case=True)
            self.model = BertForTokenClassification.from_pretrained(tokenizer,
                                                                    num_labels=len(self.tag2idx),
                                                                    output_attentions=False,
                                                                    output_hidden_states=False)

        elif mode == "ModernBert":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, do_lower_case=True)
            self.model = ModernBertForTokenClassification.from_pretrained(tokenizer,
                                                                          num_labels=len(self.tag2idx),
                                                                          output_attentions=False,
                                                                          output_hidden_states=False)
        elif mode == "Custom":
            assert self.custom_model is not None, "custom_model is empty"
            self.model = self.custom_model
        
        print("Model Loaded")

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

    def _tokenize_text(self, sentences, labels):
        tokenized_texts_and_labels = [self.tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(sentences, labels) ]
        tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]

        Nlabels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

        input_ids = pad_sequences_torch([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                        maxlen=self.MAX_LEN, padding_value=0)

        tags = pad_sequences_torch(
                                        [[self.tag2idx.get(l) for l in lab] for lab in Nlabels],
                                        maxlen=self.MAX_LEN,
                                        padding_value=self.tag2idx["PAD"]
                                    )

        attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

        return (input_ids, tags, attention_masks)    

    def df_to_loader(self, df, bs=16):
        labels, sentences, tag_values, _ = prepare_dataset(df)
        dataloader = self._get_dataloader(labels, sentences, tag_values, bs=bs)
        return dataloader

    def _get_dataloader(self, labels, sentences, tag_values, bs=16, return_logs=False):
        (input_ids, tags,
         attention_masks) = self._tokenize_text(sentences, labels)

        inputs = torch.tensor(input_ids)
        tags = torch.tensor(tags)
        masks = torch.tensor(attention_masks)
        data = TensorDataset(inputs, masks, tags)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=bs)
        if return_logs:
            return (dataloader, input_ids, tags, attention_masks)

        return dataloader

    def _create_dataloader(self, inputs, masks, tags, bs, shuffle=True):
        dataset = TensorDataset(torch.tensor(inputs), torch.tensor(masks), torch.tensor(tags))
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=bs)

    def preprocess(self, random_state=100, test_size=0.1, bs=32, FULL_FINETUNING=True,
                   lr=3e-5, eps=1e-8, verbose=False):

        if verbose:
            print("Loading Dataset and Model ...")

        input_ids, tags, attention_masks = self._tokenize_text(self.sentences, self.labels)

        if self.test_sentences is None:
            tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(
                input_ids, tags, test_size=test_size, random_state=random_state)
            tr_masks, val_masks, _, _ = train_test_split(
                attention_masks, input_ids, test_size=test_size, random_state=random_state)
        else:
            tr_inputs, tr_tags, tr_masks = input_ids, tags, attention_masks
            val_inputs, val_tags, val_masks = self._tokenize_text(self.test_sentences,
                                                                  self.test_labels)

        self.train_dataloader = self._create_dataloader(tr_inputs, tr_masks,
                                                        tr_tags, bs, shuffle=True)
        self.valid_dataloader = self._create_dataloader(val_inputs, val_masks,
                                                        val_tags, bs, shuffle=False)

        self._load_tokenizer(self.tokname, mode=self.mode)

        params = self.model.named_parameters() if FULL_FINETUNING else self.model.classifier.named_parameters()
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ] if FULL_FINETUNING else [{"params": [p for n, p in params]}]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)

        if verbose:
            print("Loading Complete !")

    def train_eval(self, epochs=1, max_grad_norm=1.0,
                   weight=None, currloss=np.inf, path='./models/model1', save_logs=None, verbose=True):

        loss_values, validation_loss_values, self.batch_loss = [], [], []
        self.model.to(self.device)
        total_steps = len(self.train_dataloader) * epochs
        if weight is None:
            weight = [1] * len(self.tag2idx)

        if isinstance(weight, list):
            weight = torch.tensor(weight, dtype=torch.float).to(self.device)

        class_weights = weight.to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        #scheduler = get_linear_schedule_with_warmup(
        #    self.optimizer,
        #    num_warmup_steps=20000,
        #    num_training_steps=total_steps
        #)

        if save_logs is not None:
            log_file = open(save_logs, 'a', encoding='utf-8')

        for epoch in range(epochs):#, desc="Epoch"):
            self.model.train()
            total_loss = 0

            for step, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):

                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                self.model.zero_grad()

                if self.mode in ["Bert", "ModernBert"]:
                    outputs = self.model(b_input_ids.long(),
                                         attention_mask=b_input_mask, labels=b_labels.long())
                    scores = outputs.logits

                elif self.mode == "Custom":
                    scores = self.model(b_input_ids.long())

                targets = b_labels[:, :scores.shape[1]]

                loss = loss_fn(scores.view(-1, scores.shape[-1]), targets.view(-1).long())
                loss.backward()

                total_loss += loss.item()
                self.batch_loss.append(loss.item())

                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                               max_norm=max_grad_norm)

                self.optimizer.step()

                #scheduler.step()

            avg_train_loss = total_loss / len(self.train_dataloader)

            if verbose:
                print("Average train loss: {}".format(avg_train_loss))

            loss_values.append(avg_train_loss)

            (eval_loss, acc,
             f1, classification_rep) = self._test_model(self.valid_dataloader, loss_fn)

            validation_loss_values.append(eval_loss)
            if verbose:
                print("Validation loss: {}".format(eval_loss))
                print("Validation Accuracy: {}".format(acc))
                print("Validation F1-Score: {}".format(f1))
                print("Classification Report:\n", classification_rep)
                print()

            self.loss_values = loss_values
            self.validation_loss_values = validation_loss_values
            if currloss > eval_loss:
                self.save_model(path)
                print("Model saved successfully.")
                currloss = eval_loss
                if save_logs:
                    with open(save_logs, 'w', encoding='utf-8') as log_file:
                        write_file(log_file, epoch, epochs, avg_train_loss, eval_loss,
                                   acc, f1, classification_rep)

    def _test_model(self, dataloader, loss_fn):
        self.model.eval()

        eval_loss = 0
        predictions, true_labels = [], []
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():

                if self.mode in ["Bert", "ModernBert"]:
                    outputs = self.model(b_input_ids.long(),
                                         attention_mask=b_input_mask, labels=b_labels.long())
                    scores = outputs.logits

                elif self.mode == "Custom":
                    scores = self.model(b_input_ids.long())

            logits = scores.cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            targets = b_labels[:, :scores.shape[1]]
            eval_loss += loss_fn(scores.view(-1, scores.shape[-1]), targets.view(-1).long()).mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(self.valid_dataloader)

        pred_tags = [self.tag_values[p_i] for p, l in zip(predictions, true_labels)
                     for p_i, l_i in zip(p, l) if self.tag_values[l_i] != "PAD"]
        valid_tags = [self.tag_values[l_i] for l in true_labels
                      for l_i in l if self.tag_values[l_i] != "PAD"]
        acc = accuracy_score(pred_tags, valid_tags)
        f1 = f1_score(pred_tags, valid_tags, average='weighted')
        labels = [label for label in self.tag_values if label != "PAD"]
        classification_rep = classification_report(valid_tags, pred_tags, labels=labels)

        return (eval_loss, acc, f1, classification_rep)


    def save_model(self, path):
        if self.mode != "Custom":
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        if self.mode == "ModernBert":
            self.model = ModernBertForTokenClassification.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            #self.model.eval()
        elif self.mode == "Bert":
            self.model = BertForTokenClassification.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)

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
            outputs = self.model(tokens_tensor, attention_mask=attention_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_indices = np.argmax(logits, axis=2)[0]
        labels = [self.tag_values[label_idx] for label_idx in label_indices]
        return list(zip(tokenized_sentence, labels))

    def batch_predict(self, sentences, batch_size=32):
        """
        Batch predict function optimized for speed.
        """
        self.model.to(self.device)
        self.model.eval()

        all_tokenized = []
        sentence_maps = []

        for sent in sentences:
            tokens = self.tokenizer.tokenize(sent)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            all_tokenized.append(indexed_tokens)
            sentence_maps.append(tokens)

        max_len = min(max(len(seq) for seq in all_tokenized), self.MAX_LEN)
        input_ids = pad_sequences_torch(all_tokenized, maxlen=max_len, padding_value=0)
        attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

        input_ids = torch.tensor(input_ids).to(self.device)
        attention_masks = torch.tensor(attention_masks).to(self.device)

        dataset = TensorDataset(input_ids, attention_masks)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        all_predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                batch_input_ids, batch_attention_masks = batch
                outputs = self.model(batch_input_ids.long(), attention_mask=batch_attention_masks)
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                batch_predictions = np.argmax(logits, axis=2)

                for pred, tokens in zip(batch_predictions, sentence_maps):
                    pred_labels = [self.tag_values[label_idx] for label_idx in pred[:len(tokens)]]
                    all_predictions.append(list(zip(tokens, pred_labels)))

        return all_predictions

    def plotloss(self, save_fig=None):
        sns.set(style='darkgrid')

        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12,6)

        plt.plot(self.loss_values, 'b-o', label="training loss")
        plt.plot(self.validation_loss_values, 'r-o', label="validation loss")

        plt.title("Learning curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        if save_fig:
            plt.savefig(save_fig)
        else:
            plt.show()

    def plot_batchloss(self, save_fig=None):
        sns.set(style='darkgrid')

        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12,6)

        plt.plot(self.batch_loss, label="training batch loss")

        plt.title("Learning curve")
        plt.ylabel("Loss")
        plt.legend()

        if save_fig:
            plt.savefig(save_fig)

        else:
            plt.show()
