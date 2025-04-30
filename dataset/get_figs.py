import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from transformers import AutoModel, AutoTokenizer
from sklearn.manifold import TSNE
import torch
import numpy as np


def get_n_words_per_setence(df, save=False):
    w_p_sentence = df['sentence'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    _ = sns.histplot(w_p_sentence,
                     kde=True,
                     bins=30,
                     color="#4CB391",
                     edgecolor="white",
                     linewidth=0.5)

    plt.title('', fontsize=14, pad=20)

    plt.xlabel('Words per sentences', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    median_val = w_p_sentence.median()
    plt.axvline(median_val, color='#E94B3C', linestyle='--', linewidth=1.5)
    plt.text(median_val*1.05, plt.ylim()[1]*0.9,
             f'Median: {int(median_val)} words',
             color='#E94B3C')

    sns.despine(left=True)
    plt.tight_layout()

    if save:
        plt.savefig(save)
    else:
        plt.show()


def encode_word(word, embedder, tokenizer, device="cuda"):
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = embedder(**inputs)
    return outputs.last_hidden_state[0, 0, :].cpu().numpy()


def get_tsne_bert(df, model='bert-base-uncased', n=300, random_state=42,
                  save_fig=None, device="cuda", model_name="base"):
    df["Label"] = df["Label"].apply(lambda x: x[2:] if x!="O" else x)
    df_sampled = df.groupby("Label").apply(lambda x: x.sample(n=n,
                                                              random_state=random_state)).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(model)
    embedder = AutoModel.from_pretrained(model).to(device)
    embedder.eval()

    embeddings = [encode_word(word, embedder, tokenizer, device) for word in df_sampled['Mot']]

    labels = df_sampled['Label'].tolist()
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = np.array(embeddings)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=embeddings_2d[:,0], y=embeddings_2d[:,1], hue=labels, palette='tab10')
    plt.title(f"Bert '{model_name}' Representation")
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(title='Label')

    if save_fig:
        plt.savefig(save_fig)
    else:
        plt.show()


def get_wordcloud(df, n_words=200, save_fig=None):
    text = ' '.join(df['Mot'].astype(str))

    stop_words = set(STOPWORDS)

    wordcloud = WordCloud(
        stopwords=stop_words,
        background_color='white',
        width=1600,
        height=800,
        max_words=n_words
    ).generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud)
    plt.axis('off')
    if save_fig:
        plt.savefig(save_fig)
    else:
        plt.show()


if __name__ == "__main__":

    df = pd.read_csv("../data/train.csv", index_col=0)
    get_n_words_per_setence(df, "./figs/distribution_words.png")
    get_wordcloud(df, save_fig="./figs/wordcloud.png")
    get_tsne_bert(df, model='bert-base-uncased', n=300, random_state=42,
                  save_fig="./figs/tsne_bert.png", device="cuda", model_name="base")
