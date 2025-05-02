import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from transformers import AutoModel, AutoTokenizer
from sklearn.manifold import TSNE
import torch
import numpy as np
from src.rope import RoPE


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
    df_m = df.copy()
    df_m["Label"] = df_m["Label"].apply(lambda x: x[2:] if x!="O" else x)
    df_sampled = df_m.groupby("Label").apply(lambda x: x.sample(n=n,
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


def show_pos_embedding_base(max_pos=50, d_model=128, save_fig=None):
    pos = np.arange(max_pos)[:, np.newaxis]
    dim = np.arange(0, d_model, 2)[np.newaxis, :]
    inv_freq = 1.0 / (10000 ** (dim / d_model))
    theta = pos * inv_freq

    rope_pe = np.zeros((max_pos, d_model))
    rope_pe[:, 0::2] = np.cos(theta)
    rope_pe[:, 1::2] = np.sin(theta)
    rope_pe = np.flip(rope_pe, 0)

    plt.figure(figsize=(12, 6))
    plt.imshow(rope_pe, cmap='viridis', aspect='equal')
    plt.title("")
    plt.ylabel("Position in the sequence")
    plt.xlabel("Dimension")
    plt.yticks([])
    if save_fig:
        plt.savefig(save_fig)
    else:
        plt.show()


def get_rope_similarity(rope, max_pos=25, save_fig=None):
    similarities = np.zeros((max_pos, max_pos))
    base_vector = np.random.randn(1, rope.d_model)
    
    for m in range(max_pos):
        q = rope.rotate(base_vector, m)
        for n in range(max_pos):
            k = rope.rotate(base_vector, n)
            similarities[m, n] = q @ k.T
            
    plt.figure(figsize=(10, 8))
    plt.imshow(similarities, cmap='viridis')
    plt.title("RoPE Attention Similarities")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
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
    get_tsne_bert(df, model='dmis-lab/biobert-v1.1', n=300, random_state=42,
                  save_fig="./figs/tsne_biobert.png", device="cuda", model_name="base")
    get_tsne_bert(df, model='dmis-lab/biobert-v1.1', n=300, random_state=42,
                  save_fig="./figs/tsne_biobert.png", device="cuda", model_name="BioBert")
    get_tsne_bert(df, model='answerdotai/ModernBERT-base', n=300, random_state=42,
                  save_fig="./figs/tsne_modernbert.png", device="cuda", model_name="ModernBertBase")
    get_tsne_bert(df, model='Simonlee711/Clinical_ModernBERT', n=300, random_state=42,
                  save_fig="./figs/clinical_modernbert.png", device="cuda", model_name="ClinicalModernBert")
    show_pos_embedding_base(save_fig = "./figs/pos_embd_base.png")
    get_rope_similarity(RoPE(d_model=64), 25, "./figs/rope_sim.png")
