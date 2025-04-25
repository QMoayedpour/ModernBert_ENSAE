import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import json


def load_colormap():
    with open("./colormap.json", "r") as f:
        cmap = json.load(f)

    cmap = {label: tuple(color) for label, color in cmap.items()}

    return cmap


def load_cmap():
    df = pd.read_csv("../data/train.csv", index_col=0)

    unique_labels = df['Label'].unique()
    unique_labels = [lab[2:] if lab != "O" else lab for lab in unique_labels]

    cmap = plt.get_cmap('tab10', len(unique_labels))

    cmap_dic = {label: cmap(i) for i, label in enumerate(unique_labels)}

    return (cmap, cmap_dic)


def show_sentence(df, cmap_dic, figsize=(6, 2), fontsize_txt=30, fontsize_lab=20, save=False):
    fig, ax = plt.subplots(figsize=figsize)

    ax.axis('off')

    for i, (mot, label) in enumerate(zip(df['Mot'], df['Label'])):
        color = cmap_dic[label[2:]] if label != "O" else cmap_dic["O"]
        ax.text(i, 1, mot, ha='center', va='center', fontsize=fontsize_txt, color=color, fontweight='bold')

        ax.text(i, 0, label, ha='center', va='center', fontsize=fontsize_lab)

    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches='tight')

    else:
        plt.show()


if __name__ == "__main__":
    _, cmap_dic = load_cmap()

    cmap_dic = {label: list(color) for label, color in cmap_dic.items()}

    with open("./colormap.json", "w") as f:
        json.dump(cmap_dic, f)

    _, cmap_dic = load_cmap()

    df_train = pd.read_csv("../data/train.csv", index_col=0)

    show_sentence(df_train[df_train["sentence"]==38], cmap_dic, save="figs/example_1.png")
