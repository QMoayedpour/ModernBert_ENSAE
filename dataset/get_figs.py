import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_n_words_per_setence(df, save=False):
    w_p_sentence = df['sentence'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    ax = sns.histplot(w_p_sentence, 
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


if __name__ == "__main__":

    df = pd.read_csv("../data/train.csv", index_col=0)
    get_n_words_per_setence(df, "./figs/distribution_words.png")
