from setuptools import setup, find_packages

setup(
    name="ModernBert_Project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pandas", "torch", "numpy",
                      "matplotlib", "scikit-learn",
                      "tqdm", "transformers", "spacy", "seaborn", "seqeval",
                      "keras_preprocessing", "wordcloud", "protobuf"],
)
