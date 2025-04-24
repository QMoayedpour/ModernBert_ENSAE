# Data Extraction

We will use clinical trial data. To begin with, first download the dataset [here](https://figshare.com/articles/dataset/Chia_Annotated_Datasets/11855817?file=21728853) and download the folder ``chia_without_scope.zip``. After you unzipped it, put all the files in ``data_chia`` then run:


```bash
cd data

python data_to_bio.py
```

Please, make sure you are in the directory ``data`` before running the command. Once the data extraction is finished, you should have 3 csv (train, test and dev).