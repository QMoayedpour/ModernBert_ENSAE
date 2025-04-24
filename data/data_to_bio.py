import os
from shutil import copyfile
import shutil
from data_extraction import (brat_to_bio,
                             split_train_test,
                             txt_to_df,
                             clean_up)
from spacy.lang.en import English

print('Processing ...')

nlp = English()

print(os.getcwd())

inputpath = f"./data_chia"
outputpath = f"./data_bio"
trainpath = f"./trains"
testpath = f"./tests"


inputfiles = set()
for f in os.listdir(inputpath):
    if f.endswith('.ann'):
        inputfiles.add(f.split('.')[0].split('_')[0])

select_types = ['Condition', 'Drug', 'Procedure', 'Observation', 'Person', 'Mood']

brat_to_bio(inputfiles, inputpath, outputpath, select_types)

chia_datasets = split_train_test(inputfiles, train=0.8)

with open("./train.txt", "w", encoding="utf-8") as f:
    for fid in chia_datasets["train"]:
        copyfile(f"{outputpath}/{fid}_exc.bio.txt", f"{trainpath}/{fid}_exc.bio.txt")
        copyfile(f"{outputpath}/{fid}_inc.bio.txt", f"{trainpath}/{fid}_inc.bio.txt")
        with open(f"{outputpath}/{fid}_exc.bio.txt", "r", encoding="utf-8") as fr:
            txt = fr.read().strip()
            if txt != '':
                f.write(txt)
                f.write("\n\n")
        with open(f"{outputpath}/{fid}_inc.bio.txt", "r", encoding="utf-8") as fr:
            txt = fr.read().strip()
            if txt != '':
                f.write(txt)
                f.write("\n\n")

with open("./dev.txt", "w", encoding="utf-8") as f:
    for fid in chia_datasets["dev"]:
        copyfile(f"{outputpath}/{fid}_exc.bio.txt", f"{trainpath}/{fid}_exc.bio.txt")
        copyfile(f"{outputpath}/{fid}_inc.bio.txt", f"{trainpath}/{fid}_inc.bio.txt")
        with open(f"{outputpath}/{fid}_exc.bio.txt", "r", encoding="utf-8") as fr:
            txt = fr.read().strip()
            if txt != '':
                f.write(txt)
                f.write("\n\n")
        with open(f"{outputpath}/{fid}_inc.bio.txt", "r", encoding="utf-8") as fr:
            txt = fr.read().strip()
            if txt != '':
                f.write(txt)
                f.write("\n\n")

with open("./test.txt", "w", encoding="utf-8") as f:
    for fid in chia_datasets["test"]:
        copyfile(f"{outputpath}/{fid}_exc.bio.txt", f"{testpath}/{fid}_exc.bio.txt")
        copyfile(f"{outputpath}/{fid}_inc.bio.txt", f"{testpath}/{fid}_inc.bio.txt")
        with open(f"{outputpath}/{fid}_exc.bio.txt", "r", encoding="utf-8") as fr:
            txt = fr.read().strip()
            if txt != '':
                f.write(txt)
                f.write("\n\n")
        with open(f"{outputpath}/{fid}_inc.bio.txt", "r", encoding="utf-8") as fr:
            txt = fr.read().strip()
            if txt != '':
                f.write(txt)
                f.write("\n\n")

txt_to_df("./train.txt", "./train2.csv")
txt_to_df("./test.txt", "./test2.csv")
txt_to_df("./dev.txt", "./dev2.csv")

print('Done !')


### Delete useless files


clean_up()