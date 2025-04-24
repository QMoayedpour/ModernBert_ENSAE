import json
import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.lang.en import English
import os
import shutil


nlp = English()


def get_annotation_entities(ann_file, select_types=None):
    entities = []
    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith('T'):
                term = line.strip().split('\t')[1].split()
                if (select_types is not None) and (term[0] not in select_types):
                    continue
                if int(term[-1]) <= int(term[1]):
                    continue
                entities.append((int(term[1]), int(term[-1]), term[0]))
    return sorted(entities, key=lambda x: (x[0], x[1]))


def remove_overlap_entities(sorted_entities):
    keep_entities = []
    for idx, entity in enumerate(sorted_entities):
        if idx == 0:
            keep_entities.append(entity)
            last_keep = entity
            continue
        if entity[0] < last_keep[1]:
            if entity[1]-entity[0] > last_keep[1]-last_keep[0]:
                last_keep = entity
                keep_entities[-1] = last_keep
        elif entity[0] == last_keep[1]:
            last_keep = (last_keep[0], entity[1], last_keep[-1])
            keep_entities[-1] = last_keep
        else:
            last_keep = entity
            keep_entities.append(entity)
    return keep_entities


def entity_dictionary(keep_entities, txt_file, file):
    f_ann = {}
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.readlines()
        if file in ['NCT02348918_exc', 'NCT02348918_inc', 'NCT01735955_exc']:
            text = ' '.join([i.strip() for i in text])
        else:
            text = '  '.join([i.strip() for i in text])
    for entity in keep_entities:
        entity_text = text[entity[0]:entity[1]]
        doc = nlp(entity_text)
        token_starts = [(i, doc[i:].start_char) for i in range(len(doc))]
        term_type = entity[-1]
        term_offset = entity[0]
        for i, token in enumerate(doc):
            ann_offset = token_starts[i][1]+term_offset
            if ann_offset not in f_ann:
                f_ann[ann_offset] = [i, token.text, term_type]
    return f_ann


def brat_to_bio(inputfiles, inputpath, outputpath, select_types):
    for infile in inputfiles:
        for t in ["exc", "inc"]:
            file = f"{infile}_{t}"
            ann_file = f"{inputpath}/{file}.ann"
            txt_file = f"{inputpath}/{file}.txt"
            out_file = f"{outputpath}/{file}.bio.txt"
            sorted_entities = get_annotation_entities(ann_file, select_types)
            keep_entities = remove_overlap_entities(sorted_entities)
            f_ann = entity_dictionary(keep_entities, txt_file, file)
            with open(out_file, "w", encoding="utf-8") as f_out:
                with open(txt_file, "r", encoding="utf-8") as f:
                    sent_offset = 0
                    for line in f:
                        if '⁄' in line:
                            line = line.replace('⁄', '/')
                        doc = nlp(line.strip())
                        token_starts = [(i, doc[i:].start_char) for i in range(len(doc))]
                        for token in doc:
                            token_sent_offset = token_starts[token.i][1]
                            token_doc_offset = token_starts[token.i][1]+sent_offset
                            if token_doc_offset in f_ann:
                                if f_ann[token_doc_offset][0] == 0:
                                    label = f"B-{f_ann[token_doc_offset][2]}"
                                else:
                                    label = f"I-{f_ann[token_doc_offset][2]}"
                            else:
                                label = f"O"

                            f_out.write(f"{token.text} {token_sent_offset} {token_sent_offset+len(token.text)} {token_doc_offset} {token_doc_offset+len(token.text)} {label}\n")
                        f_out.write('\n')
                        if file in ['NCT02348918_exc', 'NCT02348918_inc', 'NCT01735955_exc']:  # 3 trials with inconsistent offsets
                            sent_offset += (len(line.strip())+1)
                        else:
                            sent_offset += (len(line.strip())+2)


def split_train_test(inputfiles, train=0.8, randomstate=42):
    train_ids, dev_ids = train_test_split(list(inputfiles), train_size=train,
                                          random_state=randomstate, shuffle=True)
    dev_ids, test_ids = train_test_split(dev_ids, train_size=0.5,
                                         random_state=randomstate, shuffle=True)

    print(len(train_ids), len(dev_ids), len(test_ids))
    chia_datasets = {"train": train_ids, "dev": dev_ids, "test": test_ids}
    json.dump(chia_datasets, open("./chia_datasets.json", "w", encoding="utf-8"))
    return chia_datasets


def txt_to_df(path, output_path):
    fichier_chemin = path
    donnees = {'sentence': [], 'Mot': [], 'Label': []}
    with open(fichier_chemin, 'r') as fichier:
        ligne = fichier.readline()
        phrase_index = 1

        while ligne:
            mots_labels = ligne.strip().split()

            if len(mots_labels) >= 2:
                donnees['sentence'].append(phrase_index)
                donnees['Mot'].append(mots_labels[0])
                donnees['Label'].append(mots_labels[-1])
            else:
                phrase_index += 1

            ligne = fichier.readline()
    df = pd.DataFrame(donnees)
    df.to_csv(output_path)
    return df


def clean_up():
    for file in ["train.txt", "test.txt", "dev.txt", "chia_dataset.json"]:
        if os.path.exists(file):
            os.remove(file)

    for folder in ["./data_bio", "./trains", "./tests"]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f == "__init__.py":
                    continue
                path = os.path.join(folder, f)
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
