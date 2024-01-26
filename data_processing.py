import pandas as pd
import numpy as np
import os
import json

np.random.seed(42)

train_index_label = pd.read_csv("dataset/train.txt")

train_neg_label = train_index_label.loc[train_index_label["tag"] == "negative"]
train_pos_label = train_index_label.loc[train_index_label["tag"] == "positive"]
train_neu_label = train_index_label.loc[train_index_label["tag"] == "neutral"]

total_count_neg = len(train_neg_label)
total_count_pos = len(train_pos_label)
total_count_neu = len(train_neu_label)
print(total_count_neg, total_count_pos, total_count_neu)

train_count_neg = int(total_count_neg * 0.8)
train_count_pos = int(total_count_pos * 0.8)
train_count_neu = int(total_count_neu * 0.8)
print(train_count_neg, train_count_pos, train_count_neu)

train_neg = pd.DataFrame(train_neg_label).sample(n=train_count_neg).values
train_pos = pd.DataFrame(train_pos_label).sample(n=train_count_pos).values
train_neu = pd.DataFrame(train_neu_label).sample(n=train_count_neu).values
print(train_neg[:10], train_pos[:10], train_neu[:10])

val_neg = train_neg_label.loc[(~train_index_label["guid"].isin(train_neg[:, 0]))].values
val_pos = train_pos_label.loc[(~train_index_label["guid"].isin(train_pos[:, 0]))].values
val_neu = train_neu_label.loc[(~train_index_label["guid"].isin(train_neu[:, 0]))].values

print(val_neg[:10], val_pos[:10], val_neu[:10])

test_index_label = pd.read_csv("dataset/test_without_label.txt").values
print(test_index_label)

train_data = []
val_data = []
test_data = []

def read_data(data, dataset):
    for i in range(data.shape[0]):
        guid_str = str(int(data[i, 0]))
        current_dir = os.getcwd()
        image_path = os.path.join(current_dir + "/dataset/data/", f"{guid_str}.jpg")
        with open(current_dir + "/dataset/data/" + guid_str + ".txt", "rb") as f:
            encoding = "gb18030"

        with open(current_dir + "/dataset/data/" + guid_str + ".txt", encoding=encoding) as f:
            dataset.append({
                "guid": guid_str,
                "text": f.read().rstrip("\n"),
                "label": (data[i, 1] if data[0, 1] != "NaN" else None),
                "image": image_path
            })
    return dataset

current_dir = os.getcwd()
train_data = read_data(train_neg, train_data)
train_data = read_data(train_pos, train_data)
train_data = read_data(train_neu, train_data)

val_data = read_data(val_neg, val_data)
val_data = read_data(val_pos, val_data)
val_data = read_data(val_neu, val_data)

test_data = read_data(test_index_label, test_data)

with open(current_dir + "/dataset/train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f)
with open(current_dir + "/dataset/test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f)
with open(current_dir + "/dataset/val.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f)

