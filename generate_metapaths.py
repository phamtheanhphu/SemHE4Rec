import os
import numpy as np
import pandas as pd

#dataset = 'amazon-baby-100k'
dataset = 'amazon-baby-small'

output_metapath_dir = './outputs/{}/meta-paths'.format(dataset)

ui_link_file_path = './inputs/{}/ratings.txt'.format(dataset)
ic_link_file_path = './inputs/{}/item_cat.txt'.format(dataset)

user_dict = {}
item_dict = {}
cat_dict = {}

user_item_rel_csv = pd.read_csv(ui_link_file_path, encoding='utf-8', sep='\t', header=None)
for i in user_item_rel_csv.values:
    if int(i[0]) not in user_dict.keys():
        user_dict[int(i[0])] = 1
    if int(i[1]) not in item_dict.keys():
        item_dict[int(i[1])] = 1

item_cat_rel_csv = pd.read_csv(ic_link_file_path, encoding='utf-8', sep='\t', header=None)
for i in item_cat_rel_csv.values:
    if int(i[1]) not in cat_dict.keys():
        cat_dict[int(i[1])] = 1

user_num = max(user_dict.keys()) + 1
print('Loading total users: {}'.format(user_num))

item_num = max(item_dict.keys()) + 1
print('Loading total items: {}'.format(item_num))

genre_num = max(cat_dict.keys()) + 1
print('Loading total categories: {}'.format(genre_num))

ui = np.zeros((user_num, item_num))
for i in user_item_rel_csv.values:
    ui[int(i[0])][int(i[1])] = 1

ic = np.zeros((item_num, genre_num))
for i in item_cat_rel_csv.values:
    ic[int(i[0])][int(i[1])] = 1

uiu = ui.dot(ui.T)
with open(os.path.join(output_metapath_dir, 'uiu.txt'), 'w', encoding='utf-8') as f:
    for i in range(uiu.shape[0]):
        for j in range(uiu.shape[1]):
            if uiu[i][j] != 0 and i != j:
                f.write(str(i) + '\t' + str(j) + '\t' + str(int(uiu[i][j])) + '\n')

ici = ic.dot(ic.T)
with open(os.path.join(output_metapath_dir, 'ici.txt'), 'w', encoding='utf-8') as f:
    for i in range(ic.shape[0]):
        for j in range(ic.shape[1]):
            if ic[i][j] != 0 and i != j:
                try:
                    f.write(str(i) + '\t' + str(j) + '\t' + str(int(ici[i][j])) + '\n')
                except Exception:
                    continue

uiciu = ui.dot(ic).dot(ic.T).dot(ui.T)
with open(os.path.join(output_metapath_dir, 'uiciu.txt'), 'w', encoding='utf-8') as f:
    for i in range(uiciu.shape[0]):
        for j in range(uiciu.shape[1]):
            if uiciu[i][j] != 0 and i != j:
                f.write(str(i) + '\t' + str(j) + '\t' + str(int(uiciu[i][j])) + '\n')
