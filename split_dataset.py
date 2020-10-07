import os
import random

# dataset = 'amazon-baby-100k'
dataset = 'amazon-baby-small'

rating_data_file_path = './inputs/{}/ratings.txt'.format(dataset)
output_dir = './outputs/{}/rating_data'.format(dataset)

train_rate = .8

R = []
with open(rating_data_file_path, 'r', encoding='utf-8') as infile:
    line_count = 0
    for line in infile.readlines():
        line_count += 1
        if line_count == 1:
            continue
        user, item, rating, timestamp = line.strip().split('\t')
        R.append([user, item, rating, timestamp])

random.shuffle(R)
train_rating_num = int(len(R) * train_rate)

with open(os.path.join(output_dir, 'ratings.train.txt'), 'w', encoding='utf-8') as train_data_file, \
        open(os.path.join(output_dir, 'ratings.test.txt'), 'w', encoding='utf-8') as test_data_file:
    for r in R[:train_rating_num]:
        train_data_file.write('\t'.join(r) + '\n')
    for r in R[train_rating_num:]:
        test_data_file.write('\t'.join(r) + '\n')
