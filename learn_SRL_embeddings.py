import os
import pandas as pd
import gensim
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# dataset = 'amazon-baby-100k'
dataset = 'amazon-baby-small'

rating_data_file_path = './inputs/{}/ratings.txt'.format(dataset)
output_embeddings_dir = './outputs/{}/embeddings'.format(dataset)
item_desc_data_file_path = './inputs/{}/item_desc_stemmed.json'.format(dataset)
word_embedding_file_path = os.path.join(output_embeddings_dir, 'words.embedding')

user_dict = {}
item_user_ratings = {}
ratings_csv = pd.read_csv(rating_data_file_path, sep='\t', encoding='utf-8')
for row in ratings_csv.values:
    user_id = int(row[0]) - 1
    item_id = int(row[1]) - 1
    rating = int(row[2]) - 1
    if user_id not in user_dict.keys():
        user_dict[user_id] = 1
    if item_id not in item_user_ratings.keys():
        item_user_ratings[item_id] = [(user_id, rating)]
    else:
        item_user_ratings[item_id].append((user_id, rating))

user_num = len(user_dict.keys())
item_num = len(item_user_ratings.keys())

print('Reading total users: {:d}'.format(user_num))
print('Reading total items: {:d}'.format(item_num))


def get_item_ratings_as_one_hot(item_id):
    user_ratings = item_user_ratings[item_id]
    ratings = np.zeros(len(user_ratings))
    acc = 0
    for (user_id, rating) in user_ratings:
        ratings[acc] = rating
        acc += 1
    return torch.LongTensor(ratings)


def save_emb_to_file(emb, file_path):
    emb = emb.weight.detach().numpy()
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('{} {}\n'.format(len(emb), len(emb[0])))
        for idx, vector in enumerate(emb):
            f.write('{} {}\n'.format(idx + 1, ' '.join([str(i) for i in vector])))


item_summary_words_embeddings = gensim.models.KeyedVectors.load_word2vec_format(word_embedding_file_path)
item_summary_embeddings = {}

with open(item_desc_data_file_path, 'r', encoding='utf-8') as f:
    item_summary_dict = json.load(f)
    for item_id in item_summary_dict.keys():
        item_id = int(item_id) - 1
        item_summary_embeddings[item_id] = []
        item_summary = item_summary_dict[str(item_id + 1)]
        for word in item_summary.split(' '):
            item_summary_embeddings[item_id].append(item_summary_words_embeddings.wv.get_vector(word))


class SRL_Encoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, item_num, user_num, rating_range):
        super(SRL_Encoder, self).__init__()
        self.is_cuda_available = False
        if torch.cuda.is_available():
            print('CUDA/GPU device is available: [{}]'.format(torch.cuda.get_device_name(0)))
            self.is_cuda_available = True

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.item_num = item_num
        self.user_num = user_num
        self.rating_range = rating_range

        self.item_embeddings = nn.Embedding(item_num, emb_dim)
        if self.is_cuda_available:
            self.item_embeddings = self.item_embeddings.cuda()

        self.user_embeddings = nn.Embedding(user_num, emb_dim)
        if self.is_cuda_available:
            self.user_embeddings = self.user_embeddings.cuda()

        self.gru_encoder = nn.GRU(emb_dim, hidden_dim)

        self.hidden2rating = nn.Linear(hidden_dim, rating_range)

    def forward(self, item_id, user_ids, word_embeddings):
        gru_out, _ = self.gru_encoder(word_embeddings)
        gru_out = torch.mean(gru_out, 0)
        gru_out = gru_out.view(1, self.emb_dim)

        item_emb = self.item_embeddings(torch.LongTensor([item_id]))
        user_emb = self.user_embeddings(torch.LongTensor([user_ids]))

        combined_item_emb = torch.mul(item_emb, gru_out)
        mul_item_user_emb = torch.mul(user_emb, combined_item_emb)

        rating_space = self.hidden2rating(mul_item_user_emb).view(len(user_ids), -1)
        rating_scores = F.softmax(rating_space, dim=1)
        return rating_scores


EMB_DIM = 128
HIDDEN_DIM = 128
rating_range = 5

print_every_epoch = 10

model = SRL_Encoder(EMB_DIM, HIDDEN_DIM, item_num, user_num, rating_range)

if torch.cuda.is_available():
    print('CUDA/GPU device is available: [{}]'.format(torch.cuda.get_device_name(0)))
    model.cuda()

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=.1)

epoch_num = 100

for epoch in range(epoch_num):
    total_loss = 0
    for item_id in item_user_ratings.keys():
        model.zero_grad()
        word_embeddings = torch.FloatTensor(item_summary_embeddings[item_id])
        word_embeddings = word_embeddings.view(len(item_summary_embeddings[item_id]), 1, -1)
        ground_truth_rating_scores = get_item_ratings_as_one_hot(item_id)
        user_ids = [user_id - 1 for (user_id, rating) in item_user_ratings[item_id]]
        predicted_rating_scores = model(item_id, user_ids, word_embeddings)
        loss = loss_func(predicted_rating_scores, ground_truth_rating_scores)
        total_loss += loss.detach().numpy()
        loss.backward()
        optimizer.step()

    if epoch % print_every_epoch == 0:
        print('Epoch: {:d} / loss: {:.5f}'.format(epoch, total_loss))

save_emb_to_file(model.item_embeddings, os.path.join(output_embeddings_dir, 'item.embedding'))
save_emb_to_file(model.user_embeddings, os.path.join(output_embeddings_dir, 'user.embedding'))
