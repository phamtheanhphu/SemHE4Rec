import os
import json
from gensim.models import word2vec

#dataset = 'amazon-baby-100k'
dataset = 'amazon-baby-small'

output_embeddings_dir = './outputs/{}/embeddings'.format(dataset)

item_desc_data_file_path = './inputs/{}/item_desc_stemmed.json'.format(dataset)
item_desc_data_file = open(item_desc_data_file_path, 'r', encoding='utf-8')
item_desc_data_dict = json.load(item_desc_data_file)

max_iter = 10
win_size = 5
emb_dim = 128

documents = []

output_word_embeddings_file = os.path.join(output_embeddings_dir, 'words.embedding')

for item_id in item_desc_data_dict.keys():
    item_desc = item_desc_data_dict[item_id]
    documents.append(item_desc.split(' '))

model = word2vec.Word2Vec(documents,
                          size=emb_dim,
                          window=win_size,
                          min_count=0,
                          sg=1,
                          hs=1,
                          iter=max_iter,
                          workers=8)

model.wv.save_word2vec_format(output_word_embeddings_file)
