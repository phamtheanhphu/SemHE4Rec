import os, sys
from SemHE4Rec import SemHE4Rec


def main():

    dataset = 'amazon-baby-100k'
    # dataset = 'amazon-baby-small'

    rating_data_dir = './outputs/{}/rating_data'.format(dataset)

    embeddings_dir_path = './outputs/{}/embeddings'.format(dataset)
    train_rating_file_path = os.path.join(rating_data_dir, 'ratings.train.txt')
    test_rating_file_path = os.path.join(rating_data_dir, 'ratings.test.txt')

    user_emb_dim = 128
    user_metapaths = ['uiu', 'uiciu']

    item_emb_dim = 128
    item_metapaths = ['ici', ]

    max_iterations = 100

    alpha_hyper_param = .01
    beta_hyper_param = 1
    lambda_constant = 1.0

    model = SemHE4Rec(
        train_rating_file_path,
        test_rating_file_path,
        embeddings_dir_path,
        user_emb_dim,
        user_metapaths,
        item_emb_dim,
        item_metapaths,
        max_iterations,
        alpha_hyper_param,
        beta_hyper_param,
        lambda_constant
    )

    model.run()


if __name__ == "__main__":
    sys.exit(main())
