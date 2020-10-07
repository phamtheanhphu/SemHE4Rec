import os

import numpy as np
from math import sqrt, fabs


class SemHE4Rec:
    def __init__(self,
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
                 lambda_constant):

        self.train_rating_file_path = train_rating_file_path
        self.test_rating_file_path = test_rating_file_path
        self.embeddings_dir_path = embeddings_dir_path

        self.rating_range = 5

        self.user_emb_dim = user_emb_dim
        self.user_metapaths = user_metapaths

        self.item_emb_dim = item_emb_dim
        self.item_metapaths = item_metapaths

        self.max_iterations = max_iterations

        self.alpha_hyper_param = alpha_hyper_param
        self.beta_hyper_param = beta_hyper_param

        self.lambda_constant = lambda_constant

        users_dict = {}
        items_dict = {}

        with open(self.train_rating_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    splits = line.split('\t')
                    if int(splits[0]) not in users_dict.keys():
                        users_dict[int(splits[0])] = 1
                    if int(splits[1]) not in items_dict.keys():
                        items_dict[int(splits[1])] = 1

        self.user_num = max(users_dict.keys()) + 1
        self.item_num = max(users_dict.keys()) + 1

        self.user_metapath_num = len(self.user_metapaths)
        self.item_metapath_num = len(self.item_metapaths)

        print('Load user/item embedding data...')
        self.X, self.user_metapath_dims = self.load_all_embeddings(user_metapaths, self.user_num, node_type='user')
        print('-> Total users: {}'.format(len(self.X.keys())))
        self.Y, self.item_metapath_dims = self.load_all_embeddings(item_metapaths, self.item_num, node_type='item')
        print('-> Total items: {}'.format(len(self.Y.keys())))

        print('Load train/test data...')
        self.R, self.T, self.ba = self.load_rating_data(self.train_rating_file_path, self.test_rating_file_path)
        print('-> Total train rating size: {}'.format(len(self.R)))
        print('-> Total test rating size: {}'.format(len(self.T)))

        print('Initializing SemHE4Rec model ...')
        self.init()
        print('-> Done')

    def load_all_embeddings(self, metapaths, node_num, node_type='user'):
        X = {}
        for i in range(node_num):
            X[i] = {}
        emb_dims = []

        acc = 0

        # Loading CoRL-based embeddings
        for metapath in metapaths:
            metapath_emb_file_path = os.path.join(self.embeddings_dir_path, '{}.embedding'.format(metapath))
            with open(metapath_emb_file_path, 'r', encoding='utf-8') as infile:
                k = int(infile.readline().strip().split(' ')[1])
                emb_dims.append(k)
                for i in range(node_num):
                    X[i][acc] = np.zeros(k)

                n = 0
                for line in infile.readlines():
                    n += 1
                    arr = line.strip().split(' ')
                    i = int(arr[0]) - 1
                    if i in X:
                        for j in range(k):
                            if j in X[i][acc]:
                                X[i][acc][j] = float(arr[j + 1])
                print('--> Reading total CoRL-based embeddings '
                      'for {}/metapath:[{}] = {}'.format(node_type, metapath, n))
            acc += 1

        # Loading SRL-based embeddings
        if node_type == 'user':
            sem_emb_file_path = os.path.join(self.embeddings_dir_path, 'user.embedding')
        else:
            sem_emb_file_path = os.path.join(self.embeddings_dir_path, 'item.embedding')

        with open(sem_emb_file_path) as infile:
            k = int(infile.readline().strip().split(' ')[1])
            for i in range(node_num):
                X[i][len(emb_dims)] = np.zeros(k)
            n = 0
            for line in infile.readlines():
                n += 1
                arr = line.strip().split(' ')
                i = int(arr[0]) - 1
                if i in X:
                    for j in range(k):
                        if j in X[i][len(emb_dims)]:
                            X[i][len(emb_dims)][j] = float(arr[j + 1])
            print('--> Reading total SRL-based embeddings for {} = {}'.format(node_type, n))
        return X, emb_dims

    def load_rating_data(self, train_rating_file_path, testfile):
        R_train = []
        R_test = []
        ba = 0.0
        n = 0
        with open(train_rating_file_path) as infile:
            for line in infile.readlines():
                user, item, rating, timestamp = line.strip().split('\t')
                R_train.append([int(user) - 1, int(item) - 1, int(rating.replace('.0', ''))])
                ba += int(rating.replace('.0', ''))
                n += 1
        ba = ba / n
        ba = 0
        with open(testfile) as infile:
            for line in infile.readlines():
                user, item, rating, timestamp = line.strip().split('\t')
                R_test.append([int(user) - 1, int(item) - 1, int(rating.replace('.0', ''))])

        return R_train, R_test, ba

    def cal_sem_u(self, i):
        ui_sem = np.zeros(self.user_emb_dim)
        ui_sem += self.p_usem[i] * self.sigmod_func(
            (self.W_usem.dot(self.X[i][self.user_metapath_num]) + self.b_usem))
        return self.sigmod_func(ui_sem)

    def cal_u(self, i):
        ui = np.zeros(self.user_emb_dim)
        for k in range(self.user_metapath_num):
            ui += self.pu[i][k] * (self.Wu[k].dot(self.X[i][k]) + self.bu[k])

        ui_sem = self.cal_sem_u(i)
        ui += ui_sem

        return ui

    def cal_sem_v(self, j):
        vj_sem = np.zeros(self.item_emb_dim)
        vj_sem += self.p_vsem[j] * self.sigmod_func(
            (self.W_vsem.dot(self.X[j][self.item_metapath_num]) + self.b_vsem))
        return self.sigmod_func(vj_sem)

    def cal_v(self, j):
        vj = np.zeros(self.item_emb_dim)
        for k in range(self.item_metapath_num):
            vj += self.pv[j][k] * (self.Wv[k].dot(self.Y[j][k]) + self.bv[k])

        vj_sem = self.cal_sem_v(j)
        vj += vj_sem

        return vj

    def get_rating(self, i, j):
        try:
            ui = self.cal_u(i)
            vj = self.cal_v(j)
            return self.U[i, :].dot(self.V[j, :]) + self.lambda_constant * ui.dot(
                self.H[j, :]) + self.lambda_constant * self.E[i, :].dot(
                vj)
        except IndexError:
            return None

    def evaluate_mae_rmse(self):
        m = 0.0
        mae_list = 0.0
        rmse_list = 0.0
        n = 0
        for t in self.T:
            try:
                n += 1
                i = t[0]
                j = t[1]
                r = t[2]
                r_p = self.get_rating(i, j)
                if r_p is None:
                    continue

                if r_p > 5: r_p = 5
                if r_p < 1: r_p = 1
                m = fabs(r_p - r)
                mae_list += m
                rmse_list += m * m
            except IndexError:
                continue

        mae_list = mae_list * 1.0 / n
        rmse_list = sqrt(rmse_list * 1.0 / n)
        return mae_list, rmse_list

    def sigmod_func(self, x):
        return 1 / (1 + np.exp(-x))

    def init(self):

        self.E = np.random.randn(self.user_num, self.item_emb_dim) * 0.1
        self.H = np.random.randn(self.item_num, self.user_emb_dim) * 0.1
        self.U = np.random.randn(self.user_num, self.rating_range) * 0.1
        self.V = np.random.randn(self.user_num, self.rating_range) * 0.1

        self.pu = np.ones((self.user_num, self.user_metapath_num)) * 1.0 / self.user_metapath_num
        self.pv = np.ones((self.item_num, self.item_metapath_num)) * 1.0 / self.item_metapath_num

        self.Wu = {}
        self.bu = {}

        for k in range(self.user_metapath_num):
            self.Wu[k] = np.random.randn(self.user_emb_dim, self.user_metapath_dims[k]) * 0.1
            self.bu[k] = np.random.randn(self.user_emb_dim) * 0.1

        self.Wv = {}
        self.bv = {}
        for k in range(self.item_metapath_num):
            self.Wv[k] = np.random.randn(self.item_emb_dim, self.item_metapath_dims[k]) * 0.1
            self.bv[k] = np.random.randn(self.item_emb_dim) * 0.1

        self.W_vsem = np.random.randn(self.item_emb_dim, 128) * 0.1
        self.b_vsem = np.random.randn(self.item_emb_dim) * 0.1
        self.p_vsem = np.ones((self.item_num, 1)) * 1.0 / self.item_metapath_num

        self.W_usem = np.random.randn(self.user_emb_dim, 128) * 0.1
        self.b_usem = np.random.randn(self.user_emb_dim) * 0.1
        self.p_usem = np.ones((self.user_num, 1)) * 1.0 / self.user_metapath_num

    def run(self):

        print('Training...')

        mae_list = []
        rmse_list = []

        c_error = 9999
        n = len(self.R)

        for iteration in range(self.max_iterations):
            total_error = 0.0

            for t in self.R:
                i = t[0]
                j = t[1]
                rij = t[2]

                rij_t = self.get_rating(i, j)
                if rij_t is None:
                    continue
                eij = rij - rij_t
                total_error += eij * eij

                U_g = -eij * self.V[j, :] + self.beta_hyper_param * self.U[i, :]
                V_g = -eij * self.U[i, :] + self.beta_hyper_param * self.V[j, :]

                self.U[i, :] -= self.alpha_hyper_param * U_g
                self.V[j, :] -= self.alpha_hyper_param * V_g

                ui = self.cal_u(i)
                for k in range(self.user_metapath_num):
                    pu_g = self.lambda_constant * -eij * self.H[j, :].dot(
                        self.Wu[k].dot(self.X[i][k]) + self.bu[k]) + self.beta_hyper_param * self.pu[i][k]
                    Wu_g = self.lambda_constant * -eij * self.pu[i][k] * np.array([self.H[j, :]]).T.dot(
                        np.array([self.X[i][k]])) + self.beta_hyper_param * self.Wu[k]
                    bu_g = self.lambda_constant * -eij * self.pu[i][k] * self.H[j, :] + self.beta_hyper_param * self.bu[
                        k]

                    self.pu[i][k] -= 0.1 * self.alpha_hyper_param * pu_g
                    self.Wu[k] -= 0.1 * self.alpha_hyper_param * Wu_g
                    self.bu[k] -= 0.1 * self.alpha_hyper_param * bu_g

                # Optimizing the CoRL-based embeddings fusion function
                x_sem_t = self.sigmod_func(self.W_usem.dot(self.X[i][self.user_metapath_num]) + self.b_usem)
                p_usem_g = self.lambda_constant * -eij * (ui * (1 - ui) * self.H[j, :]).dot(
                    x_sem_t) + self.beta_hyper_param * \
                           self.p_usem[
                               i]
                W_usem_g = self.lambda_constant * -eij * self.p_usem[i] * np.array(
                    [ui * (1 - ui) * x_sem_t * (1 - x_sem_t) * self.H[j, :]]).T.dot(
                    np.array([self.X[i][self.user_metapath_num]])) + self.beta_hyper_param * self.W_usem
                b_usem_g = self.lambda_constant * -eij * ui * (1 - ui) * self.p_usem[i] * self.H[j, :] * x_sem_t * (
                        1 - x_sem_t) + self.beta_hyper_param * self.b_usem

                self.p_usem[i] -= 0.1 * self.alpha_hyper_param * p_usem_g
                self.W_usem -= 0.1 * self.alpha_hyper_param * W_usem_g
                self.b_usem -= 0.1 * self.alpha_hyper_param * b_usem_g

                H_g = self.lambda_constant * -eij * ui + self.beta_hyper_param * self.H[j, :]
                self.H[j, :] -= self.alpha_hyper_param * H_g

                vj = self.cal_v(j)
                for k in range(self.item_metapath_num):
                    pv_g = self.lambda_constant * -eij * self.E[i, :].dot(
                        self.Wv[k].dot(self.Y[j][k]) + self.bv[k]) + self.beta_hyper_param * self.pv[j][k]
                    Wv_g = self.lambda_constant * -eij * self.pv[j][k] * np.array([self.E[i, :]]).T.dot(
                        np.array([self.Y[j][k]])) + self.beta_hyper_param * self.Wv[k]
                    bv_g = self.lambda_constant * -eij * self.pv[j][k] * self.E[i, :] + self.beta_hyper_param * self.bv[
                        k]

                    self.pv[j][k] -= 0.1 * self.alpha_hyper_param * pv_g
                    self.Wv[k] -= 0.1 * self.alpha_hyper_param * Wv_g
                    self.bv[k] -= 0.1 * self.alpha_hyper_param * bv_g

                # Optimizing the SRL-based embeddings fusion function
                y_sem_t = self.sigmod_func(self.W_vsem.dot(self.Y[j][self.item_metapath_num]) + self.b_vsem)
                p_vsem_g = self.lambda_constant * -eij * (vj * (1 - vj) * self.E[i, :]).dot(
                    y_sem_t) + self.beta_hyper_param * \
                           self.p_vsem[
                               j]
                W_vsem_g = self.lambda_constant * -eij * self.p_vsem[j] * np.array(
                    [vj * (1 - vj) * y_sem_t * (1 - y_sem_t) * self.E[i, :]]).T.dot(
                    np.array([self.Y[j][self.item_metapath_num]])) + self.beta_hyper_param * self.W_vsem
                b_vsem_g = self.lambda_constant * -eij * vj * (1 - vj) * self.p_vsem[j] * self.E[i, :] * y_sem_t * (
                        1 - y_sem_t) + self.beta_hyper_param * self.b_vsem

                self.p_vsem[j] -= 0.1 * self.alpha_hyper_param * p_vsem_g
                self.W_vsem -= 0.1 * self.alpha_hyper_param * W_vsem_g
                self.b_vsem -= 0.1 * self.alpha_hyper_param * b_vsem_g

                E_g = self.lambda_constant * -eij * vj + 0.01 * self.E[i, :]
                self.E[i, :] -= self.alpha_hyper_param * E_g

            p_error = c_error
            c_error = total_error / n

            self.alpha_hyper_param = 0.93 * self.alpha_hyper_param

            if abs(p_error - c_error) < 0.0001:
                break

            MAE, RMSE = self.evaluate_mae_rmse()
            mae_list.append(MAE)
            rmse_list.append(RMSE)
            print('- >Step: [{} -> MAE: [{:.6f}], RMSE: [{:.6f}]'.format(iteration, MAE, RMSE))

        print('-> Done !')
