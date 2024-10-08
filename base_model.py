import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models_KGRACDA import KGRACDA
from utils_KGRACDA import cal_ranks, make_zeros_one_matrix, calc_metrics, calc_metrics2, merge_rows, filtration


class BaseModel(object):
    def __init__(self, args, loader):
        self.model = KGRACDA(args, loader)
        self.model.cuda()
        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_ent_ind = loader.n_ent_ind
        self.n_batch = args.n_batch
        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test = loader.n_test
        self.n_layer = args.n_layer
        # 创建一个Adam优化器对象，并赋值给self.optimizer属性，表示优化器对象，
        # 参数有self.model.parameters()（模型参数）、lr（学习率）、weight_decay（权重衰减）
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        # 参数有self.optimizer（优化器对象）、args.decay_rate（衰减率）
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        # 将0赋值给self.t_time属性，表示训练时间
        self.t_time = 0

        self.disease_number = len(loader.disease_list)
        self.circ_number = len(loader.circ_list)
        self.path = loader.trans_dir

    # 定义一个方法train_batch，用于训练一个批次的数据，并返回验证集上的平均倒数排名和输出字符串
    def train_batch(self, ):
        # 将0赋值给epoch_loss变量，表示一个周期的损失值
        epoch_loss = 0
        # 将0赋值给i变量，表示批次编号
        i = 0
        # 将self.n_batch属性赋值给batch_size变量，表示批次大小
        batch_size = self.n_batch
        # 将self.n_train属性除以batch_size变量，再加上self.n_train属性对batch_size变量取余是否大于0的布尔值，得到一个整数，并赋值给n_batch变量，表示批次数量
        n_batch = self.n_train // batch_size + (self.n_train % batch_size > 0)

        # 将self.model对象设置为训练模式
        self.model.train()

        # 遍历从0到n_batch（批次数量）的整数序列，每个整数作为一个批次编号，并赋值给i变量
        for i in range(n_batch):
            start = i * batch_size
            end = min(self.n_train, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            # 调用self.loader对象的get_batch方法，传入batch_idx作为参数，返回一个训练集三元组数组，并赋值给triple变量
            triple = self.loader.get_batch(batch_idx)
            # 调用self.model对象的zero_grad方法，将模型参数的梯度清零
            self.model.zero_grad()

            # 调用self.model对象的forward方法，传入triple数组中的第一列（头实体）和第二列（关系）作为参数，返回一个得分矩阵，并赋值给scores变量
            scores = self.model(triple[:, 0], triple[:, 1]).cuda()

            # 从scores矩阵中按行和列索引出正例得分，
            # 并赋值给pos_scores变量，行索引为torch.arange(len(scores)).cuda()（从0到scores矩阵长度的整数序列），
            # 列索引为torch.LongTensor(triple[:,2]).cuda()（triple数组中的第三列（尾实体））
            pos_scores = scores[[torch.arange(len(scores)).cuda(), torch.LongTensor(triple[:, 2]).cuda()]]

            # 从scores矩阵中按行求最大值，并保持维度不变，得到一个最大值矩阵，并赋值给max_n变量
            max_n = torch.max(scores, 1, keepdim=True)[0]

            # 将负的pos_scores变量加上max_n变量再加上scores矩阵减去max_n矩阵后按行求指数再求和再取对数，
            # 得到一个损失向量，并求和得到一个损失值，并赋值给loss变量
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n), 1)))

            # 调用loss变量的backward方法，计算损失值对模型参数的梯度
            loss.backward()
            # 调用self.optimizer对象的step方法，更新模型参数
            self.optimizer.step()

            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                # 判断X变量中是否有NaN（非数字）元素，并返回一个布尔矩阵，并赋值给flag变量
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)

            epoch_loss += loss.item()
        self.scheduler.step()

        valid_auc, out_str = self.evaluate()
        return valid_auc, out_str

    def evaluate(self, ):
        # 对valid数据集
        batch_size = self.n_batch
        n_data = self.n_valid
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()

        for i in range(n_batch):
            start = i * batch_size
            end = min(n_data, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')
            scores = self.model(subs, rels).data.cpu().numpy()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.val_filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent,))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)

            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks

        ranking = np.array(ranking)
        zero_one_matrix = make_zeros_one_matrix(ranking, len(ranking), len(scores[0]))
        length = int(len(zero_one_matrix)/2)
        zero_one_matrix = zero_one_matrix[:length]
        end_zero_matrix = merge_rows(zero_one_matrix, self.path, self.circ_number)

        tpr_list, fpr_list, accuracy_list, recall_list, precision_list, f1_list = calc_metrics(end_zero_matrix)
        v_mean_accuracy, v_mean_recall, v_mean_precision, v_mean_f1, v_auc, v_aupr = calc_metrics2(tpr_list, fpr_list, recall_list, precision_list, accuracy_list, f1_list)

        # 对test数据集
        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        self.model.eval()
        ranking = []
        for i in range(n_batch):
            start = i * batch_size
            end = min(n_data, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='test')

            scores = self.model(subs, rels).data.cpu().numpy()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.tst_filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent_ind,))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)
            filters = np.array(filters)

            scores = filtration(scores,self.circ_number,self.disease_number)
            filters = filtration(filters,self.circ_number,self.disease_number)
            objs = filtration(objs,self.circ_number,self.disease_number)

            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        zero_one_matrix = make_zeros_one_matrix(ranking, len(ranking), len(scores[0]))
        length = int(len(zero_one_matrix)/2)
        zero_one_matrix = zero_one_matrix[:length]

        end_zero_matrix = merge_rows(zero_one_matrix, self.path, self.circ_number)

        tpr_list, fpr_list, accuracy_list, recall_list, precision_list, f1_list = calc_metrics(end_zero_matrix)
        t_mean_accuracy, t_mean_recall, t_mean_precision, t_mean_f1, t_auc, t_aupr = calc_metrics2(tpr_list, fpr_list,recall_list,precision_list,accuracy_list,f1_list)

        out_str = '[VALID] accuracy:%.4f recall:%.4f precision:%.4f f1:%.4f auc:%.4f aupr:%.4f \t [TEST]  accuracy:%.4f\
         recall:%.4f precision:%.4f f1:%.4f auc:%.4f aupr:%.4f  \n' % (v_mean_accuracy,\
        v_mean_recall, v_mean_precision,v_mean_f1, v_auc, v_aupr, t_mean_accuracy, t_mean_recall, t_mean_precision,
            t_mean_f1, t_auc, t_aupr,)
        return t_auc, out_str
