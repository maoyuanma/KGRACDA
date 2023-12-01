import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
from utils_KGRACDA import select_gpu

parser = argparse.ArgumentParser(description="Parser for KGRACDA")
parser.add_argument('--data_path', type=str, default='data/dataset3')
parser.add_argument('--seed', type=str, default=2023)

args = parser.parse_args()

class Options(object):
    pass


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
# 创建一个Options对象，并设置其perf_file属性为结果保存的文件路径，
#文件名由数据集名称和'_perf.txt'组成
    opts = Options
    opts.perf_file = os.path.join(results_dir,  dataset + '_perf.txt')

    gpu = select_gpu()
    #gpu=2
    torch.cuda.set_device(gpu)
    print("current gpu device:",torch.cuda.current_device())

# 创建一个DataLoader对象，并传入数据集路径作为参数，
# 用于加载和处理数据集
    loader = DataLoader(args.data_path)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    
# 根据不同的数据集名称，设置不同的超参数值，
#包括学习率（lr）、权重衰减（lamb）、衰减率（decay_rate）、
#隐藏层维度（hidden_dim）、注意力维度（attn_dim）、模型层数（n_layer）
#批次大小（n_batch）、丢弃率（dropout）和激活函数（act）
    if dataset == 'dataset1':
        opts.lr = 0.005
        opts.lamb = 0.0002
        opts.decay_rate = 0.991
        opts.hidden_dim = 16
        opts.attn_dim = 8
        opts.dropout = 0.21
        opts.act = 'idd'
        opts.n_layer = 26
        opts.n_batch = 100


    elif dataset == 'dataset2':
        opts.lr = 0.0016
        opts.lamb = 0.0004
        opts.decay_rate = 0.994
        opts.hidden_dim = 8
        opts.attn_dim = 8
        opts.dropout = 0.02
        opts.act = 'relu'
        opts.n_layer = 32
        opts.n_batch = 20

    elif dataset == 'dataset3':
        opts.lr = 0.0014
        opts.lamb = 0.000034
        opts.decay_rate = 0.991

        opts.hidden_dim = 64
        opts.attn_dim = 32
        opts.dropout = 0.28
        opts.act = 'tanh'
        opts.n_layer = 32
        opts.n_batch = 20

    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str)

# 创建一个BaseModel对象，并传入Options对象和DataLoader对象作为参数，
#用于定义和训练模型
    model = BaseModel(opts, loader)

    best_auc = 0
    
    for epoch in range(5):
        
        # 调用model对象的train_batch方法，
        #返回当前轮次的MRR指标和输出字符串，分别赋值给mrr和out_str变量
        auc, out_str = model.train_batch()
        with open(opts.perf_file, 'a+') as f:
            f.write(out_str)

        if auc > best_auc:
            best_auc = auc
            best_str = out_str
            print(str(epoch) + '\t' + best_str)
    print(best_str)


