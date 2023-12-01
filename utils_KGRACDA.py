import numpy as np
from scipy.stats import rankdata
# 导入subprocess模块，用于创建和管理子进程，执行系统命令等
import subprocess
import logging

# 定义一个函数cal_ranks，用于计算得分的排名，
# 参数有scores（得分矩阵）、labels（标签矩阵）和filters（过滤矩阵）
def cal_ranks(scores, labels, filters):
    
    # 将得分矩阵减去每一行的最小值，并加上一个很小的正数（1e-8），
    # 避免出现负数或零
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    
    # 对得分矩阵按行进行降序排名，使用平均法处理相同的得分，
    # 得到完整的排名矩阵
    full_rank = rankdata(-scores, method='average', axis=1)
    
    filter_scores = scores * filters
    
    filter_rank = rankdata(-filter_scores, method='min', axis=1)

    ranks = (full_rank - filter_rank + 1) * labels

    ranks = ranks[np.nonzero(ranks)]
    
    return list(ranks)

def select_gpu():
    nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
    gpu_info = False
    gpu_info_line = 0
    proc_info = False
    gpu_mem = []
    gpu_occupied = set()
    i = 0
    for line in nvidia_info.stdout.split(b'\n'):
        line = line.decode().strip()
        if gpu_info:
            gpu_info_line += 1
            if line == '':
                gpu_info = False
                continue
            if gpu_info_line % 3 == 2:
                mem_info = line.split('|')[2]
                used_mem_mb = mem_info.strip().split()[0][:-3]
                gpu_mem.append(used_mem_mb)
        if proc_info:
            if line == '|  No running processes found                                                 |':
                continue
            if line == '+-----------------------------------------------------------------------------+':
                proc_info = False
                continue
            proc_gpu = int(line.split()[1])
            # proc_type = line.split()[3]
            gpu_occupied.add(proc_gpu)
        if line == '|===============================+======================+======================|':
            gpu_info = True
        if line == '|=============================================================================|':
            proc_info = True
        i += 1
    for i in range(0,len(gpu_mem)):
        if i not in gpu_occupied:
            logging.info('Automatically selected GPU %d because it is vacant.', i)
            return i
    for i in range(0,len(gpu_mem)):
        if gpu_mem[i] == min(gpu_mem):
            logging.info('All GPUs are occupied. Automatically selected GPU %d because it has the most free memory.', i)
            return i
def make_zeros_one_matrix(ranks,n,m):
    zeros_one_matrix= np.zeros((n, m))
    for i in range(n):
        zeros_one_matrix[i, int(ranks[i])-1] = 1

    return zeros_one_matrix


def calc_metrics(binary_hit):

    tpr_list, fpr_list = [], []
    accuracy_list, recall_list, precision_list, f1_list = [], [], [], []
    for i in range(binary_hit.shape[1]):
        p_matrix, n_matrix = binary_hit[:, 0:i+1], binary_hit[:, i+1:binary_hit.shape[1]+1]
        tp = np.sum(p_matrix == 1)
        fp = np.sum(p_matrix == 0)
        tn = np.sum(n_matrix == 0)
        fn = np.sum(n_matrix == 1)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = (tn+tp) / (tn+tp+fn+fp)
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1 = (2*tp) / (2*tp + fp + fn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)

    return tpr_list, fpr_list, accuracy_list, recall_list, precision_list, f1_list

def calc_metrics2(all_tpr, all_fpr, all_recall, all_precision, all_accuracy, all_f1):
    
    mean_accuracy = np.mean(np.mean(np.array(all_accuracy)))
    mean_recall = np.mean(np.mean(np.array(all_recall)))
    mean_precision = np.mean(np.mean(np.array(all_precision)))
    mean_f1 = np.mean(np.mean(np.array(all_f1)))
    auc = np.trapz(all_tpr, all_fpr)  
    aupr = np.trapz(all_precision, all_recall)

    return mean_accuracy, mean_recall, mean_precision, mean_f1, auc, aupr
 

        

