import numpy as np
from scipy.stats import rankdata


# 定义一个函数cal_ranks，用于计算得分的排名，参数有scores（得分矩阵）、labels（标签矩阵）和filters（过滤矩阵）
def cal_ranks(scores, labels, filters):
    # 将得分矩阵减去每一行的最小值，并加上一个很小的正数（1e-8），避免出现负数或零
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8

    # 对得分矩阵按行进行降序排名，使用平均法处理相同的得分，得到完整的排名矩阵
    full_rank = rankdata(-scores, method='average', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='min', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)


def make_zeros_one_matrix(ranks, n, m):
    zeros_one_matrix = np.zeros((n, m))
    for i in range(n):
        zeros_one_matrix[i, int(ranks[i]) - 1] = 1
    return zeros_one_matrix


def calc_metrics(binary_hit):
    tpr_list, fpr_list = [], []
    accuracy_list, recall_list, precision_list, f1_list = [], [], [], []
    for i in range(binary_hit.shape[1]):
        p_matrix, n_matrix = binary_hit[:, 0:i + 1], binary_hit[:, i + 1:binary_hit.shape[1] + 1]
        tp = np.count_nonzero(p_matrix)
        fp = np.sum(p_matrix == 0)
        tn = np.sum(n_matrix == 0)
        fn = np.count_nonzero(n_matrix)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        accuracy = (tn + tp) / (tn + tp + fn + fp)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = (2 * tp) / (2 * tp + fp + fn)
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


def load_data(filepath):
    with open(filepath, 'r') as file:
        data = np.loadtxt(file, delimiter=',', dtype=int)
    return data


# 计算每一行的非零元素个数
def count_nonzero_per_row(data):
    nonzero_counts = np.count_nonzero(data, axis=1)
    return nonzero_counts


def merge_rows(A, path , circ_number):
    # 指定文件路径
    file_path = path+"/circ_disease_association.txt"

    # 调用函数加载数据
    data_array = load_data(file_path)

    # 计算每一行的非零元素个数
    nonzero_counts = count_nonzero_per_row(data_array)
    P = nonzero_counts
    B = np.zeros((circ_number, A.shape[1]))
    current_row_in_a = 0

    for i, p in enumerate(P):
        if p == 1:
            # 如果P[i]为1，则直接复制A中的当前行到B
            B[i, :] = A[current_row_in_a, :]
            current_row_in_a += 1
        else:
            # 如果P[i]大于1，则合并接下来的p行
            B[i, :] = A[current_row_in_a:current_row_in_a + p, :].sum(axis=0)
            current_row_in_a += p
    return B


def filtration(a, start, end):
    b = a[:, start:start + end]
    return b
