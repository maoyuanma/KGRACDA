import os
import torch
#导入scipy.sparse模块，用于实现稀疏矩阵相关的功能，如创建、存储、运算等
from scipy.sparse import csr_matrix
import numpy as np
#导入collections模块，用于实现高效的数据结构，如双端队列、计数器、有序字典等
from collections import defaultdict
class DataLoader:
    def __init__(self, task_dir):
        self.trans_dir = task_dir
        self.ind_dir = task_dir + '_ind'
        with open(os.path.join(task_dir, 'entities.txt')) as f:
            # 创建一个空字典，并赋值给self.entity2id属性，
            #表示实体到编号的映射关系
            self.entity2id = dict()
            for line in f:
                entity, eid = line.strip().split('\t')
                self.entity2id[entity] = int(eid)

        with open(os.path.join(task_dir, 'relations.txt')) as f:
            
            # 创建一个空字典，并赋值给self.relation2id属性，
            #表示关系到编号的映射关系
            self.relation2id = dict()
            id2relation = []
            for line in f:
                relation, rid = line.strip().split()
                self.relation2id[relation] = int(rid)
                id2relation.append(relation)

        with open(os.path.join(self.ind_dir, 'entities.txt')) as f:
            self.entity2id_ind = dict()
            for line in f:
                entity, eid = line.strip().split('\t')
                self.entity2id_ind[entity] = int(eid)

#遍历从0到self.relation2id字典的长度（即关系数量）的整数序列，
#每个整数作为一个关系编号，并赋值给i变量
        for i in range(len(self.relation2id)):
            id2relation.append(id2relation[i] + '_inv')

# 将'idd'字符串添加到id2relation列表中，表示自身关系
        id2relation.append('idd')
        self.id2relation = id2relation

        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)
        
        # 将self.entity2id_ind字典的长度（即归纳知识图谱中实体数量）赋值给self.n_ent_ind属性
        self.n_ent_ind = len(self.entity2id_ind)

 #调用自定义的read_triples方法，传入传统知识图谱数据集目录和'valid.txt'文件名作为参数，
 #返回验证集三元组列表，并赋值给self.tra_valid属性
        self.tra_train = self.read_triples(self.trans_dir, 'train.txt')
        self.tra_valid = self.read_triples(self.trans_dir, 'valid.txt')
        self.tra_test  = self.read_triples(self.trans_dir, 'test.txt')
        
 # 调用自定义的read_triples方法，传入归纳知识图谱数据集目录和'train.txt'文件名以及'inductive'模式作为参数，
 #返回归纳训练集三元组列表，并赋值给self.ind_train属性
        self.ind_train = self.read_triples(self.ind_dir,   'train.txt')
        self.ind_valid = self.read_triples(self.ind_dir,   'valid.txt')
        self.ind_test  = self.read_triples(self.ind_dir,   'test.txt')


        self.val_filters = self.get_filter('valid')
        
        self.tst_filters = self.get_filter('test')

        for filt in self.val_filters:
            self.val_filters[filt] = list(self.val_filters[filt])
        for filt in self.tst_filters:
            self.tst_filters[filt] = list(self.tst_filters[filt])

        self.tra_KG, self.tra_sub = self.load_graph(self.tra_train)
        self.ind_KG, self.ind_sub = self.load_graph(self.ind_train)
       

        self.tra_train = np.array(self.tra_valid)
        self.tra_val_qry, self.tra_val_ans = self.load_query(self.tra_test)
        self.ind_val_qry, self.ind_val_ans = self.load_query(self.ind_valid)
        self.ind_tst_qry, self.ind_tst_ans = self.load_query(self.ind_test)
        self.valid_q, self.valid_a = self.tra_val_qry, self.tra_val_ans
        self.test_q,  self.test_a  = self.ind_val_qry+self.ind_tst_qry, self.ind_val_ans+self.ind_tst_ans

        self.n_train = len(self.tra_train)
        self.n_valid = len(self.valid_q)
        self.n_test  = len(self.test_q)


        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

    def read_triples(self, directory, filename):
        triples = []
        with open(os.path.join(directory, filename)) as f:
            for line in f:
                h, r, t = line.strip().split('\t')

                h, r, t = self.entity2id_ind[h], self.relation2id[r], self.entity2id_ind[t]

                triples.append([h,r,t])
                triples.append([t, r+self.n_rel, h])
        return triples

    def load_graph(self, triples):
        n_ent = self.n_ent_ind
        
        KG = np.array(triples)
        idd = np.concatenate([np.expand_dims(np.arange(n_ent),1), 2*self.n_rel*np.ones((n_ent, 1)), np.expand_dims(np.arange(n_ent),1)], 1)
        KG = np.concatenate([KG, idd], 0)

        n_fact = KG.shape[0]

        M_sub = csr_matrix((np.ones((n_fact,)), (np.arange(n_fact), KG[:,0])), shape=(n_fact, n_ent))
        return KG, M_sub

    def load_query(self, triples):
        triples.sort(key=lambda x:(x[0], x[1]))
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_neighbors(self, nodes):


        KG    = self.ind_KG
        M_sub = self.ind_sub
        n_ent = self.n_ent_ind

        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

        mask = sampled_edges[:,2] == (self.n_rel*2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
       
        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data=='train':
            return self.tra_train[batch_idx]
        if data=='valid':
            query=np.array(self.valid_q)
            answer=self.valid_a
            n_ent = self.n_ent
        if data=='test':
            query= np.array(self.test_q)
            n_ent = self.n_ent_ind
            answer=self.test_a

        subs = []
        rels = []
        objs = []
        
        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def shuffle_train(self,):
        rand_idx = np.random.permutation(self.n_train)
        self.tra_train = self.tra_train[rand_idx]


# 定义一个方法get_filter，用于获取过滤集合，参数为data（数据集名称）
    def get_filter(self, data='valid'):
        
# 创建一个默认字典，并赋值给filters变量，表示过滤集合，键为头实体和关系的元组，值为尾实体的集合
        filters = defaultdict(lambda: set())
        
        # 如果data变量等于'valid'，表示获取传统知识图谱的验证集过滤集合，则执行以下代码
        if data == 'valid':
            
  # 遍历self.tra_train属性中的每个三元组，并赋值给triple变量           
            for triple in self.tra_train:
                h, r, t = triple
                filters[(h,r)].add(t)
                
# 将t变量添加到filters字典中以(h,r)为键的值中，表示该尾实体是头实体和关系的一个正确答案                
            for triple in self.tra_valid:
                h, r, t = triple
                filters[(h,r)].add(t)
                
                
            for triple in self.tra_test:
                h, r, t = triple
                filters[(h,r)].add(t)
        else:
            for triple in self.ind_train:
                h, r, t = triple
                filters[(h,r)].add(t)
            for triple in self.ind_valid:
                h, r, t = triple
                filters[(h,r)].add(t)
            for triple in self.ind_test:
                h, r, t = triple
                filters[(h,r)].add(t)
        return filters