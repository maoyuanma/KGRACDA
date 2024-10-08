import torch
import torch.nn as nn


class GNN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x: x):
        super(GNN, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        # 创建一个nn.Embedding对象，并赋值给self.rela_embed属性，
        # 表示关系的嵌入矩阵，参数有2*n_rel+1（关系数量的两倍加一）、in_dim（输入维度）
        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)

        # 创建一个nn.Linear对象，并赋值给self.Ws_attn属性，表示头实体的注意力权重矩阵，
        # 参数有in_dim（输入维度）、attn_dim（注意力维度），bias为False（不使用偏置项）
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)

        # 创建一个nn.Linear对象，并赋值给self.Wr_attn属性，表示关系的注意力权重矩阵，
        # 参数有in_dim（输入维度）、attn_dim（注意力维度），bias为False（不使用偏置项）
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)

        # 创建一个nn.Linear对象，并赋值给self.Wqr_attn属性，表示查询关系的注意力权重矩阵，
        # 参数有in_dim（输入维度）、attn_dim（注意力维度）
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        # 创建一个nn.Linear对象，并赋值给self.W_h属性，表示隐藏状态的权重矩阵，参数有in_dim（输入维度）、out_dim（输出维度），bias为False（不使用偏置项）
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    # 定义一个前向传播方法
    # 参数有q_sub（查询头实体）、q_rel（查询关系）、hidden（隐藏状态矩阵）、edges（边信息矩阵）、n_node（节点数量）、old_nodes_new_idx（旧节点到新节点的索引映射）
    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        # 将edges矩阵中的第五列（旧节点索引）赋值给sub变量，表示头实体的旧节点索引
        sub = edges[:, 4]
        # 将edges矩阵中的第三列（关系索引）赋值给rel变量，表示关系的索引
        rel = edges[:, 2]
        # 将edges矩阵中的第六列（新节点索引）赋值给obj变量，表示尾实体的新节点索引
        obj = edges[:, 5]
        # 从hidden矩阵中按sub变量的值索引出头实体的隐藏状态，并赋值给hs变量
        hs = hidden[sub]
        # 从self.rela_embed矩阵中按rel变量的值索引出关系的嵌入向量，并赋值给hr变量
        hr = self.rela_embed(rel).cuda()
        # 将edges矩阵中的第一列（批次索引）赋值给r_idx变量，表示批次的索引
        r_idx = edges[:, 0]
        # 从self.rela_embed矩阵中按q_rel变量的值索引出查询关系的嵌入向量，并按r_idx变量的值索引出对应批次的查询关系，并赋值给h_qr变量
        h_qr = self.rela_embed(q_rel)[r_idx]
        # 将hs变量和hr变量相加，并赋值给message变量，表示信息传递的消息向量
        message = hs + hr

        # 将self.Ws_attn矩阵与hs变量相乘，再加上self.Wr_attn矩阵与hr变量相乘，
        # 再加上self.Wqr_attn矩阵与h_qr变量相乘，然后通过nn.ReLU函数进行非线性激活，
        # 再通过self.w_alpha向量进行线性映射，然后通过torch.sigmoid函数进行归一化，得到注意力得分，并赋值给alpha变量
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))

        # 将alpha变量和message变量相乘，并赋值给message变量，表示加权后的消息向量
        message = alpha * message.cuda()

        dim = 0

        # 创建一个空的输出张量，形状与输入张量在除了dim维度之外的所有维度上相同，而在dim维度上的大小等于dim_size
        out = torch.zeros(n_node, *message.shape[1:]).cuda()

        # 调用index_add函数，将message中的每个值加到由index指定的out中的相应位置上，并返回一个新的张量
        out = torch.index_add(out, dim=dim, index=obj, source=message).to(device='cuda')

        message_agg = out
        # 将self.W_h矩阵与message_agg矩阵相乘，并通过self.act函数进行非线性激活，
        # 得到新的隐藏状态矩阵，并赋值给hidden_new变量
        hidden_new = self.act(self.W_h(message_agg))
        return hidden_new


class KGRACDA(torch.nn.Module):
    def __init__(self, params, loader):
        super(KGRACDA, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNN(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

    # 定义一个前向传播方法，参数有subs（头实体列表）、rels（关系列表）、mode（模式）
    def forward(self, subs, rels):

        # 获取头实体列表的长度，并赋值给n变量，表示批次大小
        n = len(subs)

        # 获取头实体列表的长度，并赋值给n变量，表示批次大小
        n_ent = self.loader.n_ent_ind
        # 将头实体列表转换为torch.LongTensor类型，并移动到cuda设备上，并赋值给q_sub变量，表示查询头实体的张量
        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()

        # 创建一个全零张量，形状为(1, n, self.hidden_dim)，并移动到cuda设备上，并赋值给h0变量，表示初始的隐藏状态矩阵
        h0 = torch.zeros((1, n, self.hidden_dim)).cuda()

        # 将批次索引和查询头实体的张量按列拼接，并赋值给nodes变量，表示当前的节点信息矩阵
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)

        # 创建一个全零张量，形状为(n, self.hidden_dim)，并移动到cuda设备上，并赋值给hidden变量，表示当前的隐藏状态矩阵
        hidden = torch.zeros(n, self.hidden_dim).cuda()

        # 遍历从0到self.n_layer（图神经网络层数）的整数序列，每个整数作为一个层编号，并赋值给i变量
        for i in range(self.n_layer):
            # 调用self.loader对象的get_neighbors方法，传入nodes矩阵和mode参数作为参数，返回邻居节点信息、边信息和旧节点到新节点的索引映射，并分别赋值给nodes、edges和old_nodes_new_idx变量
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy())

            # 调用self.gnn_layers列表中第i个元素（即GNNLayer对象）的forward方法，
            # 传入q_sub、q_rel、hidden、edges、nodes.size(0)（节点数量）和old_nodes_new_idx参数作为参数，
            # 返回新的隐藏状态矩阵，并赋值给hidden变量
            hidden = self.gnn_layers[i](q_sub.cuda(), q_rel.cuda(), hidden.cuda(), edges.cuda(), nodes.size(0),
                                        old_nodes_new_idx).cuda()

            # 调用self.gnn_layers列表中第i个元素（即GNNLayer对象）的forward方法，
            # 传入q_sub、q_rel、hidden、edges、nodes.size(0)（节点数量）和old_nodes_new_idx参数作为参数，
            # 返回新的隐藏状态矩阵，并赋值给hidden变量
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)

            # 将h0张量按旧节点到新节点的索引映射进行复制，并赋值给h0变量
            hidden = self.dropout(hidden)

            # 调用self.dropout对象对hidden张量进行随机失活操作，并赋值给hidden变量
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

        # 调用self.W_final矩阵对hidden张量进行线性映射，并从最后一个维度上压缩，得到得分向量，并赋值给scores变量
        scores = self.W_final(hidden).squeeze(-1)
        # 创建一个全零张量，形状为(n, n_ent)，并移动到cuda设备上，并赋值给scores_all变量，表示所有实体的得分矩阵
        scores_all = torch.zeros((n, n_ent)).cuda()
        # 将scores_all张量按nodes矩阵中的批次索引和节点索引进行复制，并赋值给scores_all变量
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        return scores_all
