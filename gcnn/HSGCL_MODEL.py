from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from  torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import pywt  # 导入PyWavelets库
import xlrd
import torch.nn.init as init
import csv
import pandas as pd
# from count_data import Data_make
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,confusion_matrix
# from transformers import BertModel,BertTokenizer
import math
from torch.nn.parameter import Parameter
import scipy.sparse as sp
from scipy.sparse import issparse, diags, linalg
from torch.nn.modules.module import Module
# from MultiHeadSelfAttention import MultiHeadSelfAttention
from torch.optim.lr_scheduler import LambdaLR
import manifolds
import ptwt
# from attten_new import LRSA
from math import exp

torch.backends.cudnn.enabled = True  # 确保 cuDNN 已启用（默认状态）
torch.backends.cudnn.benchmark = False  # ★ 关键：禁用自动寻找最优算法 ★
torch.backends.cudnn.deterministic = True  # 可选：确保结果可复现


def read_data1(train_or_test,name,num=None):
    texts=[]
    with open("../data/"+ name +"/"+train_or_test+".csv",newline='',encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for sheet_name in reader:
            texts.append(sheet_name[0])
        if num == None:
            return texts
        else:
            return texts[:num]
#构建语料库，输入训练文本，构建出初始的embedding模型
#构建语料库，输入训练文本，构建出初始的embedding模型
def built_curpus1(train_texts,tokenizer,embedding_num):
    # pad文本填充一样长度，unk识别不出的文字
    word_2_index = {"<PAD>":0,"<UNK>":1}
    for text in train_texts:
        words = text.split()
        # a = tokenizer.tokenize(text)
        for word in words:
            # 检测字并生成字典
            #如果这个字不在word_2_index中，那么就将此字生成字典，后续文本生成语句特征是查找用。
            word_2_index[word] = word_2_index.get(word,len(word_2_index))
        #######下面的过程是利用bert训练好的矩阵嵌入到当前训练的文本数据集中。############
    # new_embedding = np.random.rand(len(word_2_index), embedding_num)
    # # 将 NumPy 数组转换为 PyTorch 张量
    # embedding_tensor = torch.from_numpy(new_embedding).float()
    # # 创建 nn.Embedding 对象
    # # embedding = torch.nn.Embedding.from_pretrained(embedding_tensor)
    # feature = torch.FloatTensor(embedding_tensor).requires_grad_(True)

    with open("vocab.txt", encoding="utf-8") as f2:
        bert_vacb = f2.read().split("\n")
        bert_word_2_index = {w: i for i, w in enumerate(bert_vacb)}
    bert_emb = np.loadtxt("bert_embedding_R52_save.txt")
    new_embedding = []
    for w in word_2_index:
        new_embedding.append(bert_emb[bert_word_2_index.get(w, 1)])
    new_embedding = np.asarray(new_embedding)
    # 将 NumPy 数组转换为 PyTorch 张量
    embedding_tensor = torch.from_numpy(new_embedding).float()
    # 创建 nn.Embedding 对象
    # embedding = torch.nn.Embedding.from_pretrained(embedding_tensor)
    feature = torch.FloatTensor(embedding_tensor).requires_grad_(True)
    return word_2_index, feature

def read_data(train_or_test,name,num=None):
    texts=[]
    labels=[]
    with open("../data/"+ name +"/"+train_or_test+".csv",newline='',encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for sheet_name in reader:
            texts.append(sheet_name[0])
            labels.append(sheet_name[1])
        if num == None:
            return texts,labels
        else:
            return texts[:num],labels[:num]



class TextDataset(Dataset):
    #将外部传入的数据提前写到自己的函数内部
    def __init__(self,all_text,all_label,word_2_index,max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.word_2_index = word_2_index
        self.max_len = max_len
        # self.tokenizer = tokenizer
    #统计抽取的batch_size数据的个数以及对每条数据做何种处理（获取数据和数据预处理）
    #1、根据index获取数据；2、文本段数据变为数字形式；3、数据长度规格化max_leb
    def __getitem__(self, index):
        # 双重确保类型安全
        text = str(self.all_text[index])
        # 1. 获取完整文本内容
        # text = self.all_text[index]

        # 2. 分割文本为单词列表
        words = text.split()

        # 3. 只保留存在于word_2_index中的字符
        valid_words = [word for word in words if word in self.word_2_index]

        # 4. 裁剪到最大长度
        valid_words = valid_words[:self.max_len]

        # 5. 转换为索引序列
        text_idx = [self.word_2_index[word] for word in valid_words]

        # 6. 填充不足的长度
        text_idx = text_idx + [0] * (self.max_len - len(text_idx))

        label = int(self.all_label[index])
        text_idx = torch.tensor(text_idx).unsqueeze(dim=0)
        return text_idx, label
    def __len__(self):
        return len(self.all_text)

class GraphConvolution2(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features,bias=True):
        super(GraphConvolution2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.relu = nn.ReLU()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)
    def forward(self, input, adj):
        output1 = torch.mm(input, self.weight) + self.bias
        output1 = torch.sparse.mm(adj, output1)
        return output1

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, manifold,in_features, out_features,dropout):
        super(GraphConvolution, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        # self.c=c
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.dropout = dropout
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)
    def forward(self, input, adj,self_c):
        output1 = self.manifold.mobius_matvec1(self.weight,input,self_c) #每个特征乘上权重

        bias1 = self.manifold.proj_tan0(self.bias.view(1, -1), self_c)
        hyp_bias = self.manifold.expmap0(bias1, self_c)
        hyp_bias = self.manifold.proj(hyp_bias, self_c)
        output1 = self.manifold.mobius_add(output1, hyp_bias, c=self_c)  # 将偏置加到结果上

        output1 = self.manifold.logmap0(output1, c=self_c)
        output1 = torch.sparse.mm(adj, output1) # 特征与网络结构相乘，聚合信息传递
        output1 = self.manifold.proj(self.manifold.expmap0(output1, c=self_c), c=self_c)

        return output1 # output2

class HypLinear(nn.Module):
    """
    超双曲线性层。
    """
    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold  # 流形空间
        self.in_features = in_features  # 输入特征维度
        self.out_features = out_features  # 输出特征维度
        self.c = c  # 曲率参数
        self.dropout = dropout  # dropout概率
        self.use_bias = use_bias  # 是否使用偏置
        self.bias = nn.Parameter(torch.Tensor(out_features))  # 偏置参数
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))  # 权重参数
        self.reset_parameters()  # 初始化参数
    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))  # 使用Xavier初始化权重
        init.constant_(self.bias, 0)  # 将偏置初始化为0
    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)  # 应用dropout
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)  # 计算加权矩阵和输入的Moebius矩阵向量乘积
        res = self.manifold.proj(mv, self.c)  # 将结果投影回流形空间
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)  # 将偏置投影到切空间
            hyp_bias = self.manifold.expmap0(bias, self.c)  # 将偏置从切空间映射回流形空间
            hyp_bias = self.manifold.proj(hyp_bias, self.c)  # 再次投影以确保在流形空间内
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)  # 将偏置加到结果上
            res = self.manifold.proj(res, self.c)  # 最终结果投影回流形空间
        return res
    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )  # 用于打印额外的层信息

class MODEL_1(nn.Module):
    def __init__(self, embedding_num, output1_dim,nclass, dropout,max_len,aaa):
        super(MODEL_1, self).__init__()
        self.output1_dim=output1_dim
        self.nclass=nclass
        self.max_len = max_len
        self.gc1 = GraphConvolution2(embedding_num-1, self.output1_dim, dropout)
        self.gc2 = GraphConvolution2(self.output1_dim, self.output1_dim,dropout)  #注意这里的gcn
        self.gc3 = GraphConvolution2(self.output1_dim, self.output1_dim, dropout)  # 注意这里的gcn
        self.n = Parameter(torch.FloatTensor(aaa, 1))  # 设定一个保留原数据特征的参数

        self.relu = nn.ReLU()

        self.Lin1 = Parameter(torch.FloatTensor(embedding_num-1, self.output1_dim))
        self.Lin1_bias = Parameter(torch.FloatTensor(self.output1_dim))

        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):

        torch.nn.init.uniform_(self.n, 0, 1)

        init.kaiming_uniform_(self.Lin1, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.Lin1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.Lin1_bias, -bound, bound)


    def forward(self, A1_tensor,adj):

        A1_tensor1 = self.dropout(A1_tensor)[:, 1:]
        A1_tensor1 = self.dropout(A1_tensor1)
        a0 = torch.mm(A1_tensor1, self.Lin1) + self.Lin1_bias

        a1 = a0 * self.n
        x_1 = self.gc1(A1_tensor1, adj)
        x_1 = a1 + (x_1 * (1-self.n))
        x_2 = self.gc2(x_1, adj)
        x_2 = self.gc3(x_2, adj)
        x_2 = torch.cat([x_2, a0], dim=-1)
        return x_2

# 模型
class MLDEL_2(nn.Module):
    def __init__(self, embedding_num, output1_dim,nclass, dropout,max_len,aaa):
        super(MLDEL_2, self).__init__()
        self.manifold = getattr(manifolds, 'Hyperboloid')()
        # ✅ 可学习曲率参数（初始化为 softplus⁻¹(1.0)）
        self.raw_c = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1)]))
        self.softplus = nn.Softplus()

        self.nclass = nclass
        self.output1_dim=output1_dim
        self.gc1 = GraphConvolution(self.manifold,embedding_num, output1_dim, dropout)
        self.gc2 = GraphConvolution(self.manifold,self.output1_dim, self.output1_dim, dropout)  # 注意这里的gcn
        self.n = Parameter(torch.Tensor(aaa, 1))  # 设定一个保留原数据特征的参数

        self.relu = nn.ReLU()

        self.Lin1 = Parameter(torch.Tensor(embedding_num, self.output1_dim))
        self.Lin1_bias = Parameter(torch.Tensor(self.output1_dim))

        self.dropout = nn.Dropout(p=dropout)


        self.reset_parameters()

    def get_c(self):
        # ✅ 返回正数曲率（始终 > 0）
        return self.softplus(self.raw_c) + 1e-5

    def reset_parameters(self):
        torch.nn.init.uniform_(self.n, 0, 1)

        init.kaiming_uniform_(self.Lin1, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.Lin1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.Lin1_bias, -bound, bound)


    def forward(self, A1_tensor,adj):
        self_c = self.get_c()  # ✅ 获取当前曲率

        A1_tensor1 = self.dropout(A1_tensor)
        x_tan = self.manifold.proj_tan0(A1_tensor1, self_c)
        x_hyp = self.manifold.expmap0(x_tan, c=self_c)
        a = self.manifold.proj(x_hyp, c=self_c)#############   注意，这里面有一个将结果投影到流

        a1 = self.manifold.mobius_matvec1(self.Lin1,a,self_c)
        # 加上偏置
        bias1 = self.manifold.proj_tan0(self.Lin1_bias.view(1, -1), self_c)  # 将偏置投影到切空间，这是以原点为参考线的切空间
        hyp_bias = self.manifold.expmap0(bias1, self_c)  # 将偏置从切空间映射回流形空间
        hyp_bias = self.manifold.proj(hyp_bias, self_c)  # 再次投影以确保在流形空间内

        a1 = self.manifold.mobius_add(a1, hyp_bias, c=self_c)  # 将偏置加到结果上，那么这个单个特征加偏置是以原点为参考点进行考虑
        a2 = self.manifold.mobius_matvec0(self.n, a1, self_c)

        x_1 = self.gc1(a, adj,self_c)
        x_1 = self.manifold.mobius_matvec0((1-self.n), x_1, self_c)
        x_1 = self.manifold.mobius_add0(x_1, a2, c=self_c)  # 优化不了

        x_2 = self.gc2(x_1, adj,self_c)
        x_2 = self.manifold.logmap0(x_2, c=self_c)
        # 这里先不进行线性操作，直接进行下一步的对比学习操作
        x_2 = torch.cat([x_2, a1], dim=-1)
        x_2 = self.manifold.expmap0(x_2, c=self_c)
        return x_2,self_c

import torch.nn as nn

class MLDEL_3(nn.Module):
    def __init__(self, nclass,max_len):
        super(MLDEL_3, self).__init__()
        self.nclass = nclass
        self.nomel_0 = nn.LayerNorm(128 * 3)
        self.max_len = max_len
        self.relu = nn.ReLU()

        self.n1 = nn.Linear(128, 128)
        self.n2 = nn.Linear(128, 128)
        self.n3 = nn.Linear(128, 128)

        self.weight1 = nn.Parameter(torch.ones(1, 128*3))

        self.liner_1 = nn.Linear(128*3, 128*2)
        self.liner_2 = nn.Linear(128*2, 128)

        self.classifier = Parameter(torch.Tensor(128*2, self.nclass))
        self.classifier_bias = Parameter(torch.Tensor(self.nclass))

        self.loss_fun = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):

        init.kaiming_uniform_(self.classifier, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.classifier)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.classifier_bias, -bound, bound)


    def forward(self, feature_1,feature_2,feature_3, batch_label=None):

        feature_1 = self.n1(feature_1)
        feature_2 = self.n2(feature_2)
        feature_3 = self.n3(feature_3)

        feature1 = feature_1 + feature_2 + feature_3


        feature2 = torch.cat([feature_1,feature_2, feature_3], dim=1)

        feature2 = self.relu(feature2)
        feature2 = self.liner_1(feature2)

        feature2 = self.liner_2(feature2)
        feature = torch.cat([feature1, feature2], dim=1)


        pre = torch.mm(feature, self.classifier) + self.classifier_bias

        if batch_label is not None:
            loss = self.loss_fun(pre, batch_label)
            return loss, pre
        else:
            return torch.argmax(pre, dim=-1), pre


def read_data0(train_or_test, name, num=None):
    # 构建文件路径
    file_path = f"../data/{name}_0/{train_or_test}.xlsx"

    # 使用pandas读取xlsx文件
    df = pd.read_excel(file_path, engine='openpyxl', header=None)

    # 假设文本在第一列，标签在第二列
    texts = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].tolist()

    if num is None:
        return texts, labels
    else:
        return texts[:num], labels[:num]

from torch.nn import Parameter




import torch
import torch.nn as nn

class DUIBI_LOSS_3(nn.Module):
    def __init__(self, temperature, aaa):
        super(DUIBI_LOSS_3, self).__init__()
        self.manifold = getattr(manifolds, 'Hyperboloid')()
        self.temperature = temperature
        self.eps = 1e-8
        self.aaa = aaa

    def euclidean_similarity(self, x, y):
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        sim = torch.matmul(x_norm, y_norm.transpose(0, 1))
        return sim / self.temperature

    def hyperbolic_distance(self, x, y, c):
        return self.manifold.sqdist(x, y, c)

    def hyperbolic_similarity(self, x, y, c):
        dist = self.hyperbolic_distance(x, y, c)
        sim = torch.exp(-dist)
        return sim

    def intra_space_loss(self, feat, adj, space_type='euclidean', c=None):
        n = feat.size(0)
        if space_type == 'euclidean':
            sim_matrix = self.euclidean_similarity(feat, feat)
        elif space_type == 'hyperbolic':
            sim_matrix = self.hyperbolic_similarity(feat, feat, c)

        adj_dense = adj.to_dense() if adj.is_sparse else adj
        eye_mask = torch.eye(n, dtype=torch.bool, device=feat.device)

        pos_mask = adj_dense.clone()
        pos_mask.fill_diagonal_(0)
        pos_counts = pos_mask.sum(dim=1)
        valid_nodes = pos_counts > 0
        numerator = (torch.exp(sim_matrix) * pos_mask).sum(dim=1)
        neg_mask = (adj_dense == 0) & ~eye_mask
        denominator = (torch.exp(sim_matrix) * neg_mask).sum(dim=1)

        loss_per_node = -torch.log(numerator / (denominator + self.eps) + self.eps)
        return loss_per_node[valid_nodes].mean() if valid_nodes.any() else torch.tensor(0.0, device=feat.device)

    def inter_space_loss(self, feat_A, feat_B, adj_A, space_type='euclidean', c=None):
        if space_type == 'euclidean':
            cross_sim = self.euclidean_similarity(feat_A, feat_B)
        elif space_type == 'hyperbolic':
            cross_sim = self.hyperbolic_similarity(feat_A, feat_B, c)

        adj_dense = adj_A.to_dense() if adj_A.is_sparse else adj_A
        pos_mask = adj_dense.clone()
        pos_mask.fill_diagonal_(1)
        neg_mask = (adj_dense == 0)
        neg_mask.fill_diagonal_(0)

        numerator = (torch.exp(cross_sim) * pos_mask).sum(dim=1)
        denominator = (torch.exp(cross_sim) * neg_mask).sum(dim=1)
        loss_per_node = -torch.log(numerator / (denominator + self.eps) + self.eps)
        return loss_per_node.mean()

    def forward(self, H, D, adj_1, adj_2, c):
        # 空间A：欧式空间内部对比损失
        loss_a = self.intra_space_loss(H, adj_1, space_type='euclidean')

        # 空间B：双曲空间内部对比损失
        loss_b = self.intra_space_loss(D, adj_2, space_type='hyperbolic', c=c)

        # 空间之间：D 对 H 映射为双曲，计算相似度
        H_hyper = self.manifold.expmap0(H, c=c)
        loss_inter_AB = self.inter_space_loss(D, H_hyper, adj_2, space_type='hyperbolic', c=c)

        # 空间之间：D 映射为欧式，与 H 进行对比
        D_euc = self.manifold.logmap0(D, c=c)
        loss_inter_BA = self.inter_space_loss(H, D_euc, adj_1, space_type='euclidean')

        total_loss_A = loss_a + loss_inter_AB
        total_loss_B = loss_b + loss_inter_BA
        last_loss = (total_loss_A + total_loss_B) / self.aaa
        return last_loss

class MLDEL_5(nn.Module):
    def __init__(self, embedding_num, output1_dim,nclass, dropout,max_len,aaa):
        super(MLDEL_5, self).__init__()
        self.output1_dim=output1_dim
        self.nclass=nclass
        self.max_len = max_len
        self.gc1 = GraphConvolution2(embedding_num, self.output1_dim, dropout)
        self.gc2 = GraphConvolution2(self.output1_dim, self.output1_dim,dropout)  #注意这里的gcn
        self.n = Parameter(torch.FloatTensor(aaa, 1))  # 设定一个保留原数据特征的参数

        self.relu = nn.ReLU()

        self.Lin1 = Parameter(torch.FloatTensor(embedding_num, self.output1_dim))
        self.Lin1_bias = Parameter(torch.FloatTensor(self.output1_dim))

        self.dropout = nn.Dropout(p=dropout)
        self.dropout0 = dropout

        self.weight2 = Parameter(torch.FloatTensor(self.output1_dim, 64))
        self.bias2 = nn.Parameter(torch.FloatTensor(64))

        self.liner_1 = nn.Linear((max_len + 1) * 64, self.output1_dim)
        self.classifier = Parameter(torch.FloatTensor(self.output1_dim, self.nclass))
        self.classifier_bias = Parameter(torch.FloatTensor(self.nclass))


        # loss，损失梯度
        self.loss_fun = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.n, 0, 1)

        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight2)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias2, -bound, bound)

        init.kaiming_uniform_(self.Lin1, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.Lin1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.Lin1_bias, -bound, bound)

        init.kaiming_uniform_(self.classifier, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.classifier)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.classifier_bias, -bound, bound)

    def forward(self, A1_tensor,adj,batch_idx, batch_label=None):
        batch_idx1=batch_idx.squeeze(dim=1)

        drop_weight2 = F.dropout(self.weight2, self.dropout0, training=self.training)  # 应用dropout

        A1_tensor1 = self.dropout(A1_tensor)
        a0 = torch.mm(A1_tensor1, self.Lin1) + self.Lin1_bias

        a1 = a0 * self.n
        x_1 = self.gc1(A1_tensor1, adj)
        x_1 = a1 + (x_1 * (1-self.n))
        x_2 = self.gc2(x_1, adj)


        select_dims1 = x_2[batch_idx1]

        sum_per_row = torch.sum(select_dims1, dim=1).unsqueeze(1)
        select_dims1 = torch.cat((select_dims1, sum_per_row), dim=1)  # 加上全局的特征

        select_dims1 = torch.matmul(select_dims1, drop_weight2) + self.bias2
        select_dims1 = select_dims1.view(batch_idx1.shape[0], -1)  # 这里将特征变为[batch_size,len*feature]
        select_dims1 = self.liner_1(select_dims1)


        pre = torch.mm(select_dims1, self.classifier) + self.classifier_bias

        if batch_label is not None:
            loss = self.loss_fun(pre, batch_label)
            return loss, select_dims1
        else:
            return torch.argmax(pre, dim=-1), select_dims1  # 返回类别最大值对应下标，及预测值


# 模型
class MLDEL_6(nn.Module):
    def __init__(self, embedding_num, output1_dim,nclass, dropout,max_len,aaa):
        super(MLDEL_6, self).__init__()
        self.manifold = getattr(manifolds, 'Hyperboloid')()
        # ✅ 可学习曲率参数（初始化为 softplus⁻¹(1.0)）
        self.raw_c = nn.Parameter(torch.tensor([math.log(math.exp(1.0) - 1)]))
        self.softplus = nn.Softplus()

        self.nclass = nclass
        self.output1_dim = output1_dim
        self.gc1 = GraphConvolution(self.manifold,embedding_num, output1_dim, dropout)
        self.gc2 = GraphConvolution(self.manifold,self.output1_dim, self.output1_dim, dropout)  # 注意这里的gcn
        self.n = Parameter(torch.Tensor(aaa, 1))  # 设定一个保留原数据特征的参数

        self.relu = nn.ReLU()

        self.Lin1 = Parameter(torch.Tensor(embedding_num, self.output1_dim))
        self.Lin1_bias = Parameter(torch.Tensor(self.output1_dim))

        self.dropout = nn.Dropout(p=dropout)
        self.dropout0 = dropout

        self.weight = Parameter(torch.Tensor(self.output1_dim, 64))
        self.bias = nn.Parameter(torch.FloatTensor(64))


        self.liner_1 = nn.Linear((max_len + 1) * 64, self.output1_dim)

        self.classifier = Parameter(torch.Tensor(self.output1_dim, self.nclass))
        self.classifier_bias = Parameter(torch.Tensor(self.nclass))

        self.loss_fun = nn.CrossEntropyLoss()

        self.reset_parameters()

    def get_c(self):
        # ✅ 返回正数曲率（始终 > 0）
        return self.softplus(self.raw_c) + 1e-5

    def reset_parameters(self):
        torch.nn.init.uniform_(self.n, 0, 1)

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

        init.kaiming_uniform_(self.Lin1, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.Lin1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.Lin1_bias, -bound, bound)
        init.kaiming_uniform_(self.classifier, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.classifier)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.classifier_bias, -bound, bound)


    def forward(self, A1_tensor,adj,batch_idx, batch_label=None):
        self_c = self.get_c()  # ✅ 获取当前曲率

        batch_idx1=batch_idx.squeeze(dim=1)

        a1 = self.manifold.mobius_matvec1(self.Lin1,A1_tensor,self_c)
        # 加上偏置
        bias1 = self.manifold.proj_tan0(self.Lin1_bias.view(1, -1), self_c)  # 将偏置投影到切空间，这是以原点为参考线的切空间
        hyp_bias = self.manifold.expmap0(bias1, self_c)  # 将偏置从切空间映射回流形空间
        hyp_bias = self.manifold.proj(hyp_bias, self_c)  # 再次投影以确保在流形空间内
        a1 = self.manifold.mobius_add(a1, hyp_bias, c=self_c)  # 将偏置加到结果上，那么这个单个特征加偏置是以原点为参考点进行考虑
        a2 = self.manifold.mobius_matvec0(self.n, a1, self_c)

        x_1 = self.gc1(A1_tensor, adj,self_c)
        x_1 = self.manifold.mobius_matvec0((1-self.n), x_1, self_c)
        x_1 = self.manifold.mobius_add0(x_1, a2, c=self_c)  # 优化不了
        x_2 = self.gc2(x_1, adj,self_c)

        select_dims2 =  x_2[batch_idx1]

        drop_weight3 = F.dropout(self.weight, self.dropout0, training=self.training)

        select_dims2 = self.manifold.logmap3(select_dims2, self_c)
        sum_per_row = torch.sum(select_dims2, dim=1).unsqueeze(1)
        select_dims2 = torch.cat((select_dims2, sum_per_row), dim=1) # 加上全局的特征

        select_dims2 = torch.matmul(select_dims2, drop_weight3)
        select_dims2 = self.relu(select_dims2)
        select_dims2 = select_dims2.view(batch_idx1.shape[0], -1)  # 这里将特征变为[batch_size,len*feature]
        select_dims2 = self.liner_1(select_dims2)


        pre = torch.mm(select_dims2, self.classifier) + self.classifier_bias
        if batch_label is not None:
            loss = self.loss_fun(pre, batch_label)
            return loss, select_dims2,self_c
        else:
            return torch.argmax(pre, dim=-1), select_dims2  # 返回类别最大值对应下标，及预测值

class Block(nn.Module):
    def __init__(self,kernel_s,embeddin_num,max_len,hidden_num):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=1,out_channels=hidden_num,kernel_size=(kernel_s,embeddin_num)) #  1 * 1 * 7 * 5 (batch *  in_channel * len * emb_num )
        self.act = nn.ReLU()
        self.mxp = nn.MaxPool1d(kernel_size=(max_len-kernel_s+1))

    def forward(self,batch_emb): # 1 * 1 * 7 * 5 (batch *  in_channel * len * emb_num )
        c = self.cnn.forward(batch_emb)
        a = self.act.forward(c)
        a = a.squeeze(dim=-1)#去维度
        m = self.mxp.forward(a)
        m = m.squeeze(dim=-1)
        return m

class TextCNNModel(nn.Module):
    #传入embedding、**、标签数
    def __init__(self,emb_num,max_len,class_num,hidden_num):#初始化
        super().__init__()#集成父类初始化函数
        self.emb_num = emb_num
        #卷积层
        self.block1 = Block(2,self.emb_num,max_len,hidden_num)
        self.block2 = Block(3,self.emb_num,max_len,hidden_num)
        self.block3 = Block(4,self.emb_num,max_len,hidden_num)
        self.block4 = Block(5, self.emb_num, max_len, hidden_num)

        #分类
        self.classifier = nn.Linear(hidden_num*4,class_num)  # 2 * 3？？？ 2*4？
        #loss，损失梯度
        self.loss_fun = nn.CrossEntropyLoss()
    #文字onhot
    def forward(self,features,batch_idx1,batch_label=None):
        #将一段文本的onhot编码转成embedding矩阵
        batch_emb = features[batch_idx1]
        b1_result = self.block1.forward(batch_emb)
        b2_result = self.block2.forward(batch_emb)
        b3_result = self.block3.forward(batch_emb)
        b4_result = self.block4.forward(batch_emb)

        feature = torch.cat([b1_result,b2_result,b3_result,b4_result],dim=1) # 1* 6 : [ batch * (3 * 2)]

        pre = self.classifier(feature)

        if batch_label is not None:
            loss = self.loss_fun(pre,batch_label)
            return loss,feature
        else:
            return torch.argmax(pre,dim=-1),feature


if __name__ == "__main__":
    epoch = 200
    batch_size = 10
    max_len = 10
    hidden_num = 64
    output1_dim = 128
    output2_dim = 128
    name = 'Twitter'   # mr Twitter  snippets   StackOverflow   ohsumed TagMyNews
    import pickle

    with open('../glove/word_index_' + name + '.pkl',
              'rb') as f:
        word_2_index = pickle.load(f)
    feature = torch.load(
        '../glove/embedding_' + name + '.pt')

    feature = feature.cpu()
    feature = feature.float()
    max_norms = torch.max(torch.abs(feature), dim=1, keepdim=True)[0]
    max_norms[max_norms == 0] = 1
    feature = feature / max_norms

    train_texts1, train_labels_p = read_data0("train-new",name)
    assert len(train_texts1) == len(train_labels_p)

    type_list = list(set(train_labels_p))
    train_labels = []
    for i in train_labels_p:
        index = type_list.index(i)
        train_labels.append(str(index))

    dev_texts, dev_labels_p = read_data0("validation-new",name)
    assert len(dev_texts) == len(dev_labels_p)

    dev_labels = []
    for i in dev_labels_p:
        index = type_list.index(i)
        dev_labels.append(str(index))

    # 构建随机领接矩阵
    text_texts1, text_labels_p = read_data0("test-new", name)
    assert len(text_texts1) == len(text_labels_p)

    text_labels = []
    for i in text_labels_p:
        index = type_list.index(i)
        text_labels.append(str(index))

    del type_list
    class_num = len(set(train_labels))
    print('class_num',class_num)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    train_dataset = TextDataset(train_texts1, train_labels, word_2_index,max_len)
    train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=False) # 这里的训练集全加入


    dev_dataset = TextDataset(dev_texts, dev_labels, word_2_index, max_len)
    dev_dataloader = DataLoader(dev_dataset, 32, shuffle=False)  # batch_size  这里有改变的

    test_dataset = TextDataset(text_texts1, text_labels, word_2_index, max_len)
    test_dataloader = DataLoader(test_dataset, 520, shuffle=False)  # batch_size  这里有改变的

    del train_texts1,train_dataset,dev_dataset,test_dataset,dev_texts,text_texts1

    aaa = len(feature)

    M = 25
    adj = torch.load('D_adj_' + name + '.pt', weights_only=False).to_dense()          # ohsumed _max  snippets 是 _new     StackOverflow _2_max   TagMyNews     _new0_fei_suiji

# #######################  择优  #########################
    adj[range(aaa, aaa)] = 0
    top_values, top_indices = torch.topk(adj, M, dim=1)
    adj.zero_()
    adj.scatter_(1, top_indices, top_values)

    adj_1 = adj.clone()
    adj_1[adj_1 != 0] = 1

    indices = torch.nonzero(adj_1).t()
    values = adj_1[indices[0], indices[1]]
    adj_1 = torch.sparse_coo_tensor(indices, values, adj_1.size())


    indices = torch.nonzero(adj)
    values = adj[indices[:, 0], indices[:, 1]]
    i = indices.t()
    adj = torch.sparse_coo_tensor(i, values, adj.size())
##########################################################
    adj2 = torch.load('H_adj_' + name + '.pt', weights_only=False)


    adj2[range(aaa, aaa)] = 0
    top_values, top_indices = torch.topk(adj2, M, dim=1)
    adj2.zero_()
    adj2.scatter_(1, top_indices, top_values)

    adj_2 = adj2.clone()
    adj_2[adj_2 != 0] = 1
    indices = torch.nonzero(adj_2).t()
    values = adj_2[indices[0], indices[1]]
    adj_2 = torch.sparse_coo_tensor(indices, values, adj_2.size())

    indices = torch.nonzero(adj2)
    values = adj2[indices[:, 0], indices[:, 1]]
    i = indices.t()
    adj2 = torch.sparse_coo_tensor(i, values, adj2.size())

    del indices,values,i
    del top_values, top_indices


    wavelet = 'bior3.5'
    level = 3

    feature = feature.to(device)
    adj = adj.to(device)
    adj2 = adj2.to(device)

    adj_1 = adj_1.to(device)
    adj_2 = adj_2.to(device)

    embedding_num1 = feature.shape[1] + 1
#######################



    for i in range(100):
        print(f"第{i}次训练")
        # model_0 = WaveletTextEnhancer(wavelet='bior3.5', level=3, D_3=D_3, H_3=H_3, H_2=H_2, H_1=H_1, feature_len=aaa)
        # model_0 = model_0.to(device)

        model_1 = MODEL_1(embedding_num=embedding_num1,
                      output1_dim=output1_dim,
                      nclass=class_num,
                      dropout=0.5,
                      max_len=max_len,
                      aaa=aaa)
        model_1 = model_1.to(device)

        model_2 = MLDEL_2(embedding_num=embedding_num1,
                      output1_dim=output1_dim,
                      nclass=class_num,
                      dropout=0.5,
                      max_len=max_len,
                      aaa=aaa)
        model_2 = model_2.to(device)

        model_5 = MLDEL_5(embedding_num=256,
                          output1_dim=128,
                          nclass=class_num,
                          dropout=0.5,
                          max_len=max_len,
                          aaa=aaa)
        model_5 = model_5.to(device)

        model_6 = MLDEL_6(embedding_num=256,
                          output1_dim=128,
                          nclass=class_num,
                          dropout=0.5,
                          max_len=max_len,
                          aaa=aaa)
        model_6 = model_6.to(device)

        model_3 = MLDEL_3(nclass=class_num,
                          max_len=max_len)
        model_3 = model_3.to(device)

        model_4 = DUIBI_LOSS_3(temperature=0.1,aaa=aaa)
        model_4 = model_4.to(device)

        model_7 = TextCNNModel(emb_num=256,max_len=max_len, class_num=class_num, hidden_num=32) # hidden_num 的值是 128/卷积核的值
        model_7 = model_7.to(device)

        lr = 0.01

        opt_4 = torch.optim.Adam([
            {'params': model_1.parameters(), 'lr': 0.01},
            {'params': model_2.parameters(), 'lr': 0.01}
        ])
        scheduler_1 = torch.optim.lr_scheduler.StepLR(opt_4, step_size=20, gamma=0.5)  # 步数调整
        #
        opt_3 = torch.optim.AdamW(model_3.parameters(), lr=lr)
        scheduler_3 = torch.optim.lr_scheduler.StepLR(opt_3, step_size=20, gamma=0.5)  # 步数调整

        opt_5 = torch.optim.AdamW(model_5.parameters(), lr=lr)
        scheduler_5 = torch.optim.lr_scheduler.StepLR(opt_5, step_size=20, gamma=0.5)  # 步数调整

        opt_6 = torch.optim.AdamW(model_6.parameters(), lr=lr)
        scheduler_6 = torch.optim.lr_scheduler.StepLR(opt_6, step_size=20, gamma=0.5)  # 步数调整

        opt_7 = torch.optim.AdamW(model_7.parameters(), lr=lr)
        scheduler_7 = torch.optim.lr_scheduler.StepLR(opt_7, step_size=20, gamma=0.5)  # 步数调整

        loss_5 = -1
        loss_6 = -1
        loss_3 = -1
        loss_4 = -1
        loss_7 = -1
        count = 0
        accuracys = 0
        loss_min = 0
        best_models = {}

        for e in range(epoch):
            accuracysss = 0
            f1_max = 0

            o = torch.zeros_like(feature)
            enhanced_features = torch.cat([o[:, 0:1], feature], dim=1)
            enhanced_features = enhanced_features.to(device)

            model_1.train()
            model_2.train()
            feature_1 = model_1.forward(enhanced_features, adj)
            feature_2,self_c = model_2.forward(enhanced_features, adj2)
            # print("self_c1",self_c)
            loss_4 = model_4(feature_1, feature_2,adj_1,adj_2,self_c)

            loss_4.backward()
            opt_4.step()
            opt_4.zero_grad()

            with torch.no_grad():
                feature_1 = feature_1.detach()
                feature_2 = feature_2.detach()

            model_5.train()
            model_6.train()

            model_3.train()
            model_7.train()
            total_task_loss = 0
            batch_count = 0
            self_c2 = 0
            for batch_idx, batch_label in train_dataloader:

                batch_idx = batch_idx.to(device)
                batch_label = batch_label.to(device)

                loss_5, feature_5 = model_5.forward(feature_1, adj, batch_idx, batch_label)
                if loss_5.item() == 0:
                    loss_5 = torch.tensor(0.01, requires_grad=True)
                loss_5.backward()
                opt_5.step()
                opt_5.zero_grad()

                loss_6, feature_6,self_c_2 = model_6.forward(feature_2, adj2, batch_idx, batch_label)
                if loss_6.item() == 0:
                    loss_6 = torch.tensor(0.01, requires_grad=True)
                loss_6.backward()
                opt_6.step()
                opt_6.zero_grad()

                loss_7, feature_7 = model_7.forward(feature_1,batch_idx,batch_label)
                if loss_7.item() == 0:
                    loss_7 = torch.tensor(0.01, requires_grad=True)
                loss_7.backward()
                opt_7.step()
                opt_7.zero_grad()

                with torch.no_grad():
                    feature_5 = feature_5.detach()
                    feature_6 = feature_6.detach()
                    feature_7 = feature_7.detach()



                loss_3, _ = model_3.forward(feature_5, feature_6,feature_7,batch_label)
                if loss_3.item() == 0:
                    loss_3 = torch.tensor(0.01, requires_grad=True)
                loss_3.backward()
                opt_3.step()
                opt_3.zero_grad()
                self_c2 += self_c_2.item()

            self_c2 = self_c2/len(train_dataloader)

            scheduler_1.step()
            scheduler_3.step()
            scheduler_5.step()
            scheduler_6.step()
            scheduler_7.step()

            model_1.eval()
            model_2.eval()
            model_3.eval()
            model_5.eval()
            model_6.eval()
            model_7.eval()

            feature_1 = model_1.forward(enhanced_features, adj)
            feature_2,_ = model_2.forward(enhanced_features, adj2)


            pre_list = []
            for batch_idx, batch_label in dev_dataloader:
                batch_idx = batch_idx.to(device)

                _, feature_5 = model_5.forward(feature_1, adj, batch_idx)

                _, feature_6 = model_6.forward(feature_2, adj2, batch_idx)
                _, feature_7 = model_7.forward(feature_1,batch_idx)

                pre, feature_3 = model_3.forward(feature_5, feature_6,feature_7)
                pre = pre.tolist()
                pre_list.extend(pre)

            dev_labels = [int(x) for x in dev_labels]
            # 计算准确率
            correct = sum(1 for l, p in zip(dev_labels, pre_list) if l == p)
            accuracy = correct / len(dev_labels)

            pre_list = []
            for batch_idx, batch_label in test_dataloader:
                batch_idx = batch_idx.to(device)

                _, feature_5 = model_5.forward(feature_1, adj, batch_idx)

                _, feature_6 = model_6.forward(feature_2, adj2, batch_idx)
                _, feature_7 = model_7.forward(feature_1, batch_idx)

                pre, feature_3 = model_3.forward(feature_5, feature_6, feature_7)
                pre = pre.tolist()
                pre_list.extend(pre)

            text_labels = [int(x) for x in text_labels]

            correct = sum(1 for l, p in zip(text_labels, pre_list) if l == p)
            accuracy = correct / len(text_labels)


            f1 = f1_score(text_labels, pre_list, average='macro')

            print(
                f"{e}: ACC：{accuracy} F1:{f1}  self_c：{self_c.item()}  self_c2：{self_c2} ")

            with open("model_results_"+name+".txt", "a") as file:
                file.write(f"{e}\t{accuracy}\t{f1}\t{self_c.item()}\t{self_c2}\n")

            if accuracysss <= accuracy:
                count = 0
                accuracysss = accuracy
                f1_max = f1

            if e == 199:
                with open("model_results_" + name + "_secor.txt", "a") as file:
                    file.write(f"{accuracysss}\t{f1_max}\n")
            count += 1