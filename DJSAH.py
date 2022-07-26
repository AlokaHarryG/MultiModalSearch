#获取训练集内容并且进行处理(Wiki)
import scipy.io as scio
import pandas as pd
import numpy as np
data_path = "datasets/WIKI.mat"
#data_path = "datasets/MIRFLICKR25K.mat"
#data_path = "datasets/NUSWIDE10.mat"
data = scio.loadmat(data_path)
i_tr = data.get("I_tr")
i_te = data.get("I_te")#训练集和测试集：图像（128维的SIFT特征向量）
t_tr = data.get("T_tr")
t_te = data.get("T_te")#训练集和测试集：文本（10维的LDA主题向量）
l_tr = data.get("L_tr")
l_te = data.get("L_te")#训练集和测试集：标签
sampleinds=data.get("sampleInds")#1-2173
#print(i_tr.shape,i_te.shape,t_tr.shape,t_te.shape,l_tr.shape,l_te.shape,i_te.shape
#     ,sampleinds.shape)
#(2173, 128) (693, 128) (2173, 10) (693, 10) (2173, 1) (693, 1) (693, 128) (1, 2173)
key_list = data.keys() #dict_keys(['__header__', '__version__', '__globals__',
#'I_tr', 'I_te', 'T_tr', 'T_te', 'L_te', 'L_tr', 'sampleInds'])
def get_oi(x):
    value = []
    for key in data.keys():
        if(len(data.get(key))>100):
            value.append(data.get(key)[x])
    return value
# 注意 oi的构造：{x1[i],x2[i],l[i]},分别为128维向量，10维向量和1维向量
def get_tilde_l(l):
    c = l.shape[1]
    n = l.shape[0]
    tldl=np.zeros((c,n),float)
    for k in range(c):
        for i in range(n):
            tldl[k][i]=l[i][k]/sum(abs(l[i])**2)**(1./2)
    return tldl
print(get_tilde_l(l_tr))
def get_S(l):
    tldl = get_tilde_l(l)
    n = l.shape[0]
    s=2*tldl.T.dot(tldl)-np.ones((n,1)).dot(np.ones((1,n)))
    return s
print(get_S(l_tr))
def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

