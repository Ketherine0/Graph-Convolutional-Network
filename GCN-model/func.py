import numpy as np
import scipy.sparse as sp

def pre_data(dataset="cora", path="../data/cora/",):

    # idx_features_labels = path+dataset.content = ./data/cora/cora.content
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    # 压缩稀疏矩阵
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    onehot_labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # j: label  i: index
    idx_map = {j: i for i, j in enumerate(idx)}

    # 返回连接两个node的edge
    edges_origin = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # 变成一维数组
    flat = edges_origin.flatten()
    print(flat)
    # 将原先的label替换为index
    edges = np.array(list(map(idx_map.get, flat)), dtype=np.int32).reshape(edges_origin.shape)
    # 生成邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
    # 转化为对称形式
    adj += adj.T - sp.diags(adj.diagonal())

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    y_train, y_val, y_test, train_mask, val_mask, test_mask = split_data(onehot_labels)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def encode_onehot(labels):
    unique = set(labels)
    seq = enumerate(unique)
    uni_dict = {id: np.identity(len(unique))[i,:] for i, id in seq}
    # 根据labels得到对应的onehot encode
    onehot_labels = np.array(list(map(uni_dict.get, labels)), dtype=np.int32)
    # print(onehot_labels)
    return onehot_labels

def split_data(label_np):
    # index list
    ind_list = np.arange(len(label_np))

    ind_train = []
    label_count = {}
    for i, label in enumerate(label_np):
        # 返回"1"所在的index(=label)
        label = np.argmax(label)
        # 7组中每组选取20个进入train
        if label_count.get(label, 0) < 20:
            ind_train.append(i)
            label_count[label] = label_count.get(label, 0) + 1

    ind_val_test = list(set(ind_list) - set(ind_train))
    ind_val = ind_val_test[0:500]
    ind_test = ind_val_test[500:1500]

    # train: 140  validation: 500  test: 1000
    y_train = np.zeros(label_np.shape, dtype=np.int32)
    y_val = np.zeros(label_np.shape, dtype=np.int32)
    y_test = np.zeros(label_np.shape, dtype=np.int32)
    y_train[ind_train] = label_np[ind_train]
    y_val[ind_val] = label_np[ind_val]
    y_test[ind_test] = label_np[ind_test]
    # boolen representation of index
    train_bool = bool_matrix(ind_train, label_np.shape[0])
    val_bool = bool_matrix(ind_val, label_np.shape[0])
    test_bool = bool_matrix(ind_test, label_np.shape[0])

    return y_train, y_val, y_test, train_bool, val_bool, test_bool

def bool_matrix(ind, l):
    m = np.zeros(l)
    m[ind] = 1
    return np.array(m, dtype=np.bool)

def pre_adj(adj, symmetric=True):
    # 邻接矩阵加上对角元素
    # 但是因为邻接矩阵的对角都是0，和特征矩阵内积相当于将邻接矩阵做了加权和，
    # 节点特征的值成为了邻接矩阵的权，自身的特征被忽略。
    # 为避免这种情况，可以先给A加上一个单位矩阵I
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj

def normalize_adj(adj, symmetric=True):
    # D^(-1/2)*A*D^(-1/2) 
    # sum(1)沿着每一行相加
    if symmetric:
        D = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        adj_norm = adj.dot(D).transpose().dot(D).tocsr()
    else:
        D = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        adj_norm = D.dot(adj).tocsr()
    return adj_norm
