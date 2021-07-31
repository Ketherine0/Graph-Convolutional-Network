from numpy.core.fromnumeric import shape
import tensorflow as tf
from tensorflow.python.keras.initializers import Identity, glorot_uniform, Zeros
from tensorflow.python.keras.layers import Dropout, Input, Layer, Embedding, Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2  

class GraphCon(Layer):

    def __init__(self, units, activation=tf.nn.relu,
                 dropout_rate=0.5, use_bias=True, l2_reg=0, 
                 feature_less=False, seed=1024, **kwargs):
        super(GraphCon, self).__init__(**kwargs)
        self.units = units
        self.feature_less = feature_less
        self.use_bias = use_bias
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.seed = seed

    def build(self, input_shapes):

        input_dim = int(input_shapes[0][-1])

        # if self.feature_less:
        #     input_dim = int(input_shapes[0][-1])
        # else:
        #     # 判断是否正确
        #     assert (input_shapes) == 2
        #     features_shape = input_shapes[0]
        #     input_dim = int(features_shape[-1])

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                             initializer=glorot_uniform(seed=self.seed),
                                             regularizer=l2(self.l2_reg), name='kernel',)
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,), initializer=Zeros(), name='bias',)

        # 每次迭代将只使用50%的神经元进行训练,防止过拟合
        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        features, A = inputs
        features = self.dropout(features, training=training)
        # AXW  稀疏矩阵 A * 稠密矩阵 X
        output = tf.matmul(tf.sparse.sparse_dense_matmul(A, features), self.kernel)

        if self.use_bias:
            output += self.bias
        # 激活函数
        act = self.activation(output)

        # 确定处于学习阶段还是训练阶段
        # act._uses_learning_phase = features._uses_learning_phase
        return act

    def get_config(self):
        config = {'units': self.units,
                  'activation': self.activation,
                  'dropout_rate': self.dropout_rate,
                  'l2_reg': self.l2_reg,
                  'use_bias': self.use_bias,
                  'feature_less': self.feature_less,
                  'seed': self.seed
                  }

        base_config = super(GraphCon, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def GCN(adj_dim, feature_dim, n_hidden, num_class, num_layers=2, activation=tf.nn.relu, dropout_rate=0.5, l2_reg=0, feature_less=True, ):
    # Input: 初始化深度学习网络输入层的tensor
    Adj = Input(shape=(None, ), sparse=True)
    # X_in = Input(shape=(1,), )

    # emb = Embedding(adj_dim, feature_dim,
    #                 embeddings_initializer=Identity(1.0), trainable=False)
    # X_emb = emb(X_in)
    # h = Reshape([X_emb.shape[-1]])(X_emb)

    # num_feature < num_node
    if feature_less:
        X_in = Input(shape=(1,), )

        # input dimL adj_dim   output_dim: feature_dim
        emb = Embedding(adj_dim, feature_dim,
                        embeddings_initializer=Identity(1.0), trainable=False)
        X_emb = emb(X_in)
        # X_emb: (None, 1, 1433)
        h = Reshape([X_emb.shape[-1]])(X_emb)
    else:
        X_in = Input(shape=(feature_dim,), )
        h = X_in

    for i in range(num_layers):
        # i == 1
        if i == num_layers - 1:
            activation = tf.nn.softmax
            n_hidden = num_class

        h = GraphCon(n_hidden, activation=activation, dropout_rate=dropout_rate, l2_reg=l2_reg)([h,Adj])

    output = h
    # 初始化函数式模型
    model = Model(inputs=[X_in, Adj], outputs=output)

    return model
