import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import Model
from layers import GCN
from func import pre_adj, pre_data

if __name__ == "__main__":

    FEATURE_LESS = False

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = pre_data('cora')

    adj = pre_adj(adj)
    # features.sum(axis-1,) 得到每一行 (每个node) features总数
    # normalize 
    features /= features.sum(axis=1,).reshape(-1,1)
#     print(features.sum(axis=1,))
#     print(features[0].sum(axis=1,))

    # dim(X) = 2708 * 1433
    X = features
    # feature_dim = 1433
    feature_dim = X.shape[-1]

    # if FEATURE_LESS:
    #     X = np.arange(adj.shape[-1])
    #     feature_dim = adj.shpe[-1]
    # else:
    #     X = features
    #     feature_dim = X.shape[-1]
    model_input = [X, adj]

    model = GCN(adj.shape[-1], feature_dim, 16, y_train.shape[1], dropout_rate=0.5, l2_reg=2.5e-4, feature_less=FEATURE_LESS,)
    # 告知训练时用的优化器、损失函数和准确率评测标准
    # learning rate = 0.01
    # 交叉熵损失函数作为 loss function
    model.compile(optimizer=adam_v2.Adam(0.01), loss='categorical_crossentropy', weighted_metrics=['categorical_crossentropy', 'acc'])

    # 训练迭代次数
    EPOCH = 200

    val_data = (model_input, y_val, val_mask)
    # 在每个epoch后保存模型到filepath
    # save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
    # save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
    callback = ModelCheckpoint('./best_model.h5', monitor='val_weighted_categorical_crossentropy', save_best_only=True, save_weights_only=True)


    # training
    print("------- start training -------")
    # verbose：日志展示  2：每个epoch输出一条记录
    # 之前储存的list中的回调函数将会在训练过程中的适当时机被调用
    model.fit(model_input, y_train, sample_weight=train_mask, validation_data=val_data,
              batch_size=adj.shape[0], epochs=EPOCH, shuffle=False, verbose=2, callbacks=[callback])
    

    # testing
    # 读取权重
    model.load_weights('./best_model.h5')
    # model.summary()
    # sample_weight的作用：为数据集中的数据分配不同的权重 此处仅为test集数据赋予权重
    eval_results = model.evaluate(model_input, y_test, sample_weight=test_mask, batch_size=adj.shape[0])
    print('Done.\n'
          'Test loss: {}\n'
          'Test weighted_loss: {}\n'
          'Test accuracy: {}'.format(*eval_results))



