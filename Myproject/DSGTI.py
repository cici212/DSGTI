import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from tensorflow import keras
#from tensorflow.keras.losses import Huber
import sklearn.metrics
from keras.layers import MultiHeadAttention

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
import tensorflow as tf
from tensorflow import keras
from data.process_domain import get_domain,get_domain_edge
from data.process_DSGTI import load_data, worker_num, claims
from data.util import update_feature, update_reliability, copier_detection
import numpy as np
import time
dataset = 'Anime'
# task_domain_file= 'data/domaindata/CopyAnime-20%/domain11.csv'
# rating_file = 'data/domaindata/CopyAnime-20%/copyrating-11-20%.csv'
rating_file='data/domaindata/Anime/rating_20object.csv'
task_domain_file= 'data/domaindata/Anime/domain11.csv'
# rating_file='data/domaindata/MovieLens/rating-101.csv'
# task_domain_file= 'data/domaindata/MovieLens/moviedomain101.csv'
# rating_file= 'data/domaindata/Anime/rating_anime_f20_148.csv'
#groundtruth_Anime=groundtruth148.csv
#groundtruth_MovieLens=groundtruth2.csv
embeddings_file='data/domaindata/Anime/embedding.csv'
# num_classes = 6
num_classes=11
object_index,source_index,task_num,worker_num,claims=get_domain_edge(rating_file)
ratings_df = pd.read_csv(rating_file, header=None, names=['task_id', 'worker_id', 'answer'])

answers = (
    ratings_df
    .pivot(index='worker_id', columns='task_id', values='answer')
    .fillna(-1)
    .values
)
print(task_num,worker_num)
node_num=task_num+worker_num
claims_one_hot = tf.one_hot(claims, depth=num_classes)
claims_one_hot = tf.cast(claims_one_hot, tf.float32)
reliability = np.ones(shape=(worker_num)) / worker_num
graph, edge_label, test_index = load_data(dataset=dataset,
                                          answer_path='./data/domaindata/{}/rating_20object.csv'.format(dataset),
                                          truth_path='./data/domaindata/{}/groundtruth11.csv'.format(dataset))
lr = 1e-3
node_num=task_num+worker_num
class TIRA(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        进行两层卷积
        '''
        self.h2 = None
        self.gcn10 = tfg.layers.GCN(512, activation=tf.nn.relu)
        self.gcn11 = tfg.layers.GCN(64, activation=None)
        self.gcn12 = tfg.layers.GCN(512, activation=tf.nn.relu)
        '''经过一次线性变换和非线性激活操作，使用了模型中定义的权重矩阵 self.w1 和偏置向量 self.b1 对输入数据进行处理。
            创建和初始化神经网络层的权重参数，为模型的学习和预测能力提供基础。
            权重矩阵对邻居节点，偏置向量对自己进行转变
        '''
        self.w1 = tf.Variable(tf.random.truncated_normal([512,512], stddev=0.1))
        self.b1 = tf.Variable(tf.random.truncated_normal([node_num, 512], stddev=0.1))
        #将图的节点特征转换为预测的类别标签。
        self.fc0 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)
        # self.fc1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        # self.fc2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.fc3 = tf.keras.layers.Dense(num_classes)
    #call 方法定义了模型的结构和信息流动的路径，通过调用该方法可以对输入数据进行前向传播，并获得模型的输出结果
    def call(self, inputs, training=None, mask=None, cache=None):

        x, edge_index = inputs

        domain_x=get_domain(task_domain_file,rating_file)

        attn_layer = tf.keras.layers.Dense(1, activation=tf.nn.softmax)
        attn_weights = attn_layer(tf.concat([x, domain_x], axis=1))  # (445, 1)
        total_x = x + tf.expand_dims(tf.reduce_sum(attn_weights * domain_x, axis=1), axis=1)  # (445, 6)
        h1 = self.gcn10([total_x, edge_index], training=training)

        h1_copier = tf.slice(h1, [task_num, 0], [worker_num, -1])
        # print(h1_copier)
        h1_copier = copier_detection(h1_copier,answers)

        h1 = tf.tensor_scatter_nd_update(h1, tf.where(tf.range(task_num, x.shape[0])), h1_copier)


        h11 = tf.nn.relu(tf.matmul(h1, self.w1) + self.b1)
        h2 = self.gcn11([h11, edge_index], training=training)
        h2 = self.gcn12([h2, edge_index], training=training)
        h = h1 + h2
        h = tf.keras.layers.BatchNormalization()(h, training=training)

        h = tf.nn.relu(h)
        # 添加Dropout层
        h = tf.keras.layers.Dropout(0.15)(h, training=training)
        h = self.fc0(h)
        h = self.fc3(h)
        return h
# x = tf.Variable(tf.random.truncated_normal([node_num, num_classes], stddev=0.1))
#执行模型的前向传播过程
def forward(graph, training=False):
    return model([graph.x, graph.edge_index], training=training)
#@tf_utils.function
def predict_edge(embedded, edge_label):
    row, col = edge_label[0], edge_label[1]
    embedded_row = tf.gather(embedded, row)#object
    embedded_col = tf.gather(embedded, col)#source
    # dot product
    logits = embedded_row * embedded_col

    return logits
import tensorflow as tf

def smooth_l1_loss(masked_labels, masked_logits):
    absolute_loss = tf.abs(masked_labels - masked_logits)
    square_loss = 0.5 * (masked_labels - masked_logits) ** 2
    loss = tf.where(absolute_loss < 1.0, square_loss, absolute_loss - 0.5)
    return tf.reduce_mean(loss)

def compute_loss(embedded,logits, vars):
    masked_logits = logits# 预测值
    object_embedd = tf.gather(embedded, object_index)
    distance1 = tf.sqrt(tf.reduce_sum(tf.square(object_embedd - claims_one_hot), axis=-1))
    weight1 = tf.math.unsorted_segment_mean(data=distance1, segment_ids=tf.cast(source_index, dtype=tf.int32),
                                            num_segments=worker_num)

    losses1 = tf.reduce_mean(weight1 * reliability)
    masked_labels = tf.one_hot(edge_label[-1,:],depth=num_classes)
    mse_loss = tf.losses.mean_squared_error(masked_labels, masked_logits)
    mae_loss = tf.keras.losses.MeanAbsoluteError()(masked_labels, masked_logits)
    huber_loss = tf.keras.losses.Huber()(masked_labels, masked_logits)
    huber_weight = 2e-2
    smooth_l1_weight = 2e-2
    # smooth_l1_loss = tf.keras.losses.SmoothL1Loss()(masked_labels, masked_logits)
    #求smoothloss
    absolute_loss = tf.abs(masked_labels - masked_logits)
    square_loss = 0.5 * (masked_labels - masked_logits) ** 2
    loss = tf.where(absolute_loss < 1.0, square_loss, absolute_loss - 0.5)
    smooth_l1_loss = tf.reduce_mean(loss)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=masked_labels
    )
    kernel_vals = [var for var in vars if "kernel" in var.name]
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]

    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 5e-4
    # return huber_weight * huber_loss + smooth_l1_weight * smooth_l1_loss + tf.add_n(l2_losses) * 2e-4 + losses1+tf.reduce_mean(losses)
from sklearn.model_selection import KFold
def evaluate(mask):
    logits = tf.nn.softmax(forward(graph))
    masked_logits = tf.gather(logits, mask)
    y_pred= tf.argmax(masked_logits, axis=-1,output_type=tf.int32)

    model = LinearRegression()
    model.fit(tf.expand_dims(tf.cast(y_pred, tf.float32), axis=-1), graph.y)
    y_pred_float = model.predict(tf.expand_dims(tf.cast(y_pred, tf.float32), axis=-1))
    # 计算 RMSE 和 MAE
    rmse = tf.sqrt(tf.keras.losses.mean_squared_error(graph.y, y_pred_float))
    mae = tf.keras.losses.mean_absolute_error(graph.y, y_pred_float)
    # 计算 R-squared
    y_true_mean = tf.reduce_mean(graph.y)
    y_pred_mean = tf.reduce_mean(y_pred_float)
    return rmse,mae

model =TIRA()
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint.restore(tf.train.latest_checkpoint('./model/{}-1'.format(dataset)))

best_test_acc  = 0
best_weight = 0
best_macro = 0
best_rmse=10
best_mae=10
best_r2=0
y_pred_best = []
re=[]
start_time = time.time()
for step in range(150):
    with tf.GradientTape() as tape:
        embedded = forward(graph, training=True)#encoder
        reliability = update_reliability(embedded, [object_index, source_index, claims_one_hot],task_num,worker_num, reliability)
        re.append(reliability)
        logits = predict_edge(embedded, edge_label[:2,:])#decoder

        loss = compute_loss(embedded,logits, tape.watched_variables())
    vars = tape.watched_variables()#监视的变量
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))
    x = update_feature([object_index, source_index, claims_one_hot], reliability, task_num, worker_num)
    test_rmse,test_mae= evaluate(test_index)
    if test_rmse<best_rmse:
        best_rmse=test_rmse
    if test_mae<best_mae:
        best_mae=test_mae

    print("step={}\tbest_rmse={}\tbest_mae={}".format(step,best_rmse,best_mae))
time_end = time.time()
time_run = time_end - start_time
print('timecost:', time_run, 's')
