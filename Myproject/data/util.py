from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import euclidean, cityblock, cdist
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
import csv
from tensorflow import keras
from sklearn.metrics import f1_score, classification_report
import tensorflow.python.ops.numpy_ops.np_config as np_config
from textdistance import jaccard
from sklearn.metrics.pairwise import pairwise_distances

def update_reliability(embedding, triple, task_num,source_num, reliability):
    # re=[]
    object_index, source_index, claims = triple
    object_num = task_num
    source_num = source_num
    node_num = object_num + source_num
    weights = tf.gather(reliability, source_index)
    weights = tf.cast(weights, tf.float32)
    # 对象索引提取对象嵌入 object_embedd。
    object_embedd = tf.gather(embedding, object_index)
    # 对象嵌入和声明值之间的距离,并用源可靠性权重进行加权,然后平方得到 weight。
    distance = tf.sqrt(tf.reduce_sum(tf.square(object_embedd - claims), axis=-1)) * weights  # * weights tf.sqrt
    weight = tf.square(tf.math.unsorted_segment_sum(data=distance, segment_ids=source_index, num_segments=source_num))
    # 可靠性权重归一化到 [0, 1] 的范围内
    weight /= tf.reduce_sum(weight)
    # 将归一化后的权重通过指数函数转换为更新后的可靠性值 r。
    r = tf.math.exp(-tf.sqrt(weight))
    #  1D 的可靠性值 r 转换成 2D 张量 r_temp,并将其数据类型转换为 32 位浮点数
    r_temp = tf.cast(tf.reshape(r, (-1, 1)), dtype=tf.float32)
    source_embedd = tf.gather(embedding, np.array(range(object_num, node_num)))
    #  f1 是一个对称矩阵,表示源嵌入之间的相关性
    f1 = tf.matmul(source_embedd, source_embedd, transpose_b=True)
    a = tf.reshape(tf.reduce_sum(tf.math.square(source_embedd), axis=-1), (-1, 1))
    # 计算源嵌入向量的欧几里德范数
    f2 = tf.math.sqrt(tf.matmul(a, a, transpose_b=True))
    # 相关性矩阵 f1 归一化,得到最终的相关性矩阵 cor。
    cor = tf.nn.softmax(f1 / f2)
    # 相关性矩阵 cor 与可靠性向量 r_temp 相乘,得到更新后的可靠性向量。
    reliability = tf.reshape(tf.matmul(cor, r_temp), (tf.shape(r_temp)[0],))
    # re.append(reliability)
    return reliability
def update_feature(triple, reliability, object_num, source_num):
    object_index, source_index, claims = triple
    weights = tf.reshape(tf.gather(reliability, source_index), (-1,1))
    weights=tf.cast(weights,tf.float32)
    # weight = tf.reshape(reliability, (-1,1))
    weighted_claims = claims * weights
    # 函数利用 tf.math.unsorted_segment_sum 函数对加权的声明 weighted_claims 进行分段求和
    objects_feature = tf.math.unsorted_segment_sum(data=weighted_claims, segment_ids=object_index, num_segments=object_num)
    sources_feature = tf.math.unsorted_segment_sum(data=weighted_claims, segment_ids=source_index, num_segments=source_num)
    # 将小于等于 0 的值设为 0,大于 0 的值设为 1
    # 然后将该二值特征与源可靠性 reliability 相乘
    reliability_=tf.cast(tf.reshape(reliability, shape=(-1, 1)),tf.float32)
    sources_feature = tf.where(sources_feature>0.0, x=1.0, y=0.0) * reliability_
    features = tf.nn.softmax(tf.concat([objects_feature, sources_feature], axis=0))
    # print(features.shape)
    # 对象特征和源特征拼接在一起,并使用 tf.nn.softmax 对拼接后的特征进行归一化,得到最终的特征向量 features
    return features
# def calculate_error_similarity(answers):
#     # 计算工人之间的错误模式相似性
#     error_similarity_matrix = 1 - cdist(answers, answers, metric='hamming')
#     return error_similarity_matrix
# def jaccard_similarity(x, y):
#     intersection = np.logical_and(x, y)
#     union = np.logical_or(x, y)
#     return intersection.sum() / float(union.sum())
# def calculate_jaccard_similarity_matrix(answers):
#     n_workers = answers.shape[0]
#     similarity_matrix = np.zeros((n_workers, n_workers))
#
#     for i in range(n_workers):
#         for j in range(i, n_workers):
#             sim = jaccard_similarity(answers[i], answers[j])
#             similarity_matrix[i, j] = sim
#             similarity_matrix[j, i] = sim
#
#     return similarity_matrix
def copier_detection(h2_copier,answers):
    if tf.is_tensor(h2_copier):
        h2_copier = h2_copier.numpy()
#     error_similarity_matrix = calculate_error_similarity(answers)
    # 计算 Jaccard 相似性矩阵
    # jaccard_similarity_matrix = calculate_jaccard_similarity_matrix(answers > 0)
    block_size = 32
    num_blocks = (h2_copier.shape[0] + block_size - 1) // block_size
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, h2_copier.shape[0])
        block = h2_copier[start:end]
        # Calculate the similarity matrix for the current block
        # feature_similarity_matrix = 1 - cdist(block, block, metric='euclidean')
        feature_similarity_matrix = 1 - cdist(block,block, metric='euclidean')
        # 综合特征相似性和错误模式相似性
        # combined_similarity_matrix = (feature_similarity_matrix + error_similarity_matrix[start:end, :][:,
        #                                                           start:end]) / 2
        # 综合特征相似性和错误模式相似性
        # 综合特征相似性和错误模式相似性
        # combined_similarity_matrix = (feature_similarity_matrix + error_similarity_matrix[start:end, :][:, start:end] + jaccard_similarity_matrix[start:end, :][:, start:end]) / 3
        copier_threshold = 0.8
        potential_copiers = np.where(np.any(feature_similarity_matrix > copier_threshold, axis=1))[0]
        for copier_idx in potential_copiers:
            non_copier_indices = np.setdiff1d(np.arange(block.shape[0]), copier_idx)
            # similarity_weights = 1 - combined_similarity_matrix[copier_idx, non_copier_indices]
            # similarity_weights /= np.sum(similarity_weights)  # 归一化权重
            # # # 自适应权重更新
            # adaptive_weights = similarity_weights * (
            #             1 - np.exp(-combined_similarity_matrix[copier_idx, non_copier_indices]))
            # adaptive_weights /= np.sum(adaptive_weights)  # 再次归一化
            # 加入 L2 正则化
            l2_reg = 0.01
            regularization_term = l2_reg * block[copier_idx]
            # # 使用加权平均的方式更新复制者的特征
            # update_tensor = np.average(block[non_copier_indices], axis=0, weights=adaptive_weights)-regularization_term
            update_tensor = np.mean(block[non_copier_indices], axis=0)-regularization_term
            block[copier_idx] = update_tensor
        h2_copier[start:end] = block
        # h2_copier[start:end].assign(tf.convert_to_tensor(block))
    return h2_copier
# def update_tensor(h2_copier, copier_idx):
#     non_copier_indices = np.setdiff1d(np.arange(h2_copier.shape[0]), copier_idx)
#     update_tensor = np.mean(h2_copier[non_copier_indices], axis=0)
#     return copier_idx, update_tensor
# def copier_detection(h2_copier, copier_threshold=0.8):
#     # Convert to numpy array if it's a TensorFlow tensor
#     if tf.is_tensor(h2_copier):
#         h2_copier = h2_copier.numpy()
#     similarity_matrix = cosine_similarity(h2_copier)
#     potential_copiers = np.where(np.any(similarity_matrix > copier_threshold, axis=1))[0]
#     h2_copier_tensor = tf.convert_to_tensor(h2_copier)
#     for copier_idx in potential_copiers:
#         # Indices of non-copier rows
#         non_copier_indices = np.where(np.arange(h2_copier.shape[0]) != copier_idx)[0]
#         # Convert indices to TensorFlow tensor
#         non_copier_indices_tf = tf.convert_to_tensor(non_copier_indices, dtype=tf.int32)
#         # Calculate the mean of the non-copier rows
#         non_copier_rows = tf.gather(h2_copier_tensor, non_copier_indices_tf)
#         update_tensor = tf.reduce_mean(non_copier_rows, axis=0)
#         # Update the copier row with the mean values
#         h2_copier_tensor = tf.tensor_scatter_nd_update(
#             h2_copier_tensor,
#             indices=tf.expand_dims([copier_idx], axis=0),
#             updates=tf.expand_dims(update_tensor, axis=0)
#         )
#     # print(h2_copier_tensor)
#     return h2_copier_tensor
