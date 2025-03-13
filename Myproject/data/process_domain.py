import math
from collections import defaultdict
from itertools import chain
import tensorflow as tf
import numpy as np
from sympy.abc import lamda
from tensorflow import keras
from sklearn.metrics import f1_score, classification_report
import tensorflow.python.ops.numpy_ops.np_config as np_config
import pandas as pd
import numpy as np
import csv
from tf_geometric.utils.graph_utils import convert_edge_to_directed, remove_self_loop_edge
#from tf_geometric.tf_geometric import Graph

# task_domain_file='./data/domaindata/MovieLens/moviedomain101.csv'
# work_rating_file ='./data/domaindata/MovieLens/rating-101.csv'
# rating_file='./data/domaindata/CopyAnime-20%/copyrating-11-20%.csv'
rating_file='./data/domaindata/Anime/rating_20object.csv'
task_domain_file='./data/domaindata/Anime/domain11.csv'
def get_task_features(task_domain_file):
    with open(task_domain_file, 'r', newline='', encoding='utf-8') as file:
        task_data = csv.reader(file)
        all_domains = set()
        for row in task_data:
            domains = row[1].split('|')
            all_domains.update(domains)
        all_domains = sorted(list(all_domains))
        file.seek(0)
        task_data = csv.reader(file)
        task_domain_matrix = []
        task_id_to_index = defaultdict(lambda: -1)
        for row in task_data:
            task_id = row[0]
            domains = row[1].split('|')
            domain_vector = [0] * len(all_domains)
            for domain in domains:
                domain_index = all_domains.index(domain)
                domain_vector[domain_index] = 1
            task_domain_matrix.append(domain_vector)
            if task_id in task_id_to_index:
                continue
            task_id_to_index[task_id] = len(task_domain_matrix) - 1
        # print("domain_len:",len(all_domains))
    return task_domain_matrix,len(all_domains),task_id_to_index,
def sigmoid(x, a=1, b=0):
    return 1 / (1 + np.exp(-(x - b) / a))

def calculate_regularization(worker_domain_matrix, workid, lamda):
    worker_domain_vector = worker_domain_matrix[workid]
    regularization = lamda * np.linalg.norm(worker_domain_vector)
    return regularization
vectorized_calculate_regularization = np.vectorize(calculate_regularization)
#
from numba import jit
import numpy as np
@jit(nopython=True)
def _linear_update(truth, result, max_abs_error, max_rel_error, worker_abs_error, worker_rel_error, worker_accuracy, update_rate, worker_domain_vector, lamda):
    abs_error = abs(truth - round(result))
    rel_error = abs_error / truth if truth != 0 else 0

    beta=1.5
    gamma=2.0
    abs_error_score = 1 / (1 + np.exp(beta * (abs_error - gamma)))
    rel_error_score = 1 / (1 + np.exp(beta * (rel_error - gamma)))
    worker_abs_error_score = 1 - (1 - update_rate) * (max_abs_error - worker_abs_error) / max_abs_error
    worker_rel_error_score = 1 - (1 - update_rate) * (max_rel_error - worker_rel_error) / max_rel_error
    worker_skill_score = worker_accuracy - lamda * np.sqrt(np.sum(worker_domain_vector ** 2))
    return min(abs_error_score, rel_error_score, worker_abs_error_score, worker_rel_error_score) * worker_skill_score
def linear_update(truth, result, max_abs_error,max_rel_error, worker_abs_error, worker_rel_error, worker_accuracy, update_rate,worker_domain_matrix,workid,lamda):
    worker_domain_vector = np.array(worker_domain_matrix[workid])
    return _linear_update(truth, result, max_abs_error, max_rel_error, worker_abs_error, worker_rel_error,
                          worker_accuracy, update_rate, worker_domain_vector, lamda)
def get_worker_features(work_rating_file,domain_num,task_id_to_index,task_domain_matrix,update_rate=0.2):
    # 读取工人数据
    with open(work_rating_file, 'r', newline='', encoding='utf-8') as file:
        # worker_data = csv.reader(file)
        worker_data = list(csv.reader(file))
        worker_count = max(int(row[1]) for row in worker_data)

        # file.seek(0)
        # worker_data = csv.reader(file)
        # 初始化工人特征字典
        worker_features = {}
        sum = {}
        number = {}
        worker_avg_error = {}
        for row in worker_data:
            movie = row[0]
            value = int(row[2])
            sum[movie] = sum.get(movie, 0) + value
            number[movie] = number.get(movie, 0) + 1
        aver = {}  # 每个movie平均分 aver是字典，从1开始
        for key in sum.keys():
            aver[key] = sum[key] / number[key]
        # file.seek(0)
        # worker_data = csv.reader(file)
        worker_task_domains = [[] for _ in range(worker_count)]
        for row in worker_data:
            workid = int(row[1]) - 1
            movieid = int(row[0])
            movieid_index = task_id_to_index[movieid]
            for domain_index in range(domain_num):
                if task_domain_matrix[movieid_index][domain_index] == 1:
                    worker_task_domains[workid].append(domain_index)
        # 2.初始化
        worker_domain_matrix = [[0.5] * domain_num for _ in range(worker_count)]
        # file.seek(0)

        for row in worker_data:
            workid = int(row[1]) - 1
            movieid = int(row[0])
            movieid_index = task_id_to_index[movieid]
            truth = (aver[(row[0])])
            result = int(row[2])
            abs_error = abs(truth - round(result))
            rel_error = abs_error / truth if truth != 0 else 0
            if workid not in worker_avg_error:
                worker_avg_error[workid] = {
                    'abs_error': abs_error,
                    'rel_error': rel_error,
                    'count': 1,
                    'total_tasks': 1,
                    'correct_tasks': 1 if abs_error <= 2 else 0
                }
            else:
                worker_avg_error[workid]['abs_error'] += abs_error
                worker_avg_error[workid]['rel_error'] += rel_error
                worker_avg_error[workid]['count'] += 1
                worker_avg_error[workid]['total_tasks'] += 1
                worker_avg_error[workid]['correct_tasks'] += 1 if abs_error <= 2 else 0
            worker_abs_error = worker_avg_error[workid]['abs_error'] / worker_avg_error[workid]['count']
            worker_rel_error = worker_avg_error[workid]['rel_error'] / worker_avg_error[workid]['count']
            worker_accuracy = worker_avg_error[workid]['correct_tasks'] / worker_avg_error[workid]['total_tasks']
            # 更新相关领域
            for domain_index in worker_task_domains[workid]:
                if task_domain_matrix[movieid_index][domain_index] == 1:
                    worker_domain_matrix[workid][domain_index] =  linear_update(truth, result, 2.0, 0.4,worker_abs_error, worker_rel_error,worker_accuracy,update_rate,worker_domain_matrix,workid,0.1)
                    # worker_domain_matrix[workid][domain_index] = bayesian_update(truth, result, 0.5, 0.1,
                    #                                                              worker_abs_error,
                    #                                                              worker_rel_error, worker_accuracy, update_rate,
                    #                                                              worker_domain_matrix, workid, domain_index)
    return worker_domain_matrix
def get_domain_edge(work_rating_file):
    with open(work_rating_file, 'r', newline='', encoding='utf-8') as truth_file:
        answer = csv.reader(truth_file)
        object_index = []
        source_index = []
        edge_label = []
        object_set = []
        source_set = []
        claims=[]
        for line in answer:
            object = int(line[0])-1
            source = int(line[1])-1
            if object not in object_set:
                object_set.append(object)
            if source not in source_set:
                source_set.append(source)
            claims.append(int(line[2]))
            edge_label.append(int(line[2]))
            #object_index.append(object_set.index(object))
            #source_index.append(source_set.index(source))
            object_index.append(object)
            source_index.append(source)
        object_index = np.array(object_index, dtype=np.int32)
        object_num = object_index.max() + 1
        source_index = np.array(source_index, dtype=np.int32)
        claims = np.array(claims, dtype=np.int32)
        edge_label = np.vstack([object_index, source_index, claims])
        edge_index, _ = remove_self_loop_edge(edge_label[:2, :])
        edge_index, _ = convert_edge_to_directed(edge_index)
        worker_num = len(source_set)
        task_num = len(object_set)
    return object_index,source_index,object_num,worker_num,claims
def get_domain(task_domain_file,work_rating_file):
    task_domain_matrix,domain_num,task_id_to_index=get_task_features(task_domain_file)
    worker_domain_matrix=get_worker_features(work_rating_file,domain_num,task_id_to_index,task_domain_matrix)
    worker_domain_matrix_norm = tf.nn.l2_normalize(worker_domain_matrix, axis=1)
    combined_matrix = tf.concat([task_domain_matrix, worker_domain_matrix_norm], axis=0)
    x = tf.nn.softmax(combined_matrix)
    return x
