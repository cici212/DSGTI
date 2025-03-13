import csv
import numpy as np
from tf_geometric.utils.graph_utils import convert_edge_to_directed, remove_self_loop_edge
from tf_geometric.data.graph import Graph
import tensorflow as tf

from data.process_domain import get_domain_edge
#from TIRA import object_index, source_index, claims, reliability, task_num, worker_num
from data.util import update_feature

# rating_file = './data/domaindata/MovieLens/rating-101.csv'
# rating_file='./data/domaindata/Anime/rating_anime_f20_148.csv'
rating_file='./data/domaindata/Anime/rating_20object.csv'
# rating_file='./data/domaindata/CopyAnime-20%/copyrating-11-20%.csv'
object_index,source_index,task_num,worker_num,claims=get_domain_edge(rating_file)
claims_one_hot = tf.one_hot(claims, depth=6)
claims_one_hot = tf.cast(claims_one_hot, tf.float32)
def get_feature(dataset=None, object_feature_path=None, answer_path=None, source_confidence_path=None):


    object_feature = np.loadtxt(object_feature_path)[:,:-2]
    answer = csv.reader(open(answer_path, 'r'))
    source_weight = np.loadtxt(source_confidence_path)[:,-1]
    source_index = []
    edge_label = []
    for line in answer:
        if dataset == 'valence7' or dataset == 'trec2011':
            source_index.append(int(line[1]))
        else:
            source_index.append(int(line[1])-1)
        edge_label.append(int(line[2]))
    source_num = np.array(source_index, dtype=np.int32).max() + 1
    class_num = np.array(edge_label, dtype=np.int32).max() + 1
    source_feature = np.zeros(shape=(source_num, class_num))# 创建全零矩阵
    for i in range(len(source_index)):
        source_feature[source_index[i]][edge_label[i]] = source_weight[source_index[i]]
    node_features = np.vstack([object_feature, source_feature])
    return node_features
def get_edge(dataset=None, answer_path=None):

    answer = csv.reader(open(answer_path, 'r'))
    object_index = []
    source_index = []
    edge_label = []
    for line in answer:
        if dataset == 'valence7' or dataset == 'trec2011':
            object_index.append(int(line[1]))
            source_index.append(int(line[0]))
        else:
            object_index.append(int(line[0])-1)
            source_index.append(int(line[1])-1)
        edge_label.append(int(line[2]))

    object_index = np.array(object_index, dtype=np.int32)

    object_num = object_index.max() + 1

    source_index = np.array(source_index, dtype=np.int32) + object_num

    edge_label = np.array(edge_label, dtype=np.int32)
    edge_label = np.vstack([object_index, source_index, edge_label])
    edge_index, _ = remove_self_loop_edge(edge_label[:2,:])

    edge_index, _ = convert_edge_to_directed(edge_index)

    return edge_index, edge_label

def get_label(dataset=None, object_label_path=None):
    with open(object_label_path, 'r', newline='', encoding='utf-8') as truth_file:
        truth_df = csv.reader(truth_file)
        object_index = []
        object_set = []
        truth = []
        object_truth_index = []
        for line in truth_df:
            if line[0].startswith('\ufeff'):
                line[0] = line[0][1:]
            object = int(line[0]) - 1
            truth_value = float(line[1])
            if object not in object_set:
                object_set.append(object)
            object_index.append(object)
            truth.append(truth_value)
        object_index = np.array(object_index, dtype=np.int32)
        mask = np.argsort(object_truth_index)
        truth = np.array(truth, dtype=np.float32)
        object_truth_index = np.sort(object_truth_index)

    # print(object_index)
    return object_index,np.array(truth,dtype=np.float32)

def get_graph(dataset=None,
              answer_path=None,
              truth_path=None):
    test_index, y = get_label(dataset=dataset, object_label_path=truth_path)
    edge_index, edge_label = get_edge(dataset=dataset, answer_path=answer_path)
    reliability = np.ones(shape=(worker_num)) / worker_num
    x = update_feature([object_index, source_index, claims_one_hot], reliability, task_num, worker_num)
    graph = Graph(x=x, edge_index=edge_index, y=y)
    return graph, edge_label, test_index
def load_data(dataset=None,
              answer_path=None,
              truth_path=None,
              ):
    graph, edge_label, test_index = get_graph(dataset=dataset,
                                  answer_path=answer_path,
                                  truth_path=truth_path)
    return graph, edge_label, test_index
