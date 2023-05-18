import networkx as nx
from tqdm import tqdm
import numpy as np
import json

def search_samemeaning_entities(doc,entity):
    """
    查找与该实体同义的所有实体提及
    :param entity:
    :return: list
    """
    # 输入目标实体，查找在vertexSet中的位置下标，返回该位置列表中所有实体提及集合，去重
    doc_init_entities = [doc['vertexSet'][ent_id][0]['name'] for ent_id in range(len(doc['vertexSet']))]
    entity_idx = [i for i in range(len(doc_init_entities)) if doc_init_entities[i] == entity]
    entities_name_list = [dic['name'] for index in entity_idx for dic in doc['vertexSet'][index]]
    return list(set(entities_name_list))

def get_entity_pair_path_info(doc, entity1, entity2):
    # Construct graph and calculate eigenvector centrality scores for each node
    G = nx.Graph()
    entities = doc['vertexSet']
    # print(len(doc['vertexSet'])) #文档中所有实体个数
    for entity in entities:
        entity_id = entity[0]['name']
        G.add_node(entity_id)
    for sent_idx, sent in enumerate(doc['sents']):
        nodes_in_sent = []
        for entity in entities:
            sent_id_ = []  # 元素数量为0或1
            for mention in entity:
                sent_id = mention['sent_id']
                if sent_id == sent_idx:  # 在entity中只要有一个提及sent_id==ent_idx，就把node_id加进去，仅添加一次
                    if sent_id not in sent_id_:
                        sent_id_.append(sent_id)
                        node_id = entity[0]['name']  # 仅加 entity[0]['name']
                        nodes_in_sent.append(node_id)
        nodes_in_sent = list(set(nodes_in_sent))
        for i in range(len(nodes_in_sent)):
            for j in range(i+1, len(nodes_in_sent)):
                if not G.has_edge(nodes_in_sent[i],nodes_in_sent[j]):
                    G.add_edge(nodes_in_sent[i], nodes_in_sent[j], sent=[sent_idx])
                else:
                    G[nodes_in_sent[i]][nodes_in_sent[j]]['sent'].append(sent_idx)
    eig_centrality = nx.eigenvector_centrality(G,max_iter=10000)
    # print(eig_centrality)
    # print(G.edges)
    # print(G.nodes)
    # print(G.adj)

    # Beam search with width 2 to find path between entity nodes
    initial_path = [(0,[entity1])]
    # 1.初始列表中的两个元组扩充以后，在所有的元组中只要有一个出现entity2停止搜索，返回共现句子集；
    # 2.curr_cost+np.log(eig_centrality[neighbor])对新的分数值按降序排列
    # 3.扩展路径，防止回头
    # 4.路径搜索为空，则返回entity1,2在文档中所有提及出现的句子集
    beam_width = 2
    while True:

        paths = []
        for curr_path in initial_path:
            curr_cost,curr_node = curr_path

            for neighbor in list(G.neighbors(curr_node[-1])):
                if neighbor not in curr_node:
                    new_cost = curr_cost + np.log(eig_centrality[neighbor])
                    new_path = curr_node+[neighbor] # None
                    paths.append((new_cost, new_path))
        # #排序之前，判断所有扩展的路径是否包含entity2
        if len(paths)==0:
            #头结点或两条路径无法扩展或无法到达尾结点
            all_entities_name = search_samemeaning_entities(doc,entity1)+search_samemeaning_entities(doc,entity2)
            co_occur_sent_list = list(set([mention['sent_id'] for entities in doc['vertexSet'] for mention in entities if mention['name'] in all_entities_name]))
            co_occur_sent_list.sort()
            return co_occur_sent_list
        else:
            for (cost,path) in paths:
                if path is None:
                    return "Cannot find a path between the given entities."
                elif path[-1] == entity2:
                    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                    co_occur_sent_set = set()
                    for edge in path_edges:
                        co_occur_sent_set = co_occur_sent_set.union(set(G[edge[0]][edge[1]]['sent']))
                    co_occur_sent_list = list(co_occur_sent_set)
                    co_occur_sent_list.sort()
                    return co_occur_sent_list
                continue

        paths.sort(key=lambda a:a[0],reverse=True)# descent sort
        initial_path = paths[:beam_width]
    # return co_occur_sent_list

    # return None
if __name__ == '__main__':
    with open('./DocRED-Eider/DocRED/train_annotated.json') as f:
        data = json.load(f)
    # entity1 = data[0]['vertexSet'][10][0]['name']# 修改第三个索引
    # entity2 = data[0]['vertexSet'][4][0]['name']
    # print(get_entity_pair_path_info(data[0],entity1,entity2))
    correct_num = 0
    total_num = 0
    for sample in tqdm(data):
        #采集每个文档中待验证(h,t),并根据函数得出证据，如果输出证据包括人工标注的，记为1；
        hts_list_idx = [(ht['h'],ht['t']) for ht in sample['labels']]
        total_num +=len(hts_list_idx)
        hts_list = [(sample['vertexSet'][h_idx][0]['name'],sample['vertexSet'][t_idx][0]['name']) for h_idx,t_idx in hts_list_idx]
        for i,(entity1,entity2) in enumerate(hts_list):
            evi_out = get_entity_pair_path_info(sample,entity1,entity2)
            if set(sample['labels'][i]['evidence']).issubset(set(evi_out)):
                correct_num+=1
    print(correct_num,total_num,correct_num/total_num) # 31212 38180 0.8174960712414877
