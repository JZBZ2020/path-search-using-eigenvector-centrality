# import networkx as nx
# G = nx.Graph()
# # G = nx.path_graph(4)
# # G.add_node(1) # 添加一个节点
# # G.add_nodes_from([2,3])
# # G.add_nodes_from([
# #     (4,{"color":"red"}),
# #     (5,{"color":"green"})
# # ])
# # 打印图的属性
# # print(G.graph)
# # centrality = nx.eigenvector_centrality(G)
# # print(['%s %0.2f'%(node,centrality[node]) for node in centrality]) # ['0 0.37', '1 0.60', '2 0.60', '3 0.37']
# # 合并图的节点
# H = nx.path_graph(10)
# # G.add_nodes_from(H)
# # print(len(H) ==len(G))
# # G.add_node(H)
# # print(len(H),len(G))
# ####
# G.add_edge(1,2)
# e = (2,3)
# G.add_edge(*e)
# G.add_edges_from([(1,2),(1,3)])
# # print(G[2])
# G.add_edges_from([(2,3,{'weight':3.14})])
# # print(G[2])
# G.add_edges_from(H.edges)
# G.clear()# 清空图
# G.add_edges_from([(1,2),(1,3)])
# G.add_node(1)
# G.add_edge(1,2)
# G.add_node("spam")
# G.add_nodes_from("spam")
# G.add_edge(3,'m')
# # print(G.number_of_nodes())
# # print(G.number_of_edges())
# # adjacency reporting
# DG = nx.DiGraph()
# DG.add_edge(2,1)
# DG.add_edge(1,3)
# DG.add_edge(2,4)
# DG.add_edge(1,2)
# assert list(DG.successors(2)) == [1,4]
# assert list(DG.edges) == [(2,1),(2,4),(1,3),(1,2)]
# # print(list(G.nodes))
# # print(list(G.edges))
# # print(list(G.adj[1]))
# # print(G.degree[1])
# # print(G.edges([2,'m']))
# # print(G.degree([2,3])) # [(2, 1), (3, 2)]
# G.remove_node(2)
# G.remove_nodes_from("spam")
# print(list(G.nodes))
# G.remove_edge(1,3)
# print(list(G.edges))
# def check_return(a):
#     """
#     检查return对循环的作用
#     :param a:
#     :return:
#     """
#     list =[1,2]
#     while True:
#         list.append(list[-1]+1)
#         for i in range(a):
#             list.append(i)
#             if i==4:
#                 return list # return退出整个大循环
#
#         print(list[-1])
# value  = check_return(10)
# print(value)
import torch
import random
list_a = [1,3,2,5]
list_a = torch.tensor(list_a)
value,index = torch.max(list_a,0)
# print(value,index)
# print(list_a.max())
list_b = [1,2,3,4,5]
print(max(list_b))
print(list_b[index])
rand_num = random.choices(list_a,k=1)
print(rand_num)
new_list = [[] for i in range(4)]
print(new_list)
def give(a,b,c):
    return a,b,c
dic = {'a': 0,'b': 1,'c':2}
print(give(**dic))
key = dic.keys()
a = torch.randn(3,4)
mean = torch.mean(a,1)
print("*"*10)

list1 =[[1,2,3],[4,5,6],[7,8,9]]

for i in range(3):
    result = lambda x:(x[0][i],x[1][i],x[2][i]),list1
    print(result)
str = "abcd"
dic1  = {}
for i in range(4):
    dic1[i] = 5+i
print(dic1,dic1.keys(),dic1.items())
data = list(map(lambda x:(x[1]), dic1.items()))
print(data)
