import torch
import torch.nn as nn
from torch.autograd import Variable
from check_path_search import get_entity_pair_path_info
import numpy as np
import collections
import random
"""

"""
batch_size = 16
LR = 0.01
epsilon = 0.9
gamma = 0.9
tao = 1
target_updata_freq = 4 # 超参数
iter_num = 100 #超参数
max_evi_num = 3
memory_capacity = 30
input_num = 64 # 公式9维度
hidden_num = 64 # 网络隐藏层维度
def get_net_input(h,t,evi_sents,action):
    """
    公式6，7，8，9
    """


# 根据公式9，确定输入维度，创建两个网络
class Net(nn.Module):
    def __init__(self,input_num,hidden_num):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_num,hidden_num)
        self.fc2 = nn.Linear(hidden_num,1)

    def forward(self,x):
        x = self.fc1(x)
        action_value = torch.tanh(self.fc2(x))# 公式11
        return action_value

class ReplayBuffer():
    def __init__(self, capacity):
        # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
        self.buffer = collections.deque(maxlen=capacity)

    # state,new_state=(h,t,[evi_sents])
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 随机采样batch_size行数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)  # list, len=32
        # *transitions代表取出列表中的值，即32项
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    # 目前队列长度
    def size(self):
        return len(self.buffer)

class ddqn(object):
    def __init__(self,input_num,hidden_num):
        self.eval_net,self.target_net = Net(input_num,hidden_num),Net(input_num,hidden_num)
        self.learn_step_counter = 0
        # self.memory_counter = 0

        self.memory = ReplayBuffer(memory_capacity)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss = nn.HuberLoss()

    # 确定state输入的维度
    def choose_action(self,state): # state=(h,t,[evi_sents])
        all_candidates = get_entity_pair_path_info(doc,state[0],state[1])
        candidates = [x for x in all_candidates if x not in state[2]]
        if np.random.uniform() < epsilon:
            # 随机在（h,t）剩下的候选集中选取一个句子下标
            action = random.choices(candidates,k=1)# list
        else:
            # 对候选句子根据公式9确定候选输入向量，对每个候选打分，选择分数最大的作为动作
            action_value = [self.eval_net(get_net_input(state[0],state[1],state[2],i)) for i in candidates]
            # 确定action_value维度
            action = torch.max(action_value,0)[1].data.numpy()
        return action

    # 从经验池提取数据，根据公式12,13,14,15训练模型
    def learn(self):
        if self.learn_step_counter % target_updata_freq ==0:
            updata_parameters = tao*self.eval_net.state_dict() + (1-tao)*self.target_net.state_dict()
            self.target_net.load_state_dict(updata_parameters)
        s, a, r, ns, d = self.memory.sample(batch_size)
        # 构造训练集
        transition_dict = {
            'states': s,
            'actions': a,
            'next_states': ns,
            'rewards': r,
            'dones': d,
        }
        # 获取当前时刻的状态 array_shape=[b,3]
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        # 获取当前时刻采取的动作 tuple_shape=[b]，维度扩充 [b,1]
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        # 当前状态下采取动作后得到的奖励 tuple=[b]，维度扩充 [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        # 下一时刻的状态 array_shape=[b,3]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        # 是否到达目标 tuple_shape=[b]，维度变换[b,1]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.bool).view(-1, 1)

        action_value_eval = torch.tensor(self.eval_net(get_net_input(states[:,0],states[:,1],states[:,2],actions[:,0])),dtype=torch.float) ##[b,]
        # 根据next_states获取每个实体对ht最大得分的动作
        all_hts_sents = [[] for i in range(batch_size)]
        action_list = []
        for i in range(batch_size):
            for sent in get_entity_pair_path_info(doc, next_states[i][0], next_states[i][1]):
                if sent not in next_states[i][2]:
                    all_hts_sents[i].append(sent)
            #根据ht候选动作集，选择分数最大的动作
            ht_value = [self.eval_net(get_net_input(next_states[i][0],next_states[i][1],next_states[i][2],action)) for action in all_hts_sents[i]]
            ht_value = torch.tensor(ht_value,dtype=torch.float)
            action_index = torch.max(ht_value,0)[1].data.numpy()
            action_list.append(action_index)
        action_list=torch.tensor(action_list).view(-1,1)
        #计算目标网络在nextstate下action_list的值
        action_value_target = torch.tensor(self.target_net(get_net_input(next_states[:,0],next_states[:,1],next_states[:2],action_list[:,0])),dtype=torch.float)
        action_value_target = rewards + gamma * action_value_target
        target_qvalues_for_actions = torch.where(dones, rewards, action_value_target)

        loss = self.loss(action_value_eval, target_qvalues_for_actions.detach())
        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step

    def infer(self):
        #遍历所有文档中的hts。初始化ht状态，列表values,track;遍历len(candidates)时间步，每步选择分值最大的动作，记录分数和证据集；根据values最大值下标确定最终证据集。
        infer_ht_evi = []
        for doc,ht in data: #data = (doc,[h,t])
            state = (ht[0],ht[1],[])
            values = []
            track = []
            ht_candidates = get_entity_pair_path_info(doc,ht[0],ht[1])
            for step in range(len(ht_candidates)):
                candis = [sent for sent in ht_candidates if sent not in state[2]]
                candis_values = [self.target_net(get_net_input(ht[0],ht[1],state[2],a)) for a in candis]
                candis_values = torch.tensor(candis_values,dtype=torch.float)
                value = torch.max(candis_values,0)[0].data.numpy()
                action = torch.max(candis_values,0)[1].data.numpy()
                values.append(value)
                track.append(state[2])
                state[2].append(action)
            values = torch.tensor(values,dtype=torch.float)
            _,indx = torch.max(values,0)
            infer_ht_evi.append(list(track[indx]))
        return infer_ht_evi


train_model = ddqn(input_num,hidden_num)
# 每一轮训练中，每个实体对hts初始句子集为空。选择动作，更新状态，查询奖励以及是否终止。存储信息，训练模型。
for epoc in iter_num:
    memory_counter = 0
    for i ,ht in enumerate(data): #data指代整个训练集上的数据
        state = (ht[0],ht[1],[])
        k = 0 # 超参数
        while True:
            action = train_model.choose_action(state)
            # 更新状态
            new_state = (ht[0],ht[1],state[2]+action)
            # 根据公式10，获取奖励
            ## 根据实体对获取黄金证据
            if action in ht_truth:
                reward = 1
            else:
                reward = 0
            # 根据K和ht的候选证据集，判断done
            if k >= max_evi_num or len(new_state[2])==len(get_entity_pair_path_info(doc,state[0],state[1]))
                done  = True
            else:
                done = False
            train_model.memory.add(state,action,reward,new_state,done)
            memory_counter+=1

            if memory_counter % memory_capacity ==0:
                train_model.learn()
            if done:
                break

            state = new_state


