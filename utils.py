import copy

import numpy as np
import torch
from torch.utils.data import Dataset
import pickle


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def pad_list(seq_list, beh_list, time_list, num_purchase):
    if type(num_purchase) == list:
      print(num_purchase)
    copy_seq_list = copy.deepcopy(seq_list)
    seq_list.extend([[] for i in range(num_purchase - len(seq_list))]) if len(seq_list) < num_purchase else seq_list
    mask_seq_list = [1] * len(copy_seq_list) + [0] * (num_purchase - len(copy_seq_list)) if len(
        copy_seq_list) < num_purchase else [1] * len(copy_seq_list)

    beh_list.extend([[] for i in range(num_purchase - len(beh_list))]) if len(beh_list) < num_purchase else beh_list

    time_list.extend([[] for i in range(num_purchase - len(time_list))]) if len(time_list) < num_purchase else time_list

    return seq_list, beh_list, time_list, mask_seq_list


def handle_data(inputData, behaviorData, timeData, purchaseNum, pad_items=None, pad_beh=None, dataset=None):
    if dataset == "Tmall":
        buy_behavior_flage = 0
    else:
        buy_behavior_flage = 0
    max_purchaseNum = max(purchaseNum)

    len_data = [len(nowData) for nowData in inputData]
    split_seqData = []
    split_behData = []
    split_timeData = []
    split_maskData = []
    init_seqData = []
    timestamp_seqData = []
    max_len = 0
    for i in range(len(inputData)):
        if i < 10:
            print(behaviorData[i])
        one_seqData = []
        one_behData = []
        one_timeData = []
        input_seq_list = inputData[i]
        input_beh_list = behaviorData[i]
        input_time_list = timeData[i]
        behavior_numpy = np.array(input_beh_list)
        buy_index = np.where(behavior_numpy == buy_behavior_flage)[0]
        buy_timestamp = np.array(input_time_list)[buy_index].tolist()
        if input_beh_list[-1] != buy_behavior_flage:
            buy_timestamp.append(input_time_list[-1])
        init_seqData.append(input_seq_list)
        timestamp_seqData.append(buy_timestamp)
        start_index = 0
        j = 0
        purchase_count = 0

        while (j < len(input_seq_list)):
            if input_beh_list[j] == buy_behavior_flage:
                purchase_count += 1
                end_index = j
                one_seqData.append(input_seq_list[start_index:(end_index + 1)])
                one_behData.append(input_beh_list[start_index:(end_index + 1)])
                one_timeData.append(input_time_list[start_index:(end_index + 1)])


                temp_len = (end_index - start_index + 1)
                start_index = end_index
                max_len = max_len if max_len >= temp_len else temp_len

            if j == (len(input_seq_list) - 1) and input_beh_list[j] != buy_behavior_flage:
                one_seqData.append(input_seq_list[start_index:])
                one_behData.append(input_beh_list[start_index:])

            j += 1
        user_seq_list, user_beh_list, user_time_list, user_mask_list = pad_list(one_seqData, one_behData, one_timeData,
                                                                                max_purchaseNum)
        split_seqData.append(user_seq_list)
        split_maskData.append(user_mask_list)
        split_timeData.append(user_time_list)
        split_behData.append(user_beh_list)

    us_pois = []
    us_pois_beh = []
    us_pois_time = []
    us_msks = []
    us_pois_len = []
    pad_time = 0
    for index, one_user_seq in enumerate(split_seqData):
        us_pois.append([list(reversed(upois)) + [pad_items] * (max_len - len(upois)) if len(upois) < max_len else list(
            reversed(upois[-max_len:]))
                        for upois in one_user_seq])
        us_pois_beh.append(
            [list(reversed(upois)) + [pad_beh] * (max_len - len(upois)) if len(upois) < max_len else list(
                reversed(upois[-max_len:]))
             for upois in split_behData[index]])

        us_pois_time.append(
            [list(reversed(upois)) + [pad_time] * (max_len - len(upois)) if len(upois) < max_len else list(
                reversed(upois[-max_len:]))
             for upois in split_timeData[index]])

        us_pois_len.append([len(upois) for upois in one_user_seq])
        us_msks.append([[1] * len(upois) + [0] * (max_len - len(upois)) if len(upois) < max_len else [1] * max_len
                        for upois in one_user_seq])
    us_pois_init = [[pad_items] * (50 - le) + list(upois) if le < 50 else list(upois[-50:])
                    for upois, le in zip(init_seqData, len_data)]
    us_time_init = [list(reversed(timeData_list)) + [0] * (max_purchaseNum - len(timeData_list)) if len(
        timeData_list) < max_purchaseNum
                    else list(timeData_list[-max_purchaseNum:])
                    for timeData_list in timestamp_seqData]

    return us_pois, us_pois_beh, us_pois_time, us_pois_len, us_msks, max_len, split_maskData, max_purchaseNum, us_pois_init, us_time_init


def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity


def pair_map(i, j, trans_behavior_map, behaviorIndex_dict):
    behavior_i = behaviorIndex_dict[i]
    behavior_j = behaviorIndex_dict[j]
    return trans_behavior_map[behavior_i + '2' + behavior_j]


class Data(Dataset):
    def __init__(self, data, train_len=None, pad_item=None, pad_beh=None, dataset=None):
        if dataset in "Tmall_cycle":
            self.trans_behavior_map = {'p2p': 1, 'p2c': 2, 'p2b': 3, 'c2p': 4, 'c2c': 5,
                                       'c2b': 6, 'b2p': 7, 'b2c': 8, 'b2b': 9}
            self.behaviorIndex_dict = {0: 'p', 3: 'c', 2: 'b'}
        if dataset == 'UB':
            self.trans_behavior_map = {'p2p': 1, 'p2c': 2, 'p2f': 3, 'p2b': 4, 'c2p': 5, 'c2c': 6, 'c2f': 7,
                                       'c2b': 8,
                                       'f2p': 9, 'f2c': 10, 'f2f': 11, 'f2b': 12, 'b2p': 13, 'b2c': 14, 'b2f': 15,
                                       'b2b': 16}
            self.behaviorIndex_dict = {0: 'p', 1: 'c', 2: 'b', 3: 'f'}
        if dataset == 'RC15':
            self.trans_behavior_map = {'c2c': 1,
                                       'c2b': 2, 'b2b': 3, 'b2c': 4}
            self.behaviorIndex_dict = {0: 'c', 1: 'b'}
        if dataset == 'ML1M':
            self.trans_behavior_map = {'c2c': 1,
                                       'c2b': 2, 'b2b': 3, 'b2c': 4}
            self.behaviorIndex_dict = {0: 'c', 1: 'b'}
        inputs, inputs_beh, inputs_time, inputs_len, inner_mask, max_len, out_mask, max_buy_num, init_input, timestamp_input = handle_data(
            data[0], data[1], data[2], data[-1], pad_item, pad_beh, dataset)
        self.inputs = np.asarray(inputs)
        self.inputs_len = np.asarray(inputs_len)
        self.targets = np.asarray(data[3])
        self.inputs_beh = inputs_beh
        self.inputs_time = inputs_time
        self.targets_time = np.asarray(data[5])
        self.inner_mask = np.asarray(inner_mask)
        self.out_mask = np.asarray(out_mask)
        self.length = len(data[0])
        self.max_len = max_len
        self.pad_items = pad_item
        self.pad_beh = pad_beh
        self.init_input = np.asarray(init_input)
        self.timestamp_input = np.asarray(timestamp_input)
        self.max_len_buy = max_buy_num
        self.dataset = dataset

    def __getitem__(self, index):
        u_input, u_input_beh, u_input_time, u_input_len, mask, target, target_time, out_mask = self.inputs[index], \
                                                                                               self.inputs_beh[
                                                                                                   index], \
                                                                                               self.inputs_time[index], \
                                                                                               self.inputs_len[index], \
                                                                                               self.inner_mask[index], \
                                                                                               self.targets[
                                                                                                   index], \
                                                                                               self.targets_time[index], \
                                                                                               self.out_mask[index]
        user_alias_inputs = []
        user_adj = []
        user_adj_beh = []
        user_adj_time = []
        user_items = []
        last_purchase_index_list = []
        is_first_seq = []
        for i, u_sample_input in enumerate(u_input):
            u_sample_input_len = u_input_len[i]
            max_n_node = self.max_len

            if type(u_sample_input) == list:
                sorted_set = sorted(set(u_sample_input), key=u_sample_input.index)
            else:
                sorted_set = sorted(set(u_sample_input.tolist()), key=u_sample_input.tolist().index)
            if u_sample_input[-1] != self.pad_items:
                gap = 1
            else:
                gap = 2

            node = np.array(list(sorted_set))

            last_purchase_index = node.shape[0] - gap
            u_beh_input = u_input_beh[i]
            u_time_input = u_input_time[i]
            items = node.tolist() + (max_n_node - len(node)) * [self.pad_items]
            adj = np.zeros((max_n_node, max_n_node))
            adj_beh = np.zeros((max_n_node, max_n_node))
            adj_time = -1 * np.ones((max_n_node, max_n_node))
            is_first_seq.append(0 if i == 0 else 1)
            for i in np.arange(len(u_sample_input) - 1):
                if u_sample_input[0] == self.pad_items:
                    break
                if i == u_sample_input_len - 1:
                    u = np.where(node == u_sample_input[0])[0][0]
                    v = np.where(node == u_sample_input[i])[0][0]
                    u_b = u_beh_input[0]
                    v_b = u_beh_input[i]
                    u_t = u_time_input[0]
                    v_t = u_time_input[i]
                    # adj[v][u] = 5
                    adj[v][u] = 3
                    adj_beh[v][u] = pair_map(u_b, v_b, self.trans_behavior_map, self.behaviorIndex_dict)
                    adj_time[v][u] = abs(v_t - u_t)
                u = np.where(node == u_sample_input[i])[0][0]
                adj[u][u] = 1
                adj_time[u][u] = 0

                if u_sample_input[i + 1] == self.pad_items:
                    break
                v = np.where(node == u_sample_input[i + 1])[0][0]
                if u == v or adj[u][v] == 4:
                    continue
                adj[v][v] = 1
                adj[v][u] = 2
                u_b = u_beh_input[i]
                v_b = u_beh_input[i + 1]

                u_t = u_time_input[i]
                v_t = u_time_input[i + 1]

                adj_beh[v][u] = pair_map(u_b, v_b, self.trans_behavior_map, self.behaviorIndex_dict)
                adj_time[v][u] = abs(u_t - v_t)


            user_alias_inputs.append([np.where(node == i)[0][0] for i in u_sample_input])
            user_adj.append(adj)
            user_adj_beh.append(adj_beh)
            user_adj_time.append(adj_time)
            user_items.append(items)
            last_purchase_index_list.append(last_purchase_index)

        return [torch.tensor(user_alias_inputs), torch.tensor(user_adj), torch.tensor(user_adj_beh),
                torch.tensor(user_items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(target_time),
                torch.tensor(out_mask), torch.tensor(self.init_input[index]), torch.tensor(self.timestamp_input[index]),
                torch.tensor(u_input_beh), torch.tensor(last_purchase_index_list), torch.tensor(is_first_seq)]


    def __len__(self):
        return self.length
