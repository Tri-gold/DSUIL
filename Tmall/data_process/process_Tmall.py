import numpy as np
import pandas as pd
import pickle

import random

count_cycle = 0
def process_seqs(seq_list, behavior_list, time_list):
    global count_cycle
    seq_numpy = np.array(seq_list)
    behavior_numpy = np.array(behavior_list)
    buy_index = np.where(behavior_numpy == 0)
    seq_buy_list = seq_numpy[buy_index]
    seq_buy_list_unique = np.unique(seq_buy_list)
    if len(seq_buy_list_unique) < len(seq_buy_list):
        count_cycle = count_cycle + 1




    out_seqs = []
    labs_item = []
    ids = []
    out_behavior = []
    labs_behavior = []
    seq_len = []
    purhcase_num_list = []
    time_seqs = []
    labs_time = []
    for i in range(1, len(seq_list)):
        if behavior_list[-i] == 0:
            tar_item = seq_list[-i]
            labs_item += [tar_item]
            tar_behavior = behavior_list[-i]
            labs_behavior += [tar_behavior]
            tar_time = time_list[-i]
            labs_time += [tar_time]
            out_seqs += [seq_list[:-i]]
            out_behavior += [behavior_list[:-i]]
            time_seqs += [time_list[:-i]]
            purhcase_num = np.sum(np.array(out_behavior) == 0)
            if out_behavior[-1] != 0:
                purhcase_num += 1
            purhcase_num_list.append(purhcase_num)
            seq_len.append(len(out_seqs))
            ids += [id]

    return out_seqs, out_behavior, time_seqs, labs_item, labs_behavior, labs_time, seq_len, purhcase_num_list


random.seed(0)
data_directory = '../Tmall/'
length = 50
L = 5
is_valid = False
train_df = pd.read_csv(data_directory + 'Tmall_train.csv')
valid_df = pd.read_csv(data_directory + 'Tmall_valid.csv')

test_df = pd.read_csv(data_directory + 'Tmall_test.csv')

between_val_test_df = pd.read_csv(data_directory + 'Tmall_between_val_test.csv')
# print("################", valid_df, '\n', test_df, '\n', between_val_test_df)
df = train_df.sort_values(by=['user_id', 'timestamp'])
between_val_test_df = between_val_test_df.sort_values(by=['user_id', 'timestamp'])

df_concat = pd.concat([df, valid_df, test_df, between_val_test_df], axis='index')

item_ids = df_concat.item_id.unique().tolist()
pad_item = len(item_ids)
user_num = len(df_concat.user_id.unique())
pad_behavior_num = 4
print("pad_items : {}\n user_num : {}".format(pad_item, user_num))
if is_valid:
    train_df = df
    valid_df = valid_df
    valid_users = set(valid_df['user_id'].values.tolist())
else:
    train_df = pd.concat([df, valid_df, between_val_test_df], axis='index')
    valid_df = test_df
    valid_users = set(valid_df['user_id'].values.tolist())


ids = df.user_id.unique()
train_df = train_df.sort_values(['user_id', 'timestamp'])
train_groups = train_df.groupby('user_id')
valid_groups = valid_df.groupby(['user_id'])
train_seq_list, train_beh_list, train_lab_item, train_lab_behavior, train_seq_len, train_puchase_num, train_time_list, train_lab_time =[] ,[], [], [], [], [], [], []
valid_seq_list, valid_beh_list, valid_lab_item, valid_lab_behavior, valid_seq_len, valid_puchase_num, valid_time_list, valid_lab_time =[], [], [], [], [], [], [], []
for id in ids:
    train_group = train_groups.get_group(id)
    total_len = len(train_group)
    item_list = train_group.item_id.values.tolist()
    item_list = item_list if total_len <= length else item_list[-length:]
    time_list = train_group.timestamp.values.tolist()
    time_list = time_list if total_len <= length else time_list[-length:]
    behavior_list = train_group.action_type.values.tolist()
    behavior_list = behavior_list if total_len <= length else behavior_list[-length:]
    out_seqs, out_behavior, out_time, labs_item, labs_behavior, labs_time, seq_len, puchase_num = process_seqs(item_list, behavior_list, time_list)
    train_seq_list += out_seqs
    train_beh_list += out_behavior
    train_time_list += out_time
    train_lab_item += labs_item
    train_lab_behavior += labs_behavior
    train_lab_time += labs_time
    train_seq_len.extend(seq_len)
    train_puchase_num.extend(puchase_num)
    if id in valid_users:
        valid_group = valid_groups.get_group(id)
        valid_tar_item = valid_group.item_id.values.tolist()
        valid_tar_beh = valid_group['action_type'].values.tolist()
        valid_tar_time = valid_group['timestamp'].values.tolist()
        valid_seq_list.extend([item_list])
        valid_beh_list.extend([behavior_list])
        valid_time_list.extend([time_list])
        valid_seq_len.append(len(item_list))
        valid_lab_item += valid_tar_item
        valid_lab_behavior += valid_tar_beh
        valid_lab_time += valid_tar_time

        purhcase_num = behavior_list.count(0)
        if behavior_list[-1] != 0:
            purhcase_num += 1
        valid_puchase_num.append(purhcase_num)


new_train = (train_seq_list, train_beh_list, train_time_list, train_lab_item, train_lab_behavior, train_lab_time, train_seq_len,
             train_puchase_num)
new_valid = (valid_seq_list, valid_beh_list, valid_time_list, valid_lab_item, valid_lab_behavior, valid_lab_time, valid_seq_len,
             valid_puchase_num)

pickle.dump(new_train, open('../processed_data/train.txt', 'wb'))
pickle.dump(new_valid, open('../processed_data/test.txt', 'wb'))

print(count_cycle)




