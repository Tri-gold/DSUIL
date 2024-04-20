import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, drop):
        hidden_size = dim
        num_attention_heads = num_heads
        attention_probs_dropout_prob = drop
        hidden_dropout_prob = drop

        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class CombineGraph(Module):
    def __init__(self, opt, num_node):
        super(CombineGraph, self).__init__()
        self.opt = opt
        self.layers = opt.layers
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.gamma = opt.gamma
        self.dataset = opt.dataset


        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=self.dropout_local, dataset = self.dataset, gamma = self.gamma)

        self.embedding = nn.Embedding(num_node + 1, self.dim, padding_idx=num_node)
        self.beh_embedding = nn.Embedding(5, self.dim, padding_idx=4)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)

        self.w_t1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_t2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu_t1 = nn.Linear(self.dim, self.dim)
        self.glu_t2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

        self.attention = SelfAttention(self.dim, self.opt.num_heads, self.dropout_local)
        self.time_emb_prob = nn.Linear(self.dim, 1)

        self.device = torch.device('cuda')
        self.params = []
        var = (nn.Parameter(torch.rand(self.dim, requires_grad=True, device=self.device) * 0.01),
               nn.Parameter(torch.rand(self.dim, requires_grad=True, device=self.device)))
        self.params.append(var)
        self.is_add_two_edge = opt.is_add_two_edge
        self.is_meaning = opt.is_meaning


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):

        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden, mask, time_delta):

        attention_mask = (mask > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        extended_attention_mask = trans_to_cuda(extended_attention_mask)
        subsequent_mask = trans_to_cuda(subsequent_mask)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        sequence_emb = hidden


        hidden_states = self.attention(sequence_emb, extended_attention_mask)
        if self.opt.is_time == 0:
          select = hidden_states[:, 0, :]

        b = self.embedding.weight # n_nodes x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores

    def decay(self, t):
        return torch.mul(self.params[0][0], torch.exp(torch.neg(t)).unsqueeze(-1).repeat(1,1,self.dim)) + self.params[0][1]

    def pos_attention(self, hidden, beh, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        beh_emb = self.beh_embedding(beh)

        pos_emb = pos_emb + beh_emb

        hs = hidden[:, 0, :]
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)
        return select
    def meaning(self, hidden, beh, mask):

        mask = mask.float().unsqueeze(-1)

        temp = torch.sum(mask, 1)
        one = torch.ones_like(temp)
        mask_sum = torch.where(temp==0, one, temp)
        hs = torch.sum(hidden * mask, -2) / mask_sum
        return hs

    def time_attention(self, hidden, time, mask):
        len = hidden.shape[1]
        hs = hidden[:, 0, :]
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([time, hidden], -1), self.w_t1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_t2).squeeze(-1)
        beta = (beta * mask).unsqueeze(-1)
        select = torch.sum(beta * hidden, 1)
        return select

    def compute_scores_init(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        b = self.embedding.weight  # n_nodes x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores

    def forward_gru(self, input, adj, beh_adj, lastPurchaseIndex, out_mask, alias_inputs, beh_inputs_subseq, mask_item):
        input = input.to(self.device).float()  # (batchsize,seqlen,inputsize)
        batch_size = input.size(0)
        max_num_purchase = input.size(1)

        lisths = []
        out_mask_sum = torch.sum(out_mask, dim=1)
        step_size = torch.sum(out_mask, dim=1).max().item()

        for i in range(step_size):

            x = input[:, i, :].long()
            adj_one_step = adj[:, i, :, :]
            beh_adj_one_step = beh_adj[:, i, :, :]
            lastPurchaseIndex_one_step = lastPurchaseIndex[:, i]
            alias_inputs_one_step = alias_inputs[:, i, :]
            beh_inputs_subseq_one_step = beh_inputs_subseq[:, i, :]
            mask_item_one_step = mask_item[:, i, :]
            x_hidden = self.embedding(x)
            if i != 0:
                for index in range(batch_size):
                    if lastPurchaseIndex_one_step[index].item()<0:
                        continue

                    if self.is_add_two_edge == 1:
                        x_hidden[index][lastPurchaseIndex_one_step[index]] = (x_hidden[index][lastPurchaseIndex_one_step[index]] + h[index])/2
                    else:
                        x_hidden[index][lastPurchaseIndex_one_step[index]] =  x_hidden[index][lastPurchaseIndex_one_step[index]]
            hidden = self.local_agg(x_hidden, adj_one_step, beh_adj_one_step)
            get = lambda j: hidden[j][alias_inputs_one_step[j]]
            seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs_one_step)).long()])
            if self.is_meaning == 1:
                seq_hidden = self.meaning(seq_hidden, beh_inputs_subseq_one_step, mask_item_one_step)
            else:
                seq_hidden = self.pos_attention(seq_hidden, beh_inputs_subseq_one_step, mask_item_one_step)
            h = seq_hidden
            lisths.append(h)
        hs = torch.stack(lisths, dim=1)

        temp_hidden = []
        for i in range(out_mask_sum.shape[0]):
            user_seq_hidden = hs[i][:out_mask_sum[i]]
            user_seq_hidden = torch.flip(user_seq_hidden, [0])
            if len(user_seq_hidden) < max_num_purchase:
                pad_seq = torch.zeros(max_num_purchase - len(user_seq_hidden), user_seq_hidden.shape[-1])
                pad_seq = trans_to_cuda(pad_seq)
                user_seq_hidden = torch.cat((user_seq_hidden, pad_seq))
            temp_hidden.append(user_seq_hidden)
        seq_hidden = torch.stack(temp_hidden)
        return seq_hidden

    def forward_2(self, inputs, adj, beh_adj, beh_inputs_subseq, mask_item, hidden, seq_hidden, lastPurchaseIndex, isFirstSeq, layer, out_mask, ali_input):
        h_local = self.forward_gru(inputs, adj, beh_adj, lastPurchaseIndex, out_mask, ali_input, beh_inputs_subseq, mask_item)
        return h_local

    def forward(self, inputs, adj, time_adj, mask_item, hidden, seq_hidden, lastPurchaseIndex, isFirstSeq, layer):
        if layer!=0:
            h = hidden

        else:
            h = self.embedding(inputs)
        h_local = self.local_agg(h, adj, time_adj)
        output = h_local

        return output


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, adj, beh_adj, items, mask, targets, target_time, out_mask, init_input,\
    time_inputs_all_seq, beh_inputs_subseq, lastPurchaseIndex, isFirstSeq = data

    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    beh_inputs_subseq = trans_to_cuda(beh_inputs_subseq)
    adj = trans_to_cuda(adj).float()
    beh_adj = trans_to_cuda(beh_adj)
    mask = trans_to_cuda(mask).long()
    lastPurchaseIndex = trans_to_cuda(lastPurchaseIndex)
    isFirstSeq = trans_to_cuda(isFirstSeq)
    items_init = items
    adj_init = adj
    beh_adj_init = beh_adj
    mask_init = mask
    beh_inputs_subseq_init = beh_inputs_subseq

    alias_inputs_init = alias_inputs
    lastPurchaseIndex_init = lastPurchaseIndex
    isFirstSeq = isFirstSeq.reshape(-1, 1)
    out_mask_view = out_mask.view(-1)
    out_mask_index = torch.where(out_mask_view==1)

    isFirstSeq = isFirstSeq[out_mask_index]
    hidden = 0
    seq_hidden = 0
    for i in range(model.layers):
        seq_hidden_1 = model.forward_2(items_init, adj_init, beh_adj_init, beh_inputs_subseq_init, mask_init, hidden,
                           seq_hidden, lastPurchaseIndex_init, isFirstSeq, i, out_mask, alias_inputs_init)

    time_batch_delta = (target_time.unsqueeze(1) - time_inputs_all_seq) * out_mask
    out_mask = trans_to_cuda(out_mask)
    time_batch_delta = trans_to_cuda(time_batch_delta)
    return targets, model.compute_scores(seq_hidden_1, out_mask, time_batch_delta)


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=False, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit, ndcg = [0, 0], [0, 0]
    top_k = [10, 20]
    test_count = 0

    for data in test_loader:
        targets, scores = forward(model, data)
        scores = trans_to_cpu(scores).detach().numpy()
        train_seq = data[-1]
        for i, seq in enumerate(train_seq):
            scores[i][model.num_node] = -9999
        sub_scores = np.argsort(scores)

        targets = targets.numpy()
        test_count += targets.shape[0]
        for i in range(len(top_k)):
            rec_list = sub_scores[:, -top_k[i]:]
            for j in range(targets.shape[0]):
                # print("rec_list : ", rec_list[j], "target : ", targets[j])
                if targets[j] in rec_list[j]:
                    rank = top_k[i] - np.argwhere(rec_list[j] == targets[j])[0][0]
                    hit[i] += 1.0
                    ndcg[i] += 1.0 / np.log2(rank + 1)
    for i in range(len(top_k)):
        result.append(hit[i] / test_count)
        result.append(ndcg[i] / test_count)

    return result
