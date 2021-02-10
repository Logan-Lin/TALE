import os
import math
from itertools import zip_longest

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import functional as F
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import nni

from utils import weight_init, next_batch, cal_classify_metric


class NextLocParam:
    local_model_path = os.path.join('data', 'cache', 'next_loc_{}_{}.model'.format(nni.get_experiment_id(), nni.get_trial_id()))
    local_result_path = os.path.join('data', 'cache', 'next_loc_result.h5')
    top_n_list = [1, 2, 3, 4, 5, 10, 20]


def attention(query, key, value, mask=None, dropout=None):
    """
    :param query: (B, <h>, max_length, d_k)
    :param key: (B, <h>, max_length, d_k)
    :param value: (B, <h>, max_length, d_k)
    :param mask:  (B, <1>, max_length, max_length), true/false matrix, and true means paddings
    :return: outputs:(B, <h>, max_length, d_k), att_scores:(B, <h>, max_length, max_length)
    """
    "Compute 'Scaled Dot Product Attention'"
    k_size = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(k_size)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)  # true->-1e9
    p_attn = F.softmax(scores, dim=-1)  # 每行和为1
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class DeepMove(nn.Module):
    """DeepMove: Predicting Human Mobility with Attentional Recurrent Networks."""

    def __init__(self, loc_embed_layer: nn.Module, loc_embed_size, time_embed_size, user_embed_size,
                 num_loc, num_time, num_users, hidden_size):
        super().__init__()
        self.num_loc = num_loc
        self.time_embed_size = time_embed_size
        self.user_embed_size = user_embed_size

        self.loc_embed_layer = loc_embed_layer
        self.add_module('loc_embed_layer', self.loc_embed_layer)

        self.time_embed_layer = nn.Embedding(num_time + 1, time_embed_size, padding_idx=num_time)
        self.user_embed_layer = nn.Embedding(num_users, user_embed_size)
        self.dropout = nn.Dropout(0.1)

        input_size = loc_embed_size + time_embed_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.history_cast = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
        self.out_linear = nn.Sequential(nn.Tanh(), nn.Linear(2 * hidden_size + user_embed_size, num_loc))

        self.apply(weight_init)

    def forward(self, current_poi_seq, current_time_seq, history_poi_seqs, history_time_seqs, user_seq):
        """
        :param current_poi_seq: current poi sequence, shape (batch_size, max_current_seq_len)
        :param current_time_seq: current time sequence, shape (batch_size, max_current_seq_len)
        :param history_poi_seqs: historical poi sequences, shape (batch_size, num_his_session, max_his_seq_len)
        :param history_time_seqs: historical time sequences, shape (batch_size, num_his_session, max_his_seq_len)
        :param user_seq: user sequence corresponding to poi sequences, shape (batch_size)
        :return: prediction of the next visited location, shape (batch_size, num_loc).
        """
        # Process current trajectory using GRU.
        current_valid_len = (current_poi_seq != self.num_loc).long().sum(-1)  # (batch_size)

        current_poi_embed = self.loc_embed_layer(current_poi_seq)  # (batch_size, max_current_seq_len, loc_embed_size)
        current_time_embed = self.time_embed_layer(current_time_seq)
        cat_current_embed = torch.cat([current_poi_embed, current_time_embed], dim=-1) \
            if self.time_embed_size > 0 else current_poi_embed  # (B, L, input_size)
        cat_current_embed = self.dropout(cat_current_embed)

        packed_current_embed = pack_padded_sequence(cat_current_embed, current_valid_len, batch_first=True,
                                                    enforce_sorted=False)
        _, gru_hidden = self.gru(packed_current_embed)  # (num_layers, batch_size, hidden_size)
        gru_hidden = gru_hidden.permute(1, 0, 2)  # (N, 1, hidden_size)

        # Process historical trajectory using mean-pooling.
        history_valid_mask = (history_poi_seqs != self.num_loc)  # (batch_size, num_his_session, max_his_seq_len)
        # Calculate the valid length of every historical sessions.
        history_valid_len = history_valid_mask.long().sum(-1)  # (batch_size, num_his_session)
        attn_key_mask = (history_valid_len == 0)  # (batch_size, num_his_session)

        history_poi_embed = self.loc_embed_layer(history_poi_seqs)
        history_time_embed = self.time_embed_layer(history_time_seqs)
        cat_history_embed = torch.cat([history_poi_embed, history_time_embed], dim=-1) \
            if self.time_embed_size > 0 else history_poi_embed # (N, L, L, input_size)
        cat_history_embed = self.dropout(cat_history_embed)

        sum_history_embed = cat_history_embed.masked_fill(~history_valid_mask.unsqueeze(-1), 0.0).sum(2)  # (N, num_his_session, input_size)
        mean_history_embed = sum_history_embed / history_valid_len.masked_fill(attn_key_mask, 1.0).unsqueeze(-1)  # (N, num_his_session, input_size)
        mean_history_embed = self.history_cast(mean_history_embed)  # (N, L, hidden_size)

        # Merge current and history information with attention.
        attn_history_embed, _ = attention(gru_hidden, mean_history_embed, mean_history_embed,
                                          attn_key_mask.unsqueeze(1), self.dropout)  # (N, 1, hidden_size)
        merge_history_embed = torch.cat([gru_hidden, attn_history_embed], dim=-1).squeeze(1)  # (N, hidden_size * 2)

        user_embed = self.user_embed_layer(user_seq)  # (N, user_embed_size)
        final_rep = torch.cat([merge_history_embed, user_embed], dim=-1) \
            if self.user_embed_size > 0 else merge_history_embed  # (N, hidden_size * 2 + user_embed_size)
        out = self.out_linear(final_rep)
        return out


class LstmLocPredictor(nn.Module):
    def __init__(self, loc_embed_layer, loc_embed_size, num_loc, hidden_size):
        super().__init__()
        self.num_loc = num_loc

        self.loc_embed_layer = loc_embed_layer
        self.add_module('loc_embed_layer', self.loc_embed_layer)

        # self.rnn = nn.LSTM(loc_embed_size, hidden_size, num_layers=1, batch_first=True)
        self.rnn = nn.GRU(loc_embed_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.out_linear = nn.Sequential(nn.Tanh(), nn.Linear(hidden_size, num_loc))

        self.apply(weight_init)

    def forward(self, current_poi_seq, *args, **kwargs):
        """
        :param current_poi_seq: current poi sequence, shape (batch_size, max_current_seq_len)
        :return: prediction of the next visited location, shape (batch_size, num_loc)
        """
        valid_len = (current_poi_seq != self.num_loc).long().sum(-1)  # (batch_size)

        poi_embed = self.dropout(self.loc_embed_layer(current_poi_seq))  # (batch_size, max_current_seq_len, loc_embed_size)
        packed_poi_embed = pack_padded_sequence(poi_embed, valid_len, batch_first=True, enforce_sorted=False)
        # _, (hidden, _) = self.rnn(packed_poi_embed)
        _, hidden = self.rnn(packed_poi_embed)
        hidden = hidden.squeeze(0)  # (batch_size, hidden_size)

        out = self.out_linear(hidden)
        return out


def train_next_loc(pre_model, dataset, batch_size, num_epoch,
                   lr, test_set_choice, early_stopping_round,
                   device, **kwargs):
    assert test_set_choice in [1, 2]
    num_loc = len(dataset.poi_index_map)

    pre_model = pre_model.to(device)
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    train_set = dataset.gen_split_session(set_choice=0, **kwargs)
    eval_set = dataset.gen_split_session(set_choice=1, **kwargs)
    test_set = eval_set if test_set_choice == 1 else dataset.gen_split_session(set_choice=2, **kwargs)

    def _pre_batch(input_model, batch):
        """ Give prediction of one batch. """
        user_seq, poi_sessions, time_sessions, label_seq, valid_num_session = zip(*batch)
        user_seq, label_seq = (torch.tensor(item).long().to(device) for item in [user_seq, label_seq])  # (batch_size)
        bs = user_seq.size(0)
        max_num_sessions = np.max(valid_num_session)

        def _pad_session(s, fill_value):
            # Each batch in poi_sessions contains a sequence of sessions, and one session contains a sub-sequence of location trajectory.
            # Noted that the number of history sessions and the length of each session are variable.
            flatten_session_batch = []
            for session_i, session_batch in enumerate(s):
                flatten_session_batch += ([[]] * (max_num_sessions - valid_num_session[session_i]) +
                                          session_batch)  # (batch_size * max_num_sessions, max_session_len)
            filled_session_batches = np.array(list(zip_longest(*flatten_session_batch, fillvalue=fill_value))).transpose()
            filled_session_batches = filled_session_batches.reshape((bs, max_num_sessions, -1))  # (batch_size, max_num_sessions, max_seq_len)
            return filled_session_batches

        poi_sessions, time_sessions = _pad_session(poi_sessions, fill_value=num_loc), \
                                      _pad_session(time_sessions, fill_value=24)
        current_poi_s, history_poi_s = (torch.tensor(item).long().to(device)
                                        for item in [poi_sessions[:, -1, :], poi_sessions[:, :-1, :]])
        current_time_s, history_time_s = (torch.floor(torch.tensor(item)).long().to(device)
                                          for item in [time_sessions[:, -1, :], time_sessions[:, :-1, :]])
        pre_distribution = input_model(current_poi_s, current_time_s, history_poi_s, history_time_s, user_seq)
        return pre_distribution, label_seq

    def _test_epoch(input_model, input_set):
        """ Test model on the whole input set. """
        input_model.eval()
        pre_distributions, labels = [], []
        for batch in next_batch(input_set, batch_size=256):
            pre_distribution, label = _pre_batch(input_model, batch)
            pre_distributions.append(pre_distribution.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
        pre_distributions, labels = (np.concatenate(item) for item in (pre_distributions, labels))
        pres = pre_distributions.argmax(-1)
        return pre_distributions, pres, labels

    max_metric = 0.0
    worse_round = 0
    for epoch in range(num_epoch):
        for batch in next_batch(shuffle(train_set), batch_size):
            pre_model.train()
            pre_distribution, label = _pre_batch(pre_model, batch)
            optimizer.zero_grad()
            loss = loss_func(pre_distribution, label)
            loss.backward()
            optimizer.step()

        pre_distributions, pres, labels = _test_epoch(pre_model, eval_set)
        metric = accuracy_score(labels, pres)
        nni.report_intermediate_result(metric)
        if early_stopping_round > 0 and max_metric < metric:
            max_metric = metric
            worse_round = 0
            torch.save(pre_model.state_dict(), NextLocParam.local_model_path)
        else:
            worse_round += 1

        if 0 < early_stopping_round <= worse_round:
            print('Early Stopping, best Epoch %d.' % (epoch - worse_round))
            break

    # Load the model with the best test metric value.
    if early_stopping_round > 0:
        pre_model.load_state_dict(torch.load(NextLocParam.local_model_path))
    pre_distributions, pres, labels = _test_epoch(pre_model, test_set)
    score_series = cal_classify_metric(pre_distributions, pres, labels, NextLocParam.top_n_list)
    print(score_series)
    nni.report_final_result(score_series.loc['acc@1'])
    os.remove(NextLocParam.local_model_path)
    return score_series
