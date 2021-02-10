import math
import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.nn.utils import weight_norm
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
import nni

from utils import weight_init, next_batch, cal_regression_metric
from downstream.next_loc import attention


class FlowParam:
    local_model_path = os.path.join('data', 'cache', 'flow.model')
    local_result_path = os.path.join('data', 'cache', 'flow_result.model')


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: input temporal sequence, shape (batch_size, input_channel, seq_len)
        :return: output hidden sequence, shape (batch_size, output_channel, out_seq_len)
        """
        return self.network(x)


class TcnFlowPredictor(nn.Module):
    """
    Temporal Convolutional Network-based flow predictor.
    """
    def __init__(self, input_seq_len, loc_embed_layer, loc_embed_size, hidden_size):
        super().__init__()

        self.loc_embed_layer = loc_embed_layer
        self.add_module('loc_embed_layer', self.loc_embed_layer)

        channel_sizes = [1] + [2 ** i for i in range(2, int(math.log2(hidden_size)), 2)]
        self.tcn = TemporalConvNet(num_inputs=1, num_channels=channel_sizes,
                                   kernel_size=2, dropout=0.1)
        self.tcn_hidden_linear = nn.Linear(channel_sizes[-1] * input_seq_len, hidden_size)

        self.embed_linear = nn.Linear(loc_embed_size, hidden_size)
        self.out_linear = nn.Sequential(nn.Linear(hidden_size * 2, int(hidden_size / 2)), nn.Tanh(),
                                        nn.Linear(int(hidden_size / 2), 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

        self.apply(weight_init)

    def forward(self, recent_history, loc_index, **kwargs):
        """
        :param recent_history: historical flow sequences, shape (batch_size, his_len)
        :param loc_index: location indices corresponding to flow sequences, shape (batch_size)
        :return: prediction of future flows, shape (batch_size)
        """
        recent_history = recent_history.unsqueeze(1)  # (batch_size, 1, his_len)
        tcn_hidden = self.tcn(recent_history)  # (batch_size, hidden_size, his_len)
        tcn_hidden = tcn_hidden.reshape(tcn_hidden.size(0), -1)  # (batch_size, hidden_size * his_len)
        tcn_hidden = self.tcn_hidden_linear(tcn_hidden)  # (batch_size, hidden_size)

        loc_embed = self.dropout(self.loc_embed_layer(loc_index))  # (batch_size, loc_embed_size)
        loc_h = self.embed_linear(loc_embed)  # (batch_size, hidden_size)
        cat_h = self.tanh(torch.cat([tcn_hidden, loc_h], dim=-1))  # (batch_size, hidden_size * 2)
        out = self.out_linear(self.dropout(cat_h)).squeeze(-1)
        return out


class LstmFlowPredictor(nn.Module):
    def __init__(self, loc_embed_layer, loc_embed_size, hidden_size, num_layers):
        super().__init__()

        self.loc_embed_layer = loc_embed_layer
        self.add_module('loc_embed_layer', self.loc_embed_layer)

        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=0.1)
        self.embed_linear = nn.Linear(loc_embed_size, hidden_size)
        self.out_linear = nn.Sequential(nn.Linear(hidden_size * 2, int(hidden_size / 2)), nn.Tanh(),
                                        nn.Linear(int(hidden_size / 2), 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

        self.apply(weight_init)

    def forward(self, recent_history, loc_index, **kwargs):
        """
        :param recent_history: historical flow sequences, shape (batch_size, his_len)
        :param loc_index: location indices corresponding to flow sequences, shape (batch_size)
        :return: prediction of future flows, shape (batch_size)
        """
        rnn_out, _ = self.rnn(recent_history.unsqueeze(-1))  # (batch_size, his_len, hidden_size)
        rnn_out = rnn_out[:, -1, :]  # (batch_size, hidden_size)

        loc_embed = self.dropout(self.loc_embed_layer(loc_index))  # (batch_size, loc_embed_size)
        loc_h = self.embed_linear(loc_embed)  # (batch_size, hidden_size)

        cat_h = self.dropout(self.tanh(torch.cat([rnn_out, loc_h], dim=-1)))  # (batch_size, hidden_size * 2)
        out = self.out_linear(cat_h).squeeze(-1)
        return out


class Seq2seqFlowPredictor(nn.Module):
    def __init__(self, loc_embed_layer, loc_embed_size, latent_size, hidden_size, num_layers):
        super().__init__()

        self.loc_embed_layer = loc_embed_layer
        self.add_module('loc_embed_layer', self.loc_embed_layer)

        _rnn_input_size = latent_size * 2 + 16
        self.recent_encoder = nn.GRU(input_size=_rnn_input_size, hidden_size=hidden_size, num_layers=num_layers,
                                     batch_first=True, dropout=0.1)
        self.remote_encoder = nn.GRU(input_size=_rnn_input_size, hidden_size=hidden_size, num_layers=num_layers,
                                     batch_first=True, dropout=0.1)
        self.encoder_merge_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder = nn.GRU(input_size=_rnn_input_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, dropout=0.1)
        self.time_embed = nn.Embedding(24, 16)

        self.embed_linear = nn.Linear(loc_embed_size, latent_size)
        self.flow_linear = nn.Linear(1, latent_size)
        self.out_linear = nn.Linear(hidden_size, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

        self.apply(weight_init)

    def forward(self, recent_history, remote_history, target_seq, loc_index, recent_hour, remote_hour, **kwargs):
        """
        :param recent_history: historical flow sequences, shape (batch_size, his_len)
        :param remote_history: remote historical flow sequences from previews day, shape (batch_size, his_len)
        :param target_seq: target flow sequences, shape (batch_size, pre_len)
        :param loc_index: location indices corresponding to flow sequences, shape (batch_size)
        :param recent_hour: hour indices of all sequences, shape (batch_size, his_len+pre_len)
        :return: prediction of future flows, shape (batch_size, pre_len)
        """
        loc_embed = self.loc_embed_layer(loc_index)
        loc_h = self.embed_linear(loc_embed)  # (batch_size, latent_size)
        recent_flow_h = self.flow_linear(torch.cat([recent_history, target_seq], -1).unsqueeze(-1))  # (batch_size, his_len+pre_len, latent_size)
        recent_hour_embed = self.time_embed(recent_hour)  # (batch_size, his_len+pre_len, time_embed_size)
        recent_cat_h = torch.cat([recent_flow_h, loc_h.unsqueeze(1).repeat(1, recent_flow_h.size(1), 1), recent_hour_embed], -1)  # (batch_size, his_len+pre_len, rnn_input_size)
        recent_cat_h = self.dropout(self.tanh(recent_cat_h))

        # remote_flow_h = self.flow_linear(remote_history.unsqueeze(-1))  # (batch_size, his_len, latent_size)
        # remote_hour_embed = self.time_embed(remote_hour)  # (batch_size, his_len, time_embed_size)
        # remote_cat_h = torch.cat([remote_flow_h, loc_h.unsqueeze(1).repeat(1, remote_flow_h.size(1), 1), remote_hour_embed], -1)  # (batch_size, his_len, rnn_input_size)
        # remote_cat_h = self.dropout(self.tanh(remote_cat_h))

        recent_encoder_out, recent_hc = self.recent_encoder(recent_cat_h[:, :recent_history.size(1)])
        # remote_encoder_out, remote_hc = self.remote_encoder(remote_cat_h)
        # hc = self.encoder_merge_linear(self.dropout(self.relu(torch.cat([recent_hc, remote_hc], -1))))
        decoder_out, hc = self.decoder(recent_cat_h[:, recent_history.size(1)-1:-1], recent_hc)  # (batch_size, pre_len, hidden_size)
        out = self.out_linear(decoder_out).squeeze(-1)
        return out


class RnnFuseFlowPredictor(nn.Module):
    def __init__(self, loc_embed_layer, loc_embed_size, hidden_size, latent_size, num_layers):
        super().__init__()

        self.loc_embed_layer = loc_embed_layer
        self.add_module('loc_embed_layer', self.loc_embed_layer)

        self.embed_linear = nn.Linear(loc_embed_size, latent_size)
        self.flow_linear = nn.Linear(1, latent_size)

        self.rnn = nn.GRU(input_size=latent_size * 2, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=0.1)
        self.out_linear = nn.Sequential(nn.Linear(hidden_size, int(hidden_size / 4)), nn.Tanh(),
                                        nn.Linear(int(hidden_size / 4), 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

        self.apply(weight_init)

    def forward(self, recent_history, loc_index, **kwargs):
        """
        :param recent_history: historical flow sequences, shape (batch_size, his_len)
        :param loc_index: location indices corresponding to flow sequences, shape (batch_size)
        :return: prediction of future flows, shape (batch_size)
        """
        loc_embed = self.loc_embed_layer(loc_index)  # (batch_size, embed_size)
        loc_h = self.embed_linear(loc_embed)  # (batch_size, latent_size)
        flow_h = self.flow_linear(recent_history.unsqueeze(-1))  # (batch_size, his_len, latent_size)
        cat_h = torch.cat([loc_h.unsqueeze(1).repeat(1, flow_h.size(1), 1), flow_h], -1)  # (batch_size, his_len, latent_size * 2)
        cat_h = self.dropout(self.tanh(cat_h))

        rnn_out, _ = self.rnn(cat_h)
        rnn_out = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        out = self.out_linear(rnn_out).squeeze(-1)
        return out


class RnnPeriodFlowPredictor(nn.Module):
    def __init__(self, loc_embed_layer, loc_embed_size, hidden_size, num_layers):
        super().__init__()

        self.loc_embed_layer = loc_embed_layer
        self.add_module('loc_embed_layer', self.loc_embed_layer)

        _rnn_params = {'input_size': 1, 'hidden_size': hidden_size, 'num_layers': num_layers,
                       'batch_first': True, 'dropout': 0.1}
        self.recent_rnn = nn.GRU(**_rnn_params)
        self.remote_rnn = nn.GRU(**_rnn_params)

        self.time_embed = nn.Embedding(24, embedding_dim=hidden_size)

        self.embed_linear = nn.Linear(loc_embed_size, hidden_size)
        self.out_linear = nn.Sequential(nn.Linear(hidden_size * 4, hidden_size), nn.Tanh(),
                                        nn.Linear(hidden_size, 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

        self.apply(weight_init)

    def forward(self, recent_history, remote_history, loc_index, pre_hour_index):
        """
        :param recent_history: recent historical flow sequences, shape (batch_size, his_len).
        :param remote_history: remote historical flow sequences, shape (batch_size, num_remote_days, remote_len).
        :param loc_index: location indices corresponding to flow sequences, shape (batch_size).
        :param pre_hour_index: hour indices of points to predict, shape (batch_size).
        :return: prediction of future flows, shape (batch_size).
        """
        batch_size = recent_history.size(0)
        num_remote_days = remote_history.size(1)

        recent_out, _ = self.recent_rnn(recent_history.unsqueeze(-1))  # (batch_size, his_len, hidden_size)
        recent_out = recent_out[:, -1, :]  # (batch_size, hidden_size)

        remote_out, _ = self.remote_rnn(remote_history.reshape(batch_size * num_remote_days, -1).unsqueeze(-1))
        remote_out = remote_out[:, -1, :].reshape(batch_size, num_remote_days, -1)  # (batch_size, num_remote_days, hidden_size)
        remote_out, _ = attention(remote_out, remote_out, remote_out, dropout=self.dropout)
        remote_out = remote_out.sum(1)  # (batch_size, hidden_size)

        loc_embed = self.dropout(self.loc_embed_layer(loc_index))  # (batch_size, loc_embed_size)
        loc_h = self.embed_linear(loc_embed)  # (batch_size, hidden_size)

        time_embed = self.time_embed(pre_hour_index)  # (batch_size, hidden_size)

        cat_h = self.dropout(self.tanh(torch.cat([recent_out, remote_out, loc_h, time_embed], dim=-1)))  # (batch_size, hidden_size * 4)
        out = self.out_linear(cat_h).squeeze(-1)
        return out


class NeighFlowPredictor(nn.Module):
    def __init__(self, loc_embed_layer, loc_embed_size, hidden_size, num_layers):
        super().__init__()

        self.loc_embed_layer = loc_embed_layer
        self.add_module('loc_embed_layer', self.loc_embed_layer)

        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size,
                          batch_first=True, num_layers=num_layers, dropout=0.1)
        self.embed_linear = nn.Sequential(nn.Linear(loc_embed_size, hidden_size), nn.Tanh(), nn.Dropout(0.1))
        self.attn_linear = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(0.1))
        self.out_linear = nn.Sequential(nn.Linear(hidden_size * 2, int(hidden_size / 2)),
                                        nn.Tanh(), nn.Dropout(0.1),
                                        nn.Linear(int(hidden_size / 2), 1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, recent_history, neighbor_history,
                target_poi_index, neighbor_poi_index):
        """
        :param recent_history: recent historical flow sequences of target location, shape (batch_size, his_len)
        :param neighbor_history: recent historical flow sequences of neighbor locations, shape (batch_size, num_neigh, his_len)
        :param target_poi_index: poi indices of target locations, shape (batch_size)
        :param neighbor_poi_index: poi indices of neighbor locations, shape (batch_size, num_neigh)
        :return: prediction of future flows, shape (batch_size)
        """
        batch_size = recent_history.size(0)
        num_neigh = neighbor_history.size(1)

        target_out, _ = self.rnn(recent_history.unsqueeze(-1))
        target_out = target_out[:, -1, :]  # (batch_size, hidden_size)
        neighbor_out, _ = self.rnn(neighbor_history.reshape(batch_size * num_neigh, -1).unsqueeze(-1))
        neighbor_out = neighbor_out[:, -1, :].reshape(batch_size, num_neigh, -1)  # (batch_size, num_neigh, hidden_size)

        # target_embed = self.embed_linear(self.loc_embed_layer(target_poi_index))  # (batch_size, loc_embed_size)
        # neighbor_embed = self.embed_linear(self.loc_embed_layer(neighbor_poi_index))  # (batch_size, num_neigh, loc_embed_size)
        target_embed = self.attn_linear(target_out)
        neighbor_embed = self.attn_linear(neighbor_out)

        attn_neighbor_out, _ = attention(target_embed.unsqueeze(1), neighbor_embed, neighbor_embed, dropout=self.dropout)
        attn_neighbor_out = attn_neighbor_out.sum(1)  # (batch_size, hidden_size)

        out = self.out_linear(torch.cat([target_out, attn_neighbor_out], -1)).squeeze(-1)
        # out = self.out_linear(target_out).squeeze(-1)
        return out


def train_flow_predictor(pre_model, dataset, batch_size, num_epoch,
                         lr, test_set_choice, early_stopping_round,
                         device, pre_len, **kwargs):
    assert test_set_choice in [1, 2]

    pre_model = pre_model.to(device)
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    train_set = dataset.gen_flow_history(set_choice=0, **kwargs)
    eval_set = dataset.gen_flow_history(set_choice=1, **kwargs)
    test_set = eval_set if test_set_choice == 1 else dataset.gen_flow_history(set_choice=2, **kwargs)

    def _pre_batch(input_model, batch):
        poi_index, recent_hour, remote_hour, recent_seq, remote_seq = zip(*batch)
        poi_index, recent_hour, remote_hour = (torch.tensor(item).long().to(device)
                                               for item in (poi_index, recent_hour, remote_hour))
        recent_seq, remote_seq = (torch.tensor(item).float().to(device)
                                  for item in (recent_seq, remote_seq))

        if pre_len > 1:
            recent_history, target_seq = recent_seq[:, :-pre_len], recent_seq[:, -pre_len:]

            pre = input_model(recent_history=dataset.normalize(recent_history),
                              target_seq=dataset.normalize(target_seq),
                              remote_history=dataset.normalize(remote_seq),
                              loc_index=poi_index, recent_hour=recent_hour, remote_hour=remote_hour)
            return pre.reshape(-1), target_seq.reshape(-1)
        else:
            recent_history, label = recent_seq[:, :-1], recent_seq[:, -1]
            recent_history, remote_seq = (dataset.normalize(item) for item in (recent_history, remote_seq))
            pre = input_model(recent_history=recent_history, remote_history=remote_seq,
                              loc_index=poi_index, pre_hour_index=recent_hour)
            return pre, label

    def _test_epoch(input_model, input_set):
        input_model.eval()
        pres, labels = [], []
        for batch in next_batch(input_set, batch_size=256):
            pre, label = _pre_batch(input_model, batch)
            pres.append(pre.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
        pres, labels = (np.concatenate(item) for item in (pres, labels))
        return pres, labels

    min_metric = 1e8
    worse_round = 0
    for epoch in range(num_epoch):
        for batch in next_batch(shuffle(train_set), batch_size):
            pre_model.train()
            pre, label = _pre_batch(pre_model, batch)
            optimizer.zero_grad()
            loss = loss_func(pre, label)
            loss.backward()
            optimizer.step()

        pres, labels = _test_epoch(pre_model, eval_set)
        metric = mean_absolute_error(labels, pres)
        nni.report_intermediate_result(metric)
        if early_stopping_round > 0 and min_metric > metric:
            min_metric = metric
            worse_round = 0
            torch.save(pre_model.state_dict(), FlowParam.local_model_path)
        else:
            worse_round += 1

        if 0 < early_stopping_round <= worse_round:
            print('Early stopping, best Epoch %d' % (epoch - worse_round))
            break

    if early_stopping_round > 0:
        pre_model.load_state_dict(torch.load(FlowParam.local_model_path))
    pres, labels = _test_epoch(pre_model, test_set)

    # test_poi_indices, _, _, _ = zip(*test_set)
    # with open(os.path.join('data', 'cache', 'flow_result.pkl'), 'wb') as fp:
    #     pickle.dump({'pre': pres, 'label': labels, 'poi': np.array(test_poi_indices)}, fp)

    score_series = cal_regression_metric(pres, labels)
    print(score_series)
    nni.report_final_result(score_series.loc['mae'])
    return score_series


def train_neigh_flow_predictor(pre_model, embed_layer, dataset, batch_size, num_epoch,
                               lr, test_set_choice, early_stopping_round,
                               device, nearest_k, history_len):
    assert test_set_choice in [1, 2]

    pre_model = pre_model.to(device)
    embed_layer = embed_layer.to(device)
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    train_set = dataset.gen_flow_array(set_choice=0)
    eval_set = dataset.gen_flow_array(set_choice=1)
    test_set = eval_set if test_set_choice == 1 else dataset.gen_flow_array(set_choice=2)

    def _fetch_k_nearest(poi_indices):
        num_loc = poi_indices.size(0)
        poi_embeds = embed_layer(poi_indices)  # (num_loc, embed_size)

        neighbor_orders, neighbor_indices = [], []
        for i, poi_index in enumerate(poi_indices):
            poi_embed = poi_embeds[i]  # (embed_size)
            distance = torch.sqrt(torch.square(poi_embed.unsqueeze(0).repeat(num_loc, 1) - poi_embeds).sum(-1))  # (num_loc)
            _, nearest_indices = torch.topk(distance, nearest_k + 1, largest=False)
            nearest_indices = nearest_indices[1:]

            k_neighbors = poi_indices[nearest_indices]  # (k)
            neighbor_orders.append(nearest_indices)
            neighbor_indices.append(k_neighbors)
        return torch.stack(neighbor_orders), torch.stack(neighbor_indices)  # (num_loc, k)

    def _next_array_batch(input_set, shuffle_set=False):
        poi_indices, full_array = input_set
        poi_indices = torch.tensor(poi_indices).long().to(device)  # (num_loc)
        full_array = torch.tensor(full_array).float().to(device)  # (num_loc, full_len)
        neighbor_orders, neighbor_indices = _fetch_k_nearest(poi_indices)  # (num_loc, k)

        # This tensor can be regarded as all possible combination of poi indices and sequence starting indices.
        all_comb = [[poi, start_i] for poi in range(len(poi_indices))
                    for start_i in range(full_array.shape[-1] - history_len)]
        if shuffle_set: all_comb = shuffle(all_comb)
        all_comb = torch.tensor(all_comb).long().to(device)  # (num_comb, 2)

        data_length = len(all_comb)
        num_batches = math.ceil(data_length / batch_size)
        for batch_index in range(num_batches):
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, data_length)
            comb_batch = all_comb[start_index:end_index]  # (batch_size, 2)

            target_poi_orders = comb_batch[:, 0]
            target_poi_index = poi_indices[target_poi_orders]  # (batch_size)
            neighbor_order = neighbor_orders[target_poi_orders]  # (batch_size, k)
            neighbor_poi_index = neighbor_indices[target_poi_orders]  # (batch_size, k)
            target_seq = torch.stack([full_array[poi_order, seq_order:seq_order+history_len+1]
                                      for poi_order, seq_order in comb_batch])  # (batch_size, seq_len)
            neighbor_history = torch.stack([full_array[neighbor_order[i], seq_order:seq_order+history_len+1]
                                            for i, (poi_order, seq_order) in enumerate(comb_batch)])  # (batch_size, k, history_len)
            yield target_seq, neighbor_history, target_poi_index, neighbor_poi_index

    def _pre_batch(input_model, input_batch):
        target_seq, neighbor_history, target_poi_index, neighbor_poi_index = input_batch
        target_history, label = target_seq[:, :-1], target_seq[:, -1]
        target_history, neighbor_history = (dataset.normalize(item) for item in (target_history, neighbor_history))
        pre = input_model(target_history, neighbor_history, target_poi_index, neighbor_poi_index)
        return pre, label

    def _test_epoch(input_model, input_set):
        input_model.eval()
        pres, labels = [], []
        for batch in _next_array_batch(input_set, shuffle_set=False):
            pre, label = _pre_batch(input_model, batch)
            pres.append(pre.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
        pres, labels = (np.concatenate(item) for item in (pres, labels))
        return pres, labels

    min_metric = 1e8
    worse_round = 0
    for epoch in range(num_epoch):
        for batch in _next_array_batch(train_set, shuffle_set=True):
            pre_model.train()
            pre, label = _pre_batch(pre_model, batch)
            optimizer.zero_grad()
            loss = loss_func(pre, label)
            loss.backward()
            optimizer.step()

        pres, labels = _test_epoch(pre_model, eval_set)
        metric = mean_absolute_error(labels, pres)
        nni.report_intermediate_result(metric)
        if early_stopping_round > 0 and min_metric > metric:
            min_metric = metric
            worse_round = 0
            torch.save(pre_model.state_dict(), FlowParam.local_model_path)
        else:
            worse_round += 1

        if 0 < early_stopping_round <= worse_round:
            print('Early stopping, best Epoch %d' % (epoch - worse_round))
            break

    if early_stopping_round > 0:
        pre_model.load_state_dict(torch.load(FlowParam.local_model_path))
    pres, labels = _test_epoch(pre_model, test_set)

    # test_poi_indices, _ = zip(*test_set)
    # with open(os.path.join('data', 'cache', 'flow_result.pkl'), 'wb') as fp:
    #     pickle.dump({'pre': pres, 'label': labels, 'poi': np.array(test_poi_indices)}, fp)

    score_series = cal_regression_metric(pres, labels)
    print(score_series)
    nni.report_final_result(score_series.loc['mae'])
    return score_series


def train_ml_flow_predictor(pre_model, dataset, test_set_choice, early_stopping_round, **kwargs):
    train_set = dataset.gen_flow_session(set_choice=0, **kwargs)
    eval_set = dataset.gen_flow_session(set_choice=1, **kwargs)
    test_set = dataset.gen_flow_session(set_choice=test_set_choice, **kwargs)

    def _split_set(input_set):
        _, seq = zip(*input_set)
        seq = np.array(seq)
        history_seq, label = seq[:, :-1], seq[:, -1]
        return history_seq, label

    train_history, train_label = _split_set(train_set)
    eval_history, eval_label = _split_set(eval_set)
    test_history, test_label = _split_set(test_set)
    try:
        pre_model = pre_model.fit(train_history, train_label, eval_set=[(eval_history, eval_label)],
                                  early_stopping_rounds=early_stopping_round)
    except TypeError:
        pre_model = pre_model.fit(train_history, train_label)
    pre = pre_model.predict(test_history)

    score_series = cal_regression_metric(pre, test_label)
    print(score_series)
    nni.report_final_result(score_series.loc['mae'])
    return score_series
