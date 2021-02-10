import math
from itertools import zip_longest
import random

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import init
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error


def gen_index_map(series, offset=0):
    index_map = {origin: index + offset
                 for index, origin in enumerate(series.drop_duplicates())}
    return index_map


def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true > 0)
    y_true = y_true[non_zero_index]
    y_pred = y_pred[non_zero_index]

    mape = np.abs((y_true - y_pred) / y_true)
    mape[np.isinf(mape)] = 0
    return np.mean(mape) * 100


def create_src_trg(full_seq, pre_len, fill_value):
    src_seq, trg_seq = zip(*[[s[:-pre_len], s[-pre_len:]] for s in full_seq])
    src_seq = np.transpose(np.array(list(zip_longest(*src_seq, fillvalue=fill_value))))
    return src_seq, np.array(trg_seq)


def create_src(full_seq, fill_value):
    return np.transpose(np.array(list(zip_longest(*full_seq, fillvalue=fill_value))))


def top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    successes = 0
    for i, truth in enumerate(truths):
        if truth in best_n[i, :]:
            successes += 1
    return float(successes) / truths.shape[0]


def cal_classify_metric(pre_dists, pres, labels, top_n_list):
    precision, recall, f1 = precision_score(labels, pres, average='macro'), \
                            recall_score(labels, pres, average='macro'), \
                            f1_score(labels, pres, average='macro')
    if pre_dists is not None:
        top_n_acc = [top_n_accuracy(labels, pre_dists, n) for n in top_n_list]
    else:
        top_n_acc = [accuracy_score(labels, pres)] + [-1.0 for _ in range(len(top_n_list)-1)]
    score_series = pd.Series([precision, recall, f1] + top_n_acc,
                             index=['macro-pre', 'macro-rec', 'macro-f1'] + ['acc@{}'.format(n) for n in top_n_list])
    return score_series


def cal_regression_metric(pres, labels):
    mae, mse, mape = mean_absolute_error(labels, pres), mean_squared_error(labels, pres), \
                     mean_absolute_percentage_error(labels, pres)
    rmse = math.sqrt(mse)
    score_series = pd.Series([mae, mape, rmse], index=['mae', 'mape', 'rmse'])
    return score_series


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.Embedding):
        embed_size = m.weight.size(-1)
        if embed_size > 0:
            init_range = 0.5/m.weight.size(-1)
            init.uniform_(m.weight.data, -init_range, init_range)
