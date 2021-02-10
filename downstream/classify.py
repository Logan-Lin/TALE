import os
from collections import Counter

import nni
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from torch import nn

from utils import weight_init, next_batch, cal_classify_metric


class ClassifyParam:
    local_model_path = os.path.join('data', 'cache',
                                    'classify_{}_{}.model'.format(nni.get_experiment_id(), nni.get_trial_id()))
    top_n_list = list(range(1, 11)) + [15, 20]


def split_categories(categories, test_set_choice):
    categories = shuffle(categories, random_state=0)
    length = categories.shape[0]
    train_categories = categories[:int(length * 0.6)]
    eval_categories = categories[int(length * 0.6):int(length * 0.8)]
    test_categories = categories[int(length * 0.8):] if test_set_choice == 2 else eval_categories
    return train_categories, eval_categories, test_categories


def dot_product_knn_classify(embed_layer, categories, test_set_choice, k, device):
    embed_layer = embed_layer.to(device)
    _, _, test_categories = split_categories(categories, test_set_choice)
    categories = torch.from_numpy(categories).long().to(device)

    test_poi_indices = torch.from_numpy(test_categories[:, 0]).long().to(device)
    test_labels = test_categories[:, 1]

    # Calculate a dot product value between each pairs of location embedding vectors.
    poi_indices, category_labels = categories[:, 0], categories[:, 1]
    poi_embeds = embed_layer(poi_indices).detach()  # (num_loc, embed_size)
    num_loc = poi_embeds.size(0)
    # Using Euclidean distance to measure distances between embedding vectors.
    distances = []
    for poi_index in range(poi_embeds.size(0)):
        poi_embed = poi_embeds[poi_index]  # (embed_size)
        distance = torch.sqrt(torch.square(poi_embed.unsqueeze(0).repeat(num_loc, 1) - poi_embeds).sum(-1))  # (num_loc)
        distances.append(distance)
    distances = torch.stack(distances)

    test_dot_products = distances[test_poi_indices]  # (test_set_size, num_loc)
    test_dot_products = test_dot_products.index_fill_(-1, test_poi_indices, 1e8)
    # Calculate top-k nearest locations to the testing targets.
    test_nearest = torch.argsort(test_dot_products, dim=-1)[:, :k]  # (test_set_size, k)
    # Fetch candidate labels.
    test_candidate_labels = category_labels[test_nearest].cpu().numpy()  # (test_set_size, k)
    pres = []
    for candidate_label_row in test_candidate_labels:
        np.random.shuffle(candidate_label_row)
        # Give out the most common candidate pair, (category_index, count)
        most_common = Counter(candidate_label_row.tolist()).most_common(1)
        pres.append(most_common[0][0])
    pres = np.array(pres)  # (test_set_size)
    score_series = cal_classify_metric(None, pres, test_labels, ClassifyParam.top_n_list)
    print(score_series)
    nni.report_final_result(score_series.loc['acc@1'])
    return score_series


class FCClassifier(nn.Module):
    def __init__(self, embed_layer, input_size, output_size, hidden_size):
        super().__init__()

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)

        self.input_linear = nn.Linear(input_size, hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, output_size)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

        self.apply(weight_init)

    def forward(self, x):
        """
        :param x: input batch of location tokens, shape (batch_size)
        :return: prediction of the corresponding location categories, shape (batch_size, output_size)
        """
        h = self.dropout(self.embed_layer(x))  # (batch_size, input_size)
        h = self.dropout(self.act(self.input_linear(h)))
        h = self.dropout(self.act(self.hidden_linear(h)))
        out = self.output_linear(h)
        return out


def train_classifier(pre_model, categories, batch_size, num_epoch, lr,
                     test_set_choice, early_stopping_round, device):
    """
    :param categories: categories of locations, shape (N, 2). Each row consists of POI index and category index.
    """
    assert test_set_choice in [1, 2]

    pre_model = pre_model.to(device)
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    train_categories, eval_categories, test_categories = split_categories(categories, test_set_choice)

    def _pre_batch(input_model, batch):
        batch = torch.from_numpy(batch).long().to(device)
        poi_index, label = batch[:, 0], batch[:, 1]
        pre = input_model(poi_index)
        return pre, label

    def _test_epoch(input_model, input_set):
        input_model.eval()
        pre_dists, labels = [], []
        for batch in next_batch(input_set, batch_size=batch_size):
            pre_dist, label = _pre_batch(input_model, batch)
            pre_dists.append(pre_dist.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
        pre_dists, labels = (np.concatenate(item) for item in (pre_dists, labels))
        pres = pre_dists.argmax(-1)

        return pre_dists, pres, labels

    max_metric = 0.0
    worse_round = 0
    for epoch in range(num_epoch):
        for batch in next_batch(shuffle(train_categories), batch_size=batch_size):
            pre_model.train()
            pre, label = _pre_batch(pre_model, batch)
            loss = loss_func(pre, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pre_dists, pres, labels = _test_epoch(pre_model, eval_categories)
        metric = accuracy_score(labels, pres)
        nni.report_intermediate_result(metric)
        if early_stopping_round > 0 and max_metric < metric:
            max_metric = metric
            worse_round = 0
            torch.save(pre_model.state_dict(), ClassifyParam.local_model_path)
        else:
            worse_round += 1

        if 0 < early_stopping_round <= worse_round:
            print('Early Stopping, best Epoch %d.' % (epoch - worse_round))
            break

    # Load the model with the best test metric value.
    if early_stopping_round > 0 and max_metric > 0:
        pre_model.load_state_dict(torch.load(ClassifyParam.local_model_path))
    pre_dists, pres, labels = _test_epoch(pre_model, test_categories)
    score_series = cal_classify_metric(pre_dists, pres, labels, ClassifyParam.top_n_list)
    print(score_series)
    nni.report_final_result(score_series.loc['acc@1'])
    os.remove(ClassifyParam.local_model_path)
    return score_series
