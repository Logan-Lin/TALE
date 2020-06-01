import os
import pickle
import queue
from itertools import zip_longest

import numpy as np
import torch
import tqdm
from sklearn.utils import shuffle
from torch import nn

from tools import *


class HuffmanNode:
    count = 0

    # Node in Huffman tree.
    def __init__(self, freq, left=None, right=None, label=None, leaf=False):
        super().__init__()
        self.freq = freq
        self.left = left
        self.right = right
        self.label = label
        self.id = -1
        self.leaf = leaf

        # If the node is leaf node, then its ID is redundant.
        if not leaf:
            HuffmanNode.count += 1
            self.id = HuffmanNode.count

    def __eq__(self, other):
        return self.freq == other.freq

    def __lt__(self, other):
        return self.freq < other.freq

    def __repr__(self):
        if self.leaf:
            return 'Leaf node @ label={}, freq={}'.format(self.label, self.freq)
        else:
            return 'Inner node #{} @ freq={}'.format(self.id, self.freq)


class TimeNode:
    def __init__(self, slice_id, poi_df):
        """
        Initializer of time node class.

        :param slice_id: The time slice ID this time node represents.
        :param poi_df: DataFrame containing POI records belonging to this time slice.
        """
        self.slice_id = slice_id

        # Calculate POI frequency.
        # poi_freq is a list with pairs like (poi_id, poi_frequency)
        poi_freq_counter = Counter(poi_df['poiIndex'])
        poi_freq = list(poi_freq_counter.items())

        # Build huffman tree
        # Using a PriorityQueue to store the list of POI frequencies.
        # In this way, the items in the queue will automatically be sorted.
        build_queue = queue.PriorityQueue()
        for value in poi_freq:
            build_queue.put(HuffmanNode(value[1], label=value[0], leaf=True))
        # While there are still more than 1 items in the queue, pop out two top items.
        # That will be two items with lowest value in the queue.
        while build_queue.qsize() > 1:
            left, right = build_queue.get(), build_queue.get()
            # Build a new huffman node using two items as child nodes.
            build_queue.put(HuffmanNode(left.freq + right.freq, left=left, right=right))
        # When the for-loop ends, the only item in the queue should be the root node of this huffman tree.
        self.root_node = build_queue.get()

        # Build routes
        self.routes, self.lrs = {}, {}
        self.max_route_depth = 0
        self._build_routes(self.root_node, [], [])

    def _build_routes(self, current_node, route, lr):
        """
        Build all leaf nodes' route belong to this current_node.

        :param current_node: The root node to start.
        :param route: current route list.
        :param lr: Current left-right choice list.
        """
        if current_node.leaf:
            if len(route) > 0:
                self.routes[current_node.label] = route
                self.lrs[current_node.label] = lr
                self.max_route_depth = max(self.max_route_depth, len(route))
        else:
            self._build_routes(current_node.left, route=route+[current_node.id], lr=lr+[0])
            self._build_routes(current_node.right, route=route+[current_node.id], lr=lr+[1])


class TALE(nn.Module):
    def __init__(self, check_in_records, slice_len, embed_size):
        """
        :param check_in_records: pandas DataFrame containing all check-in records.
            Each row is one check-in record. At least two columns are needed:
            - poiIndex: index of locations;
            - time: time of this check-in, UTCTimestamp (seconds)
        :param slice_len: length of one time slice, in hour.
        :param embed_size: size of embedding vectors.
        """
        super().__init__()
        HuffmanNode.count = 0

        num_slices = math.ceil(24 / slice_len)
        check_in_records['sliceIndex'] = check_in_records['time'].apply(lambda x: int(x / 60 % (24 * 60) % 24 / slice_len))
        self.time_nodes = [None] * num_slices
        for slice_index, group in check_in_records.groupby('sliceIndex'):
            time_node = TimeNode(slice_index, group)
            self.time_nodes[slice_index] = time_node
        self.data = check_in_records

        num_locations = check_in_records['poiIndex'].drop_duplicates().shape[0]
        self.input_embed = nn.Embedding(num_locations, embed_size)
        self.inner_node_embed = nn.Embedding(HuffmanNode.count + 1, embed_size, padding_idx=0)
        self.slice_node_embed = nn.Parameter(torch.rand(num_slices, embed_size), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, context, route, lr, slice):
        """
        :param context: contextual locations, shape (batch_size, context_size)
        :param route: inner node IDs, shape (batch_size, route_len)
        :param lr: left-right choices for each inner node, shape (batch_size, route_len)
        :param mask: mask indicating weather certain inner node is masked. 0 for masked, shape (batch_size, route_len)
        :param slice: index of slices corresponding to the target, shape (batch_size)
        :return: loss value of this batch.
        """
        context_embed = self.input_embed(context)  # (batch_size, context_size, embed_size)
        route_embed = self.inner_node_embed(route)  # (batch_size, route_len, embed_size)

        context_embed = context_embed.sum(dim=1, keepdim=True).transpose(1, 2)  # (batch_size, embed_size, 1)
        slice_node_embed_batch = self.slice_node_embed.unsqueeze(0).repeat(context.size(0), 1, 1)
        slice_pre = self.softmax(torch.bmm(slice_node_embed_batch, context_embed).squeeze())  # (batch_size, num_slices)
        slice_pre = torch.tensor([slice_pre[i, slice_i] for i, slice_i in enumerate(slice)]).unsqueeze(-1).to(context.device)  # (batch_size, 1)

        dot_product = torch.bmm(route_embed, context_embed).squeeze()  # (batch_size, route_len)
        lr_pre = self.sigmoid(dot_product)  # (batch_size, route_len)
        # route_hidden can be seen as the possibility of inner nodes choosing each sides.
        route_hidden = torch.pow(torch.mul(lr_pre, 2), lr) - lr_pre  # (batch_size, route_len)
        mask = torch.where(route.byte(), torch.zeros_like(route), torch.ones_like(route)).bool()
        route_hidden = route_hidden.masked_fill(mask, 1)

        pr_route = slice_pre
        for route_level in range(route_hidden.size(1)):
            pr_route = pr_route * route_hidden[:, route_level].unsqueeze(-1)
        return 1. - pr_route.mean()

    def fetch_routes(self, target):
        """
        :param target: target locations with their time slice id, shape (batch_size, 2)
        :return:
        """
        route, lr = [], []
        for poi_index, slice_index in target:
            r, l = self.time_nodes[slice_index].routes.get(poi_index, [0]), self.time_nodes[slice_index].lrs.get(poi_index, [0])
            route.append(r)
            lr.append(l)
        route, lr = (np.transpose(np.array(list(zip_longest(*item, fillvalue=0)))) for item in (route, lr))
        return route, lr
