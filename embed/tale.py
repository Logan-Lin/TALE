from embed.w2v import *


def gen_all_slots(minute, time_slice_length, influence_span_length):
    """
    :param minute: UTC timestamp in minute.
    :param time_slice_length: length of one slot in seconds.
    :param influence_span_length: length of influence span in seconds.
    """
    def _cal_slice(x):
        return int((x % (24 * 60)) / time_slice_length)

    if influence_span_length == 0:
        slices, props = [_cal_slice(minute)], [1.0]

    else:
        minute_floors = list({minute - influence_span_length / 2, minute + influence_span_length / 2} |
                             set(range((int((minute - influence_span_length/2) / time_slice_length) + 1) * time_slice_length,
                                       int(minute + influence_span_length / 2), time_slice_length)))
        minute_floors.sort()

        slices = [_cal_slice(time_minute) for time_minute in minute_floors[:-1]]
        props = [(minute_floors[index + 1] - minute_floors[index]) / influence_span_length
                 for index in range(len(minute_floors) - 1)]
    return slices, props


class TaleData(W2VData):
    def __init__(self, sentences, timestamps, slice_len, influ_len, indi_context):
        """
        :param sentences: sequences of location visiting records.
        :param timestamps: sequences of location visited timestamp (second), corresponding to sentences.
        :param slice_len: length of one time slice, in minute.
        :param influ_len: length of influence span, in minute.
        :param indi_context:
        """
        super().__init__(sentences, indi_context)
        self.sentences = sentences
        self.timestamps = timestamps

        self.visit2slice = {}  # Map a binary tuple (poi_index, timestamp) to slice indices.
        self.visit2prop = {}  # Map a binary tuple (poi_index, timestamp) to proportion corresponding to time slices.
        slice2poi = {}  # Record the included POIs for every time slice.

        for sentence, timestamp in zip(sentences, timestamps):
            for poi_index, visited_second in zip(sentence, timestamp):
                slice, prop = gen_all_slots(visited_second / 60, slice_len, influ_len)
                for s in slice:
                    slice2poi[s] = slice2poi.get(s, []) + [poi_index]
                int_minute = math.floor(visited_second % (24 * 60 * 60) / 60)
                self.visit2slice[(poi_index, int_minute)] = slice
                self.visit2prop[(poi_index, int_minute)] = prop

        self.slice2tree = {}  # Save all root node of temporal trees.
        self.slice2offset = {}  # Record node ID offset of temporal trees.
        _total_offset = 0
        for slice_index, poi_list in slice2poi.items():
            # Generate one Huffman Tree for every temporal slot.
            poi_freq = np.array(sorted(Counter(poi_list).items()))
            huffman_tree = HuffmanTree(poi_freq)
            self.slice2tree[slice_index] = huffman_tree
            self.slice2offset[slice_index] = _total_offset
            _total_offset += huffman_tree.num_inner_nodes
        self.num_inner_nodes = _total_offset + 1

    def gen_path_pairs(self, window_size):
        path_pairs = []
        for sentence, timestamp in zip(self.sentences, self.timestamps):
            for i in range(len(sentence) - (2 * window_size + 1) + 1):
                target = sentence[i+window_size]
                visit = (target, math.floor(timestamp[i+window_size] % (24 * 60 * 60) / 60))
                slice = self.visit2slice[visit]
                prop = self.visit2prop[visit]
                huffman_pos = [(np.array(self.slice2tree[s].id2pos[target]) + self.slice2offset[s]).tolist() for s in slice]
                huffman_neg = [(np.array(self.slice2tree[s].id2neg[target]) + self.slice2offset[s]).tolist() for s in slice]
                context = sentence[i:i+window_size] + sentence[i+window_size+1:i+2*window_size+1]
                if self.indi_context:
                    path_pairs += [[[c], huffman_pos, huffman_neg, slice, prop] for c in context]
                else:
                    path_pairs.append([context, huffman_pos, huffman_neg, slice, prop])
        return path_pairs


class Tale(nn.Module):
    def __init__(self, num_vocab, num_inner_nodes, num_slots, embed_dimension):
        super().__init__()
        self.num_vocab = num_vocab
        self.embed_dimension = embed_dimension

        # Input embedding.
        self.u_embeddings = nn.Embedding(num_vocab, embed_dimension, sparse=True)
        # Output embedding.
        self.w_embeddings = nn.Embedding(num_inner_nodes, embed_dimension, padding_idx=0, sparse=True)
        # Slot output embedding.
        self.slot_embeddings = nn.Embedding(num_slots, embed_dimension, sparse=True)

        init_range = 0.5 / self.embed_dimension
        self.u_embeddings.weight.data.uniform_(-init_range, init_range)
        self.w_embeddings.weight.data.uniform_(-0, 0)
        self.slot_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_w, neg_w, pos_s, prob):
        """
        :param pos_u: positive input tokens, shape (batch_size, window_size * 2)
        :param pos_w: positive output tokens, shape (batch_size, pos_num)
        :param neg_w: negative output tokens, shape (batch_size, neg_num)
        :param pos_s: positive slice tokens, shape (batch_size)
        :param prob: probability distribution, shape (batch_size)
        :return: loss value of this batch.
        """
        pos_u_embed = self.u_embeddings(pos_u)  # (batch_size, window_size * 2, embed_size)
        pos_u_embed = pos_u_embed.sum(1, keepdim=True)  # (batch_size, 1, embed_size)

        pos_w_mask = torch.where(pos_w == 0, torch.ones_like(pos_w), torch.zeros_like(pos_w)).bool()  # (batch_size, num_pos)
        pos_w_embed = self.w_embeddings(pos_w)  # (batch_size, num_pos, embed_size)
        score = torch.mul(pos_u_embed, pos_w_embed).sum(dim=-1)  # (batch_size, num_pos)
        score = F.logsigmoid(-1 * score)  # (batch_size, num_pos)
        score = score.masked_fill(pos_w_mask, torch.tensor(0.0).to(pos_u.device))

        neg_w_mask = torch.where(neg_w == 0, torch.ones_like(neg_w), torch.zeros_like(neg_w)).bool()
        neg_w_embed = self.w_embeddings(neg_w)
        neg_score = torch.mul(pos_u_embed, neg_w_embed).sum(dim=-1)  # (batch_size, num_neg)
        neg_score = F.logsigmoid(neg_score)
        neg_score = neg_score.masked_fill(neg_w_mask, torch.tensor(0.0).to(pos_u.device))

        s_embed = self.slot_embeddings(pos_s)  # (batch_size, embed_size)
        s_score = torch.mul(pos_u_embed, s_embed.unsqueeze(1)).sum(dim=-1).squeeze(-1)  # (batch_size)
        s_score = F.logsigmoid(-1 * s_score)

        return -1 * torch.mul((torch.sum(score, dim=-1) + torch.sum(neg_score, dim=-1) + s_score), prob).sum()


def train_tale(tale_model, dataset, window_size, batch_size, num_epoch, init_lr, optim_class, device):
    tale_model = tale_model.to(device)
    optimizer = optim_class(tale_model.parameters(), lr=init_lr)

    train_set = dataset.gen_path_pairs(window_size)
    trained_batches = 0
    batch_count = math.ceil(num_epoch * len(train_set) / batch_size)

    for epoch in range(num_epoch):
        loss_log = []
        for pair_batch in next_batch(shuffle(train_set), batch_size):
            flatten_batch = []
            for row in pair_batch:
                flatten_batch += [[row[0], p_w, n_w, p_s, prob] for p_w, n_w, p_s, prob in zip(*row[1:])]

            context, pos_w, neg_w, pos_s, prob = zip(*flatten_batch)
            context, pos_s = (torch.tensor(item).long().to(device) for item in [context, pos_s])  # (batch_size, window_size * 2), (batch_size)
            pos_w, neg_w = (torch.tensor(list(zip_longest(*item, fillvalue=0))).long().to(device).transpose(0, 1)
                            for item in [pos_w, neg_w])  # (batch_size, longest)
            prob = torch.tensor(prob).float().to(device)

            optimizer.zero_grad()
            loss = tale_model(context, pos_w, neg_w, pos_s=pos_s, prob=prob)
            loss.backward()
            optimizer.step()

            trained_batches += 1
            loss_log.append(loss.detach().cpu().numpy().tolist())

        if isinstance(optimizer, torch.optim.SGD):
            lr = init_lr * (1.0 - trained_batches / batch_count)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        print('Epoch %d avg loss: %.5f' % (epoch, np.mean(loss_log)))
    return tale_model.u_embeddings.weight.detach().cpu().numpy()
