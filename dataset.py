import math
from collections import Counter

import pandas as pd
import numpy as np

from utils import gen_index_map


set_splits = [[0.0, 0.6], [0.6, 0.8], [0.8, 1.0]]
set_splits_small = [[0.0, 0.8], [0.8, 0.9], [0.9, 1.0]]


class TrajectoryDataset:
    """
    Base class for trajectory datasets.
    """
    def __init__(self, data_df, poi_df):
        """
        :param data_df: contains check-in records, with three columns: user_id, poi_id and datetime.
        :param poi_df: contains POI information, with poi_id as its index, and at least two columns: lat and lng.
        """
        self.data_df = data_df.sort_values(['user_id', 'datetime']).reset_index(drop=True)
        self.poi_df = pd.DataFrame(poi_df, copy=True)

        self.user_index_map = gen_index_map(self.data_df['user_id'])
        self.poi_index_map = gen_index_map(self.poi_df.index)

        self.data_df['user_index'] = self.data_df['user_id'].map(self.user_index_map)
        self.data_df['poi_index'] = self.data_df['poi_id'].map(self.poi_index_map)
        self.data_df['timestamp'] = self.data_df['datetime'].apply(lambda x: x.timestamp())
        self.data_df['hour'] = self.data_df['timestamp'] % (24 * 60 * 60) / 60 / 60
        self.data_df['week'] = self.data_df['datetime'].dt.weekday

        self.data_df = self.data_df.join(self.poi_df, on='poi_id')
        self.poi_df['poi_index'] = self.poi_df.index.map(self.poi_index_map)

        # Split all check-in records based on their day indices.
        day_col = self.data_df['datetime'].apply(lambda x: math.floor(x.timestamp() / 24 / 60 / 60)).astype(int)
        day_range = [day_col.min(), day_col.max() - day_col.min() + 1]
        self.split_data = [self.data_df[day_col.isin(range(day_range[0] + int(set_splits[i][0] * day_range[1]),
                                                           day_range[0] + int(set_splits[i][1] * day_range[1])))] for i in range(3)] + \
                          [self.data_df]

    def gen_length_sequence(self, set_choice, seq_length):
        """
        Generate sequences of static length.

        :param set_choice: index of the chosen set, 0 for training set, 1 for evaluation set,
            2 for testing set and 3 for full set.
        :param seq_length: length of sequences.
        :return:
        """
        return_seq = []
        chosen_set = self.split_data[set_choice]

        for user_index, group in chosen_set.groupby('user_index'):
            for i in range(group.shape[0] - seq_length + 1):
                sub_group = group.iloc[i:i+seq_length]
                return_seq.append([user_index,
                                   sub_group['poi_index'].to_list(),
                                   sub_group['timestamp'].to_list(),
                                   sub_group['week'].to_list()])
        return return_seq

    def gen_full_sequence(self, set_choice):
        """
        Give out the full sequence of all users.

        :param set_choice: index of the chosen set.
        :return:
        """
        return_seq = []
        chosen_set = self.split_data[set_choice]

        for user_index, group in chosen_set.groupby('user_index'):
            return_seq.append([user_index,
                               group['poi_index'].to_list(),
                               group['timestamp'].to_list(),
                               group['week'].to_list()])
        return return_seq

    def gen_static_session(self, set_choice, session_len, num_sessions):
        """
        Generate sessions of sequences, each session with static time delta.

        :param set_choice: index of the chosen set.
        :param session_len: length of one session, in hour.
        :param num_sessions: total number of sessions in history sequence.
        """
        return_seq = []
        chosen_set = self.split_data[set_choice]

        for user_index, group in chosen_set.groupby('user_index'):
            group = group.reset_index(drop=True)
            start_i = group[(group.iloc[0]['datetime'] + pd.Timedelta(hours=session_len * num_sessions)) < group['datetime']]
            if start_i.shape[0] > 0:
                start_i = start_i.index[0]
                for i in range(start_i, group.shape[0]):
                    session_diff = np.floor((group.iloc[i-1]['datetime'] - group['datetime']).apply(lambda x: x.total_seconds() / 60 / 60 / session_len))
                    sessions = [group[session_diff == h] for h in range(num_sessions - 1, -1, -1)]  # (num_sessions)
                    return_seq.append([user_index,
                                       [session['poi_index'].to_list() for session in sessions],
                                       [session['timestamp'].to_list() for session in sessions],
                                       group.iloc[i]['poi_index'],
                                       len(sessions)])
        return return_seq

    def gen_split_session(self, set_choice, split_threshold, history_length):
        """
        Splitting the full sequences of users into sessions based on the split threshold.

        :param set_choice: index of the chosen set.
        :param split_threshold: time threshold to split long sequences into sessions, in hour.
        :param history_length: span of history to fetch, in hour.
        """
        return_seq = []
        chosen_set = self.split_data[set_choice]

        for user_index, group in chosen_set.groupby('user_index'):
            group = group.reset_index(drop=True)

            # Split full trajectories into sessions.
            # Any pair of consecutive records with time delta higher than split_threshold will be regard as splitting borders.
            dt_series = group['datetime']
            time_diff = (dt_series - dt_series.shift(1)).apply(lambda x: x.total_seconds()).fillna(0)
            split_indices = [0] + dt_series[time_diff > (60 * 60 * split_threshold)].index.tolist() + [dt_series.shape[0]]
            split_indices = np.array([split_indices[:-1], split_indices[1:]]).transpose()

            session_base_timestamp = []
            session_start_index = []
            sessions = []
            for i, split_index in enumerate(split_indices):
                session = group.iloc[split_index[0]:split_index[1]]
                this_base_timestamp = session.iloc[-1]['timestamp']
                session_base_timestamp.append(this_base_timestamp)
                sessions.append([user_index,
                                 session['poi_index'].to_list(),
                                 session['hour'].to_list()])
                this_start_index = np.where(np.array(session_base_timestamp) >= (this_base_timestamp - history_length * 60 * 60))[0][0]
                session_start_index.append(this_start_index)

            for i, session in enumerate(sessions):
                for j in range(1, len(session[1])):
                    return_seq.append([user_index,
                                       [sessions[k][1] for k in range(session_start_index[i], i)] + [session[1][:j]],
                                       [sessions[k][2] for k in range(session_start_index[i], i)] + [session[2][:j]],
                                       session[1][j],
                                       i - session_start_index[i] + 1])
        return return_seq

    def gen_coor_df(self):
        """
        Generate coordinate DataFrame.

        :return:
        """
        coor_df = self.poi_df.set_index('poi_index')
        return coor_df[['lat', 'lng']]


class CheckinDataset(TrajectoryDataset):
    """
    Dataset supporter for Check-in type data.
    """
    def __init__(self, data_df, poi_df):
        """
        :param data_df: contains check-in records, with three columns: user_id, poi_id and datetime.
        :param poi_df: contains POI information, with poi_id as its index, and three columns: lat, lng and category_name.
        """
        super().__init__(data_df, poi_df)

        self.category_index_map = gen_index_map(self.poi_df['category_name'])
        self.poi_df['category_index'] = self.poi_df['category_name'].map(self.category_index_map)

    def gen_categories(self, min_threshold=0):
        """
        Generate a category matrix of locations.

        :param min_threshold: a threshold to filter out class labels with too little examples.
        :return: category matrix with shape (num_loc, 2), each row consists of POI index and category index.
        """
        category_df = self.poi_df[['poi_index', 'category_index']].copy()
        category_counter = Counter(category_df['category_index'])
        category_counter = pd.Series(category_counter.values(), index=category_counter.keys())
        valid_category = category_counter[category_counter >= min_threshold].index
        category_df = category_df[category_df['category_index'].isin(valid_category)]
        return category_df.sort_values('poi_index').to_numpy()


class SignalDataset(TrajectoryDataset):
    """
    Dataset supporter for mobile signaling type data.
    """
    def __init__(self, data_df, poi_df, flow_df):
        super().__init__(data_df, poi_df)

        self.flow_df = pd.DataFrame(flow_df, copy=True)
        self.flow_df['poi_index'] = self.flow_df['poi_id'].map(self.poi_index_map)
        _non_zero_flow = self.flow_df[self.flow_df['flow'] > 0]['flow']
        self.mean_v, self.std_v = _non_zero_flow.mean(), _non_zero_flow.std()
        # self.flow_df.loc[:, 'flow'] = (self.flow_df['flow'] - self.mean_v) / self.std_v
        self.flow_df['index'] = self.flow_df['day'] * 24 + self.flow_df['hour']

        day_col = self.flow_df['day'].astype(int)
        day_range = [day_col.min(), day_col.max() - day_col.min() + 1]
        self.split_day = [[day_range[0] + int(set_splits[i][0] * day_range[1]),
                           day_range[0] + int(set_splits[i][1] * day_range[1])] for i in range(3)] + \
                         [[day_range[0], day_range[0] + day_range[1]]]
        self.split_flow = [self.flow_df[day_col.isin(range(self.split_day[i][0], self.split_day[i][1]))] for i in range(4)]

    def gen_flow_session(self, set_choice, seq_len):
        """
        Generate flow sequences of static length.

        :param set_choice: index of chosen set, 0 for training set, 1 for evaluation set, 2 for testing set and 3 for full set.
        :param seq_len: length of sequences.
        :return:
        """
        return_seq = []
        chosen_set = self.split_flow[set_choice]

        for poi_index, group in chosen_set.groupby('poi_index'):
            for i in range(group.shape[0] - seq_len + 1):
                flow_seq = group['flow'].iloc[i:i+seq_len]
                if flow_seq.sum() > 0:
                    return_seq.append([poi_index, flow_seq.to_list()])
        return return_seq

    def gen_flow_array(self, set_choice):
        """
        Generate flow array containing sequential flow value of all locations
        :param set_choice: index of chosen set, 0 for training set, 1 for evaluation set, 2 for testing set and 3 for full set.
        :return:
        """
        chosen_set = self.split_flow[set_choice]
        all_poi_indices = chosen_set['poi_index'].drop_duplicates().tolist()
        flow_array = chosen_set['flow'].to_numpy().reshape(len(all_poi_indices), -1)  # (num_loc, total_len)
        return all_poi_indices, flow_array

    def gen_flow_history(self, set_choice, recent_seq_len, history_seq_len, history_num_day):
        """
        Generate flow sequences which combines recent history with remote ones.

        :param set_choice: index of chosen set, 0 for training set, 1 for evaluation set, 2 for testing set.
        :param recent_seq_len: length of recent historical sequence.
        :param history_seq_len: length of remote historical sequence.
        :return:
        """
        return_seq = []
        chosen_set = self.split_flow[set_choice]
        history_shift = int((history_seq_len - 1) / 2)

        for poi_index, group in self.flow_df.groupby('poi_index'):
            group = pd.DataFrame(group, copy=True).set_index('index')
            for index in range(max(chosen_set['index'].min(), self.flow_df['index'].min() + 24 * history_num_day + history_shift),
                               chosen_set['index'].max() - recent_seq_len - history_shift + 1):
                recent_seq = group.loc[index:index + recent_seq_len - 1]
                if recent_seq['flow'].sum() > 0:
                    remote_seq = group.loc[index - 24 * history_num_day - history_shift:index - 24 * history_num_day + history_shift]
                    return_seq.append([poi_index, recent_seq['hour'].to_list(), remote_seq['hour'].to_list(),
                                       recent_seq['flow'].to_list(), remote_seq['flow'].to_list()])
        return return_seq

    def normalize(self, x):
        return (x - self.mean_v) / self.std_v

    def denormalize(self, x):
        return x * self.std_v + self.mean_v
