import pandas as pd
import numpy as np

from tools import *


class Dataset:
    """
    This is the base class of all dataset subclasses.
    All dataset will include a std_df DataFrame, which will have the following columns:
        poiIndex, userIndex, categoryIndex: indices starting from 0;
        time: utc timestamp in seconds;
        latitude, longitude: geographical coordinate;
        weekday: weekday from 0 to 6.
    """
    def __init__(self, df, venue_column, user_column, time_column,
                 min_seq_len, min_visited, category_column=None):
        df = dataframe_freq_filter(df, venue_column, min_visited)
        df = dataframe_freq_filter(df, user_column, min_seq_len)
        fetch_column = ['userIndex', 'poiIndex', 'time', 'latitude', 'longitude', 'weekday']

        df['poiIndex'] = df[venue_column].map(self.gen_index_map(df[venue_column]))
        df['userIndex'] = df[user_column].map(self.gen_index_map(df[user_column]))
        if category_column is not None:
            df['categoryIndex'] = df[category_column].map(self.gen_index_map(df[category_column]))
            fetch_column.append('categoryIndex')
        df['time'] = df[time_column].apply(lambda x: x.timestamp())
        df['weekday'] = df[time_column].apply(lambda x: x.weekday())
        df = df.sort_values(['userIndex', 'time']).reset_index(drop=True)

        self.std_df = df[fetch_column]
        self.max_len = 0

    def construct_span(self, span_len, start_prop, end_prop, temp_slot_len):
        """
        Construct arbitrary spans of contiguous sequences.

        :param span_len: length of one span in the result set.
        """
        train_sets = []
        for user_id, group in self.std_df[['userIndex', 'poiIndex', 'time', 'latitude', 'longitude', 'weekday']].groupby('userIndex'):
            group['temp_slot'] = group['time'] % (24 * 60 * 60) / (60 * 60 * temp_slot_len)
            sequence = list(map(list, group[['poiIndex', 'temp_slot', 'latitude', 'longitude', 'weekday', 'time']].to_numpy()))

            seq_len = len(sequence)
            self.max_len = max(self.max_len, seq_len)
            start_index, end_index = int(seq_len * start_prop), int(seq_len * end_prop)

            train_seq = sequence[start_index:end_index]
            train_sets += self._construct_set(train_seq, span_len)
        return train_sets if span_len == -1 else np.array(train_sets)

    def construct_user_span(self, span_len, start_prop, end_prop):
        """
        Construct user visiting sequences.

        :param span_len: length of one span in the result test.
        """
        train_sets = []
        for poi_index, group in self.std_df.sort_values(['poiIndex', 'time'])[['poiIndex', 'userIndex']].groupby('poiIndex'):
            sequence = list(group['userIndex'])
            seq_len = len(sequence)
            start_index, end_index = int(seq_len * start_prop), int(seq_len * end_prop)

            train_seq = sequence[start_index:end_index]
            train_sets += self._construct_set(train_seq, span_len, [poi_index])
        return train_sets if span_len == -1 else np.array(train_sets)

    @staticmethod
    def _construct_set(sequence, span_len, tail=None):
        sets = []
        tail = [] if tail is None else tail
        if span_len == -1:
            sets.append(sequence + tail)
        else:
            for i in range(len(sequence) - span_len):
                sets.append(sequence[i:i+span_len] + tail)
        return sets

    @staticmethod
    def gen_index_map(column, offset=0):
        index_map = {origin: index + offset
                     for index, origin in enumerate(column.drop_duplicates())}
        return index_map


class FourSquareData(Dataset):
    def __init__(self, data_path, min_seq_len=10, min_visited=10):
        super().__init__(df=pd.read_csv(data_path, parse_dates=[-1], dtype={'userId':np.int32}),
                         venue_column='venueId', user_column='userId', time_column='utcTimestamp',
                         min_seq_len=min_seq_len, min_visited=min_visited, category_column='venueCategoryId')
