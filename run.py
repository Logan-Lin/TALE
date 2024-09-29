import gc
from argparse import ArgumentParser

from sklearn.svm import SVR
from xgboost import XGBRegressor

from dataset import *
from embed.static import *
from embed.poi2vec import *
from embed.teaser import *
from embed.tale import *
from downstream.classify import *
from downstream.next_loc import *
from downstream.flow import *


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='nyc')
    parser.add_argument('--embed', type=str, default='fc')
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--embed_batch_size', type=int, default=64)
    parser.add_argument('--embed_epoch', type=int, default=10)
    parser.add_argument('--embed_set_choice', type=int, default=0)
    parser.add_argument('--embed_lr', type=float, default=1e-3)
    parser.add_argument('--task', type=str, default='classify')
    parser.add_argument('--pre_model', type=str, default='fc')
    parser.add_argument('--task_batch_size', type=int, default=64)
    parser.add_argument('--task_epoch', type=int, default=999999)
    parser.add_argument('--task_lr', type=float, default=1e-3)
    parser.add_argument('--task_hidden_size', type=int, default=256)
    parser.add_argument('--task_test_set', type=int, default=2)
    parser.add_argument('--early_stopping_round', type=int, default=5)
    args = parser.parse_args()
    
    device = args.device
    dataset_name = args.dataset
    
    """Load dataset"""
    assert dataset_name in ['nyc', 'tky', 'ist', 'jkt', 'pek']
    data_df = pd.read_hdf(os.path.join('data', '{}.h5'.format(dataset_name)), key='data')
    poi_df = pd.read_hdf(os.path.join('data', '{}.h5'.format(dataset_name)), key='poi')
    if dataset_name in ['nyc', 'tky', 'ist', 'jkt']:
        dataset = CheckinDataset(data_df, poi_df)
    else:
        flow_df = pd.read_hdf(os.path.join('data', '{}.h5'.format(dataset_name)), key='flow')
        dataset = SignalDataset(data_df, poi_df, flow_df)
    num_loc = len(dataset.poi_index_map)
    num_users = len(dataset.user_index_map)
    
    """Embed methods"""
    # General parameters of embedding methods.
    embed_name = args.embed
    embed_size = args.embed_size
    embed_batch_size = args.embed_batch_size
    embed_epoch = args.embed_epoch
    print('Dataset {}, Embed {} with '
          'embed_size={}, batch_size={}, epoch={}'.format(dataset_name, embed_name, embed_size, embed_batch_size, embed_epoch))
    
    assert embed_name in ['fc', 'random', 'ns', 'hs', 'teaser', 'p2v', 'tale']
    if embed_name == 'fc':
        # Fully-connected embedding layer trained along with downstream tasks.
        embed_layer = DownstreamEmbed(num_loc, embed_size)
    elif embed_name == 'random':
        # Randomly initialize embedding matrix.
        embed_mat = np.random.uniform(low=-0.5 / embed_size, high=0.5 / embed_size, size=(num_loc, embed_size))
        embed_layer = StaticEmbed(embed_mat)
        del embed_mat
    else:
        # Parameters only valid for pre-trained embedding methods.
        embed_set_choice = args.embed_set_choice
        window_size = 2
        indi_context = False
        ns_num_neg = 5  # number of negative sampling words
        init_lr = args.embed_lr
        embed_adam = True  # Set True to use Adam optimizer, or else SGD.
        print('Window size {}, {}initial lr={}, use {} optimizer'.format(window_size, 
            'individual context, ' if indi_context else '', init_lr, 'Adam' if embed_adam else 'SGD'))
    
        if embed_adam:
            optim_class = torch.optim.SparseAdam
        else:
            optim_class = torch.optim.SGD
    
        # Generate sequences for word2vec-based training.
        user_seq, poi_seq, timestamp_seq, week_seq = zip(*dataset.gen_full_sequence(set_choice=0))
    
        if embed_name == 'ns':
            # Set true to enable CBOW mode, else SkipGram mode.
            cbow_mode = False
    
            embed_dataset = NSData(poi_seq, indi_context=indi_context)
            embed_model = NS(num_vocab=num_loc, embed_dimension=embed_size, cbow=cbow_mode)
            embed_mat = train_ns(embed_model, embed_dataset, window_size=window_size, num_neg=ns_num_neg,
                                 batch_size=embed_batch_size, num_epoch=embed_epoch, init_lr=init_lr,
                                 optim_class=optim_class, device=device)
        elif embed_name == 'hs':
            embed_dataset = HSData(poi_seq, indi_context=indi_context)
            embed_model = HS(num_vocab=num_loc, embed_dimension=embed_size)
            embed_mat = train_hs(embed_model, embed_dataset, window_size=window_size, batch_size=embed_batch_size,
                                 num_epoch=embed_epoch, init_lr=init_lr, optim_class=optim_class, device=device)
        elif embed_name == 'teaser':
            teaser_num_ne = 2  # number of unvisited locations
            teaser_num_nn = 2  # number of non-neighbor locations
            teaser_distance_thres = 0.2  # distance threshold for non-neighbor locations
            teaser_week_embed_size = 16
            teaser_beta = 0.5  # loss value weight of unvisited and non-neighbor locations
    
            coor_df = dataset.gen_coor_df()
            embed_dataset = TeaserData(users=user_seq, sentences=poi_seq, weeks=week_seq,
                                       coor_df=coor_df, num_ne=teaser_num_ne, num_nn=teaser_num_nn,
                                       indi_context=indi_context, distance_threshold=teaser_distance_thres, sample=1e-3)
            embed_model = Teaser(num_vocab=num_loc, num_user=num_users, embed_dimension=embed_size,
                                 week_embed_dimension=teaser_week_embed_size, beta=teaser_beta)
            embed_mat = train_teaser(embed_model, embed_dataset, window_size=window_size, num_neg=ns_num_neg,
                                     batch_size=embed_batch_size, num_epoch=embed_epoch, init_lr=init_lr,
                                     optim_class=optim_class, device=device)
        elif embed_name == 'p2v':
            p2v_theta = 0.01  # length threshold of Area Node.
    
            coor_df = dataset.gen_coor_df()
            embed_dataset = P2VData(sentences=poi_seq, coor_df=coor_df, theta=p2v_theta, indi_context=indi_context)
            embed_model = POI2Vec(num_vocab=num_loc, num_inner_nodes=embed_dataset.total_offset, embed_dimension=embed_size)
            embed_mat = train_p2v(embed_model, embed_dataset, window_size=window_size, batch_size=embed_batch_size,
                                  num_epoch=embed_epoch, init_lr=init_lr, optim_class=optim_class, device=device)
        elif embed_name == 'tale':
            tale_slice = 240
            tale_influ = 0
    
            embed_dataset = TaleData(poi_seq, timestamp_seq,
                                     slice_len=tale_slice, influ_len=tale_influ, indi_context=indi_context)
            embed_model = Tale(num_vocab=num_loc, num_inner_nodes=embed_dataset.num_inner_nodes,
                               num_slots=max(embed_dataset.slice2offset.keys()) + 1,
                               embed_dimension=embed_size)
            embed_mat = train_tale(embed_model, embed_dataset, window_size=window_size, batch_size=embed_batch_size,
                                   num_epoch=embed_epoch, init_lr=init_lr, optim_class=optim_class, device=device)
    
        # Upload result embedding mat to HDFS.
        embed_layer = StaticEmbed(embed_mat)
    
        # Process garbage collection.
        del embed_dataset, embed_model, embed_mat
        del user_seq, poi_seq, timestamp_seq, week_seq
    gc.collect()
    
    
    """Downstream tasks"""
    task_name = args.task
    pre_model_name = args.pre_model
    task_batch_size = args.task_batch_size
    task_epoch = args.task_epoch
    task_lr = args.task_lr
    task_hidden_size = args.task_hidden_size
    task_test_set = args.task_test_set
    early_stopping_round = args.early_stopping_round
    
    print('Downstream task {} with model {}'.format(task_name, pre_model_name))
    
    if task_name == 'classify':
        category_threshold = 5
    
        assert pre_model_name in ['fc', 'knn'] and dataset_name in ['ist', 'jkt', 'nyc', 'tky']
        categories = dataset.gen_categories(min_threshold=category_threshold)
    
        if pre_model_name == 'fc':
            fc_classifier = FCClassifier(embed_layer=embed_layer, input_size=embed_size,
                                         output_size=len(dataset.category_index_map),
                                         hidden_size=task_hidden_size)
            score_series = train_classifier(fc_classifier, categories, batch_size=task_batch_size, num_epoch=task_epoch,
                                            lr=task_lr, test_set_choice=task_test_set, early_stopping_round=early_stopping_round, device=device)
        else:
            knn_num_nearest = 5
    
            score_series = dot_product_knn_classify(embed_layer, categories, test_set_choice=task_test_set,
                                                    k=knn_num_nearest, device=device)
    
    elif task_name == 'next_loc':
        session_split_threshold = 12
    
        assert pre_model_name in ['lstm', 'deepmove']
        if pre_model_name == 'deepmove':
            history_session_length = 7*24
            time_embed_size = 4
            user_embed_size = 4
    
            pre_model = DeepMove(loc_embed_layer=embed_layer, loc_embed_size=embed_size,
                                 time_embed_size=time_embed_size, user_embed_size=user_embed_size,
                                 num_loc=num_loc, num_time=24, num_users=num_users, hidden_size=task_hidden_size)
        else:
            history_session_length = 0
            pre_model = LstmLocPredictor(loc_embed_layer=embed_layer, loc_embed_size=embed_size,
                                         num_loc=num_loc, hidden_size=task_hidden_size)
        score_series = train_next_loc(pre_model, dataset, batch_size=task_batch_size, num_epoch=task_epoch,
                                      lr=task_lr, test_set_choice=task_test_set,
                                      early_stopping_round=early_stopping_round, device=device,
                                      split_threshold=session_split_threshold, history_length=history_session_length)
    
    elif task_name == 'flow':
        history_flow_length = 3
        history_flow_num_day = 0
        pre_flow_length = 3 if pre_model_name == 'seq2seq' else 1
        fuse_latent_size = 64
    
        assert pre_model_name in ['lstm', 'tcn', 'svr', 'xgb', 'period-rnn', 'fuse-rnn', 'neigh-rnn', 'seq2seq']
        if pre_model_name in ['lstm', 'tcn', 'period-rnn', 'fuse-rnn', 'seq2seq']:
            rnn_params = {'loc_embed_layer': embed_layer, 'loc_embed_size': embed_size,
                          'hidden_size': task_hidden_size, 'num_layers': 1}
            if pre_model_name == 'lstm':
                pre_model = LstmFlowPredictor(**rnn_params)
            elif pre_model_name == 'period-rnn':
                pre_model = RnnPeriodFlowPredictor(**rnn_params)
            elif pre_model_name == 'fuse-rnn':
                pre_model = RnnFuseFlowPredictor(**rnn_params, latent_size=fuse_latent_size)
            elif pre_model_name == 'seq2seq':
                pre_model = Seq2seqFlowPredictor(**rnn_params, latent_size=fuse_latent_size)
            else:
                pre_model = TcnFlowPredictor(input_seq_len=history_flow_length, loc_embed_layer=embed_layer,
                                             loc_embed_size=embed_size, hidden_size=task_hidden_size)
    
            dataset_params = {'recent_seq_len': history_flow_length + pre_flow_length,
                              'history_seq_len': history_flow_length if history_flow_num_day > 0 else 0,
                              'history_num_day': history_flow_num_day}
            score_series = train_flow_predictor(pre_model, dataset, batch_size=task_batch_size, num_epoch=task_epoch,
                                                lr=task_lr, test_set_choice=task_test_set,
                                                early_stopping_round=early_stopping_round, device=device,
                                                pre_len=pre_flow_length, **dataset_params)
        elif pre_model_name == 'neigh-rnn':
            flow_neighbor_k = 5
    
            pre_model = NeighFlowPredictor(embed_layer, embed_size, hidden_size=task_hidden_size, num_layers=2)
            score_series = train_neigh_flow_predictor(pre_model, embed_layer, dataset,
                                                      batch_size=task_batch_size, num_epoch=task_epoch,
                                                      lr=task_lr, test_set_choice=task_test_set,
                                                      early_stopping_round=early_stopping_round, device=device,
                                                      nearest_k=flow_neighbor_k, history_len=history_flow_length)
        else:
            if pre_model_name == 'svr':
                svr_kernel = 'linear'
    
                pre_model = SVR(kernel=svr_kernel, max_iter=task_epoch)
            else:
                xgb_n_estimators = 400
                xgb_max_depth = 5
                xgb_booster = 'gbtree'
    
                pre_model = XGBRegressor(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth,
                                         booster=xgb_booster, n_jobs=4)
            score_series = train_ml_flow_predictor(pre_model, dataset, test_set_choice=task_test_set,
                                                   early_stopping_round=early_stopping_round,
                                                   seq_len=history_flow_length+1)