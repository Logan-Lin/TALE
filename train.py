from tale import *
from dataset import *


dataset_name = 'fs_nyc'
device = 'cuda:0'
train_prop = 0.6
context_length = 2
total_batches = 200000

with open(os.path.join('data', 'dataset', '{}.pkl'.format(dataset_name)), 'rb') as fp:
    dataset = pickle.load(fp)
tale_model = TALE(dataset.std_df, slice_len=4, embed_size=64).to(device)
optimizer = torch.optim.Adam(tale_model.parameters(), lr=1e-4)

window_size = 2 * context_length + 1
train_set = []
for user_id, group in tqdm.tqdm(tale_model.data.groupby('userIndex'),
                                total=tale_model.data['userIndex'].drop_duplicates().shape[0],
                                desc='Sliding window'):
    train_length = math.ceil(group.shape[0] * train_prop)
    train_group = group.iloc[:train_length]

    for i in range(train_group.shape[0] - window_size):
        train_set.append(train_group[['poiIndex', 'sliceIndex']].to_numpy().tolist()[i:i+window_size])
train_set = np.array(train_set)  # (N, window_size, 2)

trained_batches = 0
loss_val = 0.
with tqdm.tqdm(range(total_batches), desc='Avg loss: %.9f' % 1.) as bar:
    for batch in next_batch(shuffle(train_set), batch_size=64):
        context = torch.from_numpy(np.concatenate([batch[:, :context_length, 0],
                                                    batch[:, context_length+1:, 0]],
                                                    axis=1)).long().to(device)  # (batch_size, 2 * context_len)
        target = batch[:, context_length]  # (batch_size, 2)
        route, lr = tale_model.fetch_routes(target)
        route, lr, target = (torch.from_numpy(item).long().to(device) for item in (route, lr, target))
        loss = tale_model(context, route, lr, target[:, 1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trained_batches += 1
        bar.update(1)

        loss_val += loss.cpu().detach().numpy().tolist()

        num_disp_batch = 500
        if trained_batches % num_disp_batch == 0:
            bar.set_description('Avg loss: %.9f' % (loss_val / num_disp_batch))
            loss_val = 0.

        if trained_batches >= total_batches:
            break

tale_embed = tale_model.input_embed.weight.data.cpu().detach().numpy()
