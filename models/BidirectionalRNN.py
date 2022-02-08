import argparse
import json
import logging
import os
import sys
import io
import pickle
import boto3
import json

import numpy as np
import pandas as pd

# global vars used by _window_dataset
df = pd.DataFrame()
team_mapping = {}

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class BidirectionalRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, rnn_type='LSTM', dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = eval('nn.' + self.rnn_type)(
            input_size=self.input_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers,  
            dropout=self.dropout,
            batch_first=True,
            bidirectional=True
        )

        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        out = self.linear(out)
        return out

    
class NFLDataset(Dataset):

    def __init__(self, X, y, columns=[]):
        super(NFLDataset, self).__init__()
        assert len(X) == len(y), 'Length of X and y must be the same'
        self.X = X
        self.y = y
        self.columns = columns

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        window = self.X[idx]
        label = self.y[idx]
        return torch.as_tensor(window).float(), torch.as_tensor(label).float()

    def toPandas(self):
        return pd.DataFrame(self.X.reshape((-1, len(self.columns))), columns=self.columns)
    
    
def _window_dataset(team, opponent, date, label, season, window):
    global df, team_mapping
    
    team_games = df[(df['Team'] == team) & (df['Date'] < date)].tail(window)
    opponent_games = df[(df['Team'] == opponent) & (df['Date'] < date)].tail(window)
    previous_games = pd.concat([team_games, opponent_games])
    previous_games['SampleIndex'] = f'{team}_{opponent}_{date}'
    previous_games['Team'] = previous_games['Team'].apply(lambda team: team_mapping[team])
    previous_games['Opponent'] = previous_games['Opponent'].apply(lambda team: team_mapping[team])
    previous_games['PredictionSeason'] = season
    previous_games = previous_games.drop(columns=['Date'])
    
    # only return samples with full windows 
    if len(previous_games) < window*2:
        return None, None 
    else:
        return previous_games, label
    
    
def _build_datasets(window_size, is_distributed):
    global df, team_mapping
    
    # download the data
    logger.debug('[_build_datasets]: Downloading the data...')
    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket='sprtsiq', Key="data/data.csv")
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        logger.info(f"Successful S3 get_object response. Status - {status}")
        df = pd.read_csv(response.get("Body"))
    else:
        logger.error(f"[_build_datasets]: Unsuccessful S3 get_object response. Status - {status}")
        throw(Exception)
    logger.debug('[_build_datasets]: Download complete.')
    
    sorted_df = df.sort_values(['Date'], ascending=True)
    sorted_teams = sorted(df['Team'].unique()) # would be smart to include coordinates
    team_mapping.update({k:v for v,k in enumerate(sorted_teams)})
    
    # window the dataset
    logger.debug('[_build_datasets]: Windowing the dataset..')
    X, y = zip(*df.apply(
        lambda row: _window_dataset(
            row['Team'], 
            row['Opponent'], 
            row['Date'], 
            row['MoneyLineWinLoss'], 
            row['Season'], 
            window_size
        ), 
        axis=1
    ))
    logger.debug('[_build_datasets]: Windowing complete.')
    
    logger.debug('[_build_datasets]: Dropping empty dataframes and duplicates...')
    X = pd.Series(X).dropna().iloc[::2]
    y = pd.Series(y).dropna().iloc[::2]
    logger.debug('[_build_datasets]: Dropping complete.')
    
    # split the data
    logger.debug('[_build_datasets]: Splitting the data...')
    PredictionSeason_series = pd.Series([df['PredictionSeason'].iloc[-1] for df in X])
    split = len(PredictionSeason_series[PredictionSeason_series > 2017]) / len(PredictionSeason_series)
    logger.info(f'Train-test split: {round(100*(1-split),2)}/{round(100*split,2)}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)
    X_train = pd.concat(X_train.values.tolist()).drop(columns='SampleIndex')
    X_test = pd.concat(X_test.values.tolist()).drop(columns='SampleIndex')
    y_train = y_train.to_numpy().reshape(-1,1)
    y_test = y_test.to_numpy().reshape(-1,1)
    logger.debug('[_build_datasets]: Split complete.')
    
    # scale the data
    scalers = {}
    logger.debug('[_build_datasets]: Scaling the training and testing datasets...')
    for col in X_train.columns:
        train_scaler = MinMaxScaler()
        s_s = train_scaler.fit_transform(X_train[col].values.reshape(-1,1))
        s_s = np.reshape(s_s,len(s_s))
        scalers['train_scaler_' + col] = train_scaler
        X_train[col] = s_s

        test_scaler = MinMaxScaler()
        s_s = test_scaler.fit_transform(X_test[col].values.reshape(-1,1))
        s_s = np.reshape(s_s,len(s_s))
        scalers['test_scaler_' + col] = test_scaler
        X_test[col] = s_s
    logger.debug('[_build_datasets]: Scaling complete.')
    
    # build the pytorch datasets
    logger.debug('[_build_datasets]: Building PyTorch NFLDataset objects...')
    columns = X_train.columns
    X_train = X_train.values.reshape(-1, window_size*2, X_train.shape[-1])
    X_test = X_test.values.reshape(-1, window_size*2, X_test.shape[-1])
    train_dataset = NFLDataset(X_train, y_train, columns=columns)
    test_dataset = NFLDataset(X_test, y_test, columns=columns)
    
    if is_distributed:
        train_dataset = torch.utils.data.distributed.DistributedSampler(train_dataset)
    logger.debug('[_build_datasets]: Build complete.')
    
    return train_dataset, test_dataset


def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def train(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("[train]: Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("[train]: Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.backend, dist.get_world_size()
            )
            + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(), args.num_gpus)
        )

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train, test = _build_datasets(args.window_size, is_distributed)
    train_dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=args.test_batch_size, shuffle=True)

    logger.debug(
        "[train]: Processes {}/{} ({:.0f}%) of train data".format(
            len(train_dataloader.sampler),
            len(train_dataloader.dataset),
            100.0 * len(train_dataloader.sampler) / len(train_dataloader.dataset),
        )
    )

    logger.debug(
        "[train]: Processes {}/{} ({:.0f}%) of test data".format(
            len(test_dataloader.sampler),
            len(test_dataloader.dataset),
            100.0 * len(test_dataloader.sampler) / len(test_dataloader.dataset),
        )
    )

    model = BidirectionalRNN(
        args.input_size,
        args.hidden_state_size,
        args.output_size,
        args.number_of_layers,
        args.rnn_type,
        args.dropout
    ).to(device)
    
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # obtain loss function
    try:
        loss_fn = getattr(nn.functional, args.loss_fn)
    except AttributeError:
        logging.error('[train]: Invalid value for loss_fn, must be a function of torch.nn.functional')
        
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    for epoch in range(args.epochs):
        epoch_train_loss = 0
        epoch_test_loss = 0
        epoch_train_acc = 0
        epoch_test_acc = 0

        for samples, labels in train_dataloader:
            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            raw_preds = model(samples)
            train_loss = loss_fn(raw_preds, labels)
            epoch_train_loss += train_loss.item()
            
            bin_preds = torch.round(torch.sigmoid(raw_preds))
            labels_np, bin_preds_np = labels.cpu().detach().numpy(), bin_preds.cpu().detach().numpy()
            epoch_train_acc += balanced_accuracy_score(labels_np, bin_preds_np)

            train_loss.backward()
            
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
                
            optimizer.step()

        for samples, labels in test_dataloader:
            samples = samples.to(device)
            labels = labels.to(device)

            raw_preds = model(samples)
            test_loss = loss_fn(raw_preds, labels)
            epoch_test_loss += test_loss.item()
            
            bin_preds = torch.round(torch.sigmoid(raw_preds))
            labels_np, bin_preds_np = labels.cpu().detach().numpy(), bin_preds.cpu().detach().numpy()
            epoch_test_acc += balanced_accuracy_score(labels_np, bin_preds_np)

        logger.info("[{}] train loss: {} train acc: {} test loss: {} test acc: {}".format(
            epoch+1,
            epoch_train_loss/len(train_dataloader),
            epoch_train_acc/len(train_dataloader),
            epoch_test_loss/len(test_dataloader),
            epoch_test_acc/len(test_dataloader)
        ))
        
        history['train_loss'].append(epoch_train_loss/len(train_dataloader))
        history['train_acc'].append(epoch_train_acc/len(train_dataloader))
        history['test_loss'].append(epoch_test_loss/len(test_dataloader))
        history['test_acc'].append(epoch_test_acc/len(test_dataloader))
        
    best_loss_idx = np.argmin(history['test_loss'])
    logger.info('Best test_loss [test_loss: {}][epoch: {}]'. format(
        history['test_loss'][best_loss_idx],
        best_loss_idx+1
    ))
    
    best_acc_idx = np.argmax(history['test_acc'])
    logger.info('Best test_acc [test_acc: {}][epoch: {}]'. format(
        history['test_acc'][best_acc_idx],
        best_acc_idx+1
    ))
    
    save_model(model, args.model_dir)
    save_history(history, args.model_dir)


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(BidirectionalRNN())
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
    
    
def save_history(history, model_dir):
    logger.info("Saving the history.")
    path = os.path.join(model_dir, "history.json")
    with open(path, "w") as outfile:
        json.dump(history, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.0001, 
        metavar="LR", 
        help="learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--window-size", 
        type=int, 
        default=12, 
        metavar="WS", 
        help="window size (default: 12)"
    )
    parser.add_argument(
        "--loss-fn", 
        type=str, 
        default='binary_cross_entropy_with_logits', 
        metavar="L", 
        help="loss function (default: binary_cross_entropy_with_logits)"
    )
    
    # TODO: deprecate
    parser.add_argument(
        "--input-size", 
        type=int, 
        default=210, 
        metavar="I", 
    )
    
    parser.add_argument(
        "--output-size", 
        type=int, 
        default=1, 
        metavar="OS", 
        help="output size (default: 1)"
    )
    parser.add_argument(
        "--number-of-layers", 
        type=int, 
        default=2, 
        metavar="NL", 
        help="number of layers (default: 2)"
    )
    parser.add_argument(
        "--hidden-state-size", 
        type=int, 
        default=64, 
        metavar="H", 
        help="number of neurons per layer (default: 64)"
    )
    parser.add_argument(
        "--rnn-type", 
        type=str, 
        default='LSTM', 
        metavar="RNN", 
        help="rnn type (default: LSTM)"
    )
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=0.20, 
        metavar="D", 
        help="dropout (default: 0.20)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        metavar="S", 
        help="random seed (default: 42)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())