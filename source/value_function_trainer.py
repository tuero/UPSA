import os
import logging
import numpy as np
import time, random
from sampling import *
import argparse
from utils import Option, sen2mat

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_actions, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_layers = num_layers

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.fc = nn.Linear(hidden_dim * 2, num_actions)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim * 2, num_actions)
        self.act_fn = nn.ReLU()

    def forward(self, embeds):
        h0 = torch.zeros(self.num_layers * 2, embeds.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * 2, embeds.size(0), self.hidden_dim).to(device)

        out, _ = self.lstm(embeds, (h0, c0))  # out: batch_size, seq_length, hidden_size
        out = out[:, -1, :]  # Get only last timestep
        out = self.dropout(out)
        out = self.act_fn(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class ValueDataset(Dataset):
    def __init__(self, values_path, sentence_data, id2sen, emb_word, option):
        self.y = np.load(values_path, allow_pickle=False)
        self.X = [sen2mat(sentence_data(1, idx)[0][0], id2sen, emb_word, option) for idx in range(sentence_data.length)]
        assert len(self.X) == self.y.shape[0]

        # Normalize data after log transform
        self.y = np.log(self.y)
        self.y = (self.y - np.mean(self.y)) / np.std(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()
        return X, y


def collate_fn_pad(list_pairs_seq_target):
    seqs = [seq for seq, target in list_pairs_seq_target]
    targets = [target for seq, target in list_pairs_seq_target]
    seqs_padded_batched = nn.utils.rnn.pad_sequence(seqs, batch_first=True)  # will pad at beginning of sequences
    targets_batched = torch.stack(targets)
    assert seqs_padded_batched.shape[0] == len(targets_batched)
    return seqs_padded_batched, targets_batched


def _run_epoch(model, data_loader, criterion, option, optimizer=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    for X, y in data_loader:
        if train:
            optimizer.zero_grad()

        X = X.to(device)
        y = y.to(device)

        # Model forward
        pred = model.forward(X)

        # Loss
        loss = criterion(pred, y)
        epoch_loss += loss.item()

        # Update model
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), option.clip_norm)
            optimizer.step()

    return epoch_loss / len(data_loader)


def train_network(model, train_data, valid_data, option):
    model.train()
    torch.manual_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=option.learning_rate, weight_decay=option.weight_decay)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=option.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn_pad
    )
    test_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=option.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn_pad
    )
    criterion = nn.MSELoss()

    writer_train = SummaryWriter("./runs/{}/train".format(option.exp_name))
    writer_valid = SummaryWriter("./runs/{}/test".format(option.exp_name))

    for epoch in range(option.epochs):
        # Train data
        epoch_loss_train = _run_epoch(model, train_loader, criterion, option, optimizer=optimizer)

        # Test data
        with torch.no_grad():
            epoch_loss_test = _run_epoch(model, test_loader, criterion, option, train=False)

        logging.info(
            "Epoch: {:>2}, Train Loss: {:.4f}, Validate Loss: {:.4f}".format(epoch, epoch_loss_train, epoch_loss_test)
        )
        writer_train.add_scalar("total_loss", epoch_loss_train, epoch)
        writer_valid.add_scalar("total_loss", epoch_loss_test, epoch)


def init(option):
    # Python 3 conversion for loading embeddings
    fileobj = open(option.emb_path, "r")
    emb_word, emb_id = pkl.load(StrToBytes(fileobj), encoding="bytes")
    emb_word = {k.decode("utf-8"): v for k, v in emb_word.items()}
    fileobj.close()

    # Get the dateset and keyword vector representation
    # Dataset is list of word ids
    # keyword vectors are boolean flags indicating whether a word is classified as a keyword
    dataclass = data.Data(config)
    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen

    # Calculate the range to operate on
    idx_start = option.data_start
    idx_end = option.data_end if option.data_end != -1 else use_data.length
    print("Operating in range of [{}, {})".format(idx_start, idx_end))

    torch.manual_seed(42)
    model = LSTM(300, 256, 45, num_layers=1).to(device)

    dataset = ValueDataset("data/quoradata/value_data.npy", use_data, id2sen, emb_word, option)
    split_per = 1.0 - (float(option.batch_size) / len(dataset)) if option.full_train else 0.8
    split_idx = int(len(dataset) * split_per)
    logging.info("Split index: {}".format(split_idx))
    train_set, val_set = torch.utils.data.random_split(dataset, [split_idx, len(dataset) - split_idx])

    # Train
    logging.info(
        "Training value function model on {}".format("full dataset" if option.full_train else "0.8, validating on 0.2")
    )
    train_network(model, train_set, val_set, option)

    # Save model
    logging.info("Exporting model to ./data/value_function_model.pt")
    torch.save(model.state_dict(), "./data/value_function_model.pt")


# python source/value_function_trainer.py --exps_dir exps-sampling --exp_name test --use_data_path data/quoradata/value_train.txt --mode kw-bleu --data_start 0 --data_end 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment setup")
    # misc
    parser.add_argument("--seed", default=33, type=int)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--no_train", default=False, action="store_true")
    parser.add_argument("--exps_dir", default=None, type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--load", default=None, type=str)

    # data property
    parser.add_argument("--data_path", default="data/quoradata/test.txt", type=str)
    parser.add_argument("--dict_path", default="data/quoradata/dict.pkl", type=str)
    parser.add_argument("--dict_size", default=30000, type=int)
    parser.add_argument("--vocab_size", default=30003, type=int)
    parser.add_argument("--backward", default=False, action="store_true")
    parser.add_argument("--keyword_pos", default=True, action="store_false")
    # model architecture
    parser.add_argument("--num_steps", default=15, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--emb_size", default=256, type=int)
    parser.add_argument("--hidden_size", default=300, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--model", default=0, type=int)
    # optimization
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--clip_norm", default=1e-1, type=float)
    parser.add_argument("--no_cuda", default=False, action="store_true")
    parser.add_argument("--local", default=False, action="store_true")
    parser.add_argument("--threshold", default=0.1, type=float)
    parser.add_argument("--full_train", default=False, action="store_true")

    # evaluation
    parser.add_argument("--sim", default="word_max", type=str)
    parser.add_argument("--mode", default="sa", type=str)
    parser.add_argument("--accuracy", default=False, action="store_true")
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--accumulate_step", default=1, type=int)
    parser.add_argument("--backward_path", default=None, type=str)
    parser.add_argument("--forward_path", default=None, type=str)

    # sampling
    parser.add_argument("--use_data_path", default="data/quoradata/test.txt", type=str)
    parser.add_argument("--reference_path", default=None, type=str)
    parser.add_argument("--pos_path", default="POS/english-models", type=str)
    parser.add_argument("--emb_path", default="data/quoradata/emb.pkl", type=str)
    parser.add_argument("--max_key", default=3, type=float)
    parser.add_argument("--max_key_rate", default=0.5, type=float)
    parser.add_argument("--rare_since", default=30000, type=int)
    parser.add_argument("--sample_time", default=100, type=int)
    parser.add_argument("--search_size", default=100, type=int)
    parser.add_argument("--action_prob", default=[0.3, 0.3, 0.3, 0.3], type=list)
    parser.add_argument("--just_acc_rate", default=0.0, type=float)
    parser.add_argument("--sim_mode", default="keyword", type=str)
    parser.add_argument("--save_path", default="temp.txt", type=str)
    parser.add_argument("--forward_save_path", default="data/tfmodel/forward.ckpt", type=str)
    parser.add_argument("--backward_save_path", default="data/tfmodel/backward.ckpt", type=str)
    parser.add_argument("--max_grad_norm", default=5, type=float)
    parser.add_argument("--keep_prob", default=1, type=float)
    parser.add_argument("--N_repeat", default=1, type=int)
    parser.add_argument("--C", default=0.03, type=float)
    parser.add_argument("--M_kw", default=8, type=float)
    parser.add_argument("--M_bleu", default=1, type=float)

    # Samples to work on
    # This lets us run multiple instances on separate parts of the data
    # for added parallelism
    parser.add_argument("--data_start", default=0, type=int)
    parser.add_argument("--data_end", default=-1, type=int)
    parser.add_argument("--alg", default="sa", type=str)

    d = vars(parser.parse_args())
    option = Option(d)

    random.seed(option.seed)
    np.random.seed(option.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu
    config = option

    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename="logs/{}.log".format(option.exp_name))
    formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)

    if option.exp_name is None:
        option.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
        option.tag = option.exp_name
    option.this_expsdir = os.path.join(option.exps_dir, option.tag)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)

    init(option)