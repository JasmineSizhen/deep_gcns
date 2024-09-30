import re
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.optim as optim
from model import DeeperGCN
from tqdm import tqdm
from args import ArgsInit
from utils.ckpt_util import save_ckpt
import logging
import time
import statistics
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from scipy import stats
import os

## loss function
cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

## NGS dataset
enc_dict = {"[MASK]": 0, "A": 1, "P": 2, "G": 3, "C": 4, "N": 5}

def parse_mfe_fa(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = []
    
    for i in range(0, len(lines), 3):
        seq_id = lines[i].strip()[1:] 
        sequence = lines[i+1].strip()
        structure_line = lines[i+2].strip()
        
        structure_match = re.match(r"([.()]+)\s+\(\s*(-?\d+\.\d+)\)", structure_line)
        if structure_match:
            structure = structure_match.group(1)
            min_energy = float(structure_match.group(2))
        else:
            structure = None
            min_energy = None
        
        data.append({
            "sequence_id": seq_id,
            "sequence": sequence,
            "secondary_structure": structure,
            "min_energy": min_energy
        })
    
    return pd.DataFrame(data)

def encode(df):
    # (TODO) modify the dataframe column names 
    names = df["sequence_id"].values.tolist()
    seqs = df["sequence"].values.tolist()
    strucs = df["secondary_structure"].values.tolist()
    
    data_list = []
    for name, seq, struc in zip(names, seqs, strucs):
        # node feat: nucleotide type
        x = torch.tensor([enc_dict["[MASK]"] for nuc in list(seq)], dtype=torch.long)
        y = torch.tensor([enc_dict[nuc] for nuc in list(seq)], dtype=torch.long)
        
        assert len(x) == len(y)
        # edge_weight 
#         bp_file = os.path.join("/mount/data/ViennaRNA-2.6.4/src/bin", name + "_dp.ps")
#         bp = {}
#         with open(bp_file, "r") as f:
#             lines = f.readlines()
#             tag = False
#             for line in lines:
#                 if "start of base pair probability data" in line:
#                     tag = True
#                     continue
#                 if "showpage" in line:
#                     break
#                 if tag:
#                     i, j, prob, _ = line.strip().split()
#                     i, j, prob = int(i), int(j), float(prob)
#                     i -= 1 
#                     j -= 1
#                     assert i>=0
#                     assert j>=0 
#                     if i in bp:
#                         bp[i][j] = prob * prob
#                     else:
#                         bp[i] = {}
#                         bp[i][j] = prob * prob
        
        # edges
        stack = []
        edges = []
        # edge_weight = []
        for i, s in enumerate(struc):
            if i == 0:
                edges.append((i, i+1))
                # edge_weight.append(1)
            elif i == len(seq) - 1:
                edges.append((i, i-1))
                # edge_weight.append(1)
            else:
                edges.append((i, i+1))
                # edge_weight.append(1)
                edges.append((i, i-1))
                # edge_weight.append(1)
            if s == ".":
                continue
            if s == "(":
                stack.append(i)
                continue
            assert s == ")"
            left = stack.pop()
            edges.append((left, i))
            edges.append((i, left))
#             edge_weight.append(bp[left][i])
#             edge_weight.append(bp[left][i])
        assert len(stack) == 0
        edge_index = torch.tensor(edges, dtype=torch.long)
        edge_index = torch.transpose(edge_index, 0, 1)
        # edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        # edge_attr are one for all edges
        edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.long)
        
        # position
        pos = torch.tensor([i for i in range(len(seq))], dtype=torch.long)

        # Data
        data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr, edge_weight=None, pos=pos)
        data_list.append(data)
        
    return data_list


def myData(trainfile, validfile):
    train_df = parse_mfe_fa(trainfile)
    valid_df = parse_mfe_fa(validfile)
    train_dl = encode(train_df)
    valid_dl = encode(valid_df)
    print(len(train_dl), len(valid_df))
    return train_dl, valid_dl

        
def train(model, device, loader, optimizer, task_type, grad_clip=0.):
    loss_list = []
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        
        # print("Input size: ", batch.x.shape)
        # print("Position size: ", batch.pos.shape)
        
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            pred = model(batch).squeeze()
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    grad_clip)

            optimizer.step()

            loss_list.append(loss.item())
    return statistics.mean(loss_list)


@torch.no_grad()
def eval(model, device, loader):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            pred = model(batch)
            pred = torch.reshape(pred, (-1,))
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true,
                  "y_pred": y_pred}
    
    pear = stats.pearsonr(y_true, y_pred)[0]
    spear = stats.spearmanr(y_true, y_pred)[0]

    return pear, spear


def main(args):
    logging.info('%s' % args)
    
    # device
    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')
        
    # saving folder
    sub_dir = ''

    trainfile = "/root/workdir/data/train.mfe.fa"
    validfile = "/root/workdir/data/val.mfe.fa"
    
    print("Batch size = ", args.batch_size)
    
    train_dl, valid_dl = myData(trainfile, validfile)
    train_loader = DataLoader(train_dl, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dl, batch_size=args.batch_size, shuffle=True)

    model = DeeperGCN(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results = {'highest_valid_pear': 0,
               'valid_spear': 0,}
               # 'test_pear': 0,
               # 'test_spear': 0}

    start_time = time.time()

    lst = list(range(1, args.epochs + 1))
    for epoch in tqdm(lst):
        epoch_loss = train(model, device, train_loader, optimizer, "regression", grad_clip=args.grad_clip)

        train_pear, train_spear = eval(model, device, train_loader)
        valid_pear, valid_spear = eval(model, device, valid_loader)

        logging.info({'Train': (train_pear, train_spear),
                      'Validation': (valid_pear, valid_spear),
                     })

        if valid_pear > results['highest_valid_pear']:
            results['highest_valid_pear'] = valid_pear
            results['valid_spear'] = valid_spear
            # results['test_pear'] = test_pear

            save_ckpt(model, optimizer,
                      round(epoch_loss, 4), epoch,
                      args.model_save_path,
                      sub_dir, name_post='valid_best')

    
    logging.info("%s" % results)
    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))

    return results


if __name__ == "__main__":
    # arguments 
    args = ArgsInit().save_exp()
    main(args)
