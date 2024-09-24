import os
import sys
import random
import pandas as pd

import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
from scipy import stats

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
from opt import OptInit
from architecture import DeepGCN
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
from utils.metrics import AverageMeter
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train():
    info_format = 'Epoch: [{}]\t loss: {: .6f}\
    train pear: {: .6f} train spear: {: .6f} \
    val pear: {: .6f} val spear: {: .6f} best val pear: {: .6f}'
    opt.printer.info('===> Init the optimizer ...')
    criterion = torch.nn.MSELoss().to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    scheduler = ReduceLROnPlateau(optimizer, "min", patience=opt.lr_patience, verbose=True, factor=0.5, cooldown=30,
                                  min_lr=opt.lr/100)
    opt.scheduler = 'ReduceLROnPlateau'

    optimizer, scheduler, opt.lr = load_pretrained_optimizer(opt.pretrained_model, optimizer, scheduler, opt.lr)

    opt.printer.info('===> Init Metric ...')
    opt.losses = AverageMeter()

    best_val_pear = 0.
    opt.printer.info('===> Start training ...')
    for _ in range(opt.epoch, opt.total_epochs):
        opt.epoch += 1
        loss, train_pear, train_spear = train_step(model, train_loader, optimizer, criterion, opt)
        val_pear, val_spear = test(model, valid_loader, opt)

        if val_pear > best_val_pear:
            best_val_pear = val_pear
            save_ckpt(model, optimizer, scheduler, opt.epoch, opt.save_path, opt.post, name_post='val_best')

        opt.printer.info(info_format.format(opt.epoch, loss, train_pear, train_spear, val_pear, val_spear, best_val_pear))

        if opt.scheduler == 'ReduceLROnPlateau':
            scheduler.step(opt.losses.avg)
        else:
            scheduler.step()

    opt.printer.info('Saving the final model.Finish!')


def train_step(model, train_loader, optimizer, criterion, opt):
    model.train()
    count = 0.
    pear = 0
    spear = 0
    opt.losses.reset()
    for i, data in enumerate(train_loader):
        opt.iter += 1
        if not opt.multi_gpus:
            data = data.to(opt.device)
            gt = data.y
        else:
            gt = torch.cat([data_batch.y for data_batch in data], 0).to(opt.device)

        # ------------------ zero, output, loss
        optimizer.zero_grad()
        out = model(data).squeeze()
        loss = criterion(out, gt)
        
        pear += stats.pearsonr(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())[0]
        spear += stats.spearmanr(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())[0]
        count += 1
        # ------------------ optimization
        loss.backward()
        optimizer.step()

        opt.losses.update(loss.item())
    
    return opt.losses.avg, pear/count, spear/count


def test(model, loader, opt):
    model.eval()
    count = 0
    pear = 0
    spear = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(opt.device)
            out = model(data).squeeze()

            pear += stats.pearsonr(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())[0]
            spear += stats.spearmanr(data.y.cpu().detach().numpy(), out.cpu().detach().numpy())[0]
            count += 1
    
    return pear/count, spear/count


def save_ckpt(model, optimizer, scheduler, epoch, save_path, name_pre, name_post='best'):
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    state = {
            'epoch': epoch,
            'state_dict': model_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
    filename = '{}/{}_{}.pth'.format(save_path, name_pre, name_post)
    torch.save(state, filename)

## dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
enc_dict = {"[MASK]": 0, "A": 1, "T": 2, "G": 3, "C": 4, "N": 5}

def encode(df):
    seqs = df["seq"].values.tolist()
    strucs = df["MEA_structure"].values.tolist()
    ys = df["log2FoldChange"].values.tolist()
    
    data_list = []
    for seq, struc, y in zip(seqs, strucs, ys):
        # node feat: nucleotide type
        x = [enc_dict[nuc] for nuc in list(seq)]
        x = torch.tensor(x, dtype=torch.long)
        # edges
        stack = []
        edges = []
        for i, s in enumerate(struc):
            if i == 0:
                edges.append((i, i+1))
            elif i == len(seq) - 1:
                edges.append((i, i-1))
            else:
                edges.append((i, i+1))
                edges.append((i, i-1))
            if s == ".":
                continue
            if s == "(":
                stack.append(i)
                continue
            assert s == ")"
            left = stack.pop()
            edges.append((left, i))
            edges.append((i, left))
        assert len(stack) == 0
        edge_index = torch.tensor(edges, dtype=torch.long)
        edge_index = torch.transpose(edge_index, 0, 1)
        
        # position 
        pos = torch.tensor([i for i in range(218)], dtype=torch.long)

        # Data
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.float), pos=pos)
        data_list.append(data)
        
    return data_list


def myData(data_file, test_fold=1, val_fold=2):
    df = pd.read_csv(data_file)
    train_df = df[(df["fold"] != test_fold) & (df["fold"] != val_fold)]
    valid_df = df[df["fold"] == val_fold]
    test_df  = df[df["fold"] == test_fold]
    
    train_dl = encode(train_df)
    valid_dl = encode(valid_df)
    test_dl  = encode(test_df)
    print(len(train_dl), len(valid_df), len(test_dl))
        
    return train_dl, valid_dl, test_dl
        

if __name__ == '__main__':
    opt = OptInit().initialize()
    opt.printer.info('===> Creating dataloader ...')
    data_file = "/mount/data/UDS-Full_length_mRNA_study/NGS/capping_efficiency_based_on_raw_counts_with_fold_mea.csv"
    train_dl, valid_dl, test_dl = myData(data_file)
    train_loader = DataLoader(train_dl, batch_size=opt.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dl, batch_size=opt.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dl, batch_size=opt.batch_size, shuffle=True)

    opt.printer.info('===> Loading the network ...')
    model = DeepGCN(opt).to(opt.device)
    if opt.multi_gpus:
        model = DataParallel(DeepGCN(opt)).to(opt.device)
    opt.printer.info('===> loading pre-trained ...')
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)

    if opt.phase == 'train':
        train()

        # load best model on validation dataset
        opt.printer.info('\n\n=================Below is best model testing=====================')
        # opt.printer.info('Loading best model on validation dataset')
        best_model_path = '{}/{}_val_best.pth'.format(opt.save_path, opt.post)
        model, opt.best_value, opt.epoch = load_pretrained_models(model, best_model_path, opt.phase)
        val_pear, val_spear = test(model, valid_loader, opt)
        opt.printer.info('Validation pearson/spearman of model on validation dataset: {: 6f}/{: 6f}'.format(val_pear, val_spear))

        # load best model on test_dataset
        opt.printer.info('\nLoading best model on test dataset')
        best_model_path = '{}/{}_test_best.pth'.format(opt.save_path, opt.post)
        model, opt.best_value, opt.epoch = load_pretrained_models(model, best_model_path, opt.phase)
        test_pear, test_spear = test(model, test_loader, opt)
        opt.printer.info('Test pearson/spearman of model on test dataset: {: 6f}/{: 6f}'.format(test_pear, test_spear))

    else:
        test_pear, test_spear = test(model, test_loader, opt)
        opt.printer.info('Test pearson/spearman of model on test dataset: {: 6f}/{: 6f}'.format(test_pear, test_spear))




