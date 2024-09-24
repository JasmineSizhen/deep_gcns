import os
import sys
import random
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
from sklearn.metrics import f1_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
from opt import OptInit
from architecture import DeepGCN
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
from utils.metrics import AverageMeter
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train():
    info_format = 'Epoch: [{}]\t loss: {: .6f} train mF1: {: .6f} \t val mF1: {: .6f}\t best val mF1: {: .6f}'
    opt.printer.info('===> Init the optimizer ...')
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    scheduler = ReduceLROnPlateau(optimizer, "min", patience=opt.lr_patience, verbose=True, factor=0.5, cooldown=30,
                                  min_lr=opt.lr/100)
    opt.scheduler = 'ReduceLROnPlateau'

    optimizer, scheduler, opt.lr = load_pretrained_optimizer(opt.pretrained_model, optimizer, scheduler, opt.lr)

    opt.printer.info('===> Init Metric ...')
    opt.losses = AverageMeter()

    best_val_value = 0.
    opt.printer.info('===> Start training ...')
    for _ in range(opt.epoch, opt.total_epochs):
        opt.epoch += 1
        loss, train_value = train_step(model, train_loader, optimizer, criterion, opt)
        val_value = test(model, valid_loader, opt)

        if val_value > best_val_value:
            best_val_value = val_value
            save_ckpt(model, optimizer, scheduler, opt.epoch, opt.save_path, opt.post, name_post='val_best')

        opt.printer.info(info_format.format(opt.epoch, loss, train_value, val_value, best_val_value))

        if opt.scheduler == 'ReduceLROnPlateau':
            scheduler.step(opt.losses.avg)
        else:
            scheduler.step()

    opt.printer.info('Saving the final model.Finish!')


def train_step(model, train_loader, optimizer, criterion, opt):
    model.train()
    micro_f1 = 0.
    count = 0.
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
        out = model(data)
        loss = criterion(out, gt)
        
        num_node = len(gt)
        micro_f1 += f1_score(data.y.cpu().detach().numpy(),
                             torch.argmax(out, dim=1).cpu().detach().numpy(), average='micro') * num_node
        count += num_node
        # ------------------ optimization
        loss.backward()
        optimizer.step()

        opt.losses.update(loss.item())
    
    return opt.losses.avg, micro_f1/count


def test(model, loader, opt):
    model.eval()
    count = 0
    micro_f1 = 0.
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(opt.device)
            out = model(data)

            num_node = len(data.x)
            micro_f1 += f1_score(data.y.cpu().detach().numpy(),
                                 torch.argmax(out, dim=1).cpu().detach().numpy(), average='micro') * num_node
            count += num_node
        
        micro_f1 = float(micro_f1)/count
    
    return micro_f1


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
all_pairs = ["GC", "CG", "AT", "TA", "GT", "TG"]

def myData(data_file):
    with open(data_file, "r") as f:
        lines = f.readlines()
    
    i_seq = 0
    data_list = []
    while i_seq < len(lines):
        assert lines[i_seq].startswith(">")
        seq = lines[i_seq+1].strip().replace("P", "T")
        struc = lines[i_seq+2].strip().split()[0]
        i_seq += 3
        
        # node label
        y = [enc_dict[nuc] for nuc in list(seq)]
        y = torch.tensor(y, dtype=torch.long)
        
        # edges
        edges = []
        stack = []
        pairs = []
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
            pairs.append((left, i))
        assert len(stack) == 0
        edge_index = torch.tensor(edges, dtype=torch.long)
        edge_index = torch.transpose(edge_index, 0, 1)
        
        # node features
        # initialize 
        x = torch.ones(len(seq),  dtype=torch.long)
        for (i, j) in pairs:
            sel_pair = random.choice(all_pairs)
            x[i] = enc_dict[sel_pair[0]]
            x[j] = enc_dict[sel_pair[1]]
        # mask all
        # x = torch.zeros(len(seq), dtype=torch.long) 
        # mask part of pairs
#         sel_pairs = random.sample(pairs, k=int(len(pairs)/2))
#         for (i, j) in sel_pairs:
#             x[i] = 0
#             x[j] = 0

        # Data
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
        
    return data_list
        

if __name__ == '__main__':
    opt = OptInit().initialize()
    opt.printer.info('===> Creating dataloader ...')
    train_file = "/mount/data/Aptamer_design/NOS/data/train.mfe.fa"
    val_file = "/mount/data/Aptamer_design/NOS/data/val.mfe.fa"
    data_list = myData(train_file)
    train_loader = DataLoader(data_list, batch_size=opt.batch_size, shuffle=True)
    data_list = myData(val_file)
    valid_loader = DataLoader(data_list, batch_size=opt.batch_size, shuffle=True)

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
        test_value = test(model, valid_loader, opt)
        opt.printer.info('Test m-F1 of model on validation dataset: {: 6f}'.format(test_value))

        # load best model on test_dataset
        # opt.printer.info('\nLoading best model on test dataset')
#         best_model_path = '{}/{}_test_best.pth'.format(opt.save_path, opt.post)
#         model, opt.best_value, opt.epoch = load_pretrained_models(model, best_model_path, opt.phase)
#         test_value = test(model, test_loader, opt)
#         opt.printer.info('Test m-F1 of model on test dataset: {: 6f}'.format(test_value))

    else:
        test_value = test(model, test_loader, opt)
        opt.printer.info('Test m-F1: {: 6f}'.format(test_value))




