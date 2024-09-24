import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin, Sequential as Seq
from gcn_lib.sparse import MultiSeq, MLP, GraphConv, ResGraphBlock, DenseGraphBlock
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class DeepGCN(torch.nn.Module):
    """
    static graph

    """
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        self.channels = opt.n_filters
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        conv = opt.conv
        heads = opt.n_heads
        c_growth = 0
        self.n_blocks = opt.n_blocks
        self.batch_size = opt.batch_size
        
        self.embedding = nn.Embedding(opt.vocab_size, opt.in_channels)
        # position embedding
        self.pos_encoder = nn.Embedding(218, opt.in_channels)
        
        self.head = GraphConv(opt.in_channels, self.channels, conv, act, norm, bias, heads)

        res_scale = 1 if opt.block.lower() == 'res' else 0
        if opt.block.lower() == 'dense':
            c_growth = self.channels 
            self.backbone = MultiSeq(*[DenseGraphBlock(self.channels+i*c_growth, c_growth, conv, act, norm, bias, heads)
                                       for i in range(self.n_blocks-1)])
        else:
            self.backbone = MultiSeq(*[ResGraphBlock(self.channels, conv, act, norm, bias, heads, res_scale)
                                       for _ in range(self.n_blocks-1)])
        fusion_dims = int(self.channels * self.n_blocks + c_growth * ((1 + self.n_blocks - 1) * (self.n_blocks - 1) / 2))
        self.fusion_block = MLP([fusion_dims, 1024], act, None, bias)
        self.prediction = Seq(*[MLP([1+fusion_dims, self.channels*2], act, norm, bias), torch.nn.Dropout(p=opt.dropout),
                                MLP([self.channels *2, self.channels], act, norm, bias), torch.nn.Dropout(p=opt.dropout),
                                MLP([self.channels, self.channels], None, None, bias)])
        
        # graph property prediction
        # self.dropout = torch.nn.Dropout(p=opt.dropout)
        # self.graph_pred_linear = torch.nn.Linear(self.channels, 1)
        
        # cnn
        kernal_num = 128
        kernal_sizes = [3,4,5]
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(1, kernal_num, (K, self.channels)) for K in kernal_sizes])
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(len(kernal_sizes) * kernal_num, 1)
        
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, data):
        x, edge_index, pos, batch = data.x, data.edge_index, data.pos, data.batch
        
        x = self.embedding(x) + self.pos_encoder(pos) # node embedding
        feats = [self.head(x, edge_index)]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1], edge_index)[0])
        feats = torch.cat(feats, 1)
        fusion, _ = torch.max(self.fusion_block(feats), 1, keepdim=True)
        h_node = self.prediction(torch.cat((feats, fusion), 1)) # [num_seq * seq_len, hidden_dim]
        
        # reshape the tensor, max/mean pooling
        # h_node = torch.reshape(h_node, (-1, 218, self.channels))
        # h_graph = torch.max(h_node, dim=1)[0] # torch.max outputs a turple (max_value, index)
        # h_graph = self.dropout(h_graph)
        # logit = self.graph_pred_linear(h_graph)
        
        # cnn
        h_node = torch.reshape(h_node, (-1, 218, self.channels)) # [batch_size, seq_len, hidden_dim]
        h_node = h_node.unsqueeze(1) # [batch_size, 1, seq_len, hidden_dim]
        h_node = [F.relu(conv(h_node)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        h_node = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h_node]   # [(N, Co), ...]*len(Ks)
        h_node = torch.cat(h_node, 1)
        h_node = self.dropout(h_node)  # (N, len(Ks)*Co)
        logit = self.fc(h_node)   # (N, C)
        
        return logit


