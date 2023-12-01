import torch
import torch.nn as nn
from torch import Tensor
from models import model_utils as mu
from torch_geometric.nn.inits import reset
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.nn.norm.batch_norm import BatchNorm
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import add_self_loops, remove_self_loops, to_torch_csr_tensor

def act_layer(activation_func, inplace=True, neg_slope =0.2, nprelu=1):
    activation_func = activation_func.lower()
    if activation_func == 'relu':
        layer = nn.ReLU(inplace)
    elif activation_func == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif activation_func == 'prelu':
        layer = nn.PReLU(num_parameters=nprelu, init=neg_slope)
    elif activation_func == 'tanh':
        layer = nn.Tanh()
    elif activation_func == 'sigmoid':
        layer = nn.Sigmoid()
    elif activation_func == 'softmax':
        layer = nn.Softmax(dim = 1)
    else:
        raise NotImplementedError(f'activation layer {activation_func} is not implemented')
    return layer

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func1='relu', activation_func2 = '', mid_channels = None):
        super().__init__()
        
        if mid_channels == None:    mid_channels = out_channels
        
        self.activation1  = act_layer(activation_func1)  
        if activation_func2 == '': self.activation2 = act_layer(activation_func1)
        else: self.activation2 = act_layer(activation_func2)
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(mid_channels),
            self.activation1,
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            self.activation2
        )

    def forward(self, x):
        return self.double_conv(x)

 
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func1 = ''):
        super().__init__()
        if activation_func1 == '':
            self.single_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.activation1  = act_layer(activation_func1)
            self.single_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                self.activation1
            )
    
    def forward(self, x):
        return self.single_conv(x)
'''
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func1 = ''):
        super().__init__()
        if activation_func1 == '':
            self.activation1 = None
            self.single_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            
        else:
            self.activation1 = act_layer(activation_func1)
            self.single_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.single_conv(x)
        #assert torch.isnan(x).any() == False, f'Inside single conv got nan'
        #assert torch.isinf(x).any() == False, f'Inside single conv got inf'
        #assert (x == 0).all()       == False, f'Inside single conv got all 0'
        return self.activation1(x)
'''  
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func1='relu', activation_func2='', mid_channels = None):
        super().__init__()
        self.maxpool_dconv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, activation_func1, activation_func2, mid_channels)
        )

    def forward(self, x):
        return self.maxpool_dconv(x)

# Decoding block without skip connections
class Up_noskip(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func1 = 'relu', activation_func2='', mid_channles = None, bilinear=True):
        super().__init__()
        if bilinear:
            # scale_factor same as the varible of maxpool()
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            # transposed conv should have the same padding, stride, kernel size as the
            # corresponding down layer in order to maintain the same dimension
            self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.upnoskip_dconv = DoubleConv(in_channels, out_channels, activation_func1, activation_func2, mid_channles)
    
    def forward(self, x):
        x = self.up(x)
        x = self.upnoskip_dconv(x)
        return x
    
# Decoding block with skip connections
class Up(nn.Module):
    def __init__(self, in_channels, skip_connections_channels, out_channels, activation_func1 = 'relu', activation_func2='', mid_channles = None, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.up_dconv = DoubleConv(in_channels+skip_connections_channels, out_channels, activation_func1, activation_func2, mid_channles)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim =1)
        x = self.up_dconv(x)
        return x

class InConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func='relu'):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, activation_func)
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation_func1 = ''):
        super().__init__()
        self.conv =  SingleConv(in_channels, out_channels, activation_func1)
    
    def forward(self, x):
        x = self.conv(x)
        return x

class UNetEncoder_noskips(nn.Module):
    def __init__(self, activation_func, in_channels):
        super().__init__()
        self.act         = activation_func
        self.in_channels = in_channels
        self.n1          = in_channels * 2
        self.n2          = self.n1 * 2
        self.n3          = self.n2 * 2
        self.n4          = self.n3 * 2 

        self.inc   = InConv(self.in_channels, self.n1, self.act)
        self.down1 = Down(self.n1, self.n2, self.act)
        self.down2 = Down(self.n2, self.n3, self.act)
        self.down3 = Down(self.n3, self.n4, self.act)
    
    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x
    
class UNetDecoder_noskips(nn.Module):
    def __init__(self, activation_func, in_channels, out_channels = 1, exp_type=''):
        super().__init__()
        self.exp_type     = exp_type
        self.act          = activation_func
        self.in_channels  = in_channels
        self.n1           = in_channels * 2
        self.n2           = self.n1 * 2
        self.n3           = self.n2 * 2
        self.n4           = self.n3 * 2
        self.out_channels = out_channels   

        self.up1   = Up_noskip(self.n4, self.n3, self.act)
        self.up2   = Up_noskip(self.n3, self.n2, self.act)
        self.up3   = Up_noskip(self.n2, self.n1, self.act)
        if self.exp_type == 'binary_class':
            self.outc  = OutConv(self.n1, self.out_channels, 'sigmoid')
        elif self.exp_type == 'three_class':
            self.outc  = OutConv(self.n1, self.out_channels, 'softmax')
        else:
            print(f'INFO: In UNET-DECODER use {self.act} as activation layer in the last layer')
            self.outc = OutConv(self.n1, self.out_channels, self.act)
    
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.outc(x)
        return x

# no skip connections
class SimpleUNet3D(nn.Module):
    """
    It is exactly the same as
    """
    def __init__(self, activation_func, in_channels, out_channels = 1, exp_type=''):
        super().__init__()
        self.exp_type     = exp_type
        self.act          = activation_func
        self.in_channels  = in_channels
        self.n1           = in_channels * 2
        self.n2           = self.n1 * 2
        self.n3           = self.n2 * 2
        self.n4           = self.n3 * 2
        self.out_channels = out_channels 

        self.inc   = InConv(self.in_channels, self.n1, self.act)
        self.down1 = Down(self.n1, self.n2, self.act)
        self.down2 = Down(self.n2, self.n3, self.act)
        self.down3 = Down(self.n3, self.n4, self.act)
        self.up1   = Up_noskip(self.n4, self.n3, self.act)
        self.up2   = Up_noskip(self.n3, self.n2, self.act)
        self.up3   = Up_noskip(self.n2, self.n1, self.act)
        if self.exp_type == 'binary_class':
            self.outc  = OutConv(self.n1, self.out_channels, 'sigmoid')         
        elif self.exp_type == 'three_class':
            self.outc  = OutConv(self.n1, self.out_channels, 'softmax')
        else:
            print(f'INFO: In UNET-DECODER use {self.act} as activation layer in the last layer')
            self.outc = OutConv(self.n1, self.out_channels, self.act)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.outc(x)
        return x

# with skip connections
class UNet3D(nn.Module):
    """
    Exactly like vanilla but with skip connections and a higher number of channels for the last two downsampling layers
    """
    def __init__(self, activation_func, in_channels, out_channels = 1, exp_type=''):
        super().__init__()
        self.exp_type     = exp_type
        self.act          = activation_func
        self.in_channels  = in_channels
        self.n1           = in_channels * 2
        self.n2           = self.n1 * 2
        self.n3           = self.n2 * 4
        self.n4           = self.n3 * 4
        # out_channels should be 1. Because we like to output an image,
        # which will be the corrected segmented initial image. 
        self.out_channels = out_channels 

        self.inc   = InConv(self.in_channels, self.n1, self.act)
        self.down1 = Down(self.n1, self.n2, self.act)
        self.down2 = Down(self.n2, self.n3, self.act)
        self.down3 = Down(self.n3, self.n4, self.act)
        self.up1   = Up(self.n4, self.n3, self.n3, self.act)
        self.up2   = Up(self.n3, self.n2, self.n2, self.act)
        self.up3   = Up(self.n2, self.n1, self.n1, self.act)
        if self.exp_type == 'binary_class':
            self.outc  = OutConv(self.n1, self.out_channels, 'sigmoid')
        elif self.exp_type == 'three_class':
            self.outc  = OutConv(self.n1, self.out_channels, 'softmax')
        else:
            print(f'INFO: In UNET-DECODER use {self.act} as activation layer in the last layer')
            self.outc = OutConv(self.n1, self.out_channels, self.act) 

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x

class UNetDecoder(nn.Module):
    def __init__(self, activation_func, in_channels, out_channels = 1, exp_type=''):
        super().__init__()
        self.exp_type     = exp_type
        self.act          = activation_func
        self.in_channels  = in_channels
        self.n1           = in_channels * 2
        self.n2           = self.n1 * 2
        self.n3           = self.n2 * 2
        self.n4           = self.n3 * 2
        self.out_channels = out_channels

        self.up1   = Up(self.n4, self.n3, self.n3, self.act)
        self.up2   = Up(self.n3, self.n2, self.n2, self.act)
        self.up3   = Up(self.n2, self.n1, self.n1, self.act)
        if self.exp_type == 'binary_class':
            self.outc  = OutConv(self.n1, self.out_channels, 'sigmoid')
        elif self.exp_type == 'three_class':
            self.outc  = OutConv(self.n1, self.out_channels, 'softmax')
        else:
            print(f'INFO: In UNET-DECODER use {self.act} as activation layer in the last layer')
            self.outc = OutConv(self.n1, self.out_channels, self.act)
             
        
    def forward(self, x1, x2, x3, x4):
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x

class UNetEncoder(nn.Module):
    def __init__(self, activation_func, in_channels):
        super().__init__()
        self.act          = activation_func
        self.in_channels  = in_channels
        self.n1           = in_channels * 2
        self.n2           = self.n1 * 2
        self.n3           = self.n2 * 2
        self.n4           = self.n3 * 2

        self.inc   = InConv(self.in_channels, self.n1, self.act)
        self.down1 = Down(self.n1, self.n2, self.act)
        self.down2 = Down(self.n2, self.n3, self.act)
        self.down3 = Down(self.n3, self.n4, self.act)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x1, x2, x3, x4

class GraphUNet(nn.Module):
    '''
    Implementation of the model Graph U-Net proposed in the paper
    <https://arxiv.org/abs/1905.05178>, which implements a U-Net
    architecture with graph pooling and unpooling operations.

    The current implementation was adopted from the pytorch geometric
    library <https://pytorch-geometric.readthedocs.io>
    
    Parameters
    ----------
    in_channels (int): Size of node features
    hidden_channels (int): Size of node features in the hidden layers
    out_channels (int): Size of node features in the output layer (embeddings dimension)
    depth (int): The depth of the graph unet
    pool_ratios (float, list(float)): graph pooling ratio for each depth
    sum_res (bool): How to perform the skip connections. If True use sumation if False use concatenation
    act (string): Which nonlinearity to use

    '''
    def __init__(self, in_channels, hidden_channels, out_channels, depth=3, pool_ratios=0.5, sum_res=False, act='relu'):
        super().__init__()
        assert depth>=1, f'Initializing GraphUNet, the depth parameter {depth} is invalid'
        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels    = out_channels
        self.depth           = depth
        if isinstance(pool_ratios, list):
            assert len(pool_ratios) == depth, f'Initializing GraphUNet, pool ratios length {len(pool_ratios)} mismatch with depth{depth}'
            self.pool_ratios = pool_ratios
        else:
            self.pool_ratios = [pool_ratios for i in range(depth)]
        self.act             = act_layer(act)
        self.sum_res         = sum_res

        #---------- ENCODER
        self.down_convs = nn.ModuleList()
        #self.down_bns   = nn.ModuleList()
        self.pools      = nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, hidden_channels, improved=True))
        #self.down_bns.append(BatchNorm(hidden_channels))
        for i in range(depth):
            self.pools.append(TopKPooling(hidden_channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(hidden_channels, hidden_channels, improved=True))
            #self.down_bns.append(BatchNorm(hidden_channels))
        
        #---------- DECODER
        in_channels2 = hidden_channels if sum_res else 2*hidden_channels
        self.up_convs = torch.nn.ModuleList()
        #self.up_bns   = torch.nn.ModuleList()
        for i in range(depth-1):
            self.up_convs.append(GCNConv(in_channels2, hidden_channels, improved=True))
            #self.up_bns.append(BatchNorm(hidden_channels))
        self.up_convs.append(GCNConv(in_channels2, out_channels, improved=True))
        #self.up_bns.append(BatchNorm(out_channels))

        self.reset_parameters()
    
    def reset_parameters(self):
        '''
        Reset all learnable parameters of the module
        '''
        for conv in self.down_convs:
            conv.reset_parameters()
        #for bn in self.down_bns:
        #    bn.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs: 
            conv.reset_parameters()
        #for bn in self.up_bns:
        #    bn.reset_parameters()

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')
    
    def augment_adj(self, edge_index, edge_weight, num_nodes):
        '''
        The authors of the paper proposed instead of using A use A^2, 
        where A is the adjacency matrix. By doing so, you reduce the number
        of isolated nodes created due to pooling layers

        Parameters
        ----------
        edge_index
        edg_weight
        num_nodes

        Return
        ----------
        edge_index: updated edje_index matrix
        edg_weight: updated edge_weight matrix
        '''
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes = num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


    def forward(self, x, edge_index, batch=None):
        assert x.shape[-1] == self.in_channels, f'Inside the graphUnet, there is a mismatch in dimensions between {x.shape} and {self.in_channels}'
        if batch == None:   batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        #---------- ENCODING
        #x = self.down_bns[0](self.down_convs[0](x, edge_index, edge_weight))
        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        # variables to keep track for pooling/unpooling
        xs           = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms        = []

        for i in range(1, self.depth+1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i-1](x, edge_index, edge_weight, batch)

            #x = self.down_bns[i](self.down_convs[i](x, edge_index, edge_weight))
            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]
        
        #---------- DECODING
        for i in range(self.depth):
            j           = self.depth-1-i
            
            res         = xs[j]
            edge_index  = edge_indices[j]
            edge_weight = edge_weights[j]
            perm        = perms[j]

            up          = torch.zeros_like(res)
            up[perm]    = x
            x           = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            #x           = self.up_bns[i](self.up_convs[i](x, edge_index, edge_weight))
            x           = self.up_convs[i](x, edge_index, edge_weight)
            x           = self.act(x) if i < self.depth-1 else x
        
        return x, edge_index

class GAE_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth=2, act='relu'):
        super().__init__()
        assert depth>=1, f'Initializing GAE encoder, the depth parameter {depth} is invalid'
        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels    = out_channels
        self.depth           = depth
        self.act             = act_layer(act)

        self.enc_convs = nn.ModuleList()
        self.enc_convs.append(GCNConv(in_channels, hidden_channels, improved=False, add_self_loops=True))
        for i in range(1, depth-1):
            self.enc_convs.append(GCNConv(hidden_channels, hidden_channels, improved=False, add_self_loops=True))
        self.enc_convs.append(GCNConv(hidden_channels, out_channels, improved=False, add_self_loops=True))
    
    def reset_parameters(self):
        '''
        Reset all learnable parameters of the module
        '''
        for conv in self.enc_convs:
            conv.reset_parameters()
    
    def forward(self, x, edge_index):
        
        edge_weight = x.new_ones(edge_index.size(1))
        
        for i in range(self.depth-1):
            x = self.act(self.enc_convs[i](x, edge_index, edge_weight))
        x = self.enc_convs[-1](x, edge_index, edge_weight)
        
        return x
    
class InnerProductDecoder(nn.Module):
    '''
    The inner product decoder from the "Variational Graph Auto-Encoders" paper
    <https://arxiv.org/abs/1611.07308>.

    The current implementation was adopted from the pytorch geometric
    library <https://pytorch-geometric.readthedocs.io>.
    '''
    def forward(self, z, edge_index, sigmoid=True):
        '''
        Decodes the latent variables `z` into edge probabilities for
        the given node-pairs `edge_index`

        Parameters
        ----------
        z (torch.Tensor): The latent space
        sigmoid (bool)  : If set to `False`, does not apply the logistic sigmoid function to the output.
        
        Return
        ----------
        value: updated edje_index matrix
        '''
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value
    
    def forward_all(self, z, sigmoid=True):
        '''
        Decodes the latent variables `z` into a probabilistic dense
        adjacency matrix. Here all the edges are taken into account.

         Parameters
        ----------
        z (torch.Tensor): The latent space
        sigmoid (bool)  : If set to `False`, does not apply the logistic sigmoid function to the output.
        
        Return
        ----------
        adj: updated dense adjacaency matrix
        '''
        adj         = torch.matmul(z, z.t())
        adj         = torch.sigmoid(adj) if sigmoid else adj
        adj         = adj.to_sparse_coo()
        edge_index  = adj.indices()
        edge_weight = adj.values() 
        return edge_index, edge_weight

class GAE_Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth=2, act='relu'):
        super().__init__()
        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels    = out_channels
        self.depth           = depth
        self.act             = act_layer(act)

        self.innerproducetdecoder = InnerProductDecoder()
        
        if self.depth>0:
            self.dec_convs            = nn.ModuleList()
            self.dec_convs.append(GCNConv(in_channels, hidden_channels, improved=False, add_self_loops=True))
            for i in range(1, depth-1):
                self.dec_convs.append(GCNConv(hidden_channels, hidden_channels, improved=False, add_self_loops=True))
            self.dec_convs.append(GCNConv(hidden_channels, out_channels, improved=False, add_self_loops=True))
    
    def reset_parameters(self):
        '''
        Reset all learnable parameters of the module
        '''
        reset(self.innerproducetdecoder)
        if self.depth > 0:
            for conv in self.dec_convs:
                conv.reset_parameters()
    
    def forward(self, x, edge_index):

        edge_weight = self.innerproducetdecoder.forward(x, edge_index)
        
        if self.depth > 0:
            for i in range(self.depth-1):
                x = self.act(self.dec_convs[i](x, edge_index, edge_weight))
            x = self.dec_convs[-1](x, edge_index, edge_weight)

        return x, edge_index, edge_weight

# no skip connections
class CombNet_v1(nn.Module):
    def __init__(self, 
                 activation_func_unet, 
                 activation_func_graph, 
                 in_channels_unet,
                 hidden_channels_graph,
                 depth_graph       = 3, 
                 pool_ratios_graph = 0.8, 
                 sum_res_graph     = False,
                 out_channels_unet = 1, 
                 exp_type          = ''):
        
        super().__init__()
        
        self.encoder      = UNetEncoder_noskips(activation_func_unet, 
                                                in_channels_unet)
        
        in_channels_graph = 16*4*4*2 # the flatten image that the encoder produces
        self.graph_unet   = GraphUNet(in_channels_graph, 
                                      hidden_channels_graph, 
                                      in_channels_graph, 
                                      depth_graph, 
                                      pool_ratios_graph,
                                      sum_res_graph, 
                                      activation_func_graph)
        
        self.decoder = UNetDecoder_noskips(activation_func_unet, 
                                           in_channels_unet, 
                                           out_channels_unet, 
                                           exp_type)
        
    def forward(self, node_fts, adj_mtx):
        batch_shape = node_fts.shape
        node_fts    = node_fts.view(batch_shape[0]*batch_shape[1], batch_shape[2], batch_shape[3], batch_shape[4], batch_shape[5])
        
        # pass all the patches through the unet encoder
        batch_preds = []
        minibatch   = 256
        idx         = 0
        while True:
            index_start = idx * minibatch
            index_end   = (idx + 1) * minibatch
            idx        += 1
            if index_start >= node_fts.shape[0]: break
            if index_end > node_fts.shape[0]:   index_end = node_fts.shape[0]
            preds = self.encoder(node_fts[index_start:index_end, :, :, :, :])
            batch_preds.append(preds)
        batch_preds   = torch.cat(batch_preds, dim = 0) 
        batch_preds   = batch_preds.view(batch_shape[0], batch_shape[1], batch_preds.shape[1], batch_preds.shape[2], batch_preds.shape[3], batch_preds.shape[4])
        encoder_shape = batch_preds.shape

        # flatten the patches into 1 vector
        batch_preds = batch_preds.view(batch_preds.shape[0], batch_preds.shape[1], -1)
        
        # pass the graphs through the graph unet
        batch_preds_g = []
        adj_mtx_g     = []
        # pass through the graph unet one graph at a time
        for image in range(batch_shape[0]):
            fts, adj = self.graph_unet(batch_preds[image], adj_mtx[image])
            batch_preds_g.append(fts)
            adj_mtx_g.append(adj)
        batch_preds_g = torch.stack(batch_preds_g)
        adj_mtx_g     = torch.stack(adj_mtx_g)

        #unflatten the patches
        batch_preds_g = batch_preds_g.view(encoder_shape[0]*encoder_shape[1], encoder_shape[2], encoder_shape[3], encoder_shape[4], encoder_shape[5])
        
        # pass all the patches through the unet decoder
        outputs     = []
        idx         = 0
        while True:
            index_start = idx * minibatch
            index_end   = (idx + 1) * minibatch
            idx        += 1
            if index_start >= batch_preds_g.shape[0]: break
            if index_end > batch_preds_g.shape[0]:   index_end = batch_preds_g.shape[0]
            preds = self.decoder(batch_preds_g[index_start:index_end, :, :, :, :])
            outputs.append(preds)
        outputs = torch.cat(outputs, dim = 0)
        outputs = outputs.view(batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3], batch_shape[4], batch_shape[5])
        return outputs,  adj_mtx_g
    
# skip connections
class CombNet_v2(nn.Module):
    def __init__(self, 
                 activation_func_unet, 
                 activation_func_graph, 
                 in_channels_unet,
                 hidden_channels_graph,
                 depth_graph       = 3, 
                 pool_ratios_graph = 0.8, 
                 sum_res_graph     = False,
                 out_channels_unet = 1, 
                 exp_type          = ''):
        
        super().__init__()
        
        self.encoder      = UNetEncoder(activation_func_unet, 
                                        in_channels_unet)
        
        in_channels_graph = 16*4*4*2 # the flatten image that the encoder produces
        self.graph_unet   = GraphUNet(in_channels_graph, 
                                      hidden_channels_graph, 
                                      in_channels_graph, 
                                      depth_graph, 
                                      pool_ratios_graph,
                                      sum_res_graph, 
                                      activation_func_graph)
        
        self.decoder = UNetDecoder(activation_func_unet, 
                                   in_channels_unet, 
                                   out_channels_unet, 
                                   exp_type)
        
    def forward(self, node_fts, adj_mtx):
        batch_shape = node_fts.shape
        node_fts    = node_fts.view(batch_shape[0]*batch_shape[1], batch_shape[2], batch_shape[3], batch_shape[4], batch_shape[5])
        
        # pass all the patches through the unet encoder
        x1          = []
        x2          = []
        x3          = []
        x4          = []
        minibatch   = 256
        idx         = 0
        while True:
            index_start = idx * minibatch
            index_end   = (idx + 1) * minibatch
            idx        += 1
            if index_start >= node_fts.shape[0]: break
            if index_end > node_fts.shape[0]:   index_end = node_fts.shape[0]
            px1, px2, px3, px4 = self.encoder(node_fts[index_start:index_end, :, :, :, :])
            x1.append(px1)
            x2.append(px2)
            x3.append(px3)
            x4.append(px4)
        x1 = torch.cat(x1, dim = 0)
        x2 = torch.cat(x2, dim = 0)
        x3 = torch.cat(x3, dim = 0)
        x4 = torch.cat(x4, dim = 0) 
        x4 = x4.view(batch_shape[0], batch_shape[1], x4.shape[1], x4.shape[2], x4.shape[3], x4.shape[4])
        encoder_shape = x4.shape

        # flatten the patches into 1 vector
        x4 = x4.view(x4.shape[0], x4.shape[1], -1)
        
        # -- only for debug
        #assert torch.isnan(x1).any() == False, f'Inside forward x1 gunet got nan'
        #assert torch.isnan(x2).any() == False, f'Inside forward x2 gunet got nan'
        #assert torch.isnan(x3).any() == False, f'Inside forward x3 gunet got nan'
        #assert torch.isnan(x4).any() == False, f'Inside forward x4 gunet got nan'
        
        #assert torch.isinf(x1).any() == False, f'Inside forward x1 gunet got inf'
        #assert torch.isinf(x2).any() == False, f'Inside forward x2 gunet got inf'
        #assert torch.isinf(x3).any() == False, f'Inside forward x3 gunet got inf'
        #assert torch.isinf(x4).any() == False, f'Inside forward x4 gunet got inf'
        
        #assert (x1 == 0).all()       == False, f'Inside forward x1 gunet got only zeros'
        #assert (x2 == 0).all()       == False, f'Inside forward x2 gunet got only zeros'
        #assert (x3 == 0).all()       == False, f'Inside forward x3 gunet got only zeros'
        #assert (x4 == 0).all()       == False, f'Inside forward x4 gunet got only zeros'
        # only for debug ---

        # pass the graphs through the graph unet
        batch_preds_g = []
        adj_mtx_g     = []
        # pass through the graph unet one graph at a time
        for image in range(batch_shape[0]):
            fts, adj = self.graph_unet(x4[image], adj_mtx[image])
            batch_preds_g.append(fts)
            adj_mtx_g.append(adj)
        batch_preds_g = torch.stack(batch_preds_g)
        adj_mtx_g     = torch.stack(adj_mtx_g)

        #unflatten the patches
        batch_preds_g = batch_preds_g.view(encoder_shape[0]*encoder_shape[1], encoder_shape[2], encoder_shape[3], encoder_shape[4], encoder_shape[5])
        
        # -- only for debug
        #assert torch.isnan(batch_preds_g).any() == False, f'Inside forward batch_preds_g gunet got nan'
        #assert torch.isinf(batch_preds_g).any() == False, f'Inside forward batch_preds_g gunet got inf'
        #assert (batch_preds_g == 0).all()       == False, f'Inside forward batch_preds_g gunet got only zeros'
        # only for debug ---

        # pass all the patches through the unet decoder
        outputs     = []
        idx         = 0
        while True:
            index_start = idx * minibatch
            index_end   = (idx + 1) * minibatch
            idx        += 1
            if index_start >= batch_preds_g.shape[0]: break
            if index_end > batch_preds_g.shape[0]:   index_end = batch_preds_g.shape[0]
            preds = self.decoder(x1[index_start:index_end, :, :, :, :],
                                 x2[index_start:index_end, :, :, :, :],
                                 x3[index_start:index_end, :, :, :, :],
                                 batch_preds_g[index_start:index_end, :, :, :, :])
            outputs.append(preds)
        outputs = torch.cat(outputs, dim = 0)
        outputs = outputs.view(batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3], batch_shape[4], batch_shape[5])

        # -- only for debug
        #assert torch.isnan(outputs).any() == False, f'Inside forward outputs gunet got nan'
        #assert torch.isinf(outputs).any() == False, f'Inside forward outputs gunet got inf'
        #assert (outputs == 0).all()       == False, f'Inside forward outputs gunet got only zeros'
        # only for debug ---

        return outputs,  adj_mtx_g

class CombNet_v3(nn.Module):
    def __init__(self, 
                 activation_func_unet, 
                 activation_func_graph, 
                 in_channels_unet,
                 hidden_channels_graph,
                 depth_graph_enc   = 2, 
                 depth_graph_dec   = 2,
                 out_channels_unet = 1, 
                 exp_type          = ''):
        
        super().__init__()
        
        self.encoder      = UNetEncoder(activation_func_unet, 
                                        in_channels_unet)
        
        in_channels_graph = 16*4*4*2 # the flatten image that the encoder produces

        if depth_graph_dec == 0:
            self.gae_encoder = GAE_Encoder(in_channels_graph, 
                                           hidden_channels_graph, 
                                           in_channels_graph, 
                                           depth_graph_enc, 
                                           activation_func_graph)
            
            self.gae_decoder = GAE_Decoder(in_channels_graph, 
                                            in_channels_graph, 
                                            in_channels_graph, 
                                            depth_graph_dec, 
                                            activation_func_graph)
        else:
            self.gae_encoder = GAE_Encoder(in_channels_graph, 
                                           hidden_channels_graph, 
                                           hidden_channels_graph, 
                                           depth_graph_enc, 
                                           activation_func_graph)
        
            self.gae_decoder = GAE_Decoder(hidden_channels_graph, 
                                            in_channels_graph, 
                                            in_channels_graph, 
                                            depth_graph_dec, 
                                            activation_func_graph)

        self.decoder = UNetDecoder(activation_func_unet, 
                                   in_channels_unet, 
                                   out_channels_unet, 
                                   exp_type)
        
    def forward(self, node_fts, adj_mtx):
        
        batch_shape = node_fts.shape
        node_fts    = node_fts.view(batch_shape[0]*batch_shape[1], batch_shape[2], batch_shape[3], batch_shape[4], batch_shape[5])
        
        # pass all the patches through the unet encoder
        x1          = []
        x2          = []
        x3          = []
        x4          = []
        minibatch   = 256
        idx         = 0
        while True:
            index_start = idx * minibatch
            index_end   = (idx + 1) * minibatch
            idx        += 1
            if index_start >= node_fts.shape[0]: break
            if index_end > node_fts.shape[0]:   index_end = node_fts.shape[0]
            px1, px2, px3, px4 = self.encoder(node_fts[index_start:index_end, :, :, :, :])
            x1.append(px1)
            x2.append(px2)
            x3.append(px3)
            x4.append(px4)
        x1 = torch.cat(x1, dim = 0)
        x2 = torch.cat(x2, dim = 0)
        x3 = torch.cat(x3, dim = 0)
        x4 = torch.cat(x4, dim = 0)

        x4 = x4.view(batch_shape[0], batch_shape[1], x4.shape[1], x4.shape[2], x4.shape[3], x4.shape[4])
        encoder_shape = x4.shape

        # flatten the patches into 1 vector
        x4 = x4.view(x4.shape[0], x4.shape[1], -1)
        
        # pass the graphs through the graph autoencoder
        batch_preds_g = []
        adj_mtx_g     = []
        adj_weights_g = []
        # pass through the graph autoencoder one graph at a time
        for image in range(batch_shape[0]):
            fts                   = self.gae_encoder(x4[image], adj_mtx[image])
            fts, adj, adj_weights = self.gae_decoder(fts, adj_mtx[image])
            batch_preds_g.append(fts)
            adj_mtx_g.append(adj)
            adj_weights_g.append(adj_weights)

        batch_preds_g = torch.stack(batch_preds_g)
        adj_mtx_g     = torch.stack(adj_mtx_g)
        adj_weights_g = torch.stack(adj_weights_g)

        #unflatten the patches
        batch_preds_g = batch_preds_g.view(encoder_shape[0]*encoder_shape[1], encoder_shape[2], encoder_shape[3], encoder_shape[4], encoder_shape[5])
        
        # pass all the patches through the unet decoder
        outputs     = []
        idx         = 0
        while True:
            index_start = idx * minibatch
            index_end   = (idx + 1) * minibatch
            idx        += 1
            if index_start >= batch_preds_g.shape[0]: break
            if index_end > batch_preds_g.shape[0]:   index_end = batch_preds_g.shape[0]
            preds = self.decoder(x1[index_start:index_end, :, :, :, :],
                                 x2[index_start:index_end, :, :, :, :],
                                 x3[index_start:index_end, :, :, :, :],
                                 batch_preds_g[index_start:index_end, :, :, :, :])
            outputs.append(preds)
        outputs = torch.cat(outputs, dim = 0)
        outputs = outputs.view(batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3], batch_shape[4], batch_shape[5])
        return outputs,  adj_mtx_g, adj_weights_g

class CombNet_v4(nn.Module):
    def __init__(self, 
                 activation_func_unet, 
                 activation_func_graph, 
                 in_channels_unet,
                 hidden_channels_graph,
                 depth_graph_enc   = 2, 
                 out_channels_unet = 1, 
                 exp_type          = ''):
        
        super().__init__()
        
        self.encoder      = UNetEncoder(activation_func_unet, 
                                        in_channels_unet)
        
        in_channels_graph = 16*4*4*2 # the flatten image that the encoder produces
        self.gae_encoder = GAE_Encoder(in_channels_graph, 
                                       hidden_channels_graph, 
                                       in_channels_graph, 
                                       depth_graph_enc, 
                                       activation_func_graph)
        
        self.gae_decoder = InnerProductDecoder()

        self.decoder     = UNetDecoder(activation_func_unet, 
                                       in_channels_unet, 
                                       out_channels_unet, 
                                       exp_type)
        
    def forward(self, node_fts, adj_mtx):
        
        batch_shape = node_fts.shape
        node_fts    = node_fts.view(batch_shape[0]*batch_shape[1], batch_shape[2], batch_shape[3], batch_shape[4], batch_shape[5])
        
        # pass all the patches through the unet encoder
        x1          = []
        x2          = []
        x3          = []
        x4          = []
        minibatch   = 256
        idx         = 0
        while True:
            index_start = idx * minibatch
            index_end   = (idx + 1) * minibatch
            idx        += 1
            if index_start >= node_fts.shape[0]: break
            if index_end > node_fts.shape[0]:   index_end = node_fts.shape[0]
            px1, px2, px3, px4 = self.encoder(node_fts[index_start:index_end, :, :, :, :])
            x1.append(px1)
            x2.append(px2)
            x3.append(px3)
            x4.append(px4)
        x1 = torch.cat(x1, dim = 0)
        x2 = torch.cat(x2, dim = 0)
        x3 = torch.cat(x3, dim = 0)
        x4 = torch.cat(x4, dim = 0)

        x4 = x4.view(batch_shape[0], batch_shape[1], x4.shape[1], x4.shape[2], x4.shape[3], x4.shape[4])
        encoder_shape = x4.shape

        # flatten the patches into 1 vector
        x4 = x4.view(x4.shape[0], x4.shape[1], -1)
        
        # pass the graphs through the graph autoencoder
        batch_preds_g = []
        adj_mtx_g     = []
        adj_weights_g = []
        # pass through the graph autoencoder one graph at a time
        for image in range(batch_shape[0]):
            fts              = self.gae_encoder(x4[image], adj_mtx[image])
            adj, adj_weights = self.gae_decoder.forward_all(fts)
            batch_preds_g.append(fts)
            adj_mtx_g.append(adj)
            adj_weights_g.append(adj_weights)

        batch_preds_g = torch.stack(batch_preds_g)
        adj_mtx_g     = torch.stack(adj_mtx_g)
        adj_weights_g = torch.stack(adj_weights_g)

        #unflatten the patches
        batch_preds_g = batch_preds_g.view(encoder_shape[0]*encoder_shape[1], encoder_shape[2], encoder_shape[3], encoder_shape[4], encoder_shape[5])
        
        # pass all the patches through the unet decoder
        outputs     = []
        idx         = 0
        while True:
            index_start = idx * minibatch
            index_end   = (idx + 1) * minibatch
            idx        += 1
            if index_start >= batch_preds_g.shape[0]: break
            if index_end > batch_preds_g.shape[0]:   index_end = batch_preds_g.shape[0]
            preds = self.decoder(x1[index_start:index_end, :, :, :, :],
                                 x2[index_start:index_end, :, :, :, :],
                                 x3[index_start:index_end, :, :, :, :],
                                 batch_preds_g[index_start:index_end, :, :, :, :])
            outputs.append(preds)
        outputs = torch.cat(outputs, dim = 0)
        outputs = outputs.view(batch_shape[0], batch_shape[1], -1, batch_shape[3], batch_shape[4], batch_shape[5])
        return outputs,  adj_mtx_g, adj_weights_g

class CombNet_v5(nn.Module):
    def __init__(self, 
                 activation_func_unet, 
                 activation_func_graph, 
                 in_channels_unet,
                 hidden_channels_graph,
                 depth_graph_enc   = 2, 
                 depth_graph_dec   = 0,
                 out_channels_unet = 1, 
                 exp_type          = ''):
        
        super().__init__()
        
        self.encoder      = UNetEncoder(activation_func_unet, 
                                           in_channels_unet)
        
        in_channels_graph = 16 * 4 * 4 * 2# the flatten image that the encoder produces
        self.gae_encoder  = GAE_Encoder(in_channels_graph, 
                                        hidden_channels_graph, 
                                        in_channels_graph, 
                                        depth_graph_enc, 
                                        activation_func_graph)
        
        self.gae_decoder = InnerProductDecoder()
        '''
        self.gae_decoder = GAE_Decoder(hidden_channels_graph, 
                                       in_channels_graph, 
                                       in_channels_graph, 
                                       depth_graph_dec, 
                                       activation_func_graph)
        '''

        self.decoder = UNetDecoder(activation_func_unet, 
                                   in_channels_unet, 
                                   out_channels_unet, 
                                   exp_type)
        
    def forward(self, image, adj_mtx):
        x1, x2, x3, x4 = self.encoder(image)

        # --- patch the final image, and keep the correspondance to the adjacency matrix
        s1 = x4.size()
        x4 = x4.unfold(2, 4, 4).unfold(3, 4, 4).unfold(4, 2, 2).permute(0,4,3,2,1,5,6,7)
        s2 = x4.size()
        x4 = x4.reshape(x4.shape[0], x4.shape[1], x4.shape[2], x4.shape[3], -1)
        s3 = x4.size()
        x4 = x4.reshape(x4.shape[0], -1, x4.shape[-1])

        batch_preds_g = []
        adj_mtx_g     = []
        adj_weights_g = []
        # pass through the graph autoencoder one graph at a time
        for image in range(x4.shape[0]):
            fts              = self.gae_encoder(x4[image], adj_mtx[image])
            adj, adj_weights = self.gae_decoder.forward_all(fts)
            batch_preds_g.append(fts)
            adj_mtx_g.append(adj)
            adj_weights_g.append(adj_weights)
        batch_preds_g = torch.stack(batch_preds_g)
        adj_mtx_g     = torch.stack(adj_mtx_g)
        adj_weights_g = torch.stack(adj_weights_g)
        
        # --- undo the patching after the graph autoencoder to pass the result through the unet decoder
        # step 1: restore the patches dimensions in the 3D space 
        batch_preds_g = batch_preds_g.reshape(s3)
        # step 2: unflatten the patches and move the channels to the correct dimension
        # step 3: undo the unfolding
        batch_preds_g = batch_preds_g.reshape(s2).permute(0,4,3,5,2,6,1,7).reshape(s1).contiguous()

        outputs = self.decoder(x1, x2, x3, batch_preds_g)
        return outputs, adj_mtx_g, adj_weights_g