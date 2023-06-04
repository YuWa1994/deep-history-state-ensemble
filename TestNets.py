# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 11:23:17 2021

@author: yuwa
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from tqdm import trange


class BPNN(nn.Module):
    def __init__(self, n_class):
        
        super(BPNN, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_features=1024,out_features=512), 
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512,out_features=256), 
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256,out_features=128), 
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128,out_features=n_class),
            nn.Softmax(dim=1)
            )

    def forward(self, x):
        out = self.nn(x)
        return out   

class _Autoencoder(nn.Module):
    def __init__(self, in_layer=1024, hid_layer=512):  
        super(_Autoencoder, self).__init__()
        self.ae1 = nn.Sequential(
            nn.Linear(in_features=in_layer,out_features=hid_layer), 
            nn.ReLU(inplace=True),
            )
        self.ae2 = nn.Sequential(
            nn.Linear(in_features=hid_layer,out_features=in_layer), 
            #nn.Sigmoid()
            )        
    def forward(self, x):
        hid = self.ae1(x)
        out = self.ae2(hid)
        return hid, out   


class SAE(nn.Module):
    def __init__(self, layers=[1024,512,256,128],
                 epoch=30, batch_size=50,
                 learn_rate=0.0001, 
                 optimizer='Adam'):
        self.layers = layers
        self.learn_rate = learn_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        '''
        self.AE_layers = []
        for i in range(len(self.layers)-1):
            AE = _Autoencoder(in_layer=self.layers[i], 
                              hid_layer=self.layers[i+1])
            self.AE_layers.append(AE)
        '''
    def get_para(self, model_dict):
        weight = model_dict['ae1.0.weight']
        bias = model_dict['ae1.0.bias']
        return weight, bias
        
    def train(self, x):
        x = torch.from_numpy(x).to(self.device) 
        result = {}
        result['hid'], result['hid'] = [], []
        result['weight'], result['bias'] = [], []
        for i in range(len(self.layers)-1):
            model = _Autoencoder(in_layer=self.layers[i], hid_layer=self.layers[i+1])
            model = model.to(self.device) 
            loss_func = nn.MSELoss()
            optim = eval('torch.optim.{}(model.parameters(),lr={})'
                .format(self.optimizer, self.learn_rate))
            
            ###### Train Autoencoder #####
            print('SAE-layer{}'.format(i+1))
            for epoch in range(self.epoch):
                indices = torch.randperm(x.shape[0])
                indices = torch.split(indices, self.batch_size)    
                for batch in range(len(indices)):
                    batch_x = Variable(x[indices[batch]]).float()
                    hid_, out_ = model(batch_x)     
                    loss = loss_func(out_, batch_x)
                    
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                print('epoch [{}/{}], loss:{:.4f}'
                .format(epoch + 1, self.epoch,loss.cpu().detach().numpy())) 
            x, x_rec = model(x.float())        
            weight, bias = self.get_para(model.state_dict())
            result['weight'].append(weight)
            result['bias'].append(bias)
        return result


class LeNet5_1D(nn.Module):
    def __init__(self, n_class):
        
        super(LeNet5_1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=125, stride=1),
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=12, stride=2),
            nn.BatchNorm1d(16),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=2),
            nn.ReLU(True), 
            nn.AvgPool1d(kernel_size=5, stride=5),
            nn.BatchNorm1d(32),
            )
        self.fc = nn.Sequential(
            nn.Linear(in_features=1408,out_features=n_class), 
            nn.Softmax(dim=1)
            )

    def forward(self, x):
        if len(x.size()) < 3:
            x = x.view(x.shape[0], 1, -1)         
        out = self.conv(x)
        out = torch.flatten(out, 1)
        #out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out   

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average=nn.AvgPool1d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2)) 
        else:
            self.average=nn.AvgPool1d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
    
    
    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)#这里的1.0即为bias
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x
    
class AlexNet_1D(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=125, stride=1),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(8),
            nn.AvgPool1d(kernel_size=12, stride=2), # 445
            #LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
            nn.BatchNorm1d(16)
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, 
                      groups=1, padding=3),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(16),
            nn.AvgPool1d(kernel_size=7, stride=2), # 220
            nn.BatchNorm1d(16)
            #LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(32),
            nn.AvgPool1d(kernel_size=5, stride=5),
            nn.BatchNorm1d(32)
            #LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True),            
            )
        
        self.FC = nn.Sequential(
            nn.Linear(in_features=1408, out_features=n_class),
            #nn.ReLU(inplace=True),
            #nn.Linear(in_features=1024, out_features=1024),
            #nn.ReLU(inplace=True),            
            #nn.Linear(in_features=1024, out_features=n_class),
            nn.Softmax(dim=1)
            )
        
    def forward(self, x):
        if len(x.size()) < 3:
            x = x.view(x.shape[0], 1, -1)         
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        #f = out.size()[1]*out.size()[2]
        #out = out.view(-1, f)
        out = torch.flatten(out, 1)
        out = self.FC(out)
        
        return out  


class _ResLayer(nn.Module):
    def __init__(self, planes, ks=15):
        super(_ResLayer, self).__init__()
        pad  = int((ks-1)/2)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=planes, 
                      out_channels=planes, 
                      kernel_size=ks, 
                      stride=1, 
                      padding=pad),
            nn.ReLU(True),
           
            nn.Conv1d(in_channels=planes, 
                      out_channels=planes, 
                      kernel_size=ks, 
                      stride=1,
                      padding=pad)
            )        
        
        self.relu = nn.ReLU(True)
        self.BN = nn.BatchNorm1d(planes)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        out = self.relu(out)
        #out = self.BN(out)
        return out

class ResNet_1D(nn.Module):
    def __init__(self, n_class=10, 
                 block_num=[2,2,2,2], 
                 expansion=1):
        
        super(ResNet_1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=125, 
                      stride=1, padding=0),      
            #nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=12, stride=2),       
            nn.BatchNorm1d(16), #445
            )   

        self.ResBlock_1 = self._make_ResBlock(_ResLayer(16, ks=5), block_num[0])
        self.conv2 = self._conv_block(16, 16, ks=5, s=2, pad=0) #221

        self.ResBlock_2 = self._make_ResBlock(_ResLayer(16, ks=5), block_num[1])
        self.conv3 = self._conv_block(16, 32, ks=5, s=2, pad=0) #109

        self.ResBlock_3 = self._make_ResBlock(_ResLayer(32, ks=5), block_num[2])
        self.conv4 = self._conv_block(32, 64, ks=5, s=2, pad=0) #53

        self.ResBlock_4 = self._make_ResBlock(_ResLayer(64, ks=5), block_num[3])
        self.conv5 = self._conv_block(64, 64, ks=5, s=2, pad=0) #25
        
        self.pool = nn.AvgPool1d(kernel_size=4, stride=1, padding=0) 
        self.BN = nn.BatchNorm1d(64)
        
        self.FC = nn.Sequential(
            nn.Linear(in_features=1408, out_features=n_class),
            #nn.ReLU(inplace=True),
            #nn.Linear(in_features=1024, out_features=1024),
            #nn.ReLU(inplace=True),            
            #nn.Linear(in_features=1024, out_features=n_class),
            nn.Softmax(dim=1)
            )

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        

    def _conv_block(self, inplanes, planes, ks=7, s=1, pad=0):     
         return nn.Sequential(
            nn.Conv1d(in_channels=inplanes, 
                      out_channels=planes, 
                      kernel_size=ks, 
                      stride=s, 
                      padding=pad),
            nn.ReLU(True),
            )             
        
    def _make_ResBlock(self, block, block_num):
        layers = []
        layers.append(block)
        if block_num > 1:
            for i in range(1, block_num):
                layers.append(block)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if len(x.size()) < 3:
            x = x.view(x.shape[0], 1, -1)         
        out = self.conv1(x)

        out = self.ResBlock_1(out)
        out = self.conv2(out)
        out = self.ResBlock_2(out)
        out = self.conv3(out)
        out = self.ResBlock_3(out)
        out = self.conv4(out)
        out = self.ResBlock_4(out)
        out = self.conv5(out)        
        out = self.pool(out)
        out = self.BN(out)

        out = torch.flatten(out, 1)
        out = self.FC(out)
        return out

class _SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        super(_SelfAttention, self).__init__()
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(Tensor([0.]))
        
    def _conv(self,n_in,n_out):
        return nn.Conv1d(in_channels=n_in, 
                      out_channels=n_out, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0)

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()
        
class _ChannelAttention(nn.Module):
    "Channel attention layer for `n_channels`."
    def __init__(self, n_feature, n_channels):
        super(_ChannelAttention, self).__init__()
        self.q = nn.parameter(torch.rand(n_feature, n_channels))
        self.k = nn.parameter(torch.rand(n_feature, n_channels))
        self.v = nn.parameter(torch.rand(n_feature, n_channels))

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()
        
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, 
                 growth_rate, 
                 bn_size, 
                 drop_rate, 
                 memory_efficient=False):
        super(_DenseLayer, self).__init__()
        
        self.Bottleneck = nn.Sequential(
            nn.Conv1d(num_input_features, 
                      bn_size *growth_rate, 
                      kernel_size=1, 
                      stride=1,    
                      bias=False),
            #nn.BatchNorm1d(bn_size * growth_rate),
            nn.ReLU(inplace=True),   
            )    
        self.conv = nn.Sequential(
            nn.Conv1d(bn_size * growth_rate, 
                      growth_rate,
                      kernel_size=7, 
                      stride=1, 
                      padding=3,  
                      bias=False),
            #nn.BatchNorm1d(growth_rate),
            nn.ReLU(inplace=True),      
            )  
        
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input
            
        prev_features = torch.cat(prev_features, 1)
        out = self.Bottleneck(prev_features) 
        new_features = self.conv(out)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features        

class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, 
                 num_input_features, 
                 bn_size, 
                 growth_rate, 
                 drop_rate, 
                 memory_efficient=False):   
        super(_DenseBlock, self).__init__()
        
        self.num_features = num_input_features + num_layers * growth_rate
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient)
            self.add_module('denselayer%d' % (i + 1), layer)       
                     
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet_1D(nn.Module):
    def __init__(self, n_classes, 
                 growth_rate=32, 
                 block_config=(2, 2, 2, 2),
                 num_init_features=16, 
                 bn_size=2, 
                 drop_rate=0, 
                 memory_efficient=False,
                 attention=False):
        
        super(DenseNet_1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=125, 
                      stride=1,
                      padding=0, 
                      bias=True), 
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=12, stride=2, padding=0),
            nn.BatchNorm1d(16))   # 445
        
        self.num_features = num_init_features
        self.DenseBlock1 = _DenseBlock(
            num_layers=2,
            num_input_features=self.num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient)        
        self.TransLayer1 = self._Transition(
            num_input_features=self.DenseBlock1.num_features,
            num_output_features=self.DenseBlock1.num_features // 2)

        self.DenseBlock2 = _DenseBlock(
            num_layers=2,
            num_input_features=self.num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient)        
        self.TransLayer2 = self._Transition(
            num_input_features=self.DenseBlock2.num_features,
            num_output_features=self.DenseBlock2.num_features // 2)

        self.DenseBlock3 = _DenseBlock(
            num_layers=2,
            num_input_features=self.num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient)        
        self.TransLayer3 = self._Transition(
            num_input_features=self.DenseBlock3.num_features,
            num_output_features=self.DenseBlock3.num_features // 2)

        self.DenseBlock4 = _DenseBlock(
            num_layers=2,
            num_input_features=self.num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient)  
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.DenseBlock4.num_features, 
                      out_channels=64, 
                      kernel_size=5, 
                      stride=2, 
                      padding=0),
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=4, stride=1),
            #nn.BatchNorm1d(64)
            )    
        
        # Linear layer
        self.classifier = nn.Linear(1408, n_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                
    
    def _Transition(self, num_input_features, num_output_features):
        self.num_features = num_input_features // 2
        layer =  nn.Sequential(
            nn.Conv1d(num_input_features, 
                      num_output_features,
                      kernel_size=1, 
                      stride=1, 
                      padding=0, 
                      bias=False),
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=5, stride=2),
            )              
        return layer      

    def forward(self, x):
        if len(x.size()) < 3:
            x = x.view(x.shape[0], 1, -1)         
        out = self.conv1(x)
        
        out = self.DenseBlock1(out)
        out = self.TransLayer1(out)
        out = self.DenseBlock2(out)
        out = self.TransLayer2(out)
        out = self.DenseBlock3(out)
        out = self.TransLayer3(out)
        out = self.DenseBlock4(out)
        
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out    

class RBM:
    def __init__(self, n_visible, n_hidden, 
              lr=0.001, epochs=50, mode='gb', batch_size=50, k=10, 
              optimizer='adam', 
              early_stopping_patience=5):
        self.mode = mode # bernoulli or gaussian RBM
        self.n_hidden = n_hidden #  Number of hidden nodes
        self.n_visible = n_visible # Number of visible nodes
        self.lr = lr # Learning rate for the CD algorithm
        self.epochs = epochs # Number of iterations to run the algorithm for
        self.batch_size = batch_size
        self.k = k
        self.optimizer = optimizer
        self.beta_1=0.9
        self.beta_2=0.999
        self.epsilon=1e-07
        self.m = [0, 0, 0]
        self.v = [0, 0, 0]
        self.m_batches = {0:[], 1:[], 2:[]}
        self.v_batches = {0:[], 1:[], 2:[]}
        self.early_stopping_patience = early_stopping_patience
        self.stagnation = 0
        self.previous_loss_before_stagnation = 0
        self.progress = []

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		# Initialize weights and biases
        std = 4 * np.sqrt(6. / (self.n_visible + self.n_hidden))
        self.W = torch.normal(mean=0, std=std, size=(self.n_hidden, self.n_visible), dtype=torch.float64)
        self.vb = torch.zeros(size=(1, self.n_visible), dtype=torch.float64)
        self.hb = torch.zeros(size=(1, self.n_hidden), dtype=torch.float64)

        self.W = self.W.to(self.device)
        self.vb = self.vb.to(self.device)
        self.hb = self.hb.to(self.device)
		
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.hb
        p_h_given_v = torch.sigmoid(activation)
        if self.mode == 'bernoulli':
            return p_h_given_v, torch.bernoulli(p_h_given_v)
        elif self.mode == 'gb':
            return p_h_given_v, torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape,
                                                                    device=self.device))

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.vb
        p_v_given_h =torch.sigmoid(activation)
        if self.mode == 'bernoulli':
            return p_v_given_h, torch.bernoulli(p_v_given_h)
        elif self.mode == 'gb':
            return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape,
                                                                    device=self.device))
	
    def adam(self, g, epoch, index):
        self.m[index] = self.beta_1 * self.m[index] + (1 - self.beta_1) * g
        self.v[index] = self.beta_2 * self.v[index] + (1 - self.beta_2) * torch.pow(g, 2)

        m_hat = self.m[index] / (1 - np.power(self.beta_1, epoch)) + (1 - self.beta_1) * g / (1 - np.power(self.beta_1, epoch))
        v_hat = self.v[index] / (1 - np.power(self.beta_2, epoch))
        return m_hat / (torch.sqrt(v_hat) + self.epsilon)

    def update(self, v0, vk, ph0, phk, epoch):
        dW = (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        dvb = torch.sum((v0 - vk), 0)
        dhb = torch.sum((ph0 - phk), 0)

        if self.optimizer == 'adam':
            dW = self.adam(dW, epoch, 0)
            dvb = self.adam(dvb, epoch, 1)
            dhb = self.adam(dhb, epoch, 2)

        self.W += self.lr * dW
        self.vb += self.lr * dvb
        self.hb += self.lr * dhb
        
    def train(self, x):
        if torch.is_tensor(x)==False:
            x = torch.from_numpy(x).to(self.device)
    
        learning = trange(self.epochs, desc=str('Starting...'))
        for epoch in learning:
            train_loss = 0
            counter = 0
            for batch_start_index in range(0, x.shape[0]-self.batch_size, self.batch_size):
                vk = x[batch_start_index:batch_start_index+self.batch_size]
                v0 = x[batch_start_index:batch_start_index+self.batch_size]
                ph0, _ = self.sample_h(v0)
    
                for k in range(self.k):
                    _, hk = self.sample_h(vk)
                    _, vk = self.sample_v(hk)
                    phk, _ = self.sample_h(vk)
                self.update(v0, vk, ph0, phk, epoch+1)
                train_loss += torch.mean(torch.abs(v0-vk))
                counter += 1
			
            self.progress.append(train_loss.item()/counter)
            details = {'epoch': epoch+1, 'loss': round(train_loss.item()/counter, 4)}
            learning.set_description(str(details))
            learning.refresh()
			
            if train_loss.item()/counter > self.previous_loss_before_stagnation and epoch>self.early_stopping_patience+1:
                self.stagnation += 1
                if self.stagnation == self.early_stopping_patience-1:
                    learning.close()
                    print("Not Improving the stopping training loop.")
                    break
                else:
                    self.previous_loss_before_stagnation = train_loss.item()/counter
                    self.stagnation = 0
        learning.close()                

class DBN:
    def __init__(self, layers=[1024,512,256,128], 
              mode='gb', k=1):
        self.layers = layers
        self.layer_parameters = [{'W':None, 'hb':None, 'vb':None} 
               for _ in range(len(layers))]
        self.k = k
        self.mode = mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.rbm_layers = []
        for i in range(len(self.layers)-1):
            rbm = RBM(self.layers[i], self.layers[i+1], 
                      epochs=30, 
                      mode=self.mode, 
                      lr=0.0001, k=1, batch_size=50, 
                      optimizer='adam', 
                      early_stopping_patience=10)
            self.rbm_layers.append(rbm)
    
    def sample_v(self, y, W, vb):
   		wy = torch.mm(y, W)
   		activation = wy + vb
   		p_v_given_h =torch.sigmoid(activation)
   		if self.mode == 'bernoulli':
   			return p_v_given_h, torch.bernoulli(p_v_given_h)
   		elif self.mode == 'gb':
   			return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape,
                                                              device=self.device))

    def sample_h(self, x, W, hb):
        wx = torch.mm(x, W.t())
        activation = wx + hb
        p_h_given_v = torch.sigmoid(activation)
        if self.mode == 'bernoulli':
            return p_h_given_v, torch.bernoulli(p_h_given_v)
        elif self.mode == 'gb':
            return p_h_given_v, torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape,
                                                                    device=self.device))

    def generate_input_for_layer(self, index, x):
        if index>0:
            x_gen = []
            for _ in range(self.k):
                x_dash = x.to(self.device)
                for i in range(index):
                    _, x_dash = self.sample_h(x_dash, 
                                              self.layer_parameters[i]['W'], 
                                              self.layer_parameters[i]['hb'])
                x_gen.append(x_dash)
            x_dash = torch.stack(x_gen)
            x_dash = torch.mean(x_dash, dim=0)
        else:
            x_dash = x
        return x_dash

    def train(self, x):
        result = {}
        result['weight']=[]
        result['bias']=[]
        
        x = torch.from_numpy(x).to(self.device)
        for index in range(len(self.layers)-1):
            print('DBN-layer{}'.format(index+1))
            rbm = self.rbm_layers[index]
            x_dash = self.generate_input_for_layer(index, x)
            rbm.train(x_dash)
            self.layer_parameters[index]['W'] = rbm.W
            self.layer_parameters[index]['hb'] = rbm.hb
            self.layer_parameters[index]['vb'] = rbm.vb
            
            result['weight'].append(self.layer_parameters[index]['W'])
            result['bias'].append(self.layer_parameters[index]['hb'])
            
        return result
'''
	def reconstructor(self, x):
		x_gen = []
		for _ in range(self.k):
			x_dash = x.clone()
			for i in range(len(self.layer_parameters)):
				_, x_dash = self.sample_h(x_dash, self.layer_parameters[i]['W'], self.layer_parameters[i]['hb'])
			x_gen.append(x_dash)
		x_dash = torch.stack(x_gen)
		x_dash = torch.mean(x_dash, dim=0)

		y = x_dash

		y_gen = []
		for _ in range(self.k):
			y_dash = y.clone()
			for i in range(len(self.layer_parameters)):
				i = len(self.layer_parameters)-1-i
				_, y_dash = self.sample_v(y_dash, self.layer_parameters[i]['W'], self.layer_parameters[i]['vb'])
			y_gen.append(y_dash)
		y_dash = torch.stack(y_gen)
		y_dash = torch.mean(y_dash, dim=0)

		return y_dash, x_dash

	def initialize_model(self):
		print("The Last layer will not be activated. The rest are activated using the Sigoid Function")
		modules = []
		for index, layer in enumerate(self.layer_parameters):
			modules.append(torch.nn.Linear(layer['W'].shape[1], layer['W'].shape[0]))
			if index < len(self.layer_parameters) - 1:
				modules.append(torch.nn.Sigmoid())
		model = torch.nn.Sequential(*modules)

		for layer_no, layer in enumerate(model):
			if layer_no//2 == len(self.layer_parameters)-1:
				break
			if layer_no%2 == 0:
				model[layer_no].weight = torch.nn.Parameter(self.layer_parameters[layer_no//2]['W'])
				model[layer_no].bias = torch.nn.Parameter(self.layer_parameters[layer_no//2]['hb'])

		return model
'''
        
    
class DBM(nn.Module):
    def __init__(self,
                 n_visible=1024,
                 layers=[1024,512,256,128],
                 mode = 'gb'):
        super(DBM, self).__init__()
        self.layers = layers
        self.mode = mode
        self.n_layers = len(layers)-1
        self.n_odd_layers = int((self.n_layers+1)/2)
        self.n_even_layers = int((self.n_layers+2)/2)

        self.rbm_layers = []
        for i in range(self.n_layers):
            rbm = RBM(self.layers[i], self.layers[i+1], 
                      epochs=50, 
                      mode=self.mode, 
                      lr=0.001, k=10, batch_size=50, 
                      optimizer='adam', 
                      early_stopping_patience=10)
            self.rbm_layers.append(rbm)
        
        '''
        self.rbm_layers = []
        for i in range(self.n_layers):
            if i == 0:
                input_size = n_visible
            else:
                input_size = n_hidden[i-1]
            rbm = RBM(n_visible = input_size, n_hidden = n_hidden[i])
            self.rbm_layers.append(rbm)
        '''
        self.W = [nn.Parameter(self.rbm_layers[i].W.data) 
                  for i in range(self.n_layers)]
        self.bias = ([nn.Parameter(self.rbm_layers[0].vb.data)]
                     +[nn.Parameter(self.rbm_layers[i].hb.data) 
                     for i in range(self.n_layers)])
        
        for i in range(self.n_layers):
            self.register_parameter('W%i'%i, self.W[i])
        for i in range(self.n_layers+1):
            self.register_parameter('bias%i'%i, self.bias[i])
        
    def odd_to_even(self, odd_input = None):
        even_p_output= []
        even_output= []
        for i in range(self.n_even_layers):
            if i == 0:
                even_p_output.append(torch.sigmoid(F.linear(odd_input[i],self.W[2*i].t(),self.bias[2*i])))
            elif (self.n_even_layers > self.n_odd_layers) and i == self.n_even_layers - 1:
                even_p_output.append(torch.sigmoid(F.linear(odd_input[i-1],self.W[2*i-1],self.bias[2*i])))
            else:
                even_p_output.append(torch.sigmoid(F.linear(odd_input[i-1],self.W[2*i-1],self.bias[2*i]) + F.linear(odd_input[i],self.W[2*i].t())))
        for i in even_p_output:
            even_output.append(torch.bernoulli(i))
            
        return even_p_output, even_output
    
    def even_to_odd(self, even_input = None):
        odd_p_output = [] 
        odd_output = []
        for i in range(self.n_odd_layers):
            if (self.n_even_layers == self.n_odd_layers) and i == self.n_odd_layers - 1:
                odd_p_output.append(torch.sigmoid(F.linear(even_input[i],self.W[2*i],self.bias[2*i+1])))
            else:
                odd_p_output.append(torch.sigmoid(F.linear(even_input[i],self.W[2*i],self.bias[2*i+1]) + F.linear(even_input[i+1],self.W[2*i+1].t())))
        
        for i in odd_p_output:
            odd_output.append(torch.bernoulli(i))
            
        return odd_p_output, odd_output
    
    def train(self, v_input, k_positive = 10, k_negative=10, greedy = False, ith_layer = 0, CD_k = 10): #for greedy training
        if greedy:
            v = v_input
        
            for ith in range(ith_layer):
                p_v, v = self.rbm_layers[ith].v_to_h(v)

            v, v_ = self.rbm_layers[ith_layer](v, CD_k = CD_k)

            return v, v_
        
        v = v_input
        even_layer = [v]
        odd_layer = []
        for i in range(1, self.n_even_layers):
            even_layer.append(torch.bernoulli
                (torch.sigmoid(self.bias[2*i].repeat(v.size()[0],1))))
            
        for _ in range(k_positive):
            p_odd_layer, odd_layer = self.even_to_odd(even_layer)
            p_even_layer, even_layer = self.odd_to_even(odd_layer)
            even_layer[0] = v
            
        positive_phase_even = [i.detach().clone() for i in even_layer]
        positive_phase_odd =  [i.detach().clone() for i in odd_layer]
        for i, d in enumerate(positive_phase_odd):
            positive_phase_even.insert(2*i+1, positive_phase_odd[i])
        positive_phase = positive_phase_even
        for _ in range(k_negative):
            p_odd_layer, odd_layer = self.even_to_odd(even_layer)
            p_even_layer, even_layer = self.odd_to_even(odd_layer)

        negative_phase_even = [i.detach().clone() for i in even_layer]
        negative_phase_odd =  [i.detach().clone() for i in odd_layer]
        for i, d in enumerate(negative_phase_odd):
            negative_phase_even.insert(2*i+1, negative_phase_odd[i])
        negative_phase = negative_phase_even
        return positive_phase, negative_phase    
    
    
class GRU(nn.Module):
    def __init__(self, n_class=10, hid=1024):
        self.hid = hid
        super(GRU, self).__init__()
        self._nn1 = nn.Sequential(
            nn.Linear(in_features=64,out_features=self.hid), 
            nn.ReLU(inplace=True), 
            )        
        
        self._gru = nn.Sequential(
            nn.LSTM(input_size=self.hid, hidden_size=self.hid),
            )
        
        self._nn2 = nn.Sequential(
            nn.Linear(in_features=self.hid*16,out_features=1024), 
            nn.ReLU(inplace=True), 
            nn.Linear(in_features=1024,out_features=1024), 
            nn.ReLU(inplace=True), 
            nn.Linear(in_features=1024,out_features=n_class), 
            nn.Softmax(dim=1)    
            )          

    def forward(self, x):
        if len(x.size()) < 3:
            x = x.reshape(-1, 16, 64)   
            x = x.reshape(-1, 64)
        out = self._nn1(x)
        out = out.reshape(-1, 16, self.hid) 
        out, _ = self._gru(out)
        out = out.reshape(-1, self.hid*16)
        out = self._nn2(out)
        return out    
    
    
    
    