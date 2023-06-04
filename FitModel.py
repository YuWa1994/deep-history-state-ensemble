# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:11:29 2021

@author: yuwa
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from YW_packages import YW_utils as utils
from sklearn.metrics import accuracy_score
import math
import pyswarms as ps
import time
import random


def setup_seed(S=None):
    if S != None:
        np.random.seed(S)    
        torch.manual_seed(S)
        torch.cuda.manual_seed_all(S)
        torch.backends.cudnn.deterministic = True


def set_parameter():
    para={'optimizer':'Adam', 
          'lr':0.001, 
          'epoch':50, 
          'batch_size':50,
          'val':0,
          'E_start':0, 'E_stride':0, 'E_end':None,
          'Snap':0, 'warm_up':0, 's_lr':0,   
          'boost':-1,        
          'RandLR':-1,    # random learning rate
          'is_train_accu': False,
          'normalize' : True,
          'loss_func': None ,
          'device' :torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
          'seed': None,     
          'add_snr': False,
          'lr_decay': -1,
          'lr_decay_start': 50
          }        
    
    history = {'lr_list':[],
               'loss_epoch':[],
               'loss_iter':[],
               'accu_iter':[],
               'model':[]
               }    
    history['model']={'SnapShot':[],
                      'HS':[],
                      'reset':[],
                      'Boost':[]
                      }
    
    if not os.path.exists('./history'): os.makedirs('./history')
    para['modelpath'] = './history'
    utils.empty_file('./history')    
    
    return para, history

        
def _convert_to_torch(X, para):
    X = torch.from_numpy(X).to(para['device'])
    X = Variable(X).float()
    return X
   
    
def _cosine_annealing(iteration, para):
    lr = para['s_lr']*(math.cos(math.pi*iteration/para['iter_per_cycle'])+1)/2
    return lr     


def _weight_reset(model):
    reset_parameters = getattr(model, "reset_parameters", None)
    if callable(reset_parameters):
        model.reset_parameters()
 
    
def _print_parameter(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    return


def _boost_distribution(model, para, trainX, trainY):
    #trainX=trainX.cpu().detach().numpy()
    #trainY=trainY.cpu().detach().numpy()
    valX = para['valX']
    valY = para['valY']
    
    val_out = test(model, para, valX, valY, normalize=False)
    train_out = test(model, para, trainX, trainY, normalize=False, is_train=True)            
    
    er = val_out['bad'].shape[0]/para['valX'].shape[0]
    alpha = 0.5*np.log((1-er)/(er+1e-8))+0.1*np.log(valY.shape[1]-1)
    dummy = train_out['predict']-train_out['testY']
    d=np.zeros(dummy.shape)
    d[np.where(dummy==0)[0]]=1
    d[np.where(dummy!=0)[0]]=-1
    
    prob=np.exp(alpha*d)/trainX.shape[0]
    distribution = prob/prob.sum()   
    para['alpha'].append(alpha)
    return distribution


def train(model, trainX, trainY, opts):
    para, history = set_parameter()
    
    for i, _ in opts.items(): para[i] = opts[i]        
    if para['seed'] != None: setup_seed(S=para['seed'])
    
    model = model.to(para['device'])
    indices = np.random.permutation(trainX.shape[0])
    trainX, trainY = trainX[indices], trainY[indices]
        
    if para['val']>0:
        num = math.floor(trainX.shape[0]*para['val'])
        trainX, trainY = trainX[num:], trainY[num:]        
        para['valX'] , para['valY']  = trainX[:num], trainY[:num]        
                
    if para['normalize'] == True:    
        trainX, para['mu'], para['sigma'] = utils.zscore(trainX, dim=0)
        if para['val']>0:
            para['valX'] = utils.normalize(para['valX'], para['mu'], para['sigma'])
        
    if para['loss_func']==None: loss_func = nn.MSELoss()               
        
    para['batch_num'] = int(np.shape(trainX)[0]/para['batch_size'])
    para['total_iter'] = para['epoch']*para['batch_num']
    para['train_num'] = trainX.shape[0]

    if para['E_end']==None: para['E_end']=para['total_iter']
            
    if para['Snap']>0: 
        cycle_iter = 0
        para['iter_per_cycle'] = math.floor((para['epoch']-para['warm_up'])*
                                    para['batch_num']/para['Snap'])  
    if para['boost']>0: para['alpha']=[]      
          
    optim = eval('torch.optim.{}(model.parameters(),lr={})'
            .format(para['optimizer'],para['lr']))
    
    global_iter = 0
    for epoch_num in range(para['epoch']): 
        # Shuaffle the data for every epoch
        if para['boost'] >= 0 and epoch_num>=para['boost']:
            D = _boost_distribution(model, para, trainX, trainY)      
            x_list = np.arange(para['train_num'])
            indices = np.random.choice(x_list,para['train_num'], p=D)
            indices = torch.from_numpy(indices).type(torch.long) 
        else:
            indices = torch.randperm(trainX.shape[0])
            
        indices = torch.split(indices, para['batch_size'])       
        for batch in range(len(indices)):
            if  para['lr_decay']>0 and epoch_num>=para['lr_decay_start']:
                for g in optim.param_groups: g['lr'] = g['lr']*(1-para['lr_decay'])
            
            if para['Snap']>0 and epoch_num>=para['warm_up']:
                lr = _cosine_annealing(cycle_iter, para)
                for g in optim.param_groups: g['lr'] = lr
                cycle_iter += 1

            if para['RandLR']>0 and epoch_num>=para['warm_up']:
                lr = random.randint(0.00001, 0.01)
            
            if para['add_snr']:
                batch_X = trainX[indices[batch]]
                batch_X = utils.snr_signal(batch_X, snr=6, ratio = 0.2)
                batch_X = torch.from_numpy(batch_X).to(para['device'])  
                batch_Y = torch.from_numpy(trainY[indices[batch]]).to(para['device'])
            else:
                batch_X = torch.from_numpy(trainX[indices[batch]]).to(para['device'])
                batch_Y = torch.from_numpy(trainY[indices[batch]]).to(para['device'])

            batch_X = Variable(batch_X).float()
            batch_Y = Variable(batch_Y).float()

            # ===================forward=====================
            predict = model(batch_X)
            loss = loss_func(predict, batch_Y)
            # ===================backward====================
            optim.zero_grad()
            loss.backward()
            optim.step()
    
            history['loss_iter'].append(loss.cpu().detach().numpy())   
            history['lr_list'].append(optim.state_dict()["param_groups"][0]["lr"])
            global_iter += 1   
            
            if para['Snap']>0 and cycle_iter == para['iter_per_cycle']:
                cycle_iter = 0
                if para['boost']<0:
                    Name = '{}snapshot_NN.pth'.format(global_iter)
                    history['model']['SnapShot'].append(Name)
                    torch.save(model.state_dict(), './history/'+Name)                
                
        
        if para['E_start']>0 and epoch_num>=para['E_start'] \
                and epoch_num<para['E_end'] \
                and (epoch_num-para['E_start'])%para['E_stride']==0:
            Name = '{}HS_NN.pth'.format(epoch_num)
            history['model']['HS'].append(Name)                        
            torch.save(model.state_dict(),'./history/'+Name) 
            
        if para['boost']>0 and epoch_num>=para['boost']:
            Name = '{}Boost.pth'.format(epoch_num)
            history['model']['Boost'].append(Name)
            torch.save(model.state_dict(), './history/'+Name)           
            
        if para['is_train_accu'] is True:
            train_log = test(trainX, trainY, model)
            history['accu_iter'].append(train_log['accu'])
        
        history['loss_epoch'].append(loss.cpu().detach().numpy())
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch_num + 1, para['epoch'], loss.cpu().detach().numpy()))  
    
    return model, history, para


def _ensemble_model(model, para, model_name, testX, testY, boost=False):
    result = {}
    result['accu_iter'] = []
    result['predict_iter'] = []     
    result['predict_ensemble'] = np.zeros(testY.shape)

    for i in range(len(model_name)):
        model.load_state_dict(
            torch.load(para['modelpath']+'/{}'.format(model_name[i])))                
        test_iter = test(model, para, testX, testY)
            
        result['accu_iter'].append(test_iter['accu'])
        result['predict_iter'].append(test_iter['NetOutput'])
        if boost:
            result['predict_ensemble'] += para['alpha'][i]*test_iter['NetOutput']
        else:     
            result['predict_ensemble'] += test_iter['NetOutput']
 

    result['testY'] = test_iter['testY']
    result['accu_ensemble'] = accuracy_score(result['testY'], 
                              np.argmax(result['predict_ensemble'],axis=1))      
    return result

        
def test(model, para, testX, testY, 
         batch_size=50, 
         normalize=False,
         is_train=False):
    
    if normalize == True: 
        testX = utils.normalize(testX, para['mu'], para['sigma'])
        
    model.eval()    
    result = {}            
    result['testY'] = np.argmax(testY,axis=1)
    testX = _convert_to_torch(testX, para)
    
    if batch_size != None:
        testX2 = torch.split(testX, batch_size)
        result['NetOutput'] = []
        for i in range(len(testX2)):
            out = model(testX2[i])
            result['NetOutput'].extend(out.detach().cpu().numpy())
        result['NetOutput'] = np.array(result['NetOutput'])  
    else:
        predict = model(testX)
        result['NetOutput'] = predict.detach().cpu().numpy()
        
    result['predict'] = np.argmax(result['NetOutput'],axis=1)
    result['accu'] = accuracy_score(result['testY'], result['predict']) 
    
    dummy = result['predict']-result['testY'] 
    result['bad'] = np.nonzero(dummy!=0)[0]  
    result['good'] = np.nonzero(dummy==0)[0]   
    
    if is_train==True: 
        model.train()
    return result


def test_HSE(model, para, model_name, testX, testY,
             is_valX=False, valX=None, valY=None,
             normalize=False,
             method='HSE',
             ensemble_num=None,
             add_snr=False,
             modelpath=None
             ):

    result = {}
    if normalize:
        testX = utils.normalize(testX, para['mu'], para['sigma'])
    
    if modelpath != None: para['modelpath']=modelpath
    
    if ensemble_num==None:
        ensemble_num=math.floor(len(model_name)/2)
    
    
    if is_valX: 
        if 'valX' in para.keys():
            valX, valY = para['valX'], para['valY']
        if add_snr==True:
            X = valX.reshape(valX.shape[0],valX.shape[2])
            X = utils.snr_signal(X, snr=6)
            valX = X.reshape(valX.shape[0],1,valX.shape[2])    
        start = time.time()
        val = _ensemble_model(model, para, model_name, valX, valY)
        end = time.time()
        result['val_time'] = end-start
        
    if method=='Ranking':
        start = time.time()
        indices = np.argsort(np.array(val['accu_iter']))[::-1]
        indices = indices[:ensemble_num]        
        result = _ensemble_model(model, para, [model_name[i] for i in indices],
                                 testX, testY) 
        end = time.time()
        result['Ranking_time'] = end-start        

    elif method=='Select' or method=='Weight':
        start = time.time()
        pso = pyswarm_optm(val['predict_iter'], valY, model_name)
        weight, pso_loss = pso._pso_weight()      
        end = time.time()
        result['PSO_time'] = end-start         
        if method == 'Select':
            model_name = [model_name[i] for i in np.where(weight>0)[0]] 
            start = time.time()
            result = _ensemble_model(model, para, model_name, testX, testY) 
            end = time.time()
            result['Select_time'] = end-start                
                 
        if method=='Weight':            
            start = time.time()
            result = _ensemble_model(model, para, model_name, testX, testY) 
            end = time.time()
            result['Weight_time'] = end-start                   
            result['predict_ensemble'] = pso._weighted_out(result['predict_iter'], 
                                                  testY, weight)
            result['accu_ensemble'] = accuracy_score(result['testY'] , 
                                     np.argmax(result['predict_ensemble'],axis=1))
               
        result['weight'] = weight
        result['pso_loss'] = pso_loss
            
    elif method=='HS':
        result = _ensemble_model(model, para, model_name, testX, testY)    
        
    elif method=='boost':
        result = _ensemble_model(model, para, model_name, testX, testY, boost=True)    
    return result 


class pyswarm_optm():
    def __init__(self, X, Y, model_name):    
        self.X = X
        self.Y = Y
        self.model_name = model_name

    def _pso_weight(self):
        options = {'c1': 0.5, 'c2': 0.5, 'w':0.9}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=100, 
            dimensions=len(self.model_name), 
            options=options
            )
        _, weight = optimizer.optimize(self._pso_fitness, iters=100)    
        cost_history = optimizer.cost_history
        return weight, cost_history
          
    def _pso_fitness(self, weight):        
        n_particles = weight.shape[0]
        j = [self._get_pso_loss(weight[i]) for i in range(n_particles)]        
        return j
       
    def _get_pso_loss(self, weight):
        out = np.zeros(self.X[0].shape)
        for k in range(len(self.X)):
            out = out+self.X[k]*weight[k]
        loss = sum(sum(np.abs(out-self.Y)))    
        return loss
    
    def _weighted_out(self, X, Y, weight):
        predict_ensemble = np.zeros(Y.shape)
        for k in range(len(X)):   
            predict_ensemble = predict_ensemble + X[k]*weight[k]        
        return predict_ensemble
    
    
class attention_optm():
    def __init__(self, X, Y, model_name):    
        self.X = X
        self.Y = Y
        self.model_name = model_name
        self.q = nn.parameter(torch.rand(len(model_name), Y.shape[1]))
        self.k = nn.parameter(torch.rand(len(model_name), Y.shape[1]))
        self.v = nn.parameter(torch.rand(len(model_name), Y.shape[1]))        

    def _pso_weight(self):
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=150, 
            dimensions=len(self.model_name), 
            options=options
            )
        _, weight = optimizer.optimize(self._pso_fitness, iters=100)    
        cost_history = optimizer.cost_history
        return weight, cost_history
          
    def _pso_fitness(self, weight):        
        n_particles = weight.shape[0]
        j = [self._get_pso_loss(weight[i]) for i in range(n_particles)]        
        return j
       
    def _get_pso_loss(self, weight):
        out = np.zeros(self.X[0].shape)
        for k in range(len(self.X)):
            out = out+self.X[k]*weight[k]
        loss = sum(sum(np.abs(out-self.Y)))    
        return loss
    
    def _weighted_out(self, X, Y, weight):
        predict_ensemble = np.zeros(Y.shape)
        for k in range(len(X)):   
            predict_ensemble = predict_ensemble + X[k]*weight[k]        
        return predict_ensemble    
    
    