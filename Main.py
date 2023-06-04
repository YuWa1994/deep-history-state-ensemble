# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:34:24 2021

@author: yuwa
"""

#import numpy as np
import scipy.io as sio
import TestNets
import datasets as data
from YW_packages import YW_utils as utils
import FitModel
import time
from training_sets import train_sets
   

def add_snr(X, noise=4):
    if noise == 'None':
        return X
    else:
        X_r = X.reshape(X.shape[0],X.shape[2])
        X_r = utils.snr_signal(X_r, snr=noise)
        X_r = X_r.reshape(X.shape[0],1,X.shape[2])      
    return X_r


def get_data(dataset='A', is_normalization='false'):
    filepath = './datasets/data'+dataset+'.mat'
    Data = sio.loadmat(filepath)
    if dataset == 'A' or dataset =='E':
        X, Y = Data['X'], Data['Y'].astype(float)
        trainX, testX, train_indices, test_indices = utils.split_train_test(X, test_ratio=0.5) 
        trainY, testY = Y[train_indices], Y[test_indices]        
    else:
        trainX, testX, trainY, testY = Data['trainX'], Data['testX'], Data['trainY'], Data['testY']
    
    if is_normalization:
        trainX, mu, sigma = utils.zscore(trainX, dim=0)
        testX = utils.normalize(testX, mu, sigma)    
    
    return trainX, testX, trainY, testY


models = ['LeNet5', 'AlexNet', 'ResNet', 'DenseNet', 'BPNN', 'GRU']
learn_method = ['HS', 'Ranking', 'AdaWeight', 'boost']
train_method = ['plain', 'Snapshot', 'boost', 'Snap-boost']


for val in [0.1]:
    for d in ['A']:
        for m in [0,1,2,3,4]:
            for s in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                print('val'+str(val)+','+'data'+d+','+'model'+str(m)+','+'trial'+str(s))
                save2path = './datasets/result/dataset'+d+'/'+models[m]
                
                history = {}
                result = {}
                result2 = {}
                result3 = {}
                result4 = {}
                result5 = {}
                result6 = {}
                
                utils.setup_seed(seed=s)
                trainX, testX, trainY, testY = get_data(dataset=d, is_normalization='false')
                
                opts = {'epoch':50,
                        'optimizer':'Adam','lr':0.001,'E_start':20, 'E_stride':1,
                        #'lr_decay':0.001,'lr_decay_start':20,
                        #'optimizer':'Adam','lr':0.001,'Snap':30, 'warm_up':20, 's_lr':0.001,
                        #'optimizer':'Adam','lr':0.001,'boost':20,'add_snr':False,
                        #'optimizer':'Adam','lr':0.001,'Snap':30, 'warm_up':20, 's_lr':0.001,'boost':20,
                        'val':val} 
                if m < 4:
                    Net = eval('TestNets.{}_1D({})'.format(models[m],trainY.shape[1]))
                else:
                    Net = eval('TestNets.{}({})'.format(models[m],trainY.shape[1]))
                
                start = time.time()
                Net, history, para = FitModel.train(Net, trainX, trainY, opts)
                end = time.time()
                history['train_run_time'] = end-start
     
                # Test
                #testX = utils.normalize(testX, para['mu'], para['sigma']) 
                #########################
                start = time.time()
                result = FitModel.test(Net, para, testX, testY, batch_size=50)
                end = time.time()
                result['test_run_time'] = end-start                
                result.update(history)
                name = models[m]+'_trial'+str(s)+'_plain'+'.mat'
                #sio.savemat(save2path+'/'+name, result)
                    
                his_model = []
                his_model.extend(history['model']['HS'])
                his_model.extend(history['model']['SnapShot'])                     
                his_model.extend(history['model']['Boost'])  

                
                if para['val'] == 0 or para['boost'] > 0:
                    #########################
                    start = time.time()
                    result2 = FitModel.test_HSE(Net, para, his_model, testX, testY,
                                               method='HS')
                    end = time.time()
                    if para['Snap'] > 0 or para['boost'] > 0:
                        result2.update(result)
                        
                    result2['test_run_time'] = end-start    
                    name = models[m]+'_trial'+str(s)+'_BS30_HSE'+'.mat'
                    #sio.savemat(save2path+'/'+name, result2)

                if para['val'] > 0:
                    ############ Ranking #############
                    start = time.time()
                    result3 = FitModel.test_HSE(Net, para, his_model, testX, testY, is_valX=True,
                                               method='Ranking',ensemble_num=3)
                    end = time.time()
                    result3['test_run_time'] = end-start 
                    
                    name = models[m]+'_trial'+str(s)+'_BS30_Ranking'+'.mat'
                    #sio.savemat(save2path+'/'+name, result3)
                    
                    ############ Weight #############
                    start = time.time()
                    result4 = FitModel.test_HSE(Net, para, his_model, testX, testY, is_valX=True,
                                               method='Weight')
                    end = time.time()
                    result4['test_run_time'] = end-start 
                    
                    name = models[m]+'_trial'+str(s)+'_BS30_Weight'+'.mat'
                    #sio.savemat(save2path+'/'+name, result4)                              
                
                    ############ Select #############
                    start = time.time()
                    result5 = FitModel.test_HSE(Net, para, his_model, testX, testY, is_valX=True,
                                               method='Select')
                    end = time.time()
                    result5['test_run_time'] = end-start 
                    
                    name = models[m]+'_trial'+str(s)+'_BS30_Select'+'.mat'
                    #sio.savemat(save2path+'/'+name, result5)     
    
                    ############ Boost #############
                    '''
                    if opts['boost']>0:
                        start = time.time()
                        result6 = FitModel.test_HSE(Net, para, his_model, testX, testY,
                                                   method='boost')
                        end = time.time()
                        result6['test_run_time'] = end-start 
                        
                        name = models[m]+'_trial'+str(s)+'_BS'+'.mat'
                        result6.update(result)
                        sio.savemat(save2path+'/'+name, result6)     
                    '''





