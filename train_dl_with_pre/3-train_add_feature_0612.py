# coding: utf-8
# copy from the last day version and add feat:
# uid_count
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.cross_validation  import StratifiedKFold
from scipy import sparse
import gc
from sklearn.metrics import roc_auc_score
from time import time
import logging
import pprint
from datetime import datetime

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch.backends.cudnn
from torch.utils.data.sampler import *
from torch.utils.data import DataLoader

import TSAdsDataset

from loss import FocalLoss


# https://blog.csdn.net/lv26230418/article/details/46356763
#-------------------------------------------------------------------------------    
def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    # format   = 'LINE %(lineno)-4d  %(levelname)-8s %(message)s',
                    format   = '%(message)s',
                    datefmt  = '%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'a');
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler();
    console.setLevel(logging.INFO);
    # set a format which is simpler for console use
    # formatter = logging.Formatter('LINE %(lineno)-4d : %(levelname)-8s %(message)s');
    formatter = logging.Formatter('%(message)s');
    # tell the handler to use this format
    console.setFormatter(formatter);
    logging.getLogger('').addHandler(console);

def tensors2var(tpl,use_cuda):
    if use_cuda:
        return (Variable(t).cuda() for t in tpl)
    else:
        return (Variable(t) for t in tpl)
from operator import itemgetter
from itertools import groupby
def grouparr(seq):
    groups = groupby(sorted(seq,key=itemgetter(0)), itemgetter(0))
    seqs = [[item[1:] for item in data] for (key, data) in groups]
    return seqs

def ftrl_train(net,train_sets,valid_set,args,criterion = nn.CrossEntropyLoss(),eval_metric = roc_auc_score):
    '''
        Args:
            args:
                eg:
                args = {
                    'batch_size':8192,
                    'device_ids':[0],
                    'use_cuda':True,
                    'optimizer_type': 'adag',
                    'learning_rate': 0.005,
                    'weight_decay': 0.0005, 
                    'n_epochs': 10, 
                    'verbose':True,
                    'njobs':4,
                    'save_path' = None,
                    'start_epoch'=0,
                    'is_eval_on_trains':True
                }
    '''     
    save_path = args['save_path']
    start_epoch = args['start_epoch']
    is_eval_on_trains = args['is_eval_on_trains']
    if save_path and not os.path.exists('/'.join(save_path.split('/')[0:-1])):
        logging.info("Save path is not existed!")
        return
    os.makedirs(save_path+'/snap/',exist_ok = True)
    # dataloader
    train_loaders = []
    eval_train_loaders = []
    for train_set in train_sets:
        train_loader = DataLoader(
                                train_set,
                                sampler = RandomSampler(train_set),
                                batch_size = args['batch_size'],
                                drop_last   = False,
                                num_workers = args['njobs'],
                                pin_memory = True
                        )
        if is_eval_on_trains:
            eval_train_loader = DataLoader(
                            train_set,
                            sampler = SequentialSampler(train_set),
                            batch_size = args['batch_size']//8,
                            drop_last   = True,
                            num_workers = args['njobs'],
                            pin_memory = False
                    )
            eval_train_loaders.append(eval_train_loader)
        train_loaders.append(train_loader)
    is_valid=False
    if valid_set is not None:
        is_valid =True
        valid_loader = DataLoader(
                        valid_set,
                        sampler = SequentialSampler(valid_set),
                        batch_size = args['batch_size']//8,
                        drop_last   = False,
                        num_workers = args['njobs'],
                        pin_memory = True
                )
        
    if args['use_cuda'] == True:
        model = torch.nn.DataParallel(net, device_ids=args['device_ids'])
    else:
        model = net

    # resume from previous ----------------------------------
    if start_epoch !=0:
        model_file = save_path +'/snap/%03d.pth'%start_epoch
        assert os.path.exists(model_file),model_file+" is not exist"
        model.load_state_dict(torch.load(model_file))
    
    if args['use_cuda'] == True:
        model.cuda()
    

    opt_algo = args['optimizer_type']
    lr = args['learning_rate']
    weight_decay=args['weight_decay']
    # get parameters

    params = []
    embedings_params = []
    embedings_params.append('fm_second_order_embeddings')
    embedings_params.append('fm_second_order_bag_embeddings')
    if net.use_lr:
        embedings_params.append('fm_first_order_embeddings')
        embedings_params.append('fm_first_order_bag_embeddings')
    #deep
    deep_params = []
    if net.use_deep:
        for i in range(len(net.deep_layers)):
            deep_params.append('linear_'+str(i+1))
            if net.is_deep_bn:
                deep_params.append('batch_norm_'+str(i+1))
    cin_params = []
    if net.use_cin:
        for i in range(len(net.cin_layer_sizes[1:])):
            cin_params.append('conv1d_'+str(i+1))
            if net.is_cin_bn:
                cin_params.append('conv1d_bn_'+str(i+1))
        if len(net.cin_deep_layers)!=0:
            for i in range(len(net.cin_deep_layers)):
                cin_params.append('cin_linear_'+str(i+1))
                if net.is_deep_bn:
                    cin_params.append('cin_deep_bn_'+str(i+1))
    din_params = []
    if net.use_din:
        for i in range(len(net.din_key_offsets)):
            din_params.append('interestpooling_'+str(i+1))
    for embedings_param in embedings_params:
        embedings_param = getattr(net,embedings_param).parameters()
        params.append({'params':embedings_param,'lr':train_args['emb_lr'],'weight_decay':train_args['emb_l2']})
    for deep_param in deep_params:
        deep_param = getattr(net,deep_param).parameters()
        params.append({'params':deep_param,'lr':train_args['deep_lr'],'weight_decay':train_args['deep_l2']})
    for cin_param in cin_params:
        cin_param = getattr(net,cin_param).parameters()
        params.append({'params':cin_param,'lr':train_args['cin_lr'],'weight_decay':train_args['cin_l2']})
    for din_param in din_params:
        din_param = getattr(net,din_param).parameters()
        # params.append({'params':din_param,'lr':train_args['cin_lr'],'weight_decay':train_args['cin_l2']})
        params.append({'params':din_param,'lr':train_args['din_lr'],'weight_decay':train_args['din_l2']})

    params.append({'params':getattr(net,'concat_linear_layer').parameters(),'lr':train_args['deep_lr'],'weight_decay':train_args['deep_l2']})

    # embeding
    
    # params = list(net.parameters())

    optimizer = torch.optim.SGD(params, lr, weight_decay=weight_decay)
    if opt_algo == 'adam':
        optimizer = torch.optim.Adam(params, lr, weight_decay=weight_decay)
    elif opt_algo == 'rmsp':
        optimizer = torch.optim.RMSprop(params, lr, weight_decay=weight_decay)
    elif opt_algo == 'adag':
        optimizer = torch.optim.Adagrad(params, lr, weight_decay=weight_decay)
    
    train_eval_result = []
    valid_eval_result = []
    train_prob_result = []
    valid_prob_result = []
    for epoch in range(start_epoch,args['n_epochs']):
        epoch_begin_time = time()
        #multi train set
        for idx_loader,train_loader in enumerate(train_loaders):
            
            train_loader.dataset.load_cache()
            if is_eval_on_trains:
                eval_train_loader = eval_train_loaders[idx_loader]
            batch_begin_time = time()
            model.train()
            total_loss = 0.0
        # use gpus      
            for it, data in enumerate(train_loader):
                indices,eval_groupby_ids,batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts,batch_y = data
                batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts,batch_y = tensors2var((batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts,batch_y), use_cuda=args['use_cuda'])
                 
                optimizer.zero_grad()            
                outputs = model(batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts)
                batch_y = batch_y.squeeze_()
                loss = criterion(outputs, batch_y)
                loss.backward()
                # torch.nn.utils.clip_grad_norm(list(model.parameters()),50)
                optimizer.step()
                total_loss += loss.item()
                
                # logging.info every 100 mini-batches
                if args['verbose']:
                    if it % 100 == 99: 
                    # if 1: 
                        eval = evaluate_by_batch(model,batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts,batch_y,eval_groupby_ids,eval_metric)
                        logging.info('[%d, %5d, %5d] loss: %.6f metric: %.6f time: %.1f s' %
                              (epoch + 1, idx_loader+1,it + 1, total_loss/100, eval, time()-batch_begin_time))
                        total_loss = 0.0
                        batch_begin_time = time()
            # eval trainset
            if is_eval_on_trains:
                train_loss, train_eval,train_prob = eval_by_dataloader(model,eval_train_loader,criterion,eval_metric,args['use_cuda'])
                train_eval_result.append(train_eval)
                train_prob_result.append(train_prob)
                logging.info('*'*50)
                logging.info('trainset: [%d] loss: %.6f metric: %.6f time: %.1f s' %
                      (epoch + 1, train_loss, train_eval, time()-epoch_begin_time))
                logging.info('*'*50)
            train_loader.dataset.clear_cache()
         # save model
        if save_path and epoch%1==0:
            model_file = save_path +'/snap/%03d.pth'%(epoch+1)
            torch.save(model.state_dict(),model_file)
        # eval validset
        if is_valid:
            valid_loss, valid_eval,valid_prob = eval_by_dataloader(model,valid_loader,criterion,eval_metric,args['use_cuda'])
            valid_eval_result.append(valid_eval)
            valid_prob_result.append(valid_prob)
            logging.info('*' * 50)
            logging.info('validset: [%d] loss: %.6f metric: %.6f time: %.1f s' %
                  (epoch + 1, valid_loss, valid_eval,time()-epoch_begin_time))
            logging.info('*' * 50)
       
    return model,train_eval_result,valid_eval_result,train_prob_result,valid_prob_result
def inner_predict(net, batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts):
    """
    Args:
        x:         
    Returns: 
        numpy
    """
    return (inner_predict_proba(net,batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts) > 0.5)

def inner_predict_proba(net, batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts):
    """
    Args:
        x:         
    Returns: 
        numpy
    """
    model = net.eval()
    pred = model(batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts).cpu()
    activate_layer = nn.Softmax(-1)
    pred = activate_layer(pred)
    return pred.data.numpy()[:,1]
def evaluate_by_batch(net, batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts,batch_y,groupby_ids,eval_metric):
    """
    Args:
        net
        x:  
        y: tensor of labels
    Returns: metric of the evaluation
    """
    y_probs = inner_predict_proba(net,batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts)
    y_trues = batch_y.cpu().data.numpy()
    eval_score = eval_metric(y_trues,y_probs)
    return eval_score
def eval_by_dataloader(net,valid_loader,criterion,eval_metric,use_cuda):
    model = net.eval()
    total_loss = 0.0
    y_probs = []
    y_trues = []
    activate_layer = nn.Softmax(-1)
    eval_groupby_ids=[]
    for it, data in enumerate(valid_loader):
        indices,batch_eval_groupby_ids,batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts,batch_y = data
        batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts,batch_y = tensors2var((batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts,batch_y), use_cuda)
        # forward
        outputs = model(batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts)
        batch_y = batch_y.squeeze_()
        loss = criterion(outputs, batch_y)
        total_loss += loss.item()*len(indices)
        # N*C
        probs = outputs.cpu()
        probs = activate_layer(probs)
        y_probs.extend(probs.data.numpy()[:,1]) 

        eval_groupby_ids.extend(batch_eval_groupby_ids.numpy())

        _y = batch_y.cpu().data.numpy()
        y_trues.extend(list(_y.ravel()))

    # evaluate by groupby_id
    eval_groupby_ids = np.array(eval_groupby_ids).reshape(-1,1)
    y_probs = np.array(y_probs).reshape(-1,1)
    y_trues = np.array(y_trues).reshape(-1,1)

    results = np.hstack((eval_groupby_ids,y_trues,y_probs))
    group_results = grouparr(results)
    eval_score = 0
    for results in group_results:
        results = np.array(results)
        eval_score += eval_metric(results[:,0], results[:,1])
    total_metric = eval_score/len(group_results)
    return total_loss/len(y_trues), total_metric,y_probs

def split_by_SKF(trains_y,nsplits=5,random_state=None):
    skf=StratifiedKFold(trains_y,n_folds = nsplits,shuffle=False,random_state=random_state)
    for train_index,valid_index in skf:
        return train_index,valid_index


if __name__ == '__main__':
    #log

    root_dir = '../tmp/0610_get_feat_with_pre/'
    encoded_dir = root_dir+'encoded_lowcase_hash200/'
    cate_enc_pkl_dir = encoded_dir+'pkl/cate_encoder/'
    bag_dict_pkl_dir = encoded_dir+'pkl/bag_enc_dict/'

    # cate features 
    base_lbenc_feats = ['creativeId','LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus',
                'advertiserId','campaignId', 'adCategoryId', 'productId', 'productType','creativeSize']
    cate_featnames = base_lbenc_feats+['uid_count']
    cate_lbenc_dict = TSAdsDataset.get_lb_encorder(cate_featnames,cate_enc_pkl_dir)

    #bag feature
    bag_feats = ['kw1','kw2','kw3','topic1','topic2','topic3',
        'appIdAction','appIdInstall',
        'interest1',
        'interest2',
        'interest3',
        'interest4',
        'interest5']

    bag_lb_encoder = TSAdsDataset.get_lb_encorder(bag_feats,bag_dict_pkl_dir)
    bag_feature_sizes = []
    for feature in bag_feats:
        bag_feature_sizes.append(len(bag_lb_encoder[feature])  )
    bag_value_cnts =[5, 5, 5, 5, 5, 5, 50, 50, 35, 30, 10, 10, 70]
    # bag_value_cnts =[5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10]

    datasets = []
    lgb_n_trees = 0
    for i in range(9):
        cate_fn_prefix = str(i)+'_'
        pre_cate_fn_prefix = 'pre_'+str(i)+'_'
        cate_load_dir = encoded_dir+'split/trains/'
        bag_fname = cate_load_dir+'merged/'+cate_fn_prefix+'bag.pkl'
        instance_ids_fname = cate_load_dir+'/instance_ids/'+cate_fn_prefix+'instance_id.pkl'
        labels_fname = cate_load_dir+'labels/'+cate_fn_prefix+'label.pkl'
        pre_bag_fname = cate_load_dir+'merged/'+pre_cate_fn_prefix+'bag.pkl'
        pre_instance_ids_fname = cate_load_dir+'/instance_ids/'+pre_cate_fn_prefix+'instance_id.pkl'
        pre_labels_fname = cate_load_dir+'labels/'+pre_cate_fn_prefix+'label.pkl'
        dataset = TSAdsDataset.StackFileBagMeanTSAdsDataset(bag_value_cnts,bag_fname,\
                    cate_load_dir,cate_featnames, cate_fn_prefix,cate_lbenc_dict,\
                    instance_ids_fname, labels_fname,\
                    is_use_pre=True,
                    pre_bag_fname=pre_bag_fname,\
                    pre_cate_fn_prefix=pre_cate_fn_prefix,\
                    pre_instance_ids_fname=pre_instance_ids_fname, pre_labels_fname=pre_labels_fname,
                    lgb_n_trees=lgb_n_trees,lgb_n_leaves=127
                    )
        datasets.append(dataset)   
    train_sets= datasets
    # valid set
    i = 9
    cate_fn_prefix = str(i)+'_'
    cate_load_dir = encoded_dir+'split/trains/'
    bag_fname = cate_load_dir+'merged/'+cate_fn_prefix+'bag.pkl'
    instance_ids_fname = cate_load_dir+'/instance_ids/'+cate_fn_prefix+'instance_id.pkl'
    labels_fname = cate_load_dir+'labels/'+cate_fn_prefix+'label.pkl'
    valid_set = TSAdsDataset.StackFileBagMeanTSAdsDataset(bag_value_cnts,bag_fname,\
                cate_load_dir,cate_featnames, cate_fn_prefix,cate_lbenc_dict,\
                instance_ids_fname, labels_fname,\
                is_use_pre=False,
                lgb_n_trees=lgb_n_trees,lgb_n_leaves=127
                )
    print("loading valid_set")
    valid_set.load_cache()

    p_sv =  valid_set.cate_p
    p_bag =  sum(bag_feature_sizes)+1
    field_num =  len(cate_featnames)+lgb_n_trees+len(bag_feats)

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
    #best params:0.741567
    train_args = {
        'batch_size':8192,
        'device_ids':[0],
        'use_cuda':True,
        'optimizer_type': 'adam',
        'learning_rate': 0.005,
        'weight_decay': 1e-7, 
        'emb_lr':0.005,
        'emb_l2': 1e-7, 
        'deep_lr':0.003,
        'deep_l2':1e-7,
        'cin_lr':0.001,
        'cin_l2':0.0,
        'din_lr':0.001,
        'din_l2':1e-7,
        'n_epochs': 5, 
        'verbose':True,
        'njobs':4,
        'save_path' :'../tmp/train_xdeepfm_0612_with_pre_all_bigbags_addFeat_default_init/',
        'start_epoch':0,
        'is_eval_on_trains':False
    }
    # show params
    os.makedirs(train_args['save_path'],exist_ok=True)
    initLogging(train_args['save_path']+'log.log')
    
    logging.info('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    logging.info('dataloader without drop last\n')
    logging.info('root_dir:\n{}\n'.format(pprint.pformat(root_dir)))
    logging.info('cate_enc_pkl_dir:\n{}\n'.format(pprint.pformat(cate_enc_pkl_dir)))
    logging.info('bag_dict_pkl_dir:\n{}\n'.format(pprint.pformat(bag_dict_pkl_dir)))
    logging.info('encoded_dir:\n{}\n'.format(pprint.pformat(encoded_dir)))
    logging.info('cate_featnames:\n{}\n'.format(pprint.pformat(cate_featnames)))
    logging.info('bag_feature_sizes:\n{}\n'.format(pprint.pformat(bag_feature_sizes)))
    logging.info('bag_value_cnts:\n{}\n'.format(pprint.pformat(bag_value_cnts)))
    logging.info('p_sv:\n{}\n'.format(pprint.pformat(p_sv)))
    logging.info('p_bag:\n{}\n'.format(pprint.pformat(p_bag)))
    logging.info('field_num:\n{}\n'.format(pprint.pformat(field_num)))
    logging.info('train_args:\n{}\n'.format(pprint.pformat(train_args)))
    from din_xdeepfm_v2 import ExDeepFM
    model = ExDeepFM(field_num,p_sv,p_bag,bag_value_cnts,
                use_cuda =True,
                use_lr = True,use_fm=True,use_deep=True,use_cin=True,
                use_din = True,

                is_deep_dropout=False,deep_layers = [240, 240,240,32],dropout_deep=[0.5,0.5,0.5,0.5,0.5],
                deep_layers_activation= 'relu',is_deep_bn=True,

                cin_layer_sizes=[50,50,50,100],
                cin_activation = 'relu',is_cin_bn = True,
                cin_direct = False,use_cin_bias = False,
                cin_deep_layers = [100,32], 
                    # cin_deep_dropouts = [0.5,0.5],
                din_query_offset = 0,din_key_offsets=[0,1,2,3,4,5,6,7,8,9,10,11,12]
                )
    logging.info('model:\n{}\n'.format(pprint.pformat(model.__dict__)))
    logging.info('init default\n')
    # CrossEntropyLoss
    ftrl_train(model,train_sets,valid_set,train_args)
    # focal loss
    ftrl_train(model,train_sets,valid_set,train_args,criterion = FocalLoss())
