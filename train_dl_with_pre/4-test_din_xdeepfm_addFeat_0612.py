import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch.backends.cudnn

from torch.utils.data.sampler import *
from torch.utils.data import DataLoader

from datetime import datetime
import logging
import pprint

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
def predict(net,test_set,args):

    '''
        args = {
            'batch_size':8192,
            'device_ids':[0],
            'use_cuda':True,
            'verbose':True,
            'njobs':4,
            'model_file':' '
        }
    '''  
    test_loader = DataLoader(
                    test_set,
                    sampler = SequentialSampler(test_set),
                    batch_size = args['batch_size'],
                    drop_last   = False,
                    num_workers = args['njobs'],
                    pin_memory = True
            )

    """
        train model
    """
    if args['use_cuda'] == True:
        print(args['device_ids'])
        print(os.environ["CUDA_VISIBLE_DEVICES"])
        model = torch.nn.DataParallel(net, device_ids=args['device_ids'])
    else:
        model = net

    model.load_state_dict(torch.load(args['model_file']))
    if args['use_cuda'] == True:
        model.cuda()

    probs = predict_by_dataloader(model,test_loader,use_cuda=args['use_cuda'])
    return probs

def predict_by_dataloader(net,test_loader,use_cuda):
    model = net.eval()
    y_probs = []
    activate_layer = nn.Softmax(-1)
    for it, data in enumerate(test_loader):
        indices,eval_groupby_ids,batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts = data
        batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts = tensors2var((batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts), use_cuda)
      
        outputs = model(batch_xi,batch_xv,batch_bag_xi,batch_bag_lenghts)
        probs = activate_layer(outputs).cpu()
        y_probs.extend(probs.data.numpy()[:,1]) 
    return y_probs
def submit(net,test_set,model_files,df_sub,save_dir):
    results = None
    args = {
        'batch_size':8192,
        'device_ids':[0],
        'use_cuda':True,
        'verbose':True,
        'njobs':8,
        'model_file':' '
    }
    for i,model_file in enumerate(model_files):
        args['model_file'] = model_file
        probs=predict(net,test_set,args)
        probs = np.array(probs)
        if i == 0:
            results=probs
        else:
            results=results+probs
    results = results/len(model_files)
    results = results.ravel()
    df_sub['score'] = results
    df_sub['score'] = df_sub['score'].apply(lambda x:float('%.6f'%x))
    
    os.makedirs(save_dir,exist_ok=True)
    df_sub.to_csv(save_dir+'/submission.csv',index = False)

if __name__ == '__main__':
    #
    
    from din_xdeepfm_v2 import ExDeepFM
    import pandas as pd
    import pickle
    import TSAdsDataset
    #data 

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

    cate_fn_prefix = ''
    cate_load_dir = encoded_dir+'split/test2/'
    bag_fname = cate_load_dir+'merged/'+cate_fn_prefix+'bag.pkl'
    instance_ids_fname = root_dir+'/sub/test2.csv'
    labels_fname = None
    lgb_n_trees = 0


    # # ce
    version_dirs = '../tmp/train_xdeepfm_0612_with_pre_all_bigbags_addFeat_default_init/'
    # fl
    # version_dirs = '../tmp/train_xdeepfm_0612_with_pre_all_bigbags_addFeat_default_init_fl/'
    bag_value_cnts =[5, 5, 5, 5, 5, 5, 50, 50, 35, 30, 10, 10, 70]
    model_files = ['004.pth','005.pth']
    
    test_set = TSAdsDataset.StackFileBagMeanTSAdsDataset(bag_value_cnts,bag_fname,\
                cate_load_dir,cate_featnames, cate_fn_prefix,cate_lbenc_dict,\
                instance_ids_fname, labels_fname,\
                is_use_pre=False,
                lgb_n_trees=lgb_n_trees,lgb_n_leaves=127
                )
    test_set.load_cache()

    p_sv =  test_set.cate_p
    p_bag =  sum(bag_feature_sizes)+1
    field_num =  len(cate_featnames)+lgb_n_trees+len(bag_feats)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    df_sub = pd.read_csv(instance_ids_fname)

    initLogging(version_dirs+'test2.log')
    logging.info('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    
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
 
    model_files = [version_dirs+'/snap/'+file for file in model_files]

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
    logging.info('model:\n{}\n'.format(pprint.pformat(model.__dict__)))

    submit(model,test_set,model_files,df_sub,save_dir=version_dirs+'/test2/')