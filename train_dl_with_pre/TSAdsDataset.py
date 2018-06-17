# -*- coding:utf-8 -*-
import torch.utils.data as data
import torch
import numpy as np
import os
import pandas as pd
import pickle
import random

import torch.utils.data as data
import torch
import numpy as np
import os
import pandas as pd
import pickle
import gc
import time


def shuffle_trp(l, n):
    '''
        Ref:
            https://stackoverflow.com/questions/30475558/padding-or-truncating-a-python-list
    '''
    random.shuffle(l)
    truncated = l[:n]
    return len(truncated),truncated + [0]*(n-len(l))
class StackFileBagMeanTSAdsDataset(data.Dataset):
    """docstring for StackFileBagMeanTSAdsDataset"""
    def __init__(self, bag_value_cnts,bag_fname,\
                cate_load_dir,cate_featnames, cate_fn_prefix,cate_lbenc_dict,\
                instance_ids_fname, labels_fname,\
                is_use_pre=False,
                pre_bag_fname=None,\
                pre_cate_fn_prefix=None,\
                pre_instance_ids_fname=None, pre_labels_fname=None,

                lgb_n_trees=0,
                lgb_n_leaves=127,
                random_seed =491001):
        super(StackFileBagMeanTSAdsDataset, self).__init__()
        self.cate_featnames = cate_featnames
        self.cate_fn_prefix = cate_fn_prefix
        self.cate_load_dir = cate_load_dir
        self.cate_lbenc_dict = cate_lbenc_dict

        self.bag_value_cnts = bag_value_cnts
        self.bag_fname = bag_fname

        self.instance_ids_fname = instance_ids_fname
        self.labels_fname = labels_fname

        self.is_use_pre = is_use_pre #whether use the data from pre-test
        self.pre_bag_fname=pre_bag_fname
        self.pre_cate_fn_prefix=pre_cate_fn_prefix
        self.pre_instance_ids_fname=pre_instance_ids_fname
        self.pre_labels_fname=pre_labels_fname
        if self.is_use_pre == True:
            assert pre_bag_fname is not None,'please make sure: pre_bag_fname is not None'
            assert pre_cate_fn_prefix is not None,'please make sure: pre_cate_fn_prefix is not None'
            assert pre_instance_ids_fname is not None,'please make sure: pre_instance_ids_fname is not None'
            assert pre_labels_fname is not None,'please make sure: pre_labels_fname is not None'

        self.lgb_n_trees = lgb_n_trees
        self.lgb_n_leaves = lgb_n_leaves
        self.Xi = None
        self.bag_feats = None
        self.groupby_ids = None
        self.y = None
        self.cate_p = 0
        self.df_instances=None

        if self.instance_ids_fname.split('.')[-1]=='pkl':
            with open(self.instance_ids_fname,'rb') as f:
                self.df_instances = pickle.load(f)
            if self.is_use_pre == True:
                with open(self.pre_instance_ids_fname,'rb') as f:
                    pre_df_instances = pickle.load(f)
                self.df_instances = pd.concat([pre_df_instances,self.df_instances])
        else:
            self.df_instances = pd.read_csv(self.instance_ids_fname)
            if self.is_use_pre == True:
                pre_df_instances = pd.read_csv(self.pre_instance_ids_fname)
                self.df_instances = pd.concat([pre_df_instances,self.df_instances])
        self.groupby_ids = self.df_instances['aid'].tolist()
        if self.labels_fname is not None:
            with open(self.labels_fname,'rb') as f:
                self.y = pickle.load(f)
            if self.is_use_pre == True:
                with open(self.pre_labels_fname,'rb') as f:
                    pre_y = pickle.load(f)
                self.y = pre_y+self.y
        self.has_label =True
        if self.y is None or len(self.y)==0:
            self.has_label=False
        random.seed(random_seed)
    def clear_cache(self):
        del self.Xi
        del self.bag_feats

        self.Xi = None
        self.bag_feats = None

    def load_cache(self):
        #stack cate featuers
        feature_sizes = []

        for feature in self.cate_featnames:
            feature_sizes.append(self.cate_lbenc_dict[feature].classes_.shape[0])
        if self.lgb_n_trees !=0 :
            for i in range(self.lgb_n_trees):
                feature_sizes.append(self.lgb_n_leaves)
        self.cate_p = sum(feature_sizes)

        features = []
        idx = 0
        for _,feat in enumerate(self.cate_featnames):
            offset = sum(feature_sizes[:idx])
            idx = idx+1
            with open(self.cate_load_dir+self.cate_fn_prefix+feat+'.pkl','rb') as f:
                cate_feat = pickle.load(f)+offset
            if self.is_use_pre ==True:
                with open(self.cate_load_dir+self.pre_cate_fn_prefix+feat+'.pkl','rb') as f:
                    pre_cate_feat = pickle.load(f)+offset
                    cate_feat = np.vstack([pre_cate_feat,cate_feat])
            features.append(cate_feat)

        if self.lgb_n_trees !=0 :
            feat = 'lgb'
            with open(self.cate_load_dir+self.cate_fn_prefix+feat+'.pkl','rb') as f:
                lgb_feats = pickle.load(f)
                for i in range(self.lgb_n_trees):
                    offset = sum(feature_sizes[:idx])
                    idx = idx+1
                    lgb_feats[:,i] = lgb_feats[:,i]+offset
                
            if self.is_use_pre==True:
                with open(self.cate_load_dir+self.pre_cate_fn_prefix+feat+'.pkl','rb') as f:
                    pre_lgb_feats = pickle.load(f)
                    for i in range(self.lgb_n_trees):
                        offset = sum(feature_sizes[:idx])
                        idx = idx+1
                        pre_lgb_feats[:,i] = pre_lgb_feats[:,i]+offset
                lgb_feats = np.vstack([pre_lgb_feats,lgb_feats])
            features.append(lgb_feats)
            
        self.Xi = np.hstack(features)
        #load bag features
        with open(self.bag_fname,'rb') as f:
            self.bag_feats = pickle.load(f)
        if self.is_use_pre==True:
            with open(self.pre_bag_fname,'rb') as f:
                pre_bag_feats = pickle.load(f)
            self.bag_feats = np.vstack([pre_bag_feats,self.bag_feats])
    def truncate_bag_feats(self,bag_feats):
        bag_Xi = []
        bag_lenghts = []
        for i,bag_feat in enumerate(bag_feats):
            splited_values = bag_feat.split(' ')
            splited_values = [int(value) for value in splited_values]
            lens,padding_values = shuffle_trp(splited_values,self.bag_value_cnts[i])
            bag_Xi.extend(padding_values)
            bag_lenghts.append(float(lens))
       
        return bag_Xi,bag_lenghts
    def __getitem__(self,index):
        Xi=self.Xi[index]
        Xv=torch.ones(Xi.shape)
        # parse bag Xi from str
        bag_Xi,bag_lenghts = self.truncate_bag_feats(self.bag_feats[index])
        if self.has_label:
            y = self.y[index]
            return index,self.groupby_ids[index],torch.from_numpy(Xi),Xv,torch.LongTensor(bag_Xi),torch.FloatTensor(bag_lenghts),torch.LongTensor([y])
        else:
            return index,self.groupby_ids[index],torch.from_numpy(Xi),Xv,torch.LongTensor(bag_Xi),torch.FloatTensor(bag_lenghts)

    def __len__(self):
        return len(self.groupby_ids)
def get_lb_encorder(label_enc_features,pkl_dir):
    label_enc_dict = {}
    for feature in label_enc_features:
        dump_fname = pkl_dir+feature+'.pkl'
        with open(dump_fname,'rb') as f:
            label_enc_dict[feature] = pickle.load(f)
    return label_enc_dict
if __name__ == '__main__':

    root_dir = '../tmp/0610_get_feat_with_pre/'
    encoded_dir = root_dir+'encoded_lowcase_hash200/'
    cate_enc_pkl_dir = encoded_dir+'pkl/cate_encoder/'
    bag_dict_pkl_dir = encoded_dir+'pkl/bag_enc_dict/'

    # cate features 
    base_lbenc_feats = ['creativeId','LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus',
                'advertiserId','campaignId', 'adCategoryId', 'productId', 'productType','creativeSize']

    cate_featnames = base_lbenc_feats

    cate_lbenc_dict = get_lb_encorder(cate_featnames,cate_enc_pkl_dir)


    bag_value_cnts =[5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10]
    i = 0
    cate_fn_prefix = str(i)+'_'
    pre_cate_fn_prefix = 'pre_'+str(i)+'_'
    cate_load_dir = encoded_dir+'split/trains/'
    bag_fname = cate_load_dir+'merged/'+cate_fn_prefix+'bag.pkl'
    instance_ids_fname = cate_load_dir+'/instance_ids/'+cate_fn_prefix+'instance_id.pkl'
    labels_fname = cate_load_dir+'labels/'+cate_fn_prefix+'label.pkl'
    pre_bag_fname = cate_load_dir+'merged/'+pre_cate_fn_prefix+'bag.pkl'
    pre_instance_ids_fname = cate_load_dir+'/instance_ids/'+pre_cate_fn_prefix+'instance_id.pkl'
    pre_labels_fname = cate_load_dir+'labels/'+pre_cate_fn_prefix+'label.pkl'
    dataset = StackFileBagMeanTSAdsDataset(bag_value_cnts,bag_fname,\
                cate_load_dir,cate_featnames, cate_fn_prefix,cate_lbenc_dict,\
                instance_ids_fname, labels_fname,\
                is_use_pre=False,
                pre_bag_fname=pre_bag_fname,\
                pre_cate_fn_prefix=pre_cate_fn_prefix,\
                pre_instance_ids_fname=pre_instance_ids_fname, pre_labels_fname=pre_labels_fname,
                lgb_n_trees=0,lgb_n_leaves=127
                )
    # dataset.is_use_pre=False
    #about 20s
    start_time = time.time()
    dataset.load_cache()
    print(time.time()-start_time)
    # input('press any key to continue..')
    # from torch.utils.data.sampler import *
    # from torch.utils.data import DataLoader
    # start_time = time.time()
    # #168s without gc 
    # train_loader = DataLoader(
    #                     dataset,
    #                     sampler = RandomSampler(dataset),
    #                     # shuffle= True,
    #                     batch_size = 8192,
    #                     drop_last   = True,
    #                     num_workers = 4,
    #                     pin_memory = False
    #             )
    # for _,data in enumerate(train_loader):
    #     pass
    #     gc.collect()
    # print(time.time()-start_time)
    # for data in dataset:
    #     pass
    input('press any key to continue..')
    print(dataset[0])
    print(len(dataset))
    # https://github.com/pytorch/pytorch/issues/5902