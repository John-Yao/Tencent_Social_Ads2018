import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import gc
import itertools
from collections import Counter
import glob
from sklearn.cross_validation  import StratifiedKFold
import time

def cate_enc_fit(df_data,label_enc_featnames,outdir,refresh=False):
    pkl_dir = outdir+'/pkl/cate_encoder/'
    label_enc_dict = {}
    os.makedirs(pkl_dir,exist_ok=True)
    for feature in label_enc_featnames:
        dump_fname = pkl_dir+feature+'.pkl'
        if os.path.exists(dump_fname) and refresh==False:
            with open(dump_fname,'rb') as f:
                label_enc_dict[feature] = pickle.load(f)
        else:
            print("%s is label encoding"%(feature))
            enc = LabelEncoder()
            enc.fit(df_data[feature].astype(str))
            label_enc_dict[feature] = enc
            with open(dump_fname,'wb') as f:
                pickle.dump(enc,f)
            del enc
            gc.collect()
    return label_enc_dict

def bag_enc_fit(df_data,label_enc_featnames,outdir,refresh=False):
    pkl_dir = outdir+'/pkl/bag_enc_dict/'
    label_enc_dict = {}
    os.makedirs(pkl_dir,exist_ok=True)
    
    for feature in label_enc_featnames:        
        dump_fname = pkl_dir+feature+'.pkl'
        if os.path.exists(dump_fname) and refresh==False:
            with open(dump_fname,'rb') as f:
                label_enc_dict[feature] = pickle.load(f)
        else:
            print("%s is label encoding"%(feature))
            split_se = df_data[feature].apply(lambda x: str(x).split(' '))
            str_list = list(split_se)
            del split_se
            gc.collect()
            ravel = list(itertools.chain.from_iterable(str_list))
            del str_list
            gc.collect()
            ravel = np.unique(ravel)
            enc = {}
            v = 0
            for key in ravel:
                enc[key] = v
                v = v+1
            
            del ravel
            gc.collect()
            label_enc_dict[feature] = enc
            with open(dump_fname,'wb') as f:
                pickle.dump(enc,f)
    return label_enc_dict
def get_lb_encorder(label_enc_features,pkl_dir):
    label_enc_dict = {}
    for feature in label_enc_features:
        dump_fname = pkl_dir+feature+'.pkl'
        with open(dump_fname,'rb') as f:
            label_enc_dict[feature] = pickle.load(f)
    return label_enc_dict
def cate_enc_transform(df_data,label_enc_features,label_enc_dict,outdir,refresh=False):
    pkl_dir = outdir
    os.makedirs(pkl_dir,exist_ok=True)
    print('---------------------label_encoder ------------------------- ')
    for i,feature in enumerate(label_enc_features): 
        dump_fname = pkl_dir+feature+'.pkl'
        if os.path.exists(dump_fname) and refresh==False:
            continue
        label_enc_value=label_enc_dict[feature].transform(list(df_data[feature].astype(str)))
        label_enc_value = label_enc_value.reshape(-1,1)
        with open(dump_fname,'wb') as f:
            pickle.dump(label_enc_value,f,protocol=4)
        print(feature,' is encoded...')

def bag_enc_transform(df_data,label_enc_features,label_enc_dict,outdir,refresh=False):
    pkl_dir = outdir
    os.makedirs(pkl_dir,exist_ok=True)
    
    print('---------------------bag label_encoder ------------------------- ')
    for i,feature in enumerate(label_enc_features): 
        dump_fname = pkl_dir+feature+'.pkl'
        if os.path.exists(dump_fname) and refresh==False:
            continue
        #(n_samples,)
        split_se = df_data[feature].apply(lambda x: str(x).split(' '))
        transform_se = split_se.apply(lambda x: [label_enc_dict[feature][_x] for _x in x])
        del split_se
        gc.collect()
        
        transform_se = transform_se.apply(lambda x: ' '.join([str(_x) for _x in x]))
        label_enc_value = transform_se.values.reshape(-1,1)
        with open(dump_fname,'wb') as f:
            pickle.dump(label_enc_value,f,protocol=4)
        print(feature,' is encoded...')
# hash lowcase
import hashlib
def hashstr(_str, nr_bins):
    return int(hashlib.md5(_str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1
def hashint(_int,nr_bins):
    return hashstr(str(_int),nr_bins)
def freq_hash_enc_lowcase(se,lowest_freq=200,low_freq_nr_bins=20,verbose=False):
    se_str = se.astype(str)
    ravel_se = list(se_str)
    freq_counter = Counter(ravel_se)
    if verbose == True:
        print(len({key:value for key,value in freq_counter.items() if value<lowest_freq}))
    
    freq_counter = { key: int(float(key))+low_freq_nr_bins+1 if value>lowest_freq else hashstr(key,low_freq_nr_bins) for key,value in freq_counter.items() }
    freq_counter = { key: value if value!=low_freq_nr_bins else -1 for key,value in freq_counter.items() }
    lbf_se = se_str.apply(lambda x:freq_counter[x])
    return lbf_se
def cross_feat_freq_hash_enc_lowcase(se,lowest_freq=200,low_freq_nr_bins=20,verbose=False):
    se_str = se.astype(str)
    ravel_se = list(se_str)
    freq_counter = Counter(ravel_se)
    if verbose == True:
        print(len({key:value for key,value in freq_counter.items() if value<lowest_freq}))
    
    freq_counter = { key: key if value>lowest_freq else str(hashstr(key,low_freq_nr_bins)) for key,value in freq_counter.items() }
    lbf_se = se_str.apply(lambda x:freq_counter[x])
    return lbf_se
#marriageStatus 
def process_marriageStatus(se,mvf_lowest_freq=200,lowest_freq=200,keep_num=1,verbose=False): 
    sub_dict = {'5 13 10':'13 10',
                '13 15 10':'13 15',
                '6 13 10' :'13 10',
                '8':'-1',
                '2 13 10':'13 10',
                '13 10 9' :'13 10',
                '1': '-1'}
#     str_se = se.astype(str)
    ravel_se = list(se)
    freq_counter = Counter(ravel_se)
    freq_counter = {key:key if key not in sub_dict.keys() else sub_dict[key] for key,value in freq_counter.items()}
    if verbose == True:
        print(freq_counter)
    se = se.apply(lambda x: freq_counter[x] )
    return se
# https://blog.csdn.net/EricLeiy/article/details/78712675
def dupe(items):
    seen = set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)
import itertools
from collections import Counter
#marriageStatus 
def mvf_freq_hash_lowcase(se,mvf_lowest_freq=20,low_freq_nr_bins=20,verbose=False): 
    se = se.fillna('-1')
    split_se = se.apply(lambda x: str(x).split(' '))
    list_se = list(split_se)
    ravel_se = list(itertools.chain.from_iterable(list_se))
    mvf_freq_counter = Counter(ravel_se)
    tmp = mvf_freq_counter
    
    mvf_freq_counter = {key: int(int(key))+low_freq_nr_bins+1 if value>mvf_lowest_freq else hashstr(key,low_freq_nr_bins) for key,value in mvf_freq_counter.items()   }
    mvf_freq_counter = {key: value if value!=low_freq_nr_bins else -1 for key,value in mvf_freq_counter.items() }
    mvf_freq_counter = {key: str(value) for key,value in mvf_freq_counter.items()}
    split_se = split_se.apply(lambda x: [mvf_freq_counter[_x] for _x in x])
#     split_se = split_se.apply(lambda x: sorted(set(x),key=x.index))
    split_se = split_se.apply(lambda x: list(dupe(x)))
    str_se = split_se.apply(lambda x: ' '.join(x))
    
    list_se = list(split_se)
    ravel_se = list(itertools.chain.from_iterable(list_se))
    mvf_freq_counter = Counter(ravel_se)
    if verbose==True:
        print(len(tmp))
        print(len(mvf_freq_counter))
    del split_se
    gc.collect()
    return str_se
## add feat
if __name__ == '__main__':
    # init variable 
    mvf_min_freq = 200
    pre_root_dir = '../data/preliminary_contest_data/'
    version_root_dir = '../tmp/0610_get_feat_with_pre/'
    root_dir = version_root_dir
    save_dir = root_dir+'encoded_lowcase_hash'+str(mvf_min_freq)+'/'
    cate_enc_pkl_dir = save_dir+'pkl/cate_encoder/'
    bag_dict_pkl_dir = save_dir+'pkl/bag_enc_dict/'
    
    os.makedirs(cate_enc_pkl_dir,exist_ok=True)
    os.makedirs(bag_dict_pkl_dir,exist_ok=True)

    # [warning] according to actual size
    pre_train_slice_last = 8798814
    pre_test1_slice_last = 11064803
    pre_test2_slice_last = 13330682

    '''
        load final 

    '''
    if 1:
        final_raw_dir = '../data/final/merged/'
        df_trains = pd.read_csv(final_raw_dir+'trains.csv')
        df_test1 = pd.read_csv(final_raw_dir+'test1.csv')
        df_test2 = pd.read_csv(final_raw_dir+'test2.csv')
        df_test1['label']=-1
        df_test2['label']=-2
        df_data = pd.concat([df_trains,df_test1,df_test2]).reset_index(drop=True)
        train_slice_last=len(df_trains)
        test1_slice_last=len(df_trains)+len(df_test1)
        del df_test1
        del df_trains
        del df_test2
        # 
        cate_features=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus',
                'advertiserId','campaignId', 'creativeId','adCategoryId', 'productId', 'productType','creativeSize']+['uid','aid','label']
        #13
        bag_features = ['kw1','kw2','kw3','topic1','topic2','topic3',
            'appIdAction','appIdInstall',
            'interest1',
            'interest2',
            'interest3',
            'interest4',
            'interest5']+['uid','aid','label']
        df_data[cate_features].to_csv(version_root_dir+'merged_cate.csv',index=False)
        df_data[bag_features].to_csv(version_root_dir+'merged_bag.csv',index=False)
        labels = df_data.iloc[:train_slice_last,:]['label'].tolist()
        with open(version_root_dir+'labels.pkl','wb') as f:
            pickle.dump(labels,f,protocol=4)

    '''
        load pre data ,merge and split to cate and bag feats
    '''
    if 1:
        ad_feats = pd.read_csv(pre_root_dir+'adFeature.csv')
        print("loading user_feats------------------")
        user_feats = pd.read_csv(pre_root_dir+'userFeature.csv')
        print("loading trains    ------------------")
        trains = pd.read_csv(pre_root_dir+'train.csv')
        print("loading tests     ------------------")
        test1 = pd.read_csv(pre_root_dir+'test1.csv')
        test2 = pd.read_csv(pre_root_dir+'test2.csv')
        train_slice_last=len(trains)
        test1_slice_last=len(trains)+len(test1)
        test2_slice_last=len(trains)+len(test1)+len(test2)
        print('train_slice_last:',train_slice_last)
        print('test1_slice_last:',test1_slice_last)
        print('test2_slice_last:',test2_slice_last)
        trains.loc[trains['label']==-1,'label']=0
        test1['label']=-1
        test2['label']=-2
        print("concat ---------------------")
        df_data=pd.concat([trains,test1,test2]).reset_index(drop=True)
        
        print("merge ---------------------")
        df_data=pd.merge(df_data,ad_feats,on='aid',how='left')
        df_data=pd.merge(df_data,user_feats,on='uid',how='left')
        # 
        cate_features=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus',
                'advertiserId','campaignId', 'creativeId','adCategoryId', 'productId', 'productType','creativeSize']+['uid','aid','label']
        #13
        bag_features = ['kw1','kw2','kw3','topic1','topic2','topic3',
            'appIdAction','appIdInstall',
            'interest1',
            'interest2',
            'interest3',
            'interest4',
            'interest5']+['uid','aid','label']
        df_data[cate_features].to_csv(version_root_dir+'pre_merged_cate.csv',index=False)
        df_data[bag_features].to_csv(version_root_dir+'pre_merged_bag.csv',index=False)
        labels = df_data.iloc[:train_slice_last,:]['label'].tolist()
        with open(version_root_dir+'pre_labels.pkl','wb') as f:
            pickle.dump(labels,f,protocol=4)

    '''
        load sub data and save to sub
    '''
    if 1:
        #save pre sub
        #save final sub
        sub_dir = pre_root_dir
        df_trains = pd.read_csv(sub_dir+'train.csv')
        df_test1 = pd.read_csv(sub_dir+'test1.csv')
        df_test2 = pd.read_csv(sub_dir+'test2.csv')
        os.makedirs(root_dir+'/sub/',exist_ok=True)
        sub_trains = df_trains[['aid','uid']]
        sub_trains.to_csv(root_dir+'/sub/pre_trains.csv',index=False)
        del sub_trains
        gc.collect()
        sub_test1 = df_test1[['aid','uid']]
        sub_test1.to_csv(root_dir+'/sub/pre_test1.csv',index=False)
        del sub_test1
        gc.collect()
        sub_test2 = df_test2[['aid','uid']]
        sub_test2.to_csv(root_dir+'/sub/pre_test2.csv',index=False)
        del sub_test2
        gc.collect()

        #save final sub
        sub_dir = '../data/final/'
        df_trains = pd.read_csv(sub_dir+'train.csv')
        df_test1 = pd.read_csv(sub_dir+'test1.csv')
        df_test2 = pd.read_csv(sub_dir+'test2.csv')
        os.makedirs(root_dir+'/sub/',exist_ok=True)
        sub_trains = df_trains[['aid','uid']]
        sub_trains.to_csv(root_dir+'/sub/trains.csv',index=False)
        del sub_trains
        gc.collect()
        sub_test1 = df_test1[['aid','uid']]
        sub_test1.to_csv(root_dir+'/sub/test1.csv',index=False)
        del sub_test1
        gc.collect()
        sub_test2 = df_test2[['aid','uid']]
        sub_test2.to_csv(root_dir+'/sub/test2.csv',index=False)
        del sub_test2
        gc.collect()
    '''
        low case
    '''

    if 1:
        # low case cate feature
        print('cate low case!')
        df_pre_data = pd.read_csv(root_dir+'pre_merged_cate.csv')
        df_data = pd.read_csv(root_dir+'merged_cate.csv')
        df_data = pd.concat([df_pre_data,df_data])
        df_data.fillna('-1',inplace=True)

        df_data['LBS']=freq_hash_enc_lowcase(df_data['LBS'])
        df_data['marriageStatus'] = process_marriageStatus(df_data['marriageStatus'])

        df_data.iloc[:df_pre_data.shape[0]].to_csv(save_dir+'pre_merged_cate.csv',index=False)
        df_data.iloc[df_pre_data.shape[0]:,].to_csv(save_dir+'merged_cate.csv',index=False)
        print('cate encoder fit!')
        label_enc_features=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus',
                'advertiserId','campaignId', 'creativeId','adCategoryId', 'productId', 'productType']+['creativeSize']
        cate_enc_fit(df_data,label_enc_features,save_dir,refresh=False)
        del df_data
  
    if 1:
        # low case
        print('bag low case!')
        df_pre_data = pd.read_csv(root_dir+'pre_merged_bag.csv')
        df_data = pd.read_csv(root_dir+'merged_bag.csv')
        df_data = pd.concat([df_pre_data,df_data])
        df_data.fillna('-1',inplace=True)
        for feat in ['appIdAction','appIdInstall','kw1','kw2','kw3','topic1','topic2','topic3']: 
            df_data[feat] = mvf_freq_hash_lowcase(df_data[feat],mvf_lowest_freq=mvf_min_freq,low_freq_nr_bins=20,verbose=True)
            print(feat,' is Done!')
        
        df_data.iloc[:df_pre_data.shape[0]].to_csv(save_dir+'pre_merged_bag.csv',index=False)
        df_data.iloc[df_pre_data.shape[0]:,].to_csv(save_dir+'merged_bag.csv',index=False)

        print('bag encoder fit!')
        bag_feats = ['kw1','kw2','kw3','topic1','topic2','topic3',
            'appIdAction','appIdInstall',
            'interest1',
            'interest2',
            'interest3',
            'interest4',
            'interest5']
  
        bag_label_enc_dict = bag_enc_fit(df_data,bag_feats,outdir = save_dir,refresh=False)
        del df_data
     
    # get all encoder
    base_lbenc_feats = ['creativeId','LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus',
                'advertiserId','campaignId', 'adCategoryId', 'productId', 'productType','creativeSize']
    
    if 1:
        # encoding base
        lb_encoder = get_lb_encorder(base_lbenc_feats,cate_enc_pkl_dir)
        
        df_pre_data = pd.read_csv(save_dir+'pre_merged_cate.csv')
        df_data = pd.read_csv(save_dir+'merged_cate.csv')
        df_data = pd.concat([df_pre_data,df_data])
        df_data.fillna('-1',inplace=True)

        cate_enc_transform(df_data,base_lbenc_feats,lb_encoder,save_dir,refresh=False)
        del df_data  
    if 1:
        # encoding bag
        bag_feats = ['kw1','kw2','kw3','topic1','topic2','topic3',
            'appIdAction','appIdInstall',
            'interest1',
            'interest2',
            'interest3',
            'interest4',
            'interest5']
        lb_encoder = get_lb_encorder(bag_feats,bag_dict_pkl_dir)
        
        df_pre_data = pd.read_csv(save_dir+'pre_merged_bag.csv')
        df_data = pd.read_csv(save_dir+'merged_bag.csv')
        df_data = pd.concat([df_pre_data,df_data])
        df_data.fillna('-1',inplace=True)

        bag_enc_transform(df_data,bag_feats,lb_encoder,save_dir)
        del df_data

    '''
    split 
    '''
    split_save_dir = save_dir+'/split/'
    os.makedirs(split_save_dir,exist_ok=True)
    os.makedirs(split_save_dir+'/test1/',exist_ok=True)
    os.makedirs(split_save_dir+'/test2/',exist_ok=True)
    os.makedirs(split_save_dir+'/trains/',exist_ok=True)
    os.makedirs(split_save_dir+'/trains/labels/',exist_ok=True)
    os.makedirs(split_save_dir+'/trains/instance_ids/',exist_ok=True)
    train_slice_last = pre_test2_slice_last+45539700
    test1_slice_last = train_slice_last+11729073
    if 1:
        # split pre labels
        with open(root_dir+'/pre_labels.pkl','rb') as f:
            print('loading '+root_dir+'/pre_labels.pkl')
            pre_trains_y = pickle.load(f)
        #save labels
        print('pre labels is being splitted!')
        skf=StratifiedKFold(pre_trains_y,n_folds = 10,shuffle=True,random_state=494101)
        arr_pre_trains_y = np.array(pre_trains_y)
        i = 0
        for _,valid_index in skf:
            with open(split_save_dir+'/trains/labels/pre_'+str(i)+'_label.pkl','wb') as f:
                pickle.dump(arr_pre_trains_y[valid_index].tolist(),f,protocol=4)
            i=i+1

        # split final label 
        trains_labels_fname = root_dir+'/labels.pkl'
        with open(trains_labels_fname,'rb') as f:
            print('loading '+trains_labels_fname)
            trains_y = pickle.load(f)

        #save labels
        print('labels is being splitted!')
        skf=StratifiedKFold(trains_y,n_folds = 10,shuffle=True,random_state=494101)
        arr_trains_y = np.array(trains_y)
        i = 0
        for _,valid_index in skf:
            with open(split_save_dir+'/trains/labels/'+str(i)+'_label.pkl','wb') as f:
                pickle.dump(arr_trains_y[valid_index].tolist(),f,protocol=4)
            i=i+1

        #split pre instance
        print('pre instance id is being splitted!')
        skf=StratifiedKFold(pre_trains_y,n_folds = 10,shuffle=True,random_state=494101)
        df_instances = pd.read_csv(root_dir+'/sub/'+'pre_trains.csv')
        i = 0
        for _,valid_index in skf:
            with open(split_save_dir+'/trains/instance_ids/pre_'+str(i)+'_instance_id.pkl','wb') as f:
                pickle.dump(df_instances.iloc[valid_index,:],f,protocol=4)
            i=i+1

        #split instance
        print('instance id is being splitted!')
        skf=StratifiedKFold(trains_y,n_folds = 10,shuffle=True,random_state=494101)
        df_instances = pd.read_csv(root_dir+'/sub/'+'trains.csv')
        i = 0
        for _,valid_index in skf:
            with open(split_save_dir+'/trains/instance_ids/'+str(i)+'_instance_id.pkl','wb') as f:
                pickle.dump(df_instances.iloc[valid_index,:],f,protocol=4)
            i=i+1
        #split feature
        fnames = glob.glob(save_dir+'*.pkl')
        for fname in fnames:
            with open(fname,'rb') as f:
                feature = pickle.load(f)
            shortname = fname.split('/')[-1]
            print(shortname+' is being splitted!')
            #save test1
            with open(split_save_dir+'/test1/'+shortname,'wb') as f:
                pickle.dump(feature[train_slice_last:test1_slice_last],f,protocol=4)
            #save test2
            with open(split_save_dir+'/test2/'+shortname,'wb') as f:
                pickle.dump(feature[test1_slice_last:],f,protocol=4)
            #split and save trains
            #save pre trains
            print('split pre trains')
            skf=StratifiedKFold(pre_trains_y,n_folds = 10,shuffle=True,random_state=494101)
            i = 0
            for _,valid_index in skf:
                with open(split_save_dir+'/trains/pre_'+str(i)+'_'+shortname,'wb') as f:
                    pickle.dump(feature[valid_index],f,protocol=4)
                i = i+1
            #save trains
            print('split trains')
            skf=StratifiedKFold(trains_y,n_folds = 10,shuffle=True,random_state=494101)
            i = 0
            for _,valid_index in skf:
                valid_index = [pre_test2_slice_last+idx for idx in valid_index]
                with open(split_save_dir+'/trains/'+str(i)+'_'+shortname,'wb') as f:
                    pickle.dump(feature[valid_index],f,protocol=4)
                i = i+1
    '''
	# merge all bag features
    '''
    if 1:
        
        bag_feats = ['kw1','kw2','kw3','topic1','topic2','topic3',
            'appIdAction','appIdInstall',
            'interest1',
            'interest2',
            'interest3',
            'interest4',
            'interest5']
    
        lb_encoder = get_lb_encorder(bag_feats,bag_dict_pkl_dir)
        feature_sizes = []
        for feature in bag_feats:
            feature_sizes.append(len(lb_encoder[feature])  )
        # merged trains
        load_dir = save_dir+'/split/trains/'
        print('merging pre bag feat!')
        for i in range(10):
            features = []
            for idx,feat in enumerate(bag_feats):
                print(i,':',feat)
                with open(load_dir+'pre_'+str(i)+'_'+feat+'.pkl','rb') as f:
                    df = pd.DataFrame( pickle.load(f),
                            columns=['a'])
                    split_se = df.a.apply(lambda x: str(x).split(' '))
                    offset = sum(feature_sizes[:idx])+1
                    transform_se = split_se.apply(lambda x: ' '.join([str(int(_x)+offset) for _x in x]))
                    features.append(transform_se.values.reshape(-1,1))
                    del df
                    del split_se
                    del transform_se
                    gc.collect()
            result = np.hstack(features)
            out_dir = load_dir+'/merged/'
            os.makedirs(out_dir,exist_ok=True)
            dump_fname = out_dir+'pre_'+str(i)+'_bag.pkl'
            with open(dump_fname,'wb') as f:
                pickle.dump(result,f)
            del result
        print('merging final bag feat!')
        for i in range(10):
            features = []
            for idx,feat in enumerate(bag_feats):
                print(i,':',feat)
                with open(load_dir+str(i)+'_'+feat+'.pkl','rb') as f:
                    df = pd.DataFrame( pickle.load(f),
                            columns=['a'])
                    split_se = df.a.apply(lambda x: str(x).split(' '))
                    offset = sum(feature_sizes[:idx])+1
                    transform_se = split_se.apply(lambda x: ' '.join([str(int(_x)+offset) for _x in x]))
                    features.append(transform_se.values.reshape(-1,1))
                    del df
                    del split_se
                    del transform_se
                    gc.collect()
            result = np.hstack(features)
            out_dir = load_dir+'/merged/'
            os.makedirs(out_dir,exist_ok=True)
            dump_fname = out_dir+str(i)+'_bag.pkl'
            with open(dump_fname,'wb') as f:
                pickle.dump(result,f)
            del result
    if 1:
        #merge tests
        bag_feats = ['kw1','kw2','kw3','topic1','topic2','topic3',
            'appIdAction','appIdInstall',
            'interest1',
            'interest2',
            'interest3',
            'interest4',
            'interest5']
        lb_encoder = get_lb_encorder(bag_feats,bag_dict_pkl_dir)
        feature_sizes = []
        for feature in bag_feats:
            feature_sizes.append(len(lb_encoder[feature])  )
        load_dir = save_dir+'/split/test1/'
        features = []
        for idx,feat in enumerate(bag_feats):
            print('test1:',feat)
            with open(load_dir+feat+'.pkl','rb') as f:
                df = pd.DataFrame( pickle.load(f),
                        columns=['a'])
                split_se = df.a.apply(lambda x: str(x).split(' '))
                offset = sum(feature_sizes[:idx])+1
                transform_se = split_se.apply(lambda x: ' '.join([str(int(_x)+offset) for _x in x]))
                features.append(transform_se.values.reshape(-1,1))
                del df
                del split_se
                del transform_se
                gc.collect()
        result = np.hstack(features)
        out_dir = load_dir+'/merged/'
        os.makedirs(out_dir,exist_ok=True)
        dump_fname = out_dir+'bag.pkl'
        with open(dump_fname,'wb') as f:
            pickle.dump(result,f)
        del result
        #test2
        load_dir = save_dir+'/split/test2/'
        features = []
        for idx,feat in enumerate(bag_feats):
            print('test2:',feat)
            with open(load_dir+feat+'.pkl','rb') as f:
                df = pd.DataFrame( pickle.load(f),
                        columns=['a'])
                split_se = df.a.apply(lambda x: str(x).split(' '))
                offset = sum(feature_sizes[:idx])+1
                transform_se = split_se.apply(lambda x: ' '.join([str(int(_x)+offset) for _x in x]))
                features.append(transform_se.values.reshape(-1,1))
                del df
                del split_se
                del transform_se
                gc.collect()
        result = np.hstack(features)
        out_dir = load_dir+'/merged/'
        os.makedirs(out_dir,exist_ok=True)
        dump_fname = out_dir+'bag.pkl'
        with open(dump_fname,'wb') as f:
            pickle.dump(result,f)

    if 1:
        '''
            # add feat uid_count
        '''
        sub_dir = pre_root_dir
        df_pre_trains = pd.read_csv(sub_dir+'train.csv')
        df_pre_trains['label'] = df_pre_trains['label'].map({1: 1, -1: 0})
        df_pre_test1 = pd.read_csv(sub_dir+'test1.csv')
        df_pre_test2 = pd.read_csv(sub_dir+'test2.csv')
        
        df_pre_test1['label']=-1
        df_pre_test2['label']=-2

        raw_dir = '../data/final/'
        df_trains = pd.read_csv(raw_dir+'train.csv')
        df_trains['label'] = df_trains['label'].map({1: 1, -1: 0})
        df_test1 = pd.read_csv(raw_dir+'test1.csv')
        df_test2 = pd.read_csv(raw_dir+'test2.csv')
        # bug fix
        df_test1['label']=-1
        df_test2['label']=-2
        df_data = pd.concat([df_pre_trains,df_pre_test1,df_pre_test2,df_trains,df_test1,df_test2]).reset_index(drop=True)
   
        feat = 'uid'
        if 1:
            df_count = df_data.groupby(feat).size().reset_index().rename(columns={0: feat+'_count'})
            df_data = pd.merge(df_data, df_count, on=feat,how='left')

            label_enc_features = ['uid_count']
        
            cate_enc_fit(df_data,label_enc_features,save_dir,refresh=False)

            lb_encoder = get_lb_encorder(label_enc_features,cate_enc_pkl_dir)
            
            cate_enc_transform(df_data,label_enc_features,lb_encoder,save_dir,refresh=False)
        #split
        label_enc_features = ['uid_count']
        fnames = [save_dir+feat+'.pkl' for feat in label_enc_features]
        split_save_dir = save_dir+'/split/'

        trains_labels_fname = root_dir+'/labels.pkl'
        with open(trains_labels_fname,'rb') as f:
            print('loading '+trains_labels_fname)
            trains_y = pickle.load(f)
      
        with open(root_dir+'/pre_labels.pkl','rb') as f:
            print('loading '+root_dir+'/pre_labels.pkl')
            pre_trains_y = pickle.load(f)

        for fname in fnames:
            with open(fname,'rb') as f:
                feature = pickle.load(f)
            shortname = fname.split('/')[-1]
            print(shortname+' is being splitted!')
            #save test1
            with open(split_save_dir+'/test1/'+shortname,'wb') as f:
                pickle.dump(feature[train_slice_last:test1_slice_last],f,protocol=4)
            #save test2
            with open(split_save_dir+'/test2/'+shortname,'wb') as f:
                pickle.dump(feature[test1_slice_last:],f,protocol=4)
            #split and save trains
            #save pre trains
            print('split pre trains')
            skf=StratifiedKFold(pre_trains_y,n_folds = 10,shuffle=True,random_state=494101)
            i = 0
            for _,valid_index in skf:
                with open(split_save_dir+'/trains/pre_'+str(i)+'_'+shortname,'wb') as f:
                    pickle.dump(feature[valid_index],f,protocol=4)
                i = i+1
            #save trains
            print('split trains')
            skf=StratifiedKFold(trains_y,n_folds = 10,shuffle=True,random_state=494101)
            i = 0
            for _,valid_index in skf:
                valid_index = [pre_test2_slice_last+idx for idx in valid_index]
                with open(split_save_dir+'/trains/'+str(i)+'_'+shortname,'wb') as f:
                    pickle.dump(feature[valid_index],f,protocol=4)
                i = i+1