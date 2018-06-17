import pandas as pd

data_dir = '../data/final/'
train_slice_num = 2
save_dir = '../data/final/merged/'
import os
import gc
os.makedirs(save_dir,exist_ok=True)

ad_feature = pd.read_csv(data_dir + 'adFeature.csv')
train = pd.read_csv(data_dir + 'train.csv')
train['label'] = train['label'].map({1: 1, -1: 0})
test1 = pd.read_csv(data_dir + 'test1.csv')
test2 = pd.read_csv(data_dir + 'test2.csv')
print('load train and test finished!')
user_feature = pd.read_csv(data_dir+'userFeature.csv')
print('load user_feature finished!')

slice_len = train.shape[0]//train_slice_num
df_slices = []
for i in range(1,train_slice_num):
    df_slices.append(train.iloc[(i-1)*slice_len:i*slice_len,:])
df_slices.append(train.iloc[(train_slice_num-1)*slice_len:,:])
# merge

# test1
print('merge on test1')
df_merge_on = test1
save_fname = save_dir+'test1.csv'
df_merge_on = pd.merge(df_merge_on, ad_feature, on='aid',how='left')
df_merge_on = pd.merge(df_merge_on, user_feature, on='uid',how='left')
df_merge_on.to_csv(save_fname,index=False)
del df_merge_on
gc.collect()
# test2
print('merge on test2')
df_merge_on = test2
save_fname = save_dir+'test2.csv'
df_merge_on = pd.merge(df_merge_on, ad_feature, on='aid',how='left')
df_merge_on = pd.merge(df_merge_on, user_feature, on='uid',how='left')
df_merge_on.to_csv(save_fname,index=False)
del df_merge_on
gc.collect()
print('merge on train')
for i,df_merge_on in enumerate(df_slices):
    save_fname = save_dir+'train_'+str(i)+'.csv'
    print('merge on train_'+str(i))
    df_merge_on = pd.merge(df_merge_on, ad_feature, on='aid',how='left')
    df_merge_on = pd.merge(df_merge_on, user_feature, on='uid',how='left')
    df_merge_on.to_csv(save_fname,index=False)
    del df_merge_on
    gc.collect()

del user_feature
del ad_feature
del df_slices
gc.collect()

df_slices = []
for i in range(train_slice_num):
    df_slices.append(pd.read_csv(save_dir+'train_'+str(i)+'.csv'))
    print('train_'+str(i)+' is read!')
df_trains = pd.concat(df_slices)
df_trains.to_csv(save_dir+'trains.csv',index=False)