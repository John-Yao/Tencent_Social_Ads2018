# Tencent_Social_Ads2018
2018腾讯社交广告33名的nn方案,单模75+
# 特征工程
- 原始特征低频hash处理
- 多值特征训练时truncate、网络里做pooling,truncate 参数如下：
```python
  bag_feats = ['kw1','kw2','kw3','topic1','topic2','topic3',
        'appIdAction','appIdInstall',
        'interest1',
        'interest2',
        'interest3',
        'interest4',
        'interest5']
  bag_value_cnts =[5, 5, 5, 5, 5, 5, 50, 50, 35, 30, 10, 10, 70]
 ```
- 增加用户的全集出现次数特征
- 统计特征主要用了下面3个，直接分箱（下面3个特征可以提1-1.5个千，因为提特征和编码过程和其他无用特征混在一起，所以训练和提特征我删除了下面3个特征）：
    - aid_age_mean,uid_creativeSize_mean,aid_creativeSize_mean
# 网络
- lr
- fm
- cin(xdeepfm,复现参考的tensorflow版，跟原论文有点出入)
- din(代替多值特征的mean pooling,max pooling没有尝试，目前网络支持mean pooling 和din共存;复现参考的tensorflow版，跟原论文有点出入)
- dnn
# Dataset(训练过程):

# 损失函数
- 交叉熵ce
- focalloss(这个线下精细调参的话可以比ce的1个千，但直接用跟ce同样参数一般都是掉几个千)，主要用于融合
# 环境配置
- Ubuntu 16.04
- python 3.6
- pytorch 0.4.0(0.2.x,0.3.x版本不支持，主要是不支持torch.where这个api)
- GPU:  12G
- Memory: >140G(主要是多值特征编码合并多列时需要用较大的内存)
- sklearn:0.19.1
- glob
# 目录介绍
- /preprocess/:     
   - 0_user_feat_transform.py：流式转化初赛和复赛的.data文件为.csv格式
   - 1_slice_merge.py：合并复赛数据（将数据切成两片，分开merge再concat）
- /feature/:
   - 2_get_feat_low_case_split_with_pre.py: 提取特征、编码、切分训练验证集
- /train_dl_with_pre：
  - 3-train_add_feature_0612.py：训练
  - 4-test_din_xdeepfm_addFeat_0612.py：测试
- /data/：
  - /final/
  - /preliminary_contest_data/
- /tmp/：
  
