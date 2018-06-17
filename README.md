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
# 训练过程:

# 训练时间：
  一个epoch，4核大概1-1.5小时
# 损失函数
- 交叉熵ce
- focalloss(这个线下精细调参的话可以比ce的1个千，但直接用跟ce同样参数一般都是掉几个千)，主要用于融合
# 环境配置
- Ubuntu 16.04
- python 3.6
- pytorch 0.4.0(0.2.x,0.3.x版本不支持，主要是不支持torch.where这个api)
- sklearn:0.19.1
- glob
- GPU:  12G(8G的应该能跑，训练集batch_size=8192时，训练显存占用5G+,验证的batch_size调小应该可以的)
- Memory: >140G(主要是多值特征编码合并多列时需要用较大的内存)
- disk: about 300G（tmp文件夹就有200+G主要是一些临时文件想重复用就没删掉）
- number of process: >4(单核能跑，只是比较慢)
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
  - /final/:
    - /merged/: 切片后保存的临时文件以及merged后的原始复赛数据
  - /preliminary_contest_data/:原始初赛数据
- /tmp/：
  - /0610_get_feat_with_pre/
    - /sub/:   初赛复赛训练集以及测试集的instance id
    - /encoded_lowcase_hash200/
      - merged_bag.csv：merge后的复赛多值特征预处理原始特征
      - merged_cate.csv:merge后的复赛类别特征预处理原始特征
      - pre_merged_bag.csv：merge后的初赛多值特征预处理原始特征
      - pre_merged_cate.csv:merge后的初赛类别特征预处理原始特征
      - .pkl: 全数据集各种编码后的单列特征
      - /pkl/
        - /bag_enc_dict/：多值特征编码器
        - /cate_encoder/：类别特征编码器
      - /split/
        - /trains/:
          - .pkl:  初赛复赛训练集各种编码后的单列特征（已经切分）
          - /instance_ids/： 训练集instance id（已经切分）
          - /merged/:
            - x_bag.pkl: concat后的复赛多值特征（已经编码了）
            - pre_x_bag.pkl: concat后的初赛多值特征（已经编码了）
        - /test1/:
          - .pkl:  测试集各种编码后的单列特征
          - /merged/:
            - bag.pkl: concat后的所有多值特征（已经编码了）
        - /test2/: 同test1
    - merged_bag.csv：merge后的复赛多值特征原始数据
    - merged_cate.csv:merge后的复赛类别特征原始数据
    - pre_merged_bag.csv：merge后的初赛多值特征原始数据
    - pre_merged_cate.csv:merge后的初赛类别特征原始数据
    - labels.pkl: 复赛的标签
    - pre_labels.pkl:初赛的标签
    
    
  
