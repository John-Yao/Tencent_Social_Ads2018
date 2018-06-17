#!/bin/sh
cd ./preprocess/
python ./0_user_feat_transform.py
python ./1_slice_merge.py
cd ..

cd ./feature/
python ./2_get_feat_low_case_split_with_pre.py
cd ..


cd ./train_dl_with_pre/
python 3-train_add_feature_0612.py
python 4-test_din_xdeepfm_addFeat_0612.py
cd ..
