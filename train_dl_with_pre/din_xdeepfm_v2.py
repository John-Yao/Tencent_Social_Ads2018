#pytorch version 0.40
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
import logging

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch.backends.cudnn
class InterestPooling(torch.nn.Module):
    """docstring for InterestPooling"""
    def __init__(self, k,use_cuda=True):
        super(InterestPooling, self).__init__()
        self.k = k
        self.activation = F.sigmoid
        self.d_l1 = nn.Linear(k*4,40)
        self.d_l2 = nn.Linear(40,20)
        self.d_l3 = nn.Linear(20,1)
        self.last_activate_layer = nn.Softmax(-1)
        self.use_cuda = use_cuda
    def forward(self,queries,keys,bags_Xi):
        '''
        Args:
            queries: [N,1,K]
            keys: [N,T,K]
            bags_Xi:[N,T]
        '''
        queries = queries.repeat(1,keys.size(1),1) #[N,T,K]
        din_all = torch.cat([queries,keys,queries-keys,queries*keys],-1) #[N,T,4K]
        #linear 
        din_att = self.d_l1(din_all)
        din_att = self.activation(din_att)
        din_att = self.d_l2(din_att)
        din_att = self.activation(din_att)
        din_att = self.d_l3(din_att)#[N,T,1]
        
        din_att = din_att.view(-1,1,keys.size(1))#[N,1,T]

        #Mask for paddings
        paddings = torch.ones(din_att.size())*(-2**32+1)
        if self.use_cuda:
            paddings= paddings.cuda()
        bags_Xi = bags_Xi.unsqueeze(dim=1)
        din_att = torch.where(bags_Xi!=0,din_att,paddings)
        #scale 
        # din_att = din_att/(keys.size(-1)**0.5)
        din_att = self.last_activate_layer(din_att)
        #[N,T,1]
        return torch.matmul(din_att,keys)#[N,1,K]

class ExDeepFM(torch.nn.Module):
    """
    Args:
    -------------
        field_num:  total num of field,including single-value(sv) and multi-value(bag)
        p_sv:       total embedding dim1 of single-value feature
        p_bag:      total embedding dim1 of bag feature
        bag_value_cnts: the value count of each bag
        embedding_size: size of the feature embedding
        n_class: number of classes. is bounded to 2
        use_cuda:
        use_lr:
        use_fm :
        use_deep :
        use_cin:
        use_din :

        is_fm_dropout: bool, shallow part(fm or ffm part) uses dropout or not?
        dropout_fm:     for the second-order fm part
        deep_layers:        a h_depth-dim array, each element is the size of corresponding hidden layers. example:[32,32] h_depth = 2
        is_deep_dropout:    bool, deep part uses dropout or not?
        dropout_deep:       an array of dropout factors,example:[0.5,0.5,0.5] h_depth=2
        deep_layers_activation: relu or sigmoid etc
        is_deep_bn：bool,  use batch_norm or not ?
        
        optimizer_type: optimizer_type, 'adam', 'rmsp', 'sgd', 'adag'
        weight_decay: weight decay (L2 penalty)
        random_seed: random_seed=491001 
        eval_metric: roc_auc_score

        din_query_offset: the offset in cate feature to interact with bag feat
        din_key_offsets: the offsets in bag feature to interact with cate feat
        
    """
    def __init__(self,field_num,p_sv, p_bag,bag_value_cnts,embedding_size = 8,n_class = 2,
                    use_cuda=True,
                    use_lr=False,use_fm = True, use_deep = True,use_cin=True,
                    use_din = False,
                    is_lr_dropout = False, dropout_lr = 0.5,
                    is_fm_dropout = False, dropout_fm = 0.5,

                    deep_layers = [32, 32], deep_layers_activation = 'relu', 
                    is_deep_dropout = False, dropout_deep=[0.5, 0.5, 0.5],is_deep_bn = False,  

                    cin_layer_sizes = [30,30,30],
                    cin_activation = 'identity',
                    is_cin_bn = False,
                    cin_direct = False,use_cin_bias = False,
                    cin_deep_layers = [], 
                    cin_deep_dropouts = [],
                    din_query_offset=0,din_key_offsets=[],
                    random_seed = 491001
                 ):
        super(ExDeepFM, self).__init__()

        self.embedding_size = embedding_size
        self.field_num = field_num
        self.p_sv = p_sv
        self.p_bag = p_bag
        self.bag_feat_num = len(bag_value_cnts)
        self.bag_value_num = sum(bag_value_cnts)
        self.bag_value_cnts = bag_value_cnts
        self.sv_feat_num = field_num-self.bag_feat_num
        self.n_class = 2

        self.use_cuda = use_cuda
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.use_cin = use_cin
        self.use_lr = use_lr
        self.use_din = use_din

        self.is_lr_dropout = is_lr_dropout
        self.dropout_lr = dropout_lr
        self.is_fm_dropout = is_fm_dropout
        self.dropout_fm = dropout_fm

        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.is_deep_bn = is_deep_bn
        
        self.cin_activation = cin_activation
        self.is_cin_bn = is_cin_bn
        self.cin_layer_sizes = [self.field_num]+cin_layer_sizes
        self.cin_direct = cin_direct
        self.use_cin_bias = use_cin_bias
        self.cin_deep_layers = cin_deep_layers
        self.cin_deep_dropouts = cin_deep_dropouts

        self.din_query_offset = din_query_offset
        self.din_key_offsets = din_key_offsets

        self.random_seed = random_seed

        torch.manual_seed(self.random_seed)

        if self.use_din:
            logging.info("Init din part")
            for i, _ in enumerate(din_key_offsets):
                setattr(self,'interestpooling_'+str(i+1),InterestPooling(self.embedding_size,self.use_cuda))
            logging.info("Init din part")            

        """
            fm part
        """
        self.fm_second_order_embeddings = nn.Embedding(self.p_sv,self.embedding_size)
        self.fm_second_order_bag_embeddings = nn.Embedding(self.p_bag,self.embedding_size,padding_idx=0) 
        if use_lr:
            logging.info("Init lr part")
            self.fm_first_order_embeddings = nn.Embedding(self.p_sv,1)
            
            self.fm_first_order_bag_embeddings = nn.Embedding(self.p_bag,1,padding_idx=0)
            if self.is_lr_dropout:
                self.fm_first_order_dropout = nn.Dropout(self.dropout_lr)
            logging.info("Init lr part succeed")
        if self.use_fm:
            
            logging.info("Init fm part")
        
            if self.dropout_fm:
                self.fm_second_order_dropout = nn.Dropout(self.dropout_fm)
            logging.info("Init fm part succeed")

        """
            deep part
        """
        if self.use_deep:
            logging.info("Init deep part")
            
            if self.is_deep_dropout:
                self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])

            self.linear_1 = nn.Linear(self.field_num*self.embedding_size,deep_layers[0])

            if self.is_deep_bn:
                self.batch_norm_1 = nn.BatchNorm1d(deep_layers[0])
            if self.is_deep_dropout:
                self.linear_1_dropout = nn.Dropout(self.dropout_deep[1])
            for i, h in enumerate(self.deep_layers[1:], 1):
                setattr(self,'linear_'+str(i+1), nn.Linear(self.deep_layers[i-1], self.deep_layers[i]))
                if self.is_deep_bn:
                    setattr(self, 'batch_norm_' + str(i + 1), nn.BatchNorm1d(deep_layers[i]))
                if self.is_deep_dropout:
                    setattr(self, 'linear_'+str(i+1)+'_dropout', nn.Dropout(self.dropout_deep[i+1]))
            
            logging.info("Init deep part succeed")
        if self.use_cin:
            logging.info("Init cin part")
            #conv1d init 
            in_channels = self.field_num
            for i,layer_size in enumerate(self.cin_layer_sizes[:-1]):
                in_channels = self.cin_layer_sizes[0]*layer_size
                if self.cin_direct or i == len(self.cin_layer_sizes[1:])-1:
                    out_channels = self.cin_layer_sizes[i+1]
                else:
                    out_channels = 2*self.cin_layer_sizes[i+1]
                setattr(self, 'conv1d_' + str(i+1), nn.Conv1d(in_channels,out_channels,kernel_size=1,bias=self.use_cin_bias))
                if self.is_cin_bn:
                    setattr(self, 'conv1d_bn_' + str(i+1), nn.BatchNorm1d(out_channels))
            if len(self.cin_deep_layers)!=0:
                if len(self.cin_deep_dropouts)!=0:
                    assert len(self.cin_deep_layers) == len(self.cin_deep_dropouts), 'make sure:len(cin_deep_layers) == len(cin_deep_dropouts)'
                for i, h in enumerate(self.cin_deep_layers):
                    if i == 0:
                        setattr(self,'cin_linear_'+str(i+1), nn.Linear(sum(self.cin_layer_sizes[1:]), self.cin_deep_layers[i]))
                    else:
                        setattr(self,'cin_linear_'+str(i+1), nn.Linear(self.cin_deep_layers[i-1], self.cin_deep_layers[i]))
                    if self.is_cin_bn:
                        setattr(self, 'cin_deep_bn_' + str(i + 1), nn.BatchNorm1d(self.cin_deep_layers[i]))
                    if len(self.cin_deep_dropouts)!=0:
                        setattr(self, 'cin_linear_'+str(i+1)+'_dropout', nn.Dropout(self.cin_deep_dropouts[i]))
            logging.info("Init cin part succeed")
        """
            linear_layer
        """
        concat_input_size = 0
        if self.use_lr:
            concat_input_size = concat_input_size+self.field_num
        if self.use_fm:
            concat_input_size = concat_input_size+self.embedding_size
        if self.use_deep:
            concat_input_size = concat_input_size + self.deep_layers[-1]
        if self.use_cin:
            if len(self.cin_deep_layers)!=0:
                concat_input_size = concat_input_size + self.cin_deep_layers[-1]
            else:
                concat_input_size = concat_input_size + sum(self.cin_layer_sizes[1:])
        self.concat_linear_layer = nn.Linear(concat_input_size,self.n_class)
        # init weight
        # for m in self.modules():
        #     if isinstance(m, nn.Embedding):
        #         nn.init.normal_(m.weight, mean=0.0,std=0.01)
                # nn.init.xavier_normal_(m.weight, gain=0.001)
                # nn.init.kaiming_normal_(m.weight, a=1,mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

        logging.info("Init succeed")

    def forward(self, Xi,Xv,bag_feat_Xi,bag_lenghts):
        """
            Args:
                bag_lenghts : FloatTensor [N,bag_feat_num]
        """
        # fm and dnn common part
        fm_embs = self.fm_second_order_embeddings(Xi.view(-1))#NxF*K
        fm_embs = fm_embs.t()*Xv.view(-1)
        fm_embs = fm_embs.t().contiguous()
        fm_embs = fm_embs.view(-1,self.sv_feat_num,self.embedding_size) #N*F*K

        fm_bag_embs = self.fm_second_order_bag_embeddings(bag_feat_Xi.view(-1))#Nxsum(bag_value_cnts)*k
        fm_bag_embs = fm_bag_embs.view(-1,self.bag_value_num,self.embedding_size)
        fm_embs_list = []
        
        #reduce by mean and interest
        pool_ly_num = 0
        ex_bag_lenghts=bag_lenghts.unsqueeze(dim=2)
        for idx,bag_feat_size in enumerate(self.bag_value_cnts,0):
            start_idx = sum(self.bag_value_cnts[:idx])
            if self.use_din==False or (idx not in self.din_key_offsets):
                pool_emb = torch.sum(fm_bag_embs[:,start_idx:start_idx+bag_feat_size,:],dim=1,keepdim=True)
                pool_emb = pool_emb/ex_bag_lenghts[:,[idx]]
                fm_embs_list.append(pool_emb)
            else:
                pooling_layer = getattr(self,'interestpooling_'+str(pool_ly_num+1))
                pool_ly_num = pool_ly_num+1
                
                queries = fm_embs[:,[self.din_query_offset]]
                keys = fm_bag_embs[:,start_idx:start_idx+bag_feat_size,:]
                bags_Xi = bag_feat_Xi[:,start_idx:start_idx+bag_feat_size]

                pool_emb = pooling_layer(queries,keys,bags_Xi)
                fm_embs_list.append(pool_emb)

        #mean 
        # bag_fm_embs = torch.cat(fm_embs_list,1)/bag_lenghts.unsqueeze(dim=2)
        bag_fm_embs = torch.cat(fm_embs_list,1)
        fm_embs = torch.cat([fm_embs,bag_fm_embs],1)
        # fm_embs_list.append()
        if self.use_lr:
            fm_first_order = self.fm_first_order_embeddings(Xi.view(-1)).view(-1)*Xv.view(-1)#Nxsv_feat_num
            fm_first_order = fm_first_order.view(-1,self.sv_feat_num)
            fm_first_order_bag = self.fm_first_order_bag_embeddings(bag_feat_Xi.view(-1))#Nxsum(bag_value_cnts)*1
            fm_first_order_bag = fm_first_order_bag.view(-1,self.bag_value_num)#N*bag_value_num
            #reduce bag feature by sum
            bag_fm_first_orders = []
            for idx,bag_feat_size in enumerate(self.bag_value_cnts,0):
                start_idx = sum(self.bag_value_cnts[:idx])
                bag_fm_first_orders.append(torch.sum(fm_first_order_bag[:,start_idx:start_idx+bag_feat_size],dim=1,keepdim=True))#N*1
            bag_fm_first_orders = torch.cat(bag_fm_first_orders,1)/bag_lenghts

            fm_first_order = torch.cat([fm_first_order,bag_fm_first_orders],1)#N*F
            if self.is_fm_dropout:
                fm_first_order = self.fm_first_order_dropout(fm_first_order)
        if self.use_fm:
            #second order

            sum_fm_embs = torch.sum(fm_embs,1,keepdim=False) #N*K
            sum_fm_embs_square = sum_fm_embs*sum_fm_embs#N*K

            square_fm_embs = fm_embs*fm_embs #N*F*K
            square_sum_fm_embs = torch.sum(square_fm_embs,1)#N*k

            fm_second_order = 0.5*(sum_fm_embs_square-square_sum_fm_embs)
            if self.is_fm_dropout:
                fm_second_order = self.fm_second_order_dropout(fm_second_order)
        if self.use_deep:
            deep_emb = fm_embs.view(-1,self.field_num*self.embedding_size)
            
            if self.deep_layers_activation == 'sigmoid':
                activation = F.sigmoid
            elif self.deep_layers_activation == 'tanh':
                activation = F.tanh
            else:
                activation = F.relu
            if self.is_deep_dropout:
                deep_emb = self.linear_0_dropout(deep_emb)
            
            x_deep = self.linear_1(deep_emb)
            
            if self.is_deep_bn:
                x_deep = self.batch_norm_1(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = self.linear_1_dropout(x_deep)
            for i in range(1, len(self.deep_layers)):
                x_deep = getattr(self, 'linear_' + str(i + 1))(x_deep)
                if self.is_deep_bn:
                    x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
                x_deep = activation(x_deep)
                if self.is_deep_dropout:
                    x_deep = getattr(self, 'linear_' + str(i + 1) + '_dropout')(x_deep)
        if self.use_cin:
            # cin_embs = fm_embs #[N,H0,K]
            
            hidden_layers = []
            final_result = []
            final_len = 0
            
            X0 = torch.transpose(fm_embs,1,2) #[N,K,H0]
            hidden_layers.append(X0)
            X0 = X0.unsqueeze(-1)#[N,K,H0,1]

            for idx,layer_size in enumerate(self.cin_layer_sizes[:-1]):
                Xk_1 = hidden_layers[-1].unsqueeze(2)#[N,K,1,H_(k-1)]
                #outer product
                out_product = torch.matmul(X0,Xk_1)#[N,K,H0,H_(k-1)]
                out_product = out_product.view(-1,self.embedding_size,layer_size*self.cin_layer_sizes[0])
                out_product = out_product.transpose(1,2)#[N,H0XH_(k-1),K]
                #conv
                conv =getattr(self,'conv1d_' + str(idx+1))
                zk = conv(out_product)#[N,Hk*2,K] or [N,Hk,K]
                if self.is_cin_bn:
                    zk = getattr(self,'conv1d_bn_'+ str(idx+1))(zk)
                if self.cin_activation == 'identity':
                    zk = zk
                else:
                    zk = F.relu(zk)
                # zk = zk.transpose(1,2)#[N,K,Hk*2] or [N,K,Hk]
                if self.cin_direct:
                    direct_connect = zk #[N,Hk,K]
                    next_hidden = zk.transpose(1,2) #[N,K,Hk]
                else:
                    if idx != len(self.cin_layer_sizes[1:])-1:
                        direct_connect,next_hidden= zk.split(self.cin_layer_sizes[idx+1],1)
                        next_hidden = next_hidden.transpose(1,2) #[N,K,Hk]
                    else:
                        direct_connect = zk
                        next_hidden = 0
                final_len = final_len+self.cin_layer_sizes[idx+1] 
                final_result.append(direct_connect)
                hidden_layers.append(next_hidden)
            #concat
            cin_result = torch.cat(final_result,1) #[N,H1+...+Hk,K]
            cin_result = torch.sum(cin_result,-1) #[N,H1+...+Hk]
            #lr
            if len(self.cin_deep_layers)!=0:
                if self.cin_activation == 'identity':
                    activation = F.relu
                else:
                    activation = F.relu
                for i in range(len(self.cin_deep_layers)):
                    cin_result = getattr(self, 'cin_linear_' + str(i + 1))(cin_result)
                    if self.is_cin_bn:
                        cin_result = getattr(self, 'cin_deep_bn_' + str(i + 1))(cin_result)
                    cin_result = activation(cin_result)
                    if len(self.cin_deep_dropouts)!=0:
                        cin_result = getattr(self, 'cin_linear_' + str(i + 1) + '_dropout')(cin_result)
        concat_input = None
        if self.use_deep:
            concat_input = x_deep
        if self.use_lr:
            if concat_input is not None:
                concat_input = torch.cat([concat_input,fm_first_order],1)
            else:
                concat_input = fm_first_order
        if self.use_fm:
            if concat_input is not None:
                concat_input = torch.cat([concat_input,fm_second_order],1)
            else:
                concat_input = fm_second_order
        if self.use_cin:
            if concat_input is not None:
                concat_input = torch.cat([concat_input,cin_result],1)
            else:
                concat_input = cin_result
        total_sum = self.concat_linear_layer(concat_input)
        return total_sum

if __name__ == '__main__':
    p_sv = 10
    sv_feature_num = 3 
    batch_size = 2
    Xv = Variable(torch.ones(batch_size,sv_feature_num))
    Xv[:,1]=0.2
    Xi = Variable(torch.LongTensor([[0,4,8],[0,4,9]]))
    #
    p_bag = 20
    bag_value_cnts = [2,3,2]
    bag_feat_Xi = Variable(torch.LongTensor([[1,2,7,8,0,10,11],[1,2,7,8,9,10,11]]))
    bag_lenghts = Variable(torch.FloatTensor([[2,2,2],[2,3,2]]))
    field_num = sv_feature_num+len(bag_value_cnts)

    model = ExDeepFM(field_num,p_sv,p_bag,bag_value_cnts,
                    use_cuda = True,
                    use_lr=True,use_fm=True,use_deep=True,use_cin=True,
                    use_din = True,
                    is_deep_bn = True,

                    cin_layer_sizes=[field_num,field_num-2,field_num-3],cin_activation = 'identity',
                    cin_direct = False,use_cin_bias = False,
                    cin_deep_layers = [32,32], 
                    cin_deep_dropouts = [0.5,0.5],

                    din_query_offset = 2,din_key_offsets=[1]
                    )

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    model = model.cuda()
    print(model(Xi.cuda(),Xv.cuda(),bag_feat_Xi.cuda(),bag_lenghts.cuda()))
