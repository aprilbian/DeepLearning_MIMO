import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.io as io
from torch.nn import init
import os
import argparse
from model import e2e_AE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

### setting the system parameter
Tx = 15
Rx = 25
M = 16  # 4 bits information
L = 1   # how many transmitted samples to transmit 4 bits
Train_SNR = 15
Test_SNR = [2*i for i in [3,4,5,6,7]]

### NN parameters
lr = 1e-4
epoches = 15
batch_size = 2048
NN_Tx = 256
NN_Rx = 2048
'''
parser = argparse.ArgumentParser()
parser.add_argument('--Tx', type=int, default=2, help='the number of transmitters')
parser.add_argument('--Rx', type=int, default=2, help='the number of transmitters')
parser.add_argument('--save_path', type=str, default='res_eq', help='models are saved here')
parser.add_argument('--L', type=int, default=1, help='the number of transmitted symbols')
parser.add_argument('--M', type=int, default=16, help='number of information bits per transmission')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epoches', type=int, default=15, help='training epoches')
parser.add_argument('--NN_Tx', type=int, default=256, help='the number of neurons at transmitter')
parser.add_argument('--NN_Rx', type=int, default=2048, help='the number of neurons at receiver')
parser.add_argument('--batch size', type=int, default=2048, help='batch size')

opt = parser.parse_args()
'''
def adjust_lr(optimizer, epoch):
    # changing the lr according to the epoch.
    if(epoch<epoches/5):
        adapt_lr = lr
    else:
        if(epoch<0.9*epoches):
            adapt_lr = 0.1*lr
        else:
            adapt_lr = 0.01*lr
    for para_group in optimizer.param_groups:
        para_group['lr'] = adapt_lr

def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=2e-2)
            #if m.bias:
            init.constant_(m.bias, 0)

'''
def loss_fun(pred,label):
    label = label.view(-1,Tx,M)
    pred = F.log_softmax(pred, dim=2)
    cost = F.nll_loss(pred[:,0,:],label[:,0,:]).data[0]
    for i in range(Tx-1):
        cost += F.nll_loss(pred[:,i+1,:],label[:,i+1,:]).data[0]
    
    return cost
'''

def generate_onehot(raw_bits,M):
    '''Generate the one-hot codeword based on the raw_bits'''
    mini_batch = raw_bits.shape[0]
    #np.random.shuffle(raw_bits)
    one_hot = np.zeros([mini_batch,M])
    for i in range(mini_batch):
        one_hot[i,raw_bits[i]] = 1
    
    return one_hot

# generate raw bits
batch = 6400*400
train_sample = np.arange(M)
train_samples = np.tile(train_sample, batch*Tx)
train_label = generate_onehot(train_samples,M)
train_label = np.reshape(train_label,(batch*M,Tx*M))

# to tensor
train_label = torch.from_numpy(train_label)
order = [i for i in range(batch*M)]


auto_encoder = e2e_AE().cuda()
auto_encoder.apply(initNetParams)

# optimizer and loss func
loss_fun = nn.MSELoss()
optimizer = optim.Adam(auto_encoder.parameters(),lr = lr)
for epoch in range(epoches):
    # shuffle the training data
    np.random.shuffle(order)
    train_label = train_label[order,:].long()
    print("epoch = ", epoch)
    adjust_lr(optimizer,epoch)
    for iter in range(int(batch*M/batch_size)):
        train_x = train_label[iter*batch_size:(iter+1)*batch_size,:].cuda()
        pred_x = auto_encoder(train_x.float(),Train_SNR)
        #pred_x = pred_x.view(-1,Tx,M)
        loss_label = torch.argmax(train_x,1)
        loss = loss_fun(pred_x,train_x.float())
        #loss = loss_fun(pred_x, train_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## monitor the loss
        if iter%1000==0:
            print('current loss is: ', loss.detach().cpu().numpy())
filepath = 'res_eq.pth'
torch.save(auto_encoder.state_dict(), filepath)
#auto_encoder.load_state_dict(torch.load(filepath))
###### for the testing phase
def SER(prob,label):
    mini_batch = prob.shape[0]
    count = 0
    for i in range(mini_batch):
        for j in range(Tx):
            index = np.argmax(prob[i,j*M:(j+1)*M])
            if label[i,j*M+index] == 1:
                count += 1
    return 1 - count/(mini_batch*Tx)

test_sample = 1024*100
test_label = train_label[0:test_sample,:]
auto_encoder.eval().cpu()
auto_encoder.is_cuda = False
Ber_list = []
with torch.no_grad():
    for t_snr in Test_SNR:
        order = [i for i in range(test_sample)]
        np.random.shuffle(order)
        test_label_iter = test_label[order,:]
        test_label_iter = test_label_iter.view(test_sample,Tx*M)
        prob_iter = auto_encoder(test_label_iter.float(),t_snr)
        #prob_iter = nn.Softmax()(prob_iter)
        ### BER
        test_label_np, prob_np = test_label_iter.cpu().detach().numpy(), prob_iter.cpu().detach().numpy()
        Ber_list.append(SER(prob_np,test_label_np))
print(Ber_list)