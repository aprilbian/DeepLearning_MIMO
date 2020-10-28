## this week: without channel estimation and explicit channel equalization, only compare the performance of 
## end to end learning
import argparse
import torch 
import torch.nn as nn
import numpy as np
import math

Tx = 15
Rx = 25
NN_Rx = 2048
NN_Tx = 256
L = 1
M = 16

class e2e_AE(nn.Module):
    def __init__(self,is_cuda = True, explicit_eq = True, eq_res = True):
        super(e2e_AE,self).__init__()

        self.is_cuda = is_cuda
        self.eq_res = False
        self.explicit_eq = False
        tx_net = [nn.Linear(M*Tx+2*Tx*Rx,NN_Tx),nn.ReLU(),nn.BatchNorm1d(NN_Tx),nn.Linear(NN_Tx,NN_Tx),nn.ReLU(),nn.BatchNorm1d(NN_Tx),nn.Linear(NN_Tx,2*Tx*L)]
        # if do the explicit equalization, the dimension of the vector should be 2*Tx, otherwise it will be 2*Rx
        if self.explicit_eq:
            rx_net = [nn.BatchNorm1d(2*Tx*L+2*Tx*Rx),nn.Linear(2*Tx*L+2*Tx*Rx,NN_Rx),nn.ReLU(),nn.BatchNorm1d(NN_Rx),nn.Linear(NN_Rx,NN_Rx),nn.ReLU(),nn.BatchNorm1d(NN_Rx),
            nn.Linear(NN_Rx,NN_Rx),nn.ReLU(),nn.BatchNorm1d(NN_Rx),nn.Linear(NN_Rx,NN_Rx),nn.ReLU(),nn.BatchNorm1d(NN_Rx),nn.Linear(NN_Rx,M*Tx)]
        else:
            # if without the explicit equalization, give more layers to the DNN for compensation.
            rx_net = [nn.BatchNorm1d(2*Rx*L+2*Tx*Rx),nn.Linear(2*Rx*L+2*Tx*Rx,NN_Rx),nn.ReLU(),nn.BatchNorm1d(NN_Rx),nn.Linear(NN_Rx,NN_Rx),nn.ReLU(),
            nn.BatchNorm1d(NN_Rx),nn.Linear(NN_Rx,NN_Rx),nn.ReLU(),nn.BatchNorm1d(NN_Rx),nn.Linear(NN_Rx,NN_Rx),nn.ReLU(),
            nn.BatchNorm1d(NN_Rx),nn.Linear(NN_Rx,NN_Rx),nn.ReLU(),nn.BatchNorm1d(NN_Rx),nn.Linear(NN_Rx,M*Tx)]

        # the residual connections for channel equalization

        res_eq = [nn.Linear(2*Rx*L+2*Tx*Rx+1,2*Tx),nn.ReLU(),nn.BatchNorm1d(2*Tx)]
        self.Tx_Net = nn.Sequential(*tx_net)
        self.Rx_Net = nn.Sequential(*rx_net)
        self.Res_EQ = nn.Sequential(*res_eq)

    def mimo_channel(self,enc,ch,snr):
        #### passing through the channel
        mini_batch = ch.shape[0]
        ch_real = ch[:,0:Tx*Rx]
        ch_imag = ch[:,Tx*Rx:2*Tx*Rx]
        ch_imag,ch_real = ch_imag.view(mini_batch,Rx,Tx),ch_real.view(mini_batch,Rx,Tx)

        enc_real = enc[:,0:Tx*L].view(mini_batch,Tx,L)
        enc_imag = enc[:,Tx*L:2*Tx*L].view(mini_batch,Tx,L)

        y_real = torch.bmm(ch_real,enc_real) - torch.bmm(ch_imag,enc_imag)
        y_imag = torch.bmm(ch_real,enc_imag) + torch.bmm(ch_imag,enc_real)
        y_real,y_imag = y_real.view(mini_batch,-1), y_imag.view(mini_batch,-1)
        y = torch.cat((y_real,y_imag),dim = -1)

        #### adding the noise
        #snr = numpy.randint(10,20,1)
        snr = 10**(snr/10)
        noise = torch.randn(y.shape)/math.sqrt(2*Rx*snr)
        if self.is_cuda:
            noise = noise.cuda()
        y = y + noise

        return y

    def normalize(self, x, pwr=1):
        '''
        Normalization function
        '''
        power = torch.sum(x**2, 1, True)
        alpha = np.sqrt(pwr)/torch.sqrt(power)
        return alpha*x

    def modulate(self,raw_bits):
        '''use QAM for the 2 antennas, equal power'''
        mini_batch = raw_bits.shape[0]
        s = np.random.randint(2,size = [mini_batch,4,1])
        s = 2 * s -1
        s = s/2      # equal-powered and normalized
        return torch.from_numpy(s).float().cuda()

    def LMMSE_CE(self,ch,raw_bits,snr):
        '''In this function, the complex-number operation is converted into real-number
        operation, following s' = [real(s),imag(s)], H' = [[real(H),-imag(H)],[imag(H),real(H)]]'''
        mini_batch = raw_bits.shape[0]
        s = self.modulate(raw_bits)
        ch_r,ch_i = ch[:,0:Tx*Rx],ch[:,Tx*Rx:2*Tx*Rx]
        revise_ch = torch.zeros([mini_batch,2*Rx,2*Tx]).cuda()  # ??
        revise_ch[:,0:Rx,0:Tx],revise_ch[:,Rx:2*Rx,0:Tx] = ch_r.view(mini_batch,Rx,Tx),ch_i.view(mini_batch,Rx,Tx),
        revise_ch[:,0:Rx,Tx:2*Tx],revise_ch[:,Rx:2*Rx,Tx:2*Tx] = -ch_i.view(mini_batch,Rx,Tx),ch_r.view(mini_batch,Rx,Tx)

        sig = torch.bmm(revise_ch,s)

        #### the LMMSE channel estimation

    def pro_channel(self,ch):
        mini_batch = ch.shape[0]
        ch_r,ch_i = ch[:,0:Tx*Rx],ch[:,Tx*Rx:2*Tx*Rx]
        
        
        ch1 = torch.cat((ch_r.view(-1,Rx,Tx),-ch_i.view(-1,Rx,Tx)), dim = 2)
        ch2 = torch.cat((ch_i.view(-1,Rx,Tx),ch_r.view(-1,Rx,Tx)), dim= 2)
        revise_ch = torch.cat((ch1,ch2),dim = 1)

        return revise_ch

    def MMSE_equalization(self,H_est, Y, snr):
        # H_est: batch*2Rx*2Tx
        # Y: batch*2Rx
        mini_batch = H_est.shape[0]
        H_est = self.pro_channel(H_est)
        Y = Y.view(mini_batch,2*Rx,L)
        snr = 10**(snr/10)
        bias = 1/snr*torch.eye(2*Rx).repeat(mini_batch,1,1)
        if self.is_cuda:
            bias = bias.cuda()
        H = torch.bmm(H_est,H_est.permute(0,2,1)) + bias
        r_H = torch.inverse(H)
        equ_s = torch.bmm(torch.bmm(H_est.permute(0,2,1),r_H),Y)
        equ_s = equ_s.view(mini_batch,-1)

        return equ_s

    def forward(self, raw_bits, snr):
        '''the channel will be generate within the class, snr for training is 15dB,
        but will change in testing phase'''
        mini_batch = raw_bits.shape[0]
        # generate the MIMO channel : for a block, generate a channel sample
        ch = torch.randn(mini_batch,2*Tx*Rx)/math.sqrt(2)
        if self.is_cuda:
            ch = ch.cuda()

        enc_input = torch.cat((raw_bits,ch),dim = -1)
        enc = self.Tx_Net(enc_input)
        # normalize the power
        sig = self.normalize(enc,L)

        # passing the channel
        rec_sig = self.mimo_channel(sig,ch,snr)

        # do explicit MMSE equalization first, 
        if self.explicit_eq:
            explicit_rec = self.MMSE_equalization(ch,rec_sig,snr)
        else:
            explicit_rec = rec_sig

        
        if self.eq_res:
            # For the residual connection
            snr_holder = snr*torch.ones(mini_batch,1)
            if self.is_cuda:
                snr_holder = snr_holder.cuda()
            res = torch.cat((rec_sig,ch,snr_holder),dim = -1)
            rec = explicit_rec + self.Res_EQ(res)
        else:
            rec = explicit_rec
        
        # feed to the receiver
        dec_input = torch.cat((rec,ch),dim = -1)
        dem_bits = self.Rx_Net(dec_input)
        # additional softmax layer
        #dem_bits = F.log_softmax(dem_bits, dim=2)
        return dem_bits