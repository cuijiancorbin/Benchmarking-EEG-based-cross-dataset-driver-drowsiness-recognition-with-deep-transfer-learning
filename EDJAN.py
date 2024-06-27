# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:17:29 2019

@author: JIAN
"""

"""
 The code implements the Entropy-Driven Joint Adaptation Network (EDJAN) for cross-dataset driver drowsiness recognition:
     
 Cui, Jian, et al. "Benchmarking EEG-based cross-dataset driver drowsiness recognition with deep transfer learning." 
 2023 45th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 2023. 
 DOI: 10.1109/EMBC40787.2023.10340982    
 
  The processed SADT dataset can be downloaded here:
  https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset/14273687
  
  The processed SEED-VIG dataset can be downloaded here:
  https://figshare.com/articles/dataset/Extracted_SEED-VIG_dataset_for_cross-dataset_driver_drowsiness_recognition/26104987
  

The proposed model achieved an accuracy of 83.68% when transfer from SADT to SEED-VIG.
It achieved an accuracy of 76.90% when transfer from SEED-VIG to SADT.
   
 Description on the backbone ICNN model can be found from:
     
 Cui J, Lan Z, Sourina O, et al. EEG-based cross-subject driver drowsiness recognition with an interpretable convolutional neural network[J]. 
 IEEE Transactions on Neural Networks and Learning Systems, 2022, 34(10): 7921-7933. DOI: 10.1109/TNNLS.2022.3147208   
  
  If you have met any problems, you can contact Dr. Cui Jian at cuijian@zhejianglab.com
"""


import torch
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim

torch.cuda.empty_cache()
torch.manual_seed(0)
np.random.seed(0)

    
class InterpretableCNN(torch.nn.Module):     
    
    def __init__(self, classes=2, sampleChannel=12, sampleLength=384 ,N1=10, d=2,kernelLength=64):
        super(InterpretableCNN, self).__init__()
        self.pointwise = torch.nn.Conv2d(1,N1,(sampleChannel,1))
        self.depthwise = torch.nn.Conv2d(N1,d*N1,(1,kernelLength),groups=N1) 
        self.activ=torch.nn.ReLU()       
        self.batchnorm = torch.nn.BatchNorm2d(d*N1,track_running_stats=False)       
        self.GAP=torch.nn.AvgPool2d((1, sampleLength-kernelLength+1))         
        self.fc = torch.nn.Linear(d*N1, classes)        
        self.softmax=torch.nn.LogSoftmax(dim=1)
         
    def forward(self, inputdata):                    
        intermediate = self.pointwise(inputdata) 
        intermediate = self.depthwise(intermediate) 
        intermediate = self.activ(intermediate) 
        intermediate = self.batchnorm(intermediate)     
        intermediate = self.GAP(intermediate)     
        intermediate = intermediate.view(intermediate.size()[0], -1)          
        intermediate = self.fc(intermediate)  
        output = self.softmax(intermediate)         
    
        return output  
    
    
    
def run():

    #########################################################################################
    #import the SADT dataset
    filename = r'dataset.mat'
    
    tmp = sio.loadmat(filename)
    xdata=np.array(tmp['EEGsample'])
    label=np.array(tmp['substate'])
    subIdx=np.array(tmp['subindex'])
    
    label.astype(int)
    subIdx.astype(int)
    
    del tmp
    
    samplenum=label.shape[0]
    sf=128

    ydata=np.zeros(samplenum,dtype=np.longlong)
    for i in range(samplenum):
        ydata[i]=int(label[i])


 # The names and their corresponding index of the original dataset are:
 # Fp1, Fp2, F7, F3, Fz, F4, F8, FT7, FC3, FCZ, FC4, FT8, T3, C3, Cz, C4, T4, TP7, CP3, CPz, CP4, TP8, T5, P3, PZ, P4, T6, O1, Oz  O2
 # 0,    1,  2,  3,  4,  5,  6,  7,   8,   9,   10,   11, 12, 13, 14, 15, 16, 17,  18,  19,  20,  21,  22,  23,24, 25, 26, 27, 28, 29

# The selected channels are 'FT7', 'FT8', 'TP7', 'TP8','CP3', 'CP4',  'P3','PZ','P4', 'O1', 'Oz', 'O2'
    selectedchannel=[    7, 11, 17,21,18, 20,   23, 24, 25, 27, 28, 29]        
    
    
    
 
    xtrain=np.zeros((xdata.shape[0],12,xdata.shape[2]))
    for kk in range(12):
        xtrain[:,kk,:]=xdata[:,selectedchannel[kk],:]   
        
        
    xdata=xtrain 
        

        
        
    #############################################################################################
    #########################################################################################
# import the SEED VIG dataset    
    
    filename1 = r'SEED_VIG.mat'
    tmp = sio.loadmat(filename1)
    xdata1=np.array(tmp['EEGsample'])
    label1=np.array(tmp['substate'])
    subIdx1=np.array(tmp['subindex'])
    
    label1.astype(int)
    subIdx1.astype(int)
    
    del tmp
    
    samplenum1=label1.shape[0]
    sf=128

    ydata1=np.zeros(samplenum1,dtype=np.longlong)
    for i in range(samplenum1):
        ydata1[i]=int(label1[i])
        
        
 # The names and their corresponding index of the original dataset are:    
 #'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8','CP1', 'CP2', 'P1','PZ','P2','PO3' ,'POZ', 'PO4', 'O1', 'Oz','O2'    
 # 0,      1,    2,     3,    4,    5,     6,    7,      8,  9,   10,  11,    12,    13,    14,   15,  16 
 # The selected channels are 'FT7', 'FT8','TP7', 'TP8','CP1', 'CP2', 'P1','PZ','P2','O1', 'Oz','O2' 
        
        
    selectedchannel1=[0,1,4,5,6,7,8,9,10,14,15,16]   
    xtrain=np.zeros((xdata1.shape[0],12,xdata1.shape[2]))
    for kk in range(12):
        xtrain[:,kk,:]=xdata1[:,selectedchannel1[kk],:]      
    
    xdata1=xtrain        
################################################################################           
    channelnum=12
    samplelength=3
    sf=128
    
#   define the learning rate, batch size and epoches
    lr=1e-3 
    batch_size = 50
    n_epoch =11 
    
############################################################################################################    

#  change this between 0 and 1 to switch the direction of transfer 
    sadt_to_seed_vig=0
    
    if sadt_to_seed_vig:
        xtrain=xdata
        ytrain=ydata
        testdatax=xdata1
        testdatay=ydata1
        testsubidx=subIdx1
        sn=samplenum1
        
        subjnum=12
    else:
        xtrain=xdata1
        ytrain=ydata1
        testdatax=xdata
        testdatay=ydata       
        testsubidx=subIdx    
        subjnum=11
        sn=samplenum
    
    results=np.zeros(subjnum)
    
    
    x_train = xtrain.reshape(xtrain.shape[0],1,channelnum, samplelength*sf)
    y_train=ytrain  
    
    train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    
#average the results by 10    
    finalresults=np.zeros(10)
    for kk in range(10):

        my_net = InterpretableCNN().double().cuda()
        optimizer = optim.Adam(my_net.parameters(), lr=lr)    
        loss_class = torch.nn.NLLLoss().cuda()
    
        for p in my_net.parameters():
            p.requires_grad = True    
      

        for epoch in range(n_epoch):   
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data   
    
                slctidx=np.random.choice(sn, labels.size()[0], replace=False)
                xtestbatch=testdatax[slctidx]
                xtestbatch = xtestbatch.reshape(xtestbatch.shape[0], 1,channelnum, samplelength*sf)
                xtestbatch  =  torch.DoubleTensor(xtestbatch).cuda()            
                
                input_data = inputs.cuda()
                class_label = labels.cuda()              
    
                my_net.zero_grad()               
                my_net.train()          
       
                class_output= my_net(input_data) 
                targetout= my_net(xtestbatch) 
                
                #the entropy loss
                Hloss=-torch.sum(torch.exp(targetout)*(targetout))/(targetout.size(0)) 
                    
                err_s_label = loss_class(class_output, class_label)
     
                err=err_s_label +Hloss
                err.backward()
                optimizer.step()    
        
        
    ################################################    
        my_net.train(False)
        with torch.no_grad():
        
            for i in range(1,subjnum+1):
        
                testindx=np.where(testsubidx == i)[0]    
                xtest=testdatax[testindx]
                x_test = xtest.reshape(xtest.shape[0], 1,channelnum, samplelength*sf)
                y_test=testdatay[testindx]  
            
        
                x_test =  torch.DoubleTensor(x_test).cuda()
                answer = my_net(x_test)
                probs=answer.cpu().numpy()
                preds       = probs.argmax(axis = -1)  
                acc=accuracy_score(y_test, preds)
        
                results[i-1]=acc
                
        finalresults[kk]= np.mean(results)       
        print('mean accuracy:',np.mean(results))

    print(np.mean(finalresults),np.std(finalresults))
if __name__ == '__main__':
    run()
    
