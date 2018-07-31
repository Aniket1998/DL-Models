import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        
        #Encoder architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,16,kernel_size=3,
                stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,kernel_size=3,
                stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32,32,kernel_size=3,
                stride=1,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32,64,kernel_size=3,
                stride=2,padding=1,bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64,64,kernel_size=3,
                stride=1,padding=1,bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64,64,kernel_size=3,
                stride=2,padding=1,bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(4*4*64,512,bias=False)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,128,bias=False)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc_mean = nn.Linear(128,128)
        self.fc_logvar = nn.Linear(128,128)
        
        #Decoder Architecture
        self.d_fc2 = nn.Linear(128,512,bias=False)
        self.d_bn_fc2 = nn.BatchNorm1d(512)
        self.d_fc1 = nn.Linear(4*4*64,bias=False)
        self.d_bn_fc1 = nn.BatchNorm1d(4*4*64)
        self.d_conv7 = nn.ConvTranspose2d(64,64,kernel_size=3,
                stride=2,padding=1,output_padding=1) #4,4,32 to 8,8,32 
        self.d_bn7 = nn.BatchNorm2d(64)
        self.d_conv6 = nn.ConvTranspose2d(64,64,kernel_size=3,
                stride=1,padding=1)
        self.d_bn6 = nn.BatchNorm2d(64)
        self.d_conv5 = nn.ConvTranspose2d(64,32,kernel_size=3,
                stride=2,padding=1,output_padding=1)
        self.d_bn5 = nn.BatchNorm2d(32)
        self.d_conv4 = nn.ConvTranspose2d(32,32,kernel_size=3,
                stride=1,padding=1)
        self.d_bn4 = nn.BatchNorm2d(32)
        self.d_conv3 = nn.ConvTranspose2d(32,16,kernel_size=3,
                stride=2,padding=1,output_padding=1)
        self.d_bn3 = nn.BatchNorm2d(16)
        self.d_conv2 = nn.ConvTranspose2d(16,16,kernel_size=3,
                stride=1,padding=1)
        self.d_bn2 = nn.BatchNorm2d(16)
        self.d_conv1 = nn.ConvTranspose2d(16,3,kernel_size=3,
                stride=2,padding=1,output_padding=1)
        self.d_bn1 = nn.BatchNorm2d(3)
        
    def encode(self,x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        conv4 = F.relu(self.bn4(self.conv4(conv3)))
        conv5 = F.relu(self.bn5(self.conv5(conv4)))
        conv6 = F.relu(self.bn6(self.conv6(conv5)))
        conv7 = F.relu(self.bn7(self.conv7(conv6)))
        fc1 = F.relu(self.bn_fc1(self.fc1(conv7.view(-1,4*4*64))))
        fc2 = F.relu(self.bn_fc2(self.fc2(fc1)))
        mean,logvar = self.fc_mean(fc2),self.fc_logvar(fc2)
        return mean,logvar,torch.normal(mean,torch.exp(logvar*0.5))
    
    def decode(self,z):
        d_fc2 = F.relu(self.d_bn_fc2(self.d_fc2(z)))
        d_fc1 = F.relu(self.d_bn_fc1(self.d_fc1(d_fc2)))
        d_conv7 = F.relu(self.d_bn7(self.d_conv7(d_fc1.view(-1,64,4,4)))) 
        d_conv6 = F.relu(self.d_bn6(self.d_conv6(d_conv7)))
        d_conv5 = F.relu(self.d_bn5(self.d_conv5(d_conv6)))
        d_conv4 = F.relu(self.d_bn4(self.d_conv4(d_conv5)))
        d_conv3 = F.relu(self.d_bn3(self.d_conv3(d_conv4)))
        d_conv2 = F.relu(self.d_bn2(self.d_conv2(d_conv3)))
        return F.relu(self.d_bn1(self.d_conv1(d_conv2))) 

class Trainer(object):
    def __init__(self,model,trainloader,testloader,epochs,batch_size,checkpoints):
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.checkpoints = checkpoints

    def save_checkpoint(self,best=False):
        torch.save()
