from functools import *
import os,sys

import numpy as np

import openml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Function


import matplotlib.pyplot as plt

def plot_data(datapoints):
    
    x=np.arange(len(datapoints))
    y=[]
    for i in range(len(datapoints[0])):
        y.append(x)
        y.append(list(map(lambda p:p[i],datapoints)))
    plt.plot(*y)
    plt.show()
    
    return

def gauss_kernel(x,y,sigma=torch.tensor(0.1),dim=1,dist=lambda x,y:(torch.Tensor.pow(x-y,2))):
    if (dim==1):
        div=torch.sqrt(np.pi*2*torch.Tensor.pow(sigma,2))
    else:
        div=torch.pow(np.pi*2*torch.Tensor.pow(sigma,2),dim/2)
    return (torch.exp(-(dist(x,y)/(2*torch.pow(sigma,2)))))/div


#if sample is one dimensional,treats samole as a population
#otherwise reats each multi-dimesional 
def entM(sample,ker=gauss_kernel):
    if (len(sample.size())==1):
        sample = sample.repeat(sample.size()[0],1)
        return ker(sample,sample.transpose(1,0))
    else:
        sample=sample.view(sample.size()[0],-1)
        
        out = []
        
        for index in range(sample.size()[1]):
            s=sample[:,index].repeat(sample.size()[0],1)
            #print(s.size())
            #print(s)
            out.append(ker(s,s.transpose(1,0)))
        
        return out
        
class MatrixEntropy(Function):
    @staticmethod
    def forward(ctx,*matrix):
        trace=torch.trace(matrix[0])    
        matrix=matrix[0]/trace
        eigen=torch.Tensor.eig(matrix,True)
        #print(eigen)
        s=0
        for i in range(eigen[0].size()[0]):
           s+=torch.pow(eigen[0][i,0],2)
        s=-torch.log2(s)
        
        ctx.save_for_backward(eigen[0],eigen[1],trace)

        
        #print("F")
        #print(ctx.saved_tensors)
        #print("\F")
        return s
    
    @staticmethod
    def backward(ctx,*grad_out):
        #print("BACKWARD")
        #print(ctx.saved_tensors)
        #print(grad_out)
        eigenvals,eigenvecs,trace = ctx.saved_tensors
        
        eigenvals = eigenvals[:,0]
        
        
        
        
        t=-2/torch.pow(trace,2)
        
        #print(eigenvals.size(),eigenvecs.size())
        t2=torch.mm(torch.diag(eigenvals),eigenvecs.transpose(1,0))
        return grad_out[0]*t*torch.mm(eigenvecs,t2)    
        
def joint_entropy(x,y):
    prod=x*y
    
    return MatrixEntropy.apply(prod/torch.trace(prod))

def conditional_entropy(x,y):
    return joint_entropy(x,y)-MatrixEntropy.apply(y)        

def mutual_information(x,y):
    me=MatrixEntropy.apply
    return me(x)+me(y)-joint_entropy(x,y)

if __name__=="__main__":
    me=MatrixEntropy.apply    
    
    x= torch.Tensor([0,0.01,0.02,0.03])
    y= torch.Tensor([0,0.01,0.02,0.03])
    x.requires_grad=True
    y.requires_grad=True
    
    xM=entM(x)
    yM=entM(y)
    
    print(me(xM))
    print(me(yM))
    print(joint_entropy(xM,yM))
    print(conditional_entropy(yM,xM))
    print(mutual_information(xM,yM))
    
    
    
    x_optim=optim.Adam([x])
    
    for i in range(500):
        x_optim.zero_grad()
        print(i+1,x)
        
        xM=entM(x)
        yM=entM(y)
        l=mutual_information(yM,xM)
        
        print(l.item())
        l.backward()
        x_optim.step()
    
    




    


