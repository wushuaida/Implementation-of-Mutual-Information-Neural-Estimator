# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:36:29 2019

@author: Dawen wu
"""

import numpy as np
from numpy import exp
from sklearn.neighbors import KernelDensity

import torch
import torch.nn as nn
import torch.nn.functional as F


def density(x, y, xy):
    """
    Create the probability density function

    Example:
        sample = [0,0,0,0,0,0]
        value =  [0,1]
    """
    # generate a sample

    # fit density
    model_x = KernelDensity(bandwidth=0.2)
    model_y = KernelDensity(bandwidth=0.2)
    model_xy = KernelDensity(bandwidth=0.2)
    
    model_x.fit(x)
    model_y.fit(y)
    model_xy.fit(xy)

    return model_x, model_y, model_xy


def cal_mi(x, y):
    """
    An auxiliary function for calculating MI.
    """
    f_xy = exp(model_xy.score([[x,y]]))
    f_x = exp(model_x.score(x))
    f_y = exp(model_y.score(y))
    
    return np.log(f_xy/(f_x*f_y))


def MC_mi(n):
    """
    Calculate the mutual information using the monte carlo method
    
    Input:
        n: int, sample size.
    Output:
        the average approximate estimation.
    """
    s = 0
    for i in range(n):
        x, y = model_xy.sample()[0]
        s += cal_mi(x,y)
    s /= n
    return s


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(1, H)
        self.fc3 = nn.Linear(H, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x)+self.fc2(y))
        h2 = self.fc3(h1)
        return h2    
  
        
def MINE(epoch):
    for i in range(epoch):
        #make data
        xy = model_xy.sample(10000)
        x = xy[:, [0]]
        y = xy[:, [1]]
        y_tilda = model_y.sample(10000)
        
        x, y, y_tilda = torch.tensor(x).float(), torch.tensor(y).float(), torch.tensor(y_tilda).float()
        #
        pred_xy = net(x, y)
        pred_x_y = net(x, y_tilda)

        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = - ret  # maximize
        net.zero_grad()
        loss.backward()
        optimizer.step()
        print(ret)
        
    return 


if __name__ == "__main__":
    n=10000 #datasize
    H=10 #the net parameter size
    
    # Define the random variable X and Y, and generate the observed dataset.
    x = np.random.normal(1.,2.,[n,1])   #X ~ N(5,4.41)
    eps = np.random.normal(0.,0.2,[n,1])
    y = x**3+eps
    x, y = x.reshape((n,1)), y.reshape((n,1))
    xy = np.hstack((x,y))
    #Using KDE method to getting the pdf from the given data, corresbonding to the MI note section 3.
    model_x, model_y, model_xy = density(x, y, xy)
    #Using MC method to estimate the MI between X and Y, corresbonding to the MI note section 4.
    mi = MC_mi(10000)
    #The MINE algo, corresbonding to the MINE note.
    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    
    MINE(1000)

