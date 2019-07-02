#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:11:31 2019

@author: quanghuy
"""
import numpy as np
import imageio
import matplotlib.pyplot as plt
np.random.seed(1)

# filename structure
path = '/Users/quanghuy/Documents/Massp/Class/20190701/YALE/centered/' # path to the database
ids = range(1, 16) # 15 persons
states = ['centerlight', 'glasses', 'happy', 'leftlight',
          'noglasses', 'normal', 'rightlight','sad',
          'sleepy', 'surprised', 'wink' ]
prefix = 'subject'
surfix = '.pgm'

# data dimension
h = 231 # hight
w = 195 # width
D = h * w
N = len(states)*15
K = 100

# collect all data
X = np.zeros((D, N))
cnt = 0
for person_id in range(1, 15):
    for state in states:
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        X[:, cnt] = imageio.imread(fn).reshape(D)
        cnt += 1

# Doing PCA, note that each row is a datapoint
from sklearn.decomposition import PCA
pca = PCA(n_components=K) # K = 100
pca.fit(X.T) #training data

# projection matrix
U = pca.components_.T

for i in range(U.shape[1]): #show each of Xi
    plt.axis('off')
    f1 = plt.imshow(U[:, i].reshape(h,w), interpolation='nearest')
    # interpolation == 'nearest' means pixel color = nearest coordinates 
    f1.axes.get_xaxis().set_visible(False)
    f1.axes.get_yaxis().set_visible(False)
#     f2 = plt.imshow(, interpolation='nearest' )
    plt.gray()
    fn = path + 'store/eigenface' + str(i).zfill(2) + '.png'
    plt.savefig(fn, bbox_inches='tight', pad_inches=0)
#    plt.show()

# See reconstruction of first 6 persons 

for person_id in [2]:
    for state in ['centerlight']:
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        im = imageio.imread(fn)
        plt.axis('off')
#       plt.imshow(im, interpolation='nearest' )
        f1 = plt.imshow(im, interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)
        plt.gray()
        fn = path + 'store/ori' + str(person_id).zfill(2) + '.png'
        # show the the original image
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.show()
        # reshape and subtract mean, don't forget 
        x = im.reshape(D, 1) - pca.mean_.reshape(D, 1) 
        # x is the X-Xo = X1*z1+X2*z2...Xk*zk
        # encode
        z = U.T.dot(x) 
        # z is the coordinates matrix
        #decode
        error = []
        for index in [1,10,20,50,100]:
            x_tilde = U[:,:index].dot(z[:index,:]) + pca.mean_.reshape(D, 1)
            # x_tilde = Xo + X1*z1 + ... X_index*z_index
    # reshape to orginal dim
            im_tilde = x_tilde.reshape(h,w)
            plt.axis('off')
    #       plt.imshow(im_tilde, interpolation='nearest' )
            f1 = plt.imshow(im_tilde, interpolation='nearest')
            f1.axes.get_xaxis().set_visible(False)
            f1.axes.get_yaxis().set_visible(False)
            plt.gray()
            fn = path + 'store/res' + str(person_id).zfill(2) + '.png'
            plt.savefig(fn, bbox_inches='tight', pad_inches=0)
            # show the im_tilde
            plt.show()
            loss = 0.0
            for i in range(D):
                loss = loss + float((x_tilde[i]-x[i]))**2
            print(index)
            print(loss**0.5)
            error.append(loss**0.5)
        print(error)
        plt.plot([1,10,20,50,100],error)

