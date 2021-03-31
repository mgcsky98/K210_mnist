#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-02-23 11:08
# @Author  : MGC
# @Site    : 
# @File    : dataset2jpg.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import os
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #MNIST数据输入
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape)
x_train = np.pad(x_train,((0,0),(2,2),(2,2)),'constant',constant_values=0)
x_train = x_train/255
print(x_train.shape)
#plt.imshow(x_train[1])
#plt.show()
save_dir = './images/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)
for i in range(1000):
    #image = mnist.train.images[i,:]
    image = x_train[i]
    #image = image.reshape(32,32)
    file = save_dir+'mnist_train_%d.jpg' % i
    Image.fromarray((image*255).astype('uint8'), mode='L').convert('L').save(file)
    #scipy.misc.toimage(image,cmin=0.0,cmax=1.0).save(file)

