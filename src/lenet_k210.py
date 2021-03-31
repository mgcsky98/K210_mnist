#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-02-21 17:00
# @Author  : MGC
# @Site    : 
# @File    : lenet_k210.py
# @Software: PyCharm


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
print(tf.__version__)

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

image_index = 1
print(y_train[image_index])
plt.imshow(x_train[image_index])
plt.show()

x_train = np.pad(x_train,((0,0),(2,2),(2,2)),'constant',constant_values=0)
x_test = np.pad(x_test,((0,0),(2,2),(2,2)),'constant',constant_values=0)
print(x_train.shape)
print(x_test.shape)

x_train = x_train.astype('float32')

x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(x_train.shape[0],32,32,1)
x_test = x_test.reshape(x_test.shape[0],32,32,1)
print(x_train.shape)
print(x_test.shape)

model =  tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters = 6, kernel_size = (3,3), padding = 'valid', activation = tf.nn.relu, input_shape = (32,32,1)),
    tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'),
    tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3), padding = 'valid', activation = tf.nn.relu),
    tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = 'same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 120, activation = tf.nn.relu),
    tf.keras.layers.Dense(units = 84, activation = tf.nn.relu),
    tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax)
])
model.summary()
#超参数设置
num_epochs = 10
batch_size = 64
learning_rate = 0.001

#优化器
adam_optimizer = tf.keras.optimizers.Adam(learning_rate)

model.compile(optimizer = adam_optimizer,
              loss = tf.keras.losses.sparse_categorical_crossentropy,
              metrics = ['accuracy'])

start_time = datetime.datetime.now()

model.fit(x = x_train,
          y = y_train,
          batch_size = batch_size,
          epochs = num_epochs)

end_time = datetime.datetime.now()
time_cost = end_time - start_time
print("time cost = ",time_cost)
#保存模型
#model.save('mnist_lenet_K210_model.h5')
#tf.keras.experimental.export_saved_model(model, 'saved_model')
tf.keras.models.save_model(model,'mnist_lenet_k210_model')

#评估指标
print(model.evaluate(x_test,y_test))

#预测
image_index = 6666
print(x_test[image_index].shape)

plt.imshow(x_test[image_index].reshape(32,32))
plt.show()
pred = model.predict(x_test[image_index].reshape(1,32,32,1))


print(pred)
print(pred.argmax())
