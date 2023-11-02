import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.keras import layers
import numpy as np
# import argparse
import cv2
import logging
import os
import re
import setproctitle
import skimage
import skimage.io
import skimage.transform

def get_value_from_checkpoint(save_path):

  reader = tf.train.load_checkpoint(save_path)
  shapes = reader.get_variable_to_shape_map()
  dtypes = reader.get_variable_to_dtype_map()
  value={}
  
  # print(f"Checkpoint at '{save_path}':")
  for key in shapes:
    # print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}, "
    #       f"value={reader.get_tensor(key)})") 
    value[key]=reader.get_tensor(key)

  return value

tensors=get_value_from_checkpoint("./tf2/")

def splat():
  model = tf.keras.Sequential()

  model.add(layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(256, 256, 3)))

  model.add(layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(128, 128, 8)))
  model.add(layers.BatchNormalization(fused=True))  # 添加批归一化层

  model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(64, 64, 16)))
  model.add(layers.BatchNormalization(fused=True))  # 添加批归一化层

  model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 32)))
  model.add(layers.BatchNormalization(fused=True))  # 添加批归一化层
  #model.summary()
  return model

def low_res_local():
  model = tf.keras.Sequential()
  
  model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(16, 16, 64)))
  model.add(layers.BatchNormalization(fused=True))  # 添加批归一化层
  
  model.add(layers.Conv2D(64, (3, 3), padding='same', activation=None, use_bias=False))
  
  return model

def low_res_global():
  model = tf.keras.Sequential()
  
  # 8x8x64
  model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(16, 16, 64)))
  model.add(layers.BatchNormalization(fused=True))  # 添加批归一化层 
  # 4x4x64
  model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(8,8,64)))
  model.add(layers.BatchNormalization(fused=True))  # 添加批归一化层

  model.add(layers.Flatten())
  
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.BatchNormalization())  # 添加批归一化层
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.BatchNormalization())  # 添加批归一化层

  model.add(layers.Dense(64, activation=None))
  
  return model

def fusion(grid_features,global_features,batch_size):
  fusion_grid = grid_features
  fusion_global = tf.reshape(global_features, [batch_size, 1, 1, 64])
  fusion = tf.nn.relu(fusion_grid+fusion_global)
  return fusion


model1=splat()
weight1 = tensors['vars/inference.Scoefficients.Ssplat.Sconv1.Sweights/.ATTRIBUTES/VARIABLE_VALUE']
bias1 = tensors['vars/inference.Scoefficients.Ssplat.Sconv1.Sbiases/.ATTRIBUTES/VARIABLE_VALUE']
model1.layers[0].set_weights([weight1, bias1])

weight2 = tensors['vars/inference.Scoefficients.Ssplat.Sconv2.Sweights/.ATTRIBUTES/VARIABLE_VALUE']
bias2 = tensors['vars/inference.Scoefficients.Ssplat.Sconv2.Sbiases/.ATTRIBUTES/VARIABLE_VALUE']
model1.layers[1].set_weights([weight2, bias2])

weight3 = tensors['vars/inference.Scoefficients.Ssplat.Sconv3.Sweights/.ATTRIBUTES/VARIABLE_VALUE']
bias3 = tensors['vars/inference.Scoefficients.Ssplat.Sconv3.Sbiases/.ATTRIBUTES/VARIABLE_VALUE']
model1.layers[3].set_weights([weight3, bias3])

weight4 = tensors['vars/inference.Scoefficients.Ssplat.Sconv4.Sweights/.ATTRIBUTES/VARIABLE_VALUE']
bias4 = tensors['vars/inference.Scoefficients.Ssplat.Sconv4.Sbiases/.ATTRIBUTES/VARIABLE_VALUE']
model1.layers[5].set_weights([weight4, bias4])


model2=low_res_local()

weightlocal1=tensors['vars/inference.Scoefficients.Slocal.Sconv1.Sweights/.ATTRIBUTES/VARIABLE_VALUE']
biaslocal1 = tensors['vars/inference.Scoefficients.Slocal.Sconv1.Sbiases/.ATTRIBUTES/VARIABLE_VALUE']
model2.layers[0].set_weights([weightlocal1, biaslocal1])

weightlocal2=tensors['vars/inference.Scoefficients.Slocal.Sconv2.Sweights/.ATTRIBUTES/VARIABLE_VALUE']
# biaslocal2 = tensors['vars/inference.Scoefficients.Slocal.Sconv2.Sbiases/.ATTRIBUTES/VARIABLE_VALUE']
model2.layers[2].set_weights([weightlocal2])


model3=low_res_global()

weightglobal1=tensors['vars/inference.Scoefficients.Sglobal.Sconv1.Sweights/.ATTRIBUTES/VARIABLE_VALUE']
biasglobal1 = tensors['vars/inference.Scoefficients.Sglobal.Sconv1.Sbiases/.ATTRIBUTES/VARIABLE_VALUE']
model3.layers[0].set_weights([weightglobal1, biasglobal1])

weightglobal2 = tensors['vars/inference.Scoefficients.Sglobal.Sconv2.Sweights/.ATTRIBUTES/VARIABLE_VALUE']
biasglobal2 = tensors['vars/inference.Scoefficients.Sglobal.Sconv2.Sbiases/.ATTRIBUTES/VARIABLE_VALUE']
model3.layers[2].set_weights([weightglobal2, biasglobal2])

weightglobal3 = tensors['vars/inference.Scoefficients.Sglobal.Sfc1.Sweights/.ATTRIBUTES/VARIABLE_VALUE']
biasglobal3 = tensors['vars/inference.Scoefficients.Sglobal.Sfc1.Sbiases/.ATTRIBUTES/VARIABLE_VALUE']
model3.layers[5].set_weights([weightglobal3, biasglobal3])

weightglobal4 = tensors['vars/inference.Scoefficients.Sglobal.Sfc2.Sweights/.ATTRIBUTES/VARIABLE_VALUE']
biasglobal4 = tensors['vars/inference.Scoefficients.Sglobal.Sfc2.Sbiases/.ATTRIBUTES/VARIABLE_VALUE']
model3.layers[7].set_weights([weightglobal4, biasglobal4])

weightglobal5 = tensors['vars/inference.Scoefficients.Sglobal.Sfc3.Sweights/.ATTRIBUTES/VARIABLE_VALUE']
biasglobal5 = tensors['vars/inference.Scoefficients.Sglobal.Sfc3.Sbiases/.ATTRIBUTES/VARIABLE_VALUE']
model3.layers[9].set_weights([weightglobal5, biasglobal5])


im_input=cv2.imread("../sample_data/input.png",-1)
print(im_input.shape)
net_shape=256
lowres_input = skimage.transform.resize(
  im_input, [net_shape, net_shape], order = 0)
lowres_input = np.expand_dims(lowres_input, axis=0)
lowres_input = skimage.img_as_float(lowres_input)
# lowres_input = lowres_input/255.0
# lowres_input = np.clip(lowres_input,0,1)

print(lowres_input)
print(lowres_input.shape)

splat_features=model1.predict(lowres_input)
print(splat_features*255.0)
print(splat_features.shape)

local_features=model2(splat_features)
print(local_features.shape)
print(local_features)

global_features=model3(splat_features)
print(global_features.shape)
print(global_features)

fused_features=fusion(local_features,global_features,1)
print(fused_features.shape)
print(fused_features)
print(tf.math.count_nonzero(fused_features))