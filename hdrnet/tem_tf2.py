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

import bilateral_slice_tf2 as bs
import bilateral_slice as bsj

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

def predic_model():

  model = tf.keras.Sequential() 
  model.add(layers.Conv2D(96,(1,1),strides=(1,1), padding='same', activation=None, input_shape=(16,16,64)))
  return model

def packed_coefficients(current_layer):
  numImgChs=3
  n_out=3
  n_in=4
  current_layer = tf.stack(
    tf.split(current_layer, n_out*n_in, axis=3), axis=4)
  current_layer = tf.stack(
    tf.split(current_layer, n_in, axis=4), axis=5)
  return current_layer

def guide_color(ful_res,ccm,ccm_bias):
  npts=16
  # nchans = ful_res.get_shape().as_list()[-1]
  nchans = ful_res.shape[-1]
  # ccm = tensors['vars/inference.Sguide.Sccm/.ATTRIBUTES/VARIABLE_VALUE']

  # ccmbias=tensors['vars/inference.Sguide.Sccm_bias/.ATTRIBUTES/VARIABLE_VALUE']
  guidemap = tf.matmul(tf.reshape(ful_res, [-1, nchans]), ccm)
  guidemap = tf.nn.bias_add(guidemap, ccm_bias, name='ccm_bias_add')
  guidemap = tf.reshape(guidemap, tf.shape(input=ful_res))

  return guidemap
  
def guide_conv():
  model=tf.keras.Sequential()
  model.add(layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same', activation=None, input_shape=(2048,2048,3)))
  return model


def guide_curve(guidemap,shifts,slopes,guide_conv_w,guide_conv_b):
  # shifts_ = np.linspace(0, 1, npts, endpoint=False, dtype=np.float32)
  # shifts_ = shifts_[np.newaxis, np.newaxis, np.newaxis, :]
  # shifts_ = np.tile(shifts_, (1, 1, nchans, 1))

  guidemap = tf.expand_dims(guidemap, 4)
  # shifts = tfv1.get_variable('shifts', dtype=tf.float32, initializer=shifts_)
  # shifts = tensors['vars/inference.Sguide.Sshifts/.ATTRIBUTES/VARIABLE_VALUE']

  # slopes_ = np.zeros([1, 1, 1, nchans, npts], dtype=np.float32)
  # slopes_[:, :, :, :, 0] = 1.0
  # slopes = tfv1.get_variable('slopes', dtype=tf.float32, initializer=slopes_)
  # slopes = tensors['vars/inference.Sguide.Sslopes/.ATTRIBUTES/VARIABLE_VALUE']
  guidemap = tf.reduce_sum(input_tensor=slopes*tf.nn.relu(guidemap-shifts), axis=[4])

  # guidemap = tf.keras.layers.Conv2D(
  #     filters=1, 
  #     kernel_size=1, 
  #     kernel_initializer=tfv1.constant_initializer(1.0/nchans),
  #     bias_initializer=tfv1.constant_initializer(0),
  #     activation=None,
  #     name='channel_mixing'
  #     )(guidemap)

  model = guide_conv()
  model.layers[0].set_weights([guide_conv_w,guide_conv_b])

  guidemap = model(guidemap)
  guidemap = tf.clip_by_value(guidemap, 0, 1)
  guidemap = tf.squeeze(guidemap, axis=[3,])
  return guidemap
# __________________________



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

model4=predic_model()

weightpredic1=tensors['vars/inference.Scoefficients.Sprediction.Sconv1.Sweights/.ATTRIBUTES/VARIABLE_VALUE']
biaspredic1=tensors['vars/inference.Scoefficients.Sprediction.Sconv1.Sbiases/.ATTRIBUTES/VARIABLE_VALUE']
model4.layers[0].set_weights([weightpredic1,biaspredic1])

ccm = tensors['vars/inference.Sguide.Sccm/.ATTRIBUTES/VARIABLE_VALUE']
ccm_bias=tensors['vars/inference.Sguide.Sccm_bias/.ATTRIBUTES/VARIABLE_VALUE']

shifts = tensors['vars/inference.Sguide.Sshifts/.ATTRIBUTES/VARIABLE_VALUE']
slopes = tensors['vars/inference.Sguide.Sslopes/.ATTRIBUTES/VARIABLE_VALUE']
guide_conv_w = tensors['vars/inference.Sguide.Schannel_mixing.Sweights/.ATTRIBUTES/VARIABLE_VALUE']
guide_conv_b = tensors['vars/inference.Sguide.Schannel_mixing.Sbiases/.ATTRIBUTES/VARIABLE_VALUE']

# ____________________



# ____________________

im_input=cv2.imread("../sample_data/input.png",-1)
print(im_input.shape)
net_shape=256
lowres_input = skimage.transform.resize(
  im_input, [net_shape, net_shape], order = 0)
lowres_input = np.expand_dims(lowres_input, axis=0)
lowres_input = skimage.img_as_float(lowres_input)
# lowres_input = lowres_input/255.0
# lowres_input = np.clip(lowres_input,0,1)
fulres_input = np.expand_dims(im_input, axis=0)
fulres_input=skimage.img_as_float(fulres_input)

print(lowres_input)
print(lowres_input.shape)

splat_features=model1(lowres_input)
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
# print(fused_features)
# print(tf.math.count_nonzero(fused_features))

predic_features=model4(fused_features)
print(predic_features)
print(predic_features.shape)
print("if less zero")
has_negative = has_negative = tf.reduce_any(predic_features < 0)
print(has_negative.numpy())

packed_features=packed_coefficients(predic_features)
print("packed_features")
print(packed_features.shape)
print("if less zero")
has_negative = has_negative = tf.reduce_any(packed_features < 0)
print(has_negative.numpy())

guidemap = guide_color(fulres_input,ccm,ccm_bias)

guidemap = guide_curve(guidemap,shifts,slopes,guide_conv_w,guide_conv_b)
# print("guidemap:")
# print(guidemap)

gray_image_array = np.array(guidemap)
packed_features_np = np.array(packed_features)

sliced = bsj.bilateral_slice(packed_features_np[0],gray_image_array[0])
# tem = bs.local_bilateral_slice(guidemap,packed_features)
print("sliced:")
print(sliced.shape)

fulres_input_tf=tf.convert_to_tensor(fulres_input,dtype=tf.float32)
sliced_tf=tf.expand_dims(tf.convert_to_tensor(sliced,dtype=tf.float32),axis=0)
# print(sliced_tf.get_shape().as_list())
output = bs.apply(sliced_tf,fulres_input_tf)
print(output)


image_array = output[0].numpy()*255.0
image_array = image_array.astype(np.uint8)
print(image_array)
# skimage.io.imsave('image.png', image_array)
# # 从数组中提取灰度图像
# gray_image_array = np.squeeze(gray_image_array, axis=0)
# gray_image_array = gray_image_array * 255.0

# # 将像素值转换为 8 位无符号整数
# gray_image_array = gray_image_array.astype(np.uint8)
# print(gray_image_array)

# # 保存图像
# skimage.io.imsave('gray_image.jpg', gray_image_array)