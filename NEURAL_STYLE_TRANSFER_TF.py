# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:11:44 2018

@author: ashima.garg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf 
import time     
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
from keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.utils.vis_utils import plot_model
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras import backend as K
from PIL import Image

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 300
COLOR_CHANNELS = 3
noise_ratio = 0.6
alpha = 10 
beta = 40
learning_rate = 2.0
num_iterations = 400
STYLE_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]

vggmodel = scipy.io.loadmat('pretrainedModel\imagenet-vgg-verydeep-19.mat')
print("model loaded")
vgg_layers = vggmodel['layers'][0]
    
def get_weights_bias(layer):
    W = vgg_layers[layer][0][0][2][0][0]
    b = vgg_layers[layer][0][0][2][0][1]
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    return W, b

def conv2d_layer(pre_layer, layer_name, layer):
    weight, bias = get_weights_bias(layer)
    conv = tf.nn.conv2d(pre_layer, weight, [1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)
    
def avg_pool_layer(pre_layer, layer_name):
    return tf.nn.avg_pool(pre_layer, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def relu_layer(conv2d_layer, layer_name, layer):
    return tf.nn.relu(conv2d_layer)
        
def build_model():
    print('\nBUILDING VGG-19 NETWORK')
    
    graph = {}
    print('INPUT')
    graph['input'] = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)), dtype = np.float32) 
    
    print('LAYER GROUP 1')
    graph['conv1_1'] = conv2d_layer(graph['input'], 'conv1_1', 0)
    graph['relu1_1'] = relu_layer(graph['conv1_1'], 'relu1_1', 0)
    graph['conv1_2'] = conv2d_layer(graph['relu1_1'], 'conv1_2', 2)
    graph['relu1_2'] = relu_layer(graph['conv1_2'], 'relu1_2', 2)
    graph['pool1'] = avg_pool_layer(graph['relu1_2'], 'pool1')
    
    print('LAYER GROUP 2')
    graph['conv2_1'] = conv2d_layer(graph['pool1'], 'conv2_1', 5)
    graph['relu2_1'] = relu_layer(graph['conv2_1'], 'relu2_1', 5)
    graph['conv2_2'] = conv2d_layer(graph['relu2_1'], 'conv2_2', 7)
    graph['relu2_2'] = relu_layer(graph['conv2_2'], 'relu2_2', 7)
    graph['pool2'] = avg_pool_layer(graph['relu2_2'], 'pool2')
    
    print('LAYER GROUP 3')
    graph['conv3_1'] = conv2d_layer(graph['pool2'], 'conv3_1', 10)
    graph['relu3_1'] = relu_layer(graph['conv3_1'], 'relu3_1', 10)
    graph['conv3_2'] = conv2d_layer(graph['relu3_1'], 'conv3_2', 12)
    graph['relu3_2'] = relu_layer(graph['conv3_2'], 'relu3_2', 12)
    graph['conv3_3'] = conv2d_layer(graph['relu3_2'], 'conv3_3', 14)
    graph['relu3_3'] = relu_layer(graph['conv3_3'], 'relu3_3', 14)
    graph['conv3_4'] = conv2d_layer(graph['relu3_3'], 'conv3_4', 16)
    graph['relu3_4'] = relu_layer(graph['conv3_4'], 'relu3_4', 16)
    graph['pool3'] = avg_pool_layer(graph['relu3_4'], 'pool3')
    
    print('LAYER GROUP 4')
    graph['conv4_1'] = conv2d_layer(graph['pool3'], 'conv4_1', 19)
    graph['relu4_1'] = relu_layer(graph['conv4_1'], 'relu4_1', 19)
    graph['conv4_2'] = conv2d_layer(graph['relu4_1'], 'conv4_2', 21)
    graph['relu4_2'] = relu_layer(graph['conv4_2'], 'relu4_2', 21)
    graph['conv4_3'] = conv2d_layer(graph['relu4_2'], 'conv4_3', 23)
    graph['relu4_3'] = relu_layer(graph['conv4_3'], 'relu4_3', 23)
    graph['conv4_4'] = conv2d_layer(graph['relu4_3'], 'conv4_4', 25)
    graph['relu4_4'] = relu_layer(graph['conv4_4'], 'relu4_4', 25)
    graph['pool4'] = avg_pool_layer(graph['relu4_4'], 'pool4')
    
    print('LAYER GROUP 5')
    graph['conv5_1'] = conv2d_layer(graph['pool4'], 'conv5_1', 28)
    graph['relu5_1'] = relu_layer(graph['conv5_1'], 'relu5_1', 28)
    graph['conv5_2'] = conv2d_layer(graph['relu5_1'], 'conv5_2', 30)
    graph['relu5_2'] = relu_layer(graph['conv5_2'], 'relu5_2', 30)
    graph['conv5_3'] = conv2d_layer(graph['relu5_2'], 'conv5_3', 32)
    graph['relu5_3'] = relu_layer(graph['conv5_3'], 'relu5_3', 32)
    graph['conv5_4'] = conv2d_layer(graph['relu5_3'], 'conv5_4', 34)
    graph['relu5_4'] = relu_layer(graph['conv5_4'], 'relu5_4', 34)
    graph['pool5'] = avg_pool_layer(graph['relu5_4'], 'pool5')
    
    return graph

MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 

def preprocess(image) :
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = image - MEANS
    return image

def generate_noise_image(contentImage):
    noiseImage = np.random.uniform(-20., 20., (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype(np.float32)
    genimage = noiseImage * noise_ratio + contentImage * (1. - noise_ratio) 
    return genimage

def compute_content_cost(a_C, a_G) :
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    con = tf.reshape(a_C,[n_H*n_W, n_C])
    gen = tf.reshape(a_G,[n_H*n_W, n_C])
    con = tf.transpose(con)
    gen = tf.transpose(gen)
    cost = (1./(4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(con, gen)))
    return cost

def gram_matrix(image, m, n_H, n_W, n_C):
    img = tf.transpose(tf.reshape(image, [n_H*n_W, n_C])) 
    gram = tf.matmul(img, tf.transpose(img))
    return gram

def compute_style_cost(styleImg, generatedImg):
    m, n_H, n_W, n_C = generatedImg.get_shape().as_list()
    styleG = gram_matrix(styleImg, m, n_H, n_W, n_C)
    generateG = gram_matrix(generatedImg, m, n_H, n_W, n_C)
    size = n_H*n_W
    cost = tf.multiply(tf.divide(1, (4 * size**2 * n_C**2)) , tf.reduce_sum(tf.square(styleG - generateG)))
    return cost

def compute_total_style_cost(model, sess):
    style_cost = 0
    for layer_name, coeff in STYLE_LAYERS:
        a_G = model[layer_name]
        a_S = sess.run(model[layer_name])
        style_cost += coeff * compute_style_cost(a_S, a_G)
    return style_cost    

def read_image(path):
    img = cv2.imread(path, 1)
    img = img.astype(np.float32)
    img = preprocess(img)
    return img
    
def postprocess(image):
    image = image + MEANS
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    return image

def display(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    return 

def trainNetwork():
    print('print style and content images')
    path = os.path.dirname(os.path.realpath(__file__))
    file = "\\Images\\"
    dirname = path + file
    stylePath = dirname + "1-style.jpg"
    styleImg = read_image(stylePath)
    contentPath = dirname + "1-content.jpg"
    contentImg = read_image(contentPath)
    generatedImg = generate_noise_image(contentImg)
    print("generated Image " + str(generatedImg.shape))
    display('STYLE', styleImg[0])
    display('CONTENT', contentImg[0])
    display('GENERATED', generatedImg[0])
    tf.reset_default_graph()    
    with tf.Session() as sess:
        model = build_model()
        print("model created ")
        sess.run(model['input'].assign(contentImg))
        a_G = model['conv4_2']
        a_C = sess.run(model['conv4_2'])
        J_content = compute_content_cost(a_C, a_G)
        sess.run(model['input'].assign(styleImg))
        J_style = compute_total_style_cost(model, sess)
        J_total = beta * J_style + alpha * 
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(J_total)
        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(generatedImg))
        print("generated image assigned")
        for i in range(num_iterations):
            sess.run(train_step)
            generatedImage = sess.run(model['input'])
            if i%20 == 0:
                J_Content, J_Style, J_Total = sess.run([J_content, J_style, J_total])
                print("J_Content " +str(J_Content))
                print("J_Style " +str(J_Style))
                print("J_total " + str(J_Total))
                generatedImage = postprocess(generatedImage)
                cv2.imwrite('Outputs\generatedImg' + str(i) + '.jpg', generatedImage)
    return
    
if __name__ == '__main__':
    trainNetwork()
    print("Completed")
