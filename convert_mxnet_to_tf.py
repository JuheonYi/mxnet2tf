import argparse
import json
import mxnet as mx
import tensorflow as tf
from tensorflow.python.framework import graph_util

import numpy as np

from converter import Converter
import os
import time

def main(model_prefix, output_prefix, input_h=128, input_w = 128):
    # Parsing JSON is easier because it contains operator name
    js_model = json.load(open(model_prefix + '-symbol.json', 'r'))
    mx_model, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
    params = arg_params
    params.update(aux_params)
    tf.reset_default_graph()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

    a = 0
    with tf.Session(config=config) as sess:
        tf_nodes = dict()
        # Workaround for input node
        input_data = tf.placeholder('float32', (1, input_h, input_w, 3), name='data')
        tf_nodes['data'] = input_data
        nodes = js_model['nodes']
        conv = Converter(tf_nodes, nodes, params)
        for node_idx, node in enumerate(nodes):
            op = node['op']
            print('Parsing node %s with operator %s and index %d' % (node['name'], op, node_idx))
            # Hack for older versions of MxNet
            if 'param' in node:
                node['attrs'] = node['param']
            if op == 'BatchNorm':
                conv.create_bn(node)
            elif op == 'elemwise_add' or op == '_Plus':
                conv.create_elementwise(node)
            elif op == 'Activation':
                conv.create_activation(node)
            elif op == 'SoftmaxOutput':
                conv.create_softmax(node)
            elif op == 'Convolution':
                conv.create_conv(node)
            elif op == 'Pooling':
                conv.create_pooling(node)
            elif op == 'Flatten':
                conv.create_flatten(node)
            elif op == 'FullyConnected':
                conv.create_fc(node)
            elif op == 'L2Normalization':
                conv.create_norm(node)
            elif op == "Concat":
                conv.create_concat(node)
            elif op == "Reshape":
                conv.create_reshape(node)
            elif op == "Crop":
                conv.create_crop(node)
            elif op == "UpSampling":
                #conv.create_upsampling(node)
                conv.create_upsampling_v2(node)
            elif op == "SoftmaxActivation":
                conv.create_softmaxactivation(node)
            elif op == 'null':
                #print("this is just a param, do not create op")
                a += 1
            else:
                print("------------------unsupported op!!!----------------------")
        
        #print("null nodes:", a)

        #test run
        output_node_names = []
        for i in range(len(js_model['heads'])):
            output_node_names.append(nodes[js_model['heads'][i][0]]['name'])
        print("output node names:", output_node_names)
        output_nodes = []
        print("------------------------------Output nodes:--------------------------")
        for i in range(len(output_node_names)):
            print(tf_nodes[output_node_names[i]])
            output_nodes.append(tf_nodes[output_node_names[i]])
        
        # test define my own conv layer after
        output_nodes_rev = []
        #################### stride 32 ################################
        output_nodes_rev.append(output_nodes[0])
        tmp = output_nodes[3]
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            bbox_stride32_w = tf.get_variable("face_rpn_bbox_pred_stride32_weight")
            bbox_stride32_b = tf.get_variable("face_rpn_bbox_pred_stride32_bias")
            landmark_stride32_w = tf.get_variable("face_rpn_landmark_pred_stride32_weight")
            landmark_stride32_b = tf.get_variable("face_rpn_landmark_pred_stride32_bias")
            #print("shape: ", bbox_stride32_w.shape, bbox_stride32_b.shape)
        tmp1 = tf.nn.conv2d(tmp, bbox_stride32_w, [1, 1, 1, 1], padding='VALID')
        tmp1 = tf.add(tmp1, bbox_stride32_b, name = "face_rpn_bbox_pred_stride32_rev")

        tmp2 = tf.nn.conv2d(tmp, landmark_stride32_w, [1, 1, 1, 1], padding='VALID')
        tmp2 = tf.add(tmp2, landmark_stride32_b, name = "face_rpn_landmark_pred_stride32_rev")

        output_nodes_rev.append(tmp1)
        output_nodes_rev.append(tmp2)
        
        #################### stride 16 ################################
        output_nodes_rev.append(output_nodes[1])
        tmp = output_nodes[4]
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            bbox_stride16_w = tf.get_variable("face_rpn_bbox_pred_stride16_weight")
            bbox_stride16_b = tf.get_variable("face_rpn_bbox_pred_stride16_bias")
            landmark_stride16_w = tf.get_variable("face_rpn_landmark_pred_stride16_weight")
            landmark_stride16_b = tf.get_variable("face_rpn_landmark_pred_stride16_bias")
            
        tmp1 = tf.nn.conv2d(tmp, bbox_stride16_w, [1, 1, 1, 1], padding='VALID')
        tmp1 = tf.add(tmp1, bbox_stride16_b, name = "face_rpn_bbox_pred_stride16_rev")

        tmp2 = tf.nn.conv2d(tmp, landmark_stride16_w, [1, 1, 1, 1], padding='VALID')
        tmp2 = tf.add(tmp2, landmark_stride16_b, name = "face_rpn_landmark_pred_stride16_rev")

        output_nodes_rev.append(tmp1)
        output_nodes_rev.append(tmp2)
        
        #################### stride 8 ################################
        output_nodes_rev.append(output_nodes[2])
        tmp = output_nodes[5]
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            bbox_stride8_w = tf.get_variable("face_rpn_bbox_pred_stride8_weight")
            bbox_stride8_b = tf.get_variable("face_rpn_bbox_pred_stride8_bias")
            landmark_stride8_w = tf.get_variable("face_rpn_landmark_pred_stride8_weight")
            landmark_stride8_b = tf.get_variable("face_rpn_landmark_pred_stride8_bias")
            
        tmp1 = tf.nn.conv2d(tmp, bbox_stride8_w, [1, 1, 1, 1], padding='VALID')
        tmp1 = tf.add(tmp1, bbox_stride8_b, name = "face_rpn_bbox_pred_stride8_rev")

        tmp2 = tf.nn.conv2d(tmp, landmark_stride8_w, [1, 1, 1, 1], padding='VALID')
        tmp2 = tf.add(tmp2, landmark_stride8_b, name = "face_rpn_landmark_pred_stride8_rev")

        output_nodes_rev.append(tmp1)
        output_nodes_rev.append(tmp2)
        
        print("------------------------output nodes rev:----------------------")
        for node in output_nodes_rev:
            print(node)

        print("---------------------test inference-------------------")
        for i in range(5):
            start = time.time()
            out = sess.run(output_nodes_rev, feed_dict = {input_data: np.zeros([1,input_h,input_w,3])})
            end = time.time()
            print("inference time: %.6f" %(end-start))
        for i in range(len(out)):
            print("output node shape:", out[i].transpose(0,3,1,2).shape)
        #'''
        
        g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        print("number of variables (trainable & global):", len(g_vars))
        saver = tf.train.Saver(g_vars)
        saver.save(sess, os.path.join("./checkpoint", "mnet.25-%dx%d"%(input_h, input_w)), write_meta_graph=True)
        
mx_prefix = "models/mnet.25"
tf_prefix = "xxx"
input_h = 1080
input_w = 1920
main(mx_prefix, tf_prefix, input_h, input_w)