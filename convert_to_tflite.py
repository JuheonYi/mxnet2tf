# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

import pylab as pl
import time
import os
import sys
from scipy.special import expit
import glob
import imageio

def freeze_graph(model_dir = "checkpoint/", model_name='mnet.25', input_node_names='input', output_node_names='output', out_name = "converted_model.tflite", quantize = False):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    input_checkpoint = model_dir+model_name

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    print(absolute_model_dir)
    output_graph = os.path.join("checkpoint/frozen_model_%s.pb"%model_name)

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    print("\nStart freezing...")
    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)
        
        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names # The output node names are used to select the usefull nodes
        )
        
        #test inference
        _in = tf.get_default_graph().get_tensor_by_name("data:0")
        print("input shape:", _in.shape)
        _out = []
        for i in range(len(output_node_names)):
            _tmp = tf.get_default_graph().get_tensor_by_name(output_node_names[i]+":0")
            print("output node:", _tmp)
            _out.append(_tmp)
        print("run test inference...")
        out = sess.run(_out, feed_dict = {_in: np.zeros([1,_in.shape[1].value, _in.shape[2].value,3])})

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            print("saving pb to ...", output_graph)
            f.write(output_graph_def.SerializeToString())
        #print("ops:",output_graph_def.node)
        print("%d ops in the final graph." % len(output_graph_def.node))
    #'''
    #'''
    print("\nStart TF lite converting...")

    converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(output_graph, input_node_names.split(","), output_node_names)
    tflite_model = converter.convert()
   
    #open(absolute_model_dir+"/"+out_name, "wb").write(tflite_model)
    out_path = absolute_model_dir+"/"+out_name+"-%dx%d.tflite"%(_in.shape[1], _in.shape[2])
    open(out_path, "wb").write(tflite_model)
    print("Finished! ", out_path)
    
    return output_graph_def


model_dir = "checkpoint/"
h = 1080
w = 1920
model_name = "mnet.25-%dx%d"%(h,w)

input_node_names = 'data'
output_node_names=['face_rpn_cls_score_stride32', 
                               'face_rpn_bbox_pred_stride32_rev', 
                               'face_rpn_landmark_pred_stride32_rev', 
                               'face_rpn_cls_score_stride16', 
                               'face_rpn_bbox_pred_stride16_rev', 
                               'face_rpn_landmark_pred_stride16_rev', 
                               'face_rpn_cls_score_stride8', 
                               'face_rpn_bbox_pred_stride8_rev', 
                               'face_rpn_landmark_pred_stride8_rev']

out_name = "RetinaFaceMobileNet"

freeze_graph(model_dir, model_name, input_node_names, output_node_names, out_name = out_name)