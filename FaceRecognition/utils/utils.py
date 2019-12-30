import tensorflow as tf 
import numpy as np 
import mxnet as mx
import pickle
import cv2
import os

from utils.exceptions import *

def load_graph_from_ckpt(sess, ckpt_dir, graph=True) -> None:
    all_ckpt_files = os.listdir(ckpt_dir)
    meta_file = [x for x in all_ckpt_files if os.path.splitext(x)[-1] == '.meta']
    if not meta_file:
        raise CkptNotComplete('Ckpt File doesn\'t contain meta file')
    meta_file = meta_file[0]
    ckpt_filename, ext = os.path.splitext(meta_file)
    
    if graph:
        saver = tf.train.import_meta_graph(os.path.join(ckpt_dir, meta_file))
    else:
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, os.path.join(ckpt_dir, ckpt_filename))

def print_all_tensor(sess, p=True) -> list:
    if p:
        print([n.name for n in sess.graph.as_graph_def().node])
    return sess.graph.as_graph_def().node

def print_global_vars(p=True) -> list:
    if p:
        print(tf.global_variables())
    return tf.global_variables()


def load_bin(db_name, image_size):
    bins, issame_list = pickle.load(open(db_name, 'rb'), encoding='bytes')
    data_list = []
    for _ in [0,1]:
        data = np.empty((len(issame_list)*2, image_size[0], image_size[1], 3))
        data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # print(img.shape)
        #
        img = cv2.resize(img, (112, 112))
        #
        for flip in [0,1]:
            if flip == 1:
                img = np.fliplr(img)
            data_list[flip][i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, issame_list

if __name__ == '__main__':
    sess = tf.Session()
    load_graph_from_ckpt(sess, '/Users/ecohnoch/Desktop/ckpt_model_d')
    g_vars = tf.global_variables()
    print(g_vars)
    graph = sess.graph
    print([n.name for n in graph.as_graph_def().node])
    sess.close()