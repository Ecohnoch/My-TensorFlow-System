import tensorflow as tf 
from utils.utils import load_graph_from_ckpt, print_all_tensor, load_bin
from utils.verification import ver_test

class InsightFace(object):
    def __init__(self, ckpt_dir, params=None):
        if not params:
            params = self.load_default_params()
        self.params = params

        # Sess start
        if not self.params['gpu']:
            self.sess = tf.Session()
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

        load_graph_from_ckpt(self.sess, ckpt_dir)
    
        self.nodes = print_all_tensor(self.sess, p=False)
        self.emb = self.sess.graph.get_tensor_by_name('resnet_v1_50/E_BN2/Identity:0')
        self.image_input = self.sess.graph.get_tensor_by_name('img_inputs:0') # img_inputs
        self.image_label_input = self.sess.graph.get_tensor_by_name('img_labels:0') # img_labels
        self.dropout_rate = self.sess.graph.get_tensor_by_name('dropout_rate:0')
        print(self.image_input)
        print(self.image_label_input)
        print(self.emb)

    def evaluation(self, bin):
        ver_list = []
        ver_name_list = []
        data_set = load_bin(bin, [112,112])
        ver_list.append(data_set)
        ver_name_list.append('lfw')
        feed_dict = {}
        feed_dict[self.dropout_rate] = 1.0
        results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=self.sess, embedding_tensor=self.emb, 
    		batch_size=32, feed_dict=feed_dict, input_placeholder=self.image_input)
        print(results)
    
    def train(self):
        pass

    def load_default_params(self) -> dict:
        params = {}
        params['gpu'] = False
        return params

if __name__ == '__main__':
    a = InsightFace('/home/ruc_tliu_1/system/ckpt')
    a.evaluation('/home/ruc_tliu_1/faces_vgg_112x112/lfw.bin')