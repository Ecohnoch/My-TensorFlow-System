import tensorflow as tf 
from utils.utils import load_graph_from_ckpt, print_all_tensor, load_bin
from utils.verification import ver_test

from models.resnet50 import resnet50

class MyInsightFace(object):
    def __init__(self, ckpt_dir, params=None):
        self.ckpt_dir = ckpt_dir
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

        self.image_input = tf.placeholder(tf.float32, [None, 112, 112, 3], name='image_inputs')
        self.image_label_input = tf.placeholder(tf.int64,   [None, ], name='labels_inputs')
        with tf.name_scope('face_resnet50'):
            self.emb = resnet50(self.image_input, is_training=False)
        print(self.image_input)
        print(self.image_label_input)
        print(self.emb)

        load_graph_from_ckpt(self.sess, ckpt_dir, graph=False)
    
        # self.nodes = print_all_tensor(self.sess, p=False)

    def evaluation(self, bin):
        ver_list = []
        ver_name_list = []
        data_set = load_bin(bin, [112,112])
        ver_list.append(data_set)
        ver_name_list.append('lfw')
        feed_dict = {}

        results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=self.sess, embedding_tensor=self.emb, 
    		batch_size=32, feed_dict=feed_dict, input_placeholder=self.image_input)
        print(results)
    
    def train(self):
        from models.resnet50 import resnet50
        from losses.loss import arcface_loss
        tf.reset_default_graph()
        num_classes = self.params['num_classes']
        batch_size = self.params['batch_size']
        ckpt_save_dir = self.params['ckpt/']
        weight_decay = self.params['weight_decay']
        max_ep = self.params['max_ep']
        w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
        tfr = self.params['tfrecords']
        dataset = tf.data.TFRecordDataset(tfr)
        dataset = dataset.map(parse_function)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        image_input = tf.placeholder(tf.float32, [None, 112, 112, 3], name='image_inputs')
        image_label_input = tf.placeholder(tf.int64,   [None, ], name='labels_inputs')
        with tf.name_scope('face_resnet50'):
            emb = resnet50(image_input, is_training=True)
            logit = arcface_loss(embedding=emb, labels=labels, w_init=w_init_method, out_num=num_classes)
        inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
        conv_var = [var for var in tf.trainable_variables() if 'conv' in var.name]
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in conv_var])
        loss = inference_loss + weight_decay * l2_loss

        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
        lr_schedule = tf.train.piecewise_constant(global_step, boundaries=self.params['lr_steps'], values=self.params['lr'], name='lr_schedule')

        opt = tf.train.MomentumOptimizer(learning_rate=lr_schedule, momentum=0.9)
        grads = opt.compute_gradients(loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(grads, global_step=global_step)
        pred = tf.nn.softmax(logit)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), dtype=tf.float32))
        saver = tf.train.Saver(max_to_keep=3)
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        counter = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for i in range(max_ep):
                sess.run(iterator.initializer)
                while True:
                    try:
                        image_train, label_train = sess.run(next_element)
                        feed_dict = {images: image_train, labels: label_train}
                        _, loss_val, acc_val, _ = sess.run([train_op, inference_loss, acc, inc_op], feed_dict=feed_dict)

                        counter += 1
                        # print('counter: ', counter, 'loss_val', loss_val, 'acc: ', acc_val)
                        if counter % 100 == 0:
                            print('counter: ', counter, 'loss_val', loss_val, 'acc: ', acc_val)
                            filename = 'Face_iter_{:d}'.format(counter) + '.ckpt'
                            filename = os.path.join(ckpt_save_dir, filename)
                            saver.save(sess, filename)
                    except tf.errors.OutOfRangeError:
                        print('End of epoch %d', i)
                        break


    def load_default_params(self) -> dict:
        params = {}
        params['gpu'] = False
        ## Training Config
        params['num_classes'] = 1500
        params['max_ep'] = 10000
        params['batch_size'] = 64
        params['ckpt_save_dir'] = 'ckpt/'
        params['tfrecords'] = 'data/'
        params['weight_decay'] = 5e-4
        params['lr_steps'] = [80000, 120000, 160000]
        params['lr'] = [0.001, 0.0005, 0.0001, 0.00001]
        
        return params
    
    def reinit(self):
        self.__init__(self.ckpt_dir)

if __name__ == '__main__':
    # a = MyInsightFace('/Users/ecohnoch/Desktop/Benchmark-git/API/face_score/ckpt')
    a = MyInsightFace('/home/ruc_tliu_1/Face/ckpt/face_real708_ckpt/')
    a.evaluation('/home/ruc_tliu_1/faces_vgg_112x112/lfw.bin')