import os
import cv2
import copy
import pickle
import numpy as np
import tensorflow as tf  

from tensorflow.data.experimental import prefetch_to_device, shuffle_and_repeat, map_and_batch

from utils import EasyDict

class Raw_Image:
    def __init__(self, config: EasyDict = None, preload:str = None, save: str = None):
        self.config = config
        self.check()
        if preload and not config.tfrecord:
            assert os.path.exists(preload)
            with open(preload, 'rb') as f:
                self.data = pickle.load(f)
            self.images = [x[0] for x in self.data]
            self.labels = [x[1] for x in self.data]
        if not preload and not config.tfrecord:
            self._traversal()
            self._labeling_and_preprocessing()

        if not preload and save and not config.tfrecord:
            self.data = list(zip(self.images, self.labels))
            with open(save, 'wb') as f:
                pickle.dump(self.data, f)

        if not config.tfrecord:
            print(' Raw Image Load Success, shape: {}x{}x{}x{}'.format(len(self.data), self.config.shape, self.config.shape, 3))

        if config.tfrecord:
            self._traversal()
            self._tfconfig(save, preload)
            print(' Raw Image Iterator Launched: ', self.iterator)
        
        # img, label = self.iterator.get_next()
        # # # print(img.shape)
        # sess = tf.Session()
        # i, l = sess.run([img, label])
        
        # print(i.shape, l.shape)




    def check(self) -> bool:
        assert self.config.dataset_root_dir != None
        assert self.config.raw_data_format != None
        assert self.config.ignore_dir_names != None

        
        def log(cond, msg):
            if cond:
                print(msg)
                return True
            return False
        
        if(log(not os.path.exists(self.config.dataset_root_dir), '*** Raw Image: Wrong dataset dir.{}'.format(self.config.dataset_root_dir))):
            return False
        return True

    def _tfconfig(self, save: str = None, preload: str = None) -> None:
        # self.all_images
        labeling_func = self.config.labeling_func
        preprocessing_func = self.config.preprocessing_func
        if not labeling_func:
            labeling_func = self._default_labeling_func
        if not preprocessing_func:
            preprocessing_func = self._default_preprocessing_func_tensor_input
        self.labels = []
        for each_image in self.all_images:
            self.labels.append(labeling_func(each_image))
        
        # preload
        if preload:
            self._load_tfrecord(preload)
            return
        if labeling_func == self._default_labeling_func:
            self._process_default_labeling_list(self.labels)
        self.inputs = tf.data.Dataset.from_tensor_slices((self.all_images, self.labels))
        self.inputs = self.inputs. \
            apply(shuffle_and_repeat(self.all_images_num)). \
            apply(map_and_batch(preprocessing_func, self.config.batch_size, num_parallel_batches=16, drop_remainder=True))
        if self.config.gpu_device:
            self.inputs = self.inputs.apply(prefetch_to_device('/gpu:{}'.format(self.config.gpu_device), None))
        self.iterator = self.inputs.make_one_shot_iterator()
        if save:
            self._save_tfrecord(save, preprocessing_func)

    def _save_tfrecord(self, save: str, preprocessing_func):
        assert os.path.exists(os.path.dirname(save))
        if preprocessing_func == self._default_preprocessing_func_tensor_input:
            preprocessing_func = self._default_preprocessing_func

        writer = tf.python_io.TFRecordWriter(save)
        for ind, (file, label) in enumerate(zip(self.all_images, self.labels)):
            img = preprocessing_func(file, self.config.shape)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            writer.write(example.SerializeToString())  # Serialize To String
        writer.close()
        print(' TFRecord save success: ', save)
    
    def _load_tfrecord(self, preload: str):
        assert os.path.exists(preload)
        self.inputs = tf.data.TFRecordDataset(preload)
        self.inputs = self.inputs.apply(map_and_batch(self._parse_func, self.config.batch_size, num_parallel_batches=16, drop_remainder=True))
    
        if self.config.gpu_device:
            self.inputs = self.inputs.apply(prefetch_to_device('/gpu:{}'.format(self.config.gpu_device), None))
        self.iterator = self.inputs.make_one_shot_iterator()
        print(' TFRecord load success: ')
        

    def _parse_func(self, example_proto):
        features = {'image_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)}
        features = tf.parse_single_example(example_proto, features)
        img = tf.decode_raw(features['image_raw'], tf.float32)
        img = tf.reshape(img, shape=(self.config.shape, self.config.shape, 3))
        label = tf.cast(features['label'], tf.int64)
        return img, label

    def _traversal(self):
        root_dir = self.config.dataset_root_dir
        data_format = self.config.raw_data_format
        ignore_dir = self.config.ignore_dir_names
        self.all_images = []
        for root, dirs, files in os.walk(root_dir):
            if os.path.split(root)[-1] in ignore_dir:
                continue
            for name in files:
                name = os.path.join(root, name)
                
                if os.path.splitext(name)[1] in data_format:
                    self.all_images.append(name)
        self.all_images_num = len(self.all_images)
        if self.all_images_num == 0:
            raise Exception('*** Dataset Error, empty image list.')

    def _labeling_and_preprocessing(self, labeling=True, preprocessing=True):
        shape_size = self.config.shape
        labeling_func = self.config.labeling_func
        preprocessing_func = self.config.preprocessing_func
        if not labeling_func:
            labeling_func = self._default_labeling_func
        if not preprocessing_func:
            preprocessing_func = self._default_preprocessing_func
        
        if not labeling:
            labeling_func = lambda x: x
        if not preprocessing:
            preprocessing_func = lambda x: x

        self.images = []
        self.labels = []
        for each_img_path in self.all_images:
            assert os.path.exists(each_img_path)
            self.labels.append(labeling_func(each_img_path))
            # img_array = cv2.imread(each_img_path)
            # if shape_size:
            #     img_array = cv2.resize(img_array, (shape_size, shape_size))
            img_array = preprocessing_func(each_img_path, shape_size)
            self.images.append(img_array)
        if labeling_func == self._default_labeling_func:
            self._process_default_labeling_list(self.labels)


    def _default_labeling_func(self, imgpath):
        # Default
        _, parent_folder = os.path.split(os.path.dirname(imgpath))
        return parent_folder
        
    def _process_default_labeling_list(self, labeling_list) -> None:
        encoder_dict = {}
        classes = 0
        for ind, item in enumerate(labeling_list):
            if item not in encoder_dict.keys():
                encoder_dict[item] = classes
                labeling_list[ind] = classes
                classes += 1
            else:
                labeling_list[ind] = encoder_dict[item]
        

    def _default_preprocessing_func(self, each_img_path, shape_size):
        img = cv2.imread(each_img_path)
        img = img.astype(np.float32)
        if shape_size:
            img = cv2.resize(img, (shape_size, shape_size))
        img = img - 127.5
        img = np.multiply(img, 0.0078125)
        return img

    def _default_preprocessing_func_tensor_input(self, *tup):
        shape_size = self.config.shape
        x = tf.read_file(tup[0])
        img = tf.image.decode_jpeg(x, channels=3, dct_method='INTEGER_ACCURATE')
        img = tf.to_float(img)
        if shape_size:
            img = tf.image.resize_images(img, size=[shape_size, shape_size], method=tf.image.ResizeMethod.BILINEAR)
        drange_in = [0.0, 255.0]
        drange_out = [-1.0, 1.0]
        scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
        bias = drange_out[0] - drange_in[0] * scale
        img = img * scale + bias

        ret = list(tup)
        ret[0] = img
        ret = tuple(ret)
        return ret





class Dataset:
    def __init__(self, config: EasyDict = None):
        self.config = config
        if not self.check_dataset_type():
            raise Exception('*** Dataset Error, check dataset config.')
        

    def check_dataset_type(self) -> bool:
        assert self.config.dataset_name != None
        assert self.config.dataset_type != None

        if self.config.dataset_type == 'Raw_Image':
            self.dataset_type = 'Raw_Image'
            self.dataset_object = Raw_Image(self.config.Raw_Image, self.config.preload, self.config.save_datapath)
            return self.dataset_object.check()
        
        return False


    def generate_batch(self, batch_size):
        pass


def load_default_config():
    config = EasyDict()
    config.dataset_name = 'name'
    config.dataset_type =  'Raw_Image'
    config.save_datapath = None
    config.preload = './1.tfrecord'

    raw_image = EasyDict()
    raw_image.dataset_root_dir = '/Users/ecohnoch/Desktop/face_gan/StyleGAN-Tensorflow-master'
    raw_image.raw_data_format = ['.png', '.jpg']
    raw_image.ignore_dir_names = ['.DS_Store']
    raw_image.shape = 32   # None, 32, 64, 128, 224...
    raw_image.labeling_func = None
    raw_image.preprocessing_func = None
    
    raw_image.tfrecord = True
    raw_image.batch_size = 32
    raw_image.gpu_device = None
    config.Raw_Image = raw_image
    return config

def labeling_func(imgpath):
    # Default
    _, parent_folder = os.path.split(os.path.dirname(imgpath))
    return parent_folder

def preprocessing_func(img):
    img = img - 127.5
    img = np.multiply(img, 0.0078125)
    return img

d = Dataset(load_default_config())