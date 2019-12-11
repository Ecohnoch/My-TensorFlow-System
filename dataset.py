import os
import cv2
import pickle
import numpy as np
import tensorflow as tf  
from utils import EasyDict

class Raw_Image:
    def __init__(self, config: EasyDict = None, preload:str = None, save: str = None):
        self.config = config
        self.check()
        if preload:
            assert os.path.exists(preload)
            with open(preload, 'rb') as f:
                self.data = pickle.load(f)
            self.images = [x[0] for x in self.data]
            self.labels = [x[1] for x in self.data]
        self._traversal()
        self._labeling_and_preprocessing()

        if not preload and save:
            self.data = list(zip(self.images, self.labels))
            with open(save, 'wb') as f:
                pickle.dump(self.data, f)
        print('Raw Image Load Success, shape: {}x{}x{}x{}'.format(len(self.data), self.config.shape, self.config.shape, 3))


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

    def _labeling_and_preprocessing(self):
        shape_size = self.config.shape
        labeling_func = self.config.labeling_func
        preprocessing_func = self.config.preprocessing_func
        if not labeling_func:
            labeling_func = self._default_labeling_func
        if not preprocessing_func:
            preprocessing_func = self._default_preprocessing_func
        self.images = []
        self.labels = []
        for each_img_path in self.all_images:
            assert os.path.exists(each_img_path)
            self.labels.append(labeling_func(each_img_path))
            img_array = cv2.imread(each_img_path)
            if shape_size:
                img_array = cv2.resize(img_array, (shape_size, shape_size))
            img_array = preprocessing_func(img_array)
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
        

    def _default_preprocessing_func(self, img):
        img = img - 127.5
        img = np.multiply(img, 0.0078125)
        return img



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
    config.preload = './1.pkl'
    config.tfrecord = None

    raw_image = EasyDict()
    raw_image.dataset_root_dir = '/Users/ecohnoch/Desktop/face_gan/StyleGAN-Tensorflow-master'
    raw_image.raw_data_format = ['.png', '.jpg']
    raw_image.ignore_dir_names = ['.DS_Store']
    raw_image.shape = 32   # None, 32, 64, 128, 224...
    raw_image.labeling_func = labeling_func
    raw_image.preprocessing_func = preprocessing_func
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