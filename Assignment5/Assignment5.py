!pip install -q -U tensorflow_addons

# Stanford 40
!wget http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip
!wget http://vision.stanford.edu/Datasets/Stanford40_ImageSplits.zip
# TV Human Interaction (TV-HI)
!wget http://www.robots.ox.ac.uk/~alonso/data/tv_human_interactions_videos.tar.gz
!wget http://www.robots.ox.ac.uk/~alonso/data/readme.txt

from matplotlib import pyplot
from os.path import isfile
from os import getcwd
import warnings
import seaborn
import pickle
import pandas
import numpy
import math
import cv2
import os

import string
import collections
from six.moves import xrange

from sklearn.exceptions import UndefinedMetricWarning
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras import Model
from tensorflow.keras import backend
from tensorflow.keras import Input

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Normalization, BatchNormalization
from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import multiply, add
from tensorflow.keras.layers import Lambda, Rescaling, Reshape, Flatten, Concatenate
from tensorflow.keras.layers import Dropout

from tensorflow_addons.optimizers import CyclicalLearningRate
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import L1


warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

# Stanford 40
!unzip Stanford40_JPEGImages.zip -d Stanford40/
!unzip Stanford40_ImageSplits.zip -d Stanford40/
# TV Human Interaction (TV-HI)
!mkdir TV-HI
!tar -xvf  'tv_human_interactions_videos.tar.gz' -C TV-HI
!mv readme.txt 'TV-HI/readme.txt'

# Stanford 40
with open('Stanford40/ImageSplits/train.txt', 'r') as f:
    train_files = list(map(str.strip, f.readlines()))
    train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
    print(f'Train files ({len(train_files)}):\n\t{train_files}')
    print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n')

with open('Stanford40/ImageSplits/test.txt', 'r') as f:
    test_files = list(map(str.strip, f.readlines()))
    test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
    print(f'Test files ({len(test_files)}):\n\t{test_files}')
    print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n')
    
action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))
print(f'Action categories ({len(action_categories)}):\n{action_categories}')

# TV Human Interaction (TV-HI)
set_1_indices = [[2,14,15,16,18,19,20,21,24,25,26,27,28,32,40,41,42,43,44,45,46,47,48,49,50],
                 [1,6,7,8,9,10,11,12,13,23,24,25,27,28,29,30,31,32,33,34,35,44,45,47,48],
                 [2,3,4,11,12,15,16,17,18,20,21,27,29,30,31,32,33,34,35,36,42,44,46,49,50],
                 [1,7,8,9,10,11,12,13,14,16,17,18,22,23,24,26,29,31,35,36,38,39,40,41,42]]
set_2_indices = [[1,3,4,5,6,7,8,9,10,11,12,13,17,22,23,29,30,31,33,34,35,36,37,38,39],
                 [2,3,4,5,14,15,16,17,18,19,20,21,22,26,36,37,38,39,40,41,42,43,46,49,50],
                 [1,5,6,7,8,9,10,13,14,19,22,23,24,25,26,28,37,38,39,40,41,43,45,47,48],
                 [2,3,4,5,6,15,19,20,21,25,27,28,30,32,33,34,37,43,44,45,46,47,48,49,50]]
classes = ['handShake', 'highFive', 'hug', 'kiss']  # we ignore the negative class

# test set
set_1 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_1_indices[c]]
set_1_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_1_indices[c]]
print(f'Set 1 to be used for test ({len(set_1)}):\n\t{set_1}')
print(f'Set 1 labels ({len(set_1_label)}):\n\t{set_1_label}\n')

# training set
set_2 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_2_indices[c]]
set_2_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_2_indices[c]]
print(f'Set 2 to be used for train and validation ({len(set_2)}):\n\t{set_2}')
print(f'Set 2 labels ({len(set_2_label)}):\n\t{set_2_label}')

def construct_data():
    """ Construct the Standfort40 and TVHII """

    with open('Stanford40/ImageSplits/train.txt', 'r') as f:
        train_files = list(map(str.strip, f.readlines()))
        train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
    with open('Stanford40/ImageSplits/test.txt', 'r') as f:
        test_files = list(map(str.strip, f.readlines()))
        test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
    action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))
    standford40 = train_files, train_labels, test_files, test_labels

    # TV Human Interaction (TV-HI)
    set_1_indices = [[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 22, 23, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39],
                     [2, 3, 4, 5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 36, 37, 38, 39, 40, 41, 42, 43, 46, 49, 50],
                     [1, 5, 6, 7, 8, 9, 10, 13, 14, 19, 22, 23, 24, 25, 26, 28, 37, 38, 39, 40, 41, 43, 45, 47, 48],
                     [2, 3, 4, 5, 6, 15, 19, 20, 21, 25, 27, 28, 30, 32, 33, 34, 37, 43, 44, 45, 46, 47, 48, 49, 50]]
    set_2_indices = [[2, 14, 15, 16, 18, 19, 20, 21, 24, 25, 26, 27, 28, 32, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                     [1, 6, 7, 8, 9, 10, 11, 12, 13, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 44, 45, 47, 48],
                     [2, 3, 4, 11, 12, 15, 16, 17, 18, 20, 21, 27, 29, 30, 31, 32, 33, 34, 35, 36, 42, 44, 46, 49, 50],
                     [1, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 22, 23, 24, 26, 29, 31, 35, 36, 38, 39, 40, 41, 42]]
    classes = ['handShake', 'highFive', 'hug', 'kiss']  # we ignore the negative class

    # training set
    set_1 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_1_indices[c]]
    set_1_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_1_indices[c]]
    # test set
    set_2 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_2_indices[c]]
    set_2_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_2_indices[c]]
    TVHII = set_1, set_1_label, set_2, set_2_label
    return standford40, TVHII, action_categories


# preprocess the training and test data (convert labels to categoricals and crop or pad the data)
def preprocess_stanford40(data, maxsize=(170, 170)):
    """ Preprocess the stanford40 data set.
        Labels are transformed to integers, then one-hot encoded.
        Features are resized."""

    train_files, train_labels, test_files, test_labels = data

    #get the number of classes
    num_classes = numpy.unique(test_labels).shape[0]

    # load, pad or crop images
    train_res = [cv2.resize(cv2.imread(f'Stanford40/JPEGImages/{train_files[image_no]}')[...,::-1], (maxsize[0], maxsize[1])) # from BGR to RGB
                 for image_no, image_name in enumerate(train_files)]
    test_res = [cv2.resize(cv2.imread(f'Stanford40/JPEGImages/{test_files[image_no]}')[...,::-1], (maxsize[0], maxsize[1])) # from BGR to RGB
                for image_no, image_name in enumerate(test_files)]

    # transform instance labels from string to integer
    train_y_int = [numpy.where(numpy.unique(train_labels) == lab)[0][0] for idx, lab in enumerate(train_labels)]
    test_y_int = [numpy.where(numpy.unique(test_labels) == lab)[0][0] for idx, lab in enumerate(test_labels)]
    # one hot encode target values
    train_y_1hot = to_categorical(train_y_int)
    test_y_1hot = to_categorical(test_y_int)
    # prepare output data
    data = [numpy.array(train_res), train_y_1hot, numpy.array(train_y_int), numpy.array(test_res), test_y_1hot, numpy.array(test_y_int), num_classes]
    return data

def preprocess_TVHI(data):
    """ Preprocess the TV-HI data set.
    Labels are transformed to integers, then one-hot encoded.
    """

    train_files, train_labels, test_files, test_labels = data

    #get the number of classes
    num_classes = numpy.unique(test_labels).shape[0]
    # transform instance labels from string to integer
    train_y_int = [numpy.where(numpy.unique(train_labels) == lab)[0][0] for idx, lab in enumerate(train_labels)]
    test_y_int = [numpy.where(numpy.unique(test_labels) == lab)[0][0] for idx, lab in enumerate(test_labels)]
    # one hot encode target values
    train_y_1hot = to_categorical(train_y_int)
    test_y_1hot = to_categorical(test_y_int)
    # prepare output data
    data = [numpy.array(train_files), train_y_1hot, numpy.array(train_y_int), numpy.array(test_files), test_y_1hot, numpy.array(test_y_int), num_classes]
    return data

def get_middle_frame(data, maxsize=(170, 170), verbose=0):
  """ Gets the middle frame for Dense Optical Flow in OpenCV """

  train_mf, test_mf = [], []
  # Get the videos from training and test set
  for set_index, video_set in enumerate(data):
    #for each video in either the training or the test set
    for index, filename in enumerate(video_set):
      video_frames = []
      if filename.endswith("avi"): 
        cap = cv2.VideoCapture(f'/content/TV-HI/tv_human_interactions_videos/{filename}')
  
        while(1):
          ret, frame2 = cap.read()
          if not ret: break
          video_frames.append(numpy.array(cv2.resize(frame2, (maxsize[0], maxsize[1]))))

      # Middle Frame 
      middle_frame = len(video_frames) // 2

      if set_index == 0: train_mf.append(video_frames[middle_frame])
      elif set_index == 1: test_mf.append(video_frames[middle_frame])
      
      if verbose==1: print(f'Next video, video numer {index}')
      cv2.destroyAllWindows()
    if verbose==1: print(f'set_index   {set_index}\n\n')
  return train_mf, test_mf

def get_optical_flow(data, maxsize=(170, 170), verbose=0):
  """ Computes the Dense Optical flow from video frames"""

  # Get the videos from training and test set
  for set_index, video_set in enumerate(data):
  
    # Lists to store the generated stacks and batches of optical flow frames alongside their corresponding actual frame
    stacks_optical_flow_images = []
    #for each video in either the training or the test set
    for index, filename in enumerate(video_set):
      
      if filename.endswith("avi"): 
        video_file = f'/content/TV-HI/tv_human_interactions_videos/{filename}'
        cap = cv2.VideoCapture(video_file)

        ret, frame1 = cap.read()
        frame1 = numpy.array(cv2.resize(frame1, (maxsize[0], maxsize[1])))
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = numpy.zeros_like(frame1)
        hsv[..., 1] = 255
        of_images = []

        while(1):
          ret, frame2 = cap.read()
          if not ret: break

          frame2 = numpy.array(cv2.resize(frame2, (maxsize[0], maxsize[1])))
          next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
          flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
          mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
          hsv[..., 0] = ang*180/numpy.pi/2
          hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
          bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

          of_images.append(bgr)
          prvs = next

        #========== END WHILE LOOP ================
        if verbose==1: print(f'Next video, video numer {index}')
      middle_frame = len(of_images) // 2
      stacks_optical_flow_images.append(of_images[middle_frame])
      cv2.destroyAllWindows()
    
    #========== END INDIVIDUAL VIDEOS FOR LOOP ================
    if set_index == 0:   train_all_optical_flow_frames = stacks_optical_flow_images
    elif set_index == 1:  test_all_optical_flow_frames = stacks_optical_flow_images
    
    if verbose==1: print(f'set_index   {set_index}\n\n')
  return train_all_optical_flow_frames, test_all_optical_flow_frames

def get_swish(**kwargs):
    def swish(x):
        """Swish activation function: x * sigmoid(x). Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941) """
        return x * backend.sigmoid(x)

    return swish

class DCNN_clfr():
    def __init__(self,
                 model_name=None,
                 batch_size=32,
                 epochs=50,
                 dropout=0.4,
                 dropout_res=0.4,
                 optimizer=RMSprop,
                 learning_rate=(1e-5, 1e-3),
                 early_stopping=10,
                 num_classes=None,):
        """ Instantiates the DCN classifier class including the preprocessing, the model, 
            the processing pipeline and performance visualization.
            # Arguments
               model_name: string, model name.
                batch_size: integer, size of the batch.
                epoch: integer, number of epochs for training and validation.
                dropout_rate: float, dropout rate before final classifier layer.
                drop_connect_rate: float, dropout rate at skip connections.
                optimizer: optimizer, specify the optimizer object for learning.
                    For instance SGD, Adam, RMSprop.
                init_learning_rate: tuple of floats, specifying the learning rate
                    (or step size) for the cyclic learning schedule
                    - first position: specifies the initial learning rate  
                      at the start of the cycle 
                    - second position: specifies the maximal learning rate 
                      at the valley of the cycle.
                max_learning_rate: float, maximal learning rate (or step size)
                    for the cyclic learning schedule at the valley of the cycle.
                early_stopping: integer, number of epochs after training is terminate
                    if the validation performance does not increase
                classes: integer, number of classes to classify images
                    into, only to be specified if `include_top` is True. """

        # General parameters
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs

        # Compilation aprameters
        self.optimizer = optimizer
        self.init_lr = learning_rate[0]
        self.max_lr = learning_rate[1]
        self.patience = early_stopping

        # Model parameters
        self.classes=num_classes
        self.dropout_rate = dropout
        self.drop_connect_rate = dropout_res
        self.conv_init = {
            'class_name': 'VarianceScaling',
            'config': {
                'scale': 2.0,
                'mode': 'fan_out',
                'distribution': 'normal'
            }
        }
        self.dense_init = {
            'class_name': 'VarianceScaling',
            'config': {
                'scale': 1. / 3.,
                'mode': 'fan_out',
                'distribution': 'uniform'
            }
        }
        self.conv_reg = {
            'class_name': 'L1',
            'config': {
                'l1': 1e-4
            }
        }
        self.block_args = collections.namedtuple('BlockArgs', [
            'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
            'expand_ratio', 'id_skip', 'strides', 'se_ratio'
        ])
        self.block_args.__new__.__defaults__ = (None,) * len(self.block_args._fields)

        self.default_block_args = [
            self.block_args(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
                      expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
            self.block_args(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
                      expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
            self.block_args(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
                      expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
            self.block_args(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
                      expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
            self.block_args(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
                      expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
            self.block_args(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
                      expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
            self.block_args(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
                      expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
        ]

        # Output parameters
        self.model = None
        self.history = []
        self.show_model_char = True

    def data_augmentation(self, data):
        """ Data is augmented with variation in brightness and color jittering"""
        augmented_data = ImageDataGenerator(fill_mode='constant',
                                            brightness_range=[0.5, 1.0],
                                            channel_shift_range=10.0,
                                            dtype=float)
        augmented_data.fit(data)
        return augmented_data

    def define_model(self,
                     width_coefficient=1.4,
                     depth_coefficient=1.8, 
                     depth_divisor=8,
                     input_shape=None,
                     pooling=None,
                     **kwargs):
        """Instantiates the EfficientNet architecture
        # Arguments
            width_coefficient: float, scaling coefficient for network width.
            depth_coefficient: float, scaling coefficient for network depth.
            depth_divisor: int.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False.
                It should have exactly 3 inputs channels.
            pooling: optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied."""

        def mb_conv_block(inputs, block_args, activation, conv_init, conv_reg, drop_rate=0, prefix='', ):
            """Mobile Inverted Residual Bottleneck."""

            has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
            bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

            # Expansion phase
            filters = block_args.input_filters * block_args.expand_ratio
            if block_args.expand_ratio != 1:
                x = Conv2D(filters, 1,
                           padding='same',
                           use_bias=False,
                           kernel_initializer=conv_init,
                           name=f'{self.model_name}_{prefix}' + 'expand_conv')(inputs)
                x = BatchNormalization(axis=bn_axis, name=f'{self.model_name}_{prefix}' + 'expand_bn')(x)
                x = Activation(activation, name=f'{self.model_name}_{prefix}' + 'expand_activation')(x)
            else:
                x = inputs

            # Depthwise Convolution
            x = DepthwiseConv2D(block_args.kernel_size,
                                strides=block_args.strides,
                                padding='same',
                                use_bias=False,
                                depthwise_initializer=conv_init,
                                name=f'{self.model_name}_{prefix}' + 'dwconv')(x)
            x = BatchNormalization(axis=bn_axis, name=f'{self.model_name}_{prefix}' + 'bn')(x)
            x = Activation(activation, name=f'{self.model_name}_{prefix}' + 'activation')(x)

            # Squeeze and Excitation phase
            if has_se:
                num_reduced_filters = max(1, int(
                    block_args.input_filters * block_args.se_ratio
                ))
                se_tensor = GlobalAveragePooling2D(name=f'{self.model_name}_{prefix}' + 'se_squeeze')(x)

                target_shape = (1, 1, filters) if backend.image_data_format() == 'channels_last' else (filters, 1, 1)
                se_tensor = Reshape(target_shape, name=f'{self.model_name}_{prefix}' + 'se_reshape')(se_tensor)
                se_tensor = Conv2D(num_reduced_filters, 1,
                                   activation=activation,
                                   padding='same',
                                   use_bias=True,
                                   kernel_initializer=conv_init,
                                   name=f'{self.model_name}_{prefix}' + 'se_reduce')(se_tensor)
                se_tensor = Conv2D(filters, 1,
                                   activation='sigmoid',
                                   padding='same',
                                   use_bias=True,
                                   kernel_initializer=conv_init,
                                   name=f'{self.model_name}_{prefix}' + 'se_expand')(se_tensor)
                if backend.backend() == 'theano':
                    # For the Theano backend, we have to explicitly make
                    # the excitation weights broadcastable.
                    pattern = ([True, True, True, False] if backend.image_data_format() == 'channels_last'
                               else [True, False, True, True])
                    se_tensor = Lambda(
                        lambda x: backend.pattern_broadcast(x, pattern),
                        name=f'{self.model_name}_{prefix}' + 'se_broadcast')(se_tensor)
                x = multiply([x, se_tensor], name=f'{self.model_name}_{prefix}' + 'se_excite')

            # Output phase
            x = Conv2D(block_args.output_filters, 1,
                       padding='same',
                       use_bias=False,
                       kernel_initializer=conv_init,
                       name=f'{self.model_name}_{prefix}' + 'project_conv')(x)
            x = BatchNormalization(axis=bn_axis, name=f'{self.model_name}_{prefix}' + 'project_bn')(x)
            if block_args.id_skip and all(
                    s == 1 for s in block_args.strides
            ) and block_args.input_filters == block_args.output_filters:
                if drop_rate and (drop_rate > 0):
                    x = Dropout(drop_rate,
                                name=f'{self.model_name}_{prefix}' + 'drop')(x)
                x = add([x, inputs], name=f'{self.model_name}_{prefix}' + 'add')
            return x

        def round_filters(filters, width_coefficient, depth_divisor):
            """Round number of filters based on width multiplier."""

            filters *= width_coefficient
            new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
            new_filters = max(depth_divisor, new_filters)

            # Make sure that round down does not go down by more than 10%.
            if new_filters < 0.9 * filters:
                new_filters += depth_divisor
            return int(new_filters)

        def round_repeats(repeats, depth_coefficient):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        # Get channel position and activation function
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        activation = get_swish(**kwargs)

        # Build stem
        input = Input(shape=input_shape, name=self.model_name)
        x = Rescaling(1. / 255., name=f'{self.model_name}_resc')(input)
        x = Normalization(axis=bn_axis, name=f'{self.model_name}_prebn')(x)

        x = Conv2D(round_filters(32, width_coefficient, depth_divisor), 3,
                   strides=(2, 2),
                   padding='same',
                   use_bias=False,
                   kernel_initializer=self.conv_init,
                   name=f'{self.model_name}_stem_conv')(x)
        x = BatchNormalization(axis=bn_axis, name=f'{self.model_name}_stem_bn')(x)
        x = Activation(activation, name=f'{self.model_name}_stem_activation')(x)

        # Build blocks
        num_blocks_total = sum(round_repeats(block_args.num_repeat,
                                             depth_coefficient) for block_args in self.default_block_args)
        block_num = 0
        for idx, block_args in enumerate(self.default_block_args):
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters,
                                            width_coefficient, depth_divisor),
                output_filters=round_filters(block_args.output_filters,
                                             width_coefficient, depth_divisor),
                num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

            # The first block needs to take care of stride and filter size increase.
            drop_rate = self.drop_connect_rate * float(block_num) / num_blocks_total
            x = mb_conv_block(x, block_args,
                              activation=activation,
                              conv_init=self.conv_init,
                              conv_reg=self.conv_reg,
                              drop_rate=drop_rate,  
                              prefix='block{}a_'.format(idx + 1))
            block_num += 1
            if block_args.num_repeat > 1:
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                # pylint: enable=protected-access
                for bidx in xrange(block_args.num_repeat - 1):
                    drop_rate = self.drop_connect_rate * float(block_num) / num_blocks_total
                    block_prefix = 'block{}{}_'.format(
                        idx + 1,
                        string.ascii_lowercase[bidx + 1]
                    )
                    x = mb_conv_block(x, block_args,
                                      activation=activation,
                                      conv_init=self.conv_init,
                                      conv_reg=self.conv_reg, 
                                      drop_rate=drop_rate, 
                                      prefix=block_prefix)
                    block_num += 1

        # Build top
        x = Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,
                   padding='same',
                   use_bias=False,
                   kernel_initializer=self.conv_init,
                   kernel_regularizer=self.conv_reg,
                   name=f'{self.model_name}_top_conv')(x)
        x = BatchNormalization(axis=bn_axis, name=f'{self.model_name}_top_bn')(x)
        output = Activation(activation, name=f'{self.model_name}_top_activation')(x)
        if not (self.classes is None):
            output = GlobalAveragePooling2D(name=f'{self.model_name}_avg_pool')(output)
            if self.dropout_rate and self.dropout_rate > 0:
                output = Dropout(self.dropout_rate, name=f'{self.model_name}_top_dropout')(output)
            output = Dense(units=self.classes,
                           activation='softmax',
                           kernel_initializer=self.dense_init,
                           name=f'{self.model_name}_probs')(output)
        else:
            if pooling == 'avg':
                output = GlobalAveragePooling2D(name=f'{self.model_name}_avg_pool')(output)
            elif pooling == 'max':
                output = GlobalMaxPooling2D(name=f'{self.model_name}_max_pool')(output)

        # Create model.
        model = Model(inputs=input, outputs=output, name=self.model_name)
        return model

    def compile_model(self, model, data_length, from_logits=False):
        """ Compiles the model with categorical crossentropy"""

        # Train from logits or from softmax activation
        if self.classes is None: from_logits=True

        # Compilation parameter
        metrics = ['accuracy', 'Precision', 'Recall', 'AUC']
        clr = CyclicalLearningRate(initial_learning_rate=self.init_lr,
                                   maximal_learning_rate=self.max_lr,
                                   scale_fn=lambda x: 1 / (2. ** (x - 1)),
                                   step_size=2 * (data_length // self.batch_size))
        opt = self.optimizer(clr, clipvalue=10)
        loss = CategoricalCrossentropy(from_logits=from_logits)

        # Compile model
        model.compile(optimizer=opt, loss=loss, metrics=metrics)

        # Graph and textual represenation of the model architecture
        if self.show_model_char:
            plot_model(model,
                       f'{self.model_name}_Graph.png',
                       show_shapes=True,
                       dpi=20 * len(model.layers),
                       expand_nested=True)
            with open(f'{self.model_name}_summary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            self.show_model_char = False
        return model

    def fit_mono_model(self, model, train_x, train_y, val_x, val_y):
        """ Trains the One-Stream-Model with data augmentation. Training uses early stopping, restoring the best weights"""

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_accuracy',
                                       restore_best_weights=True,
                                       patience=self.patience)
        
        # Data augmentation
        print("[INFO] performing 'on the fly' data augmentation")
        trainGenerator = self.data_augmentation(train_x).flow(x=train_x, y=train_y,
                                                              batch_size=self.batch_size)
        valGenerator = self.data_augmentation(val_x).flow(x=val_x, y=val_y,
                                                          batch_size=self.batch_size)
        # Trains and validates the model
        self.history = model.fit(x=trainGenerator,
                            validation_data=valGenerator,
                            shuffle='TRUE',
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            verbose=1,
                            callbacks=[early_stopping])

        # Learning curves
        self.summarize_diagnostics(self)
        # Save model with pretrained weights
        self.model.save(f'{self.model_name}_model.h5')

    def fit_dual_model(self, model, train_x, train_y, val_x, val_y):
        """ Trains the Two-Stream-Model with data augmentation. Training uses early stopping, restoring the best weights"""
        
        assert (train_y[0] == train_y[1]).all() and (val_y[0] == val_y[1]).all()
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_accuracy',
                                       restore_best_weights=True,
                                       patience=self.patience)

        # Trains and validates the model
        self.history = model.fit(x={"TVHI_frames": train_x[1], "TVHI_Optical_Flow": train_x[0]},
                            y=train_y[0],
                            validation_data=[{"TVHI_frames": val_x[1], "TVHI_Optical_Flow": val_x[0]}, val_y[0]],
                            shuffle='TRUE',
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            verbose=1,
                            callbacks=[early_stopping])
        
        # Learning curves
        self.summarize_diagnostics(self)
        # Save model with pretrained weights
        self.model.save(f'{self.model_name}_model.h5')

    def evaluate_mono_model(self, model, test_x, test_y, test_y_int):
        """ Evaluates the One-Stream-Model on the test set"""

        _, acc, precs, rec, _ = model.evaluate(x=test_x, y=test_y, verbose=0)
        print(f' \n Model evaluation: Accuracy = {numpy.round(acc * 100.0, 3)}%, '
              f'Precision = {numpy.round(precs * 100.0, 3)}%, Recall = {numpy.round(rec * 100.0, 3)}% \n')

        # Predict on test set
        predictions = numpy.argmax(self.model.predict(test_x), axis=1)
        # Classification report
        self.classification_report(self, predictions, test_y_int)

    def evaluate_dual_model(self, model, test_x, test_y, test_y_int):
        """ Evaluates the Two-Stream-Model on the test set"""

        assert (test_y_int[0] == test_y_int[1]).all()

        _, acc, precs, rec, _ = model.evaluate(x={"TVHI_frames": test_x[1], "TVHI_Optical_Flow": test_x[0]}, 
                                               y=test_y[0],
                                               verbose=0)
        print(f' \n Model evaluation: Accuracy = {numpy.round(acc * 100.0, 3)}%, '
              f'Precision = {numpy.round(precs * 100.0, 3)}%, Recall = {numpy.round(rec * 100.0, 3)}% \n')

        # Predict on test set
        predictions = numpy.argmax(self.model.predict({"TVHI_frames": test_x[1], "TVHI_Optical_Flow": test_x[0]}), axis=1)
        # Classification report
        self.classification_report(self, predictions, test_y_int[0])

    # plot diagnostic learning curves
    @staticmethod
    def summarize_diagnostics(self):
        """ Plots the learning curve of the Model Diagnostics for training and validation data"""

        pyplot.figure(figsize=(15, 9))
        pyplot.suptitle(f'Performance of model: {self.model_name}', fontsize=20)
        pyplot.subplots_adjust(top=0.85)

        # Loss
        pyplot.subplot(2, 2, 1)
        pyplot.title('Log-normalized Cross Entropy Loss', fontsize=14)
        pyplot.plot(numpy.log(self.history.history['loss']), color='blue', label='train')
        pyplot.plot(numpy.log(self.history.history['val_loss']), color='orange', label='test')
        pyplot.ylabel('Loss', fontsize=10)
        pyplot.xlabel('Epoch', fontsize=10)
        pyplot.legend()
        pyplot.tight_layout()
        # Accuracy
        pyplot.subplot(2, 2, 2)
        pyplot.title('Classification Accuracy', fontsize=14)
        pyplot.plot(self.history.history['accuracy'], color='blue', label='train')
        pyplot.plot(self.history.history['val_accuracy'], color='orange', label='test')
        pyplot.ylabel('Accuracy', fontsize=10)
        pyplot.xlabel('Epoch', fontsize=10)
        pyplot.legend()
        pyplot.tight_layout()      

        # Save plots to a file
        pyplot.savefig(f'{self.model_name}_PerfMetric.png')
        pyplot.show()
        pyplot.close()

    @staticmethod
    def draw_confusion_matrix(self, true, preds):
        """ Constructs the Confusion Matrixs. """

        # Plots the confusion matirx
        conf_matx = confusion_matrix(true, preds)
        ax = pyplot.subplot()
        seaborn.heatmap(conf_matx,
                        annot=True,
                        annot_kws={"size": 12},
                        fmt='g',
                        ax=ax,
                        cmap='Greens',
                        cbar=False, )
        pyplot.suptitle(f'Confusion Matrix of model: {self.model_name}', fontsize=16)

        # Save confusion matrix to a file
        pyplot.savefig(f'{self.model_name}_ConfMat.png')

    @staticmethod
    def classification_report(self, predictions, testLabels):
        """ Constructs the Classification Report that summarize performance on the test set. """

        # Assembels all metrics on the test set
        report = classification_report(predictions,
                                       testLabels,
                                       output_dict=True)

        # Save classification report to a file
        pandas.DataFrame(report).transpose().to_csv(f'{self.model_name}_Report.csv', index=True)

    def run_mono(self, data):
        """ Runs the test harness for building and evaluating a the One-Stream-Model. """
        
        print(f'[INFO] running {self.model_name}...')
        # Collect the data and preprocess the labels and data
        train_x, train_y, val_x, val_y, test_x, test_y, test_y_int = data

        # Compile the model
        print(f'[INFO] compiling {self.model_name}...')
        self.model = self.compile_model(model=self.model,
                                        data_length=train_x.shape[0])
        # Train and validate the model
        print(f'[INFO] training {self.model_name}...')
        self.fit_mono_model(model=self.model,
                                      train_x=train_x,
                                      train_y=train_y,
                                      val_x=val_x,
                                      val_y=val_y)
        
        # Evaluate the model on the test set
        print(f'[INFO] testing {self.model_name} on test set...\n')
        self.evaluate_mono_model(model=self.model, test_x=test_x, test_y=test_y, test_y_int=test_y_int)

    def run_dual(self, data1, data2):
        """ Runs the test harness for building and evaluating a Two-Stream-Model. """
        
        print(f'[INFO] running {self.model_name}...')
        # Collect the data and preprocess the labels and data
        train_x = (numpy.asarray(data1[0]), numpy.asarray(data2[0])); train_y = (numpy.asarray(data1[1]), numpy.asarray(data2[1]))
        val_x = (numpy.asarray(data1[2]), numpy.asarray(data2[2])); val_y = (numpy.asarray(data1[3]), numpy.asarray(data2[3]))
        test_x = (numpy.asarray(data1[4]), numpy.asarray(data2[4])); test_y = (numpy.asarray(data1[5]), numpy.asarray(data2[5]));
        test_y_int = (numpy.asarray(data1[6]), numpy.asarray(data2[6]))

        # Compile the model
        print(f'[INFO] compiling {self.model_name}...')
        self.model = self.compile_model(model=self.model,
                                        data_length=train_x[0][0].shape[0])
        # Train and validate the model
        print(f'[INFO] COMPILED MODEL \n [INFO] training {self.model_name}...')
        self.fit_dual_model(model=self.model,
                                      train_x=train_x,
                                      train_y=train_y,
                                      val_x=val_x,
                                      val_y=val_y)
        print(f'[INFO]FITTED MODEL')

        # Evaluate the model on the test set
        print(f'[INFO] testing {self.model_name} on test set...\n')
        self.evaluate_dual_model(model=self.model, test_x=test_x, test_y=test_y, test_y_int=test_y_int)

# entry poin to run all 4 models!
def main():
    # Transform data to tensor and save it as stanford40.pickle.
    # Alternatively skip preprocessing if the file exist in the directory.
    if not(isfile(getcwd() + '/stanford40.pickle') and isfile(getcwd() + '/TVHI.pickle')):
        stanford40, TVHI, stanford40_categories = construct_data()
        stanford40 = preprocess_stanford40(data=stanford40)

        with open('stanford40.pickle', 'wb') as f:
            pickle.dump([stanford40], f)
        with open('TVHI.pickle', 'wb') as f:
            pickle.dump([TVHI], f)
    else:
        with open('stanford40.pickle', 'rb') as f:
            stanford40 = pickle.load(f)[0]
        with open('TVHI.pickle', 'rb') as f:
            TVHI = pickle.load(f)[0]
   
    # ########################### Build stanford model ###########################
    if not isfile(f'{getcwd()}/STF40_frames_model.h5'):
        
        # Validation split
        train_x, val_x, train_y, val_y = train_test_split(stanford40[0], stanford40[1],
                                                          test_size=0.1, random_state=42)
        classes = stanford40[6]
        stanford40 = [train_x, train_y, val_x, val_y, 
                      stanford40[3], stanford40[4], stanford40[5]
                      ]

        STF40_frames = DCNN_clfr(model_name='STF40_frames',
                              epochs=75,
                              batch_size=32,
                              early_stopping=10, 
                              num_classes=40) 
        STF40_frames.model = STF40_frames.define_model(input_shape=train_x[0].shape)
        STF40_frames.run_mono(data=stanford40)
        backend.clear_session()
        
    # # ######################### Build TV-HI frame model ##########################
    if not isfile(f'{getcwd()}/TVHI_frames_model.h5'):

        # Preprocess TV-HI
        tv_hi_videos = [TVHI[0], TVHI[2]]
        train_tvhi, test_tvhi = get_middle_frame(tv_hi_videos)
        tvhi_data = train_tvhi, TVHI[1], test_tvhi, TVHI[3]
        final_tvhi_data = preprocess_TVHI(tvhi_data)

        # Validation split
        train_x, val_x, train_y, val_y = train_test_split(final_tvhi_data[0], final_tvhi_data[1],
                                                      test_size=0.10, random_state=42)
        tvhi_classes = final_tvhi_data[6]
        F_TVHI_data = [train_x, train_y, val_x, val_y, 
                      final_tvhi_data[3], final_tvhi_data[4], final_tvhi_data[5]
                     ]
        with open('F_TVHI_data.pickle', 'wb') as f:
            pickle.dump([F_TVHI_data], f)

        TVHI_frames = DCNN_clfr(model_name='TVHI_frames',
                              epochs=75, 
                              batch_size=32,
                              early_stopping=25, 
                              num_classes=4) 
        
        # Define the transfer-learning model
        TVHI_frames.model = TVHI_frames.define_model(input_shape=train_x[0].shape)
        ST40_model = load_model(f'{getcwd()}/STF40_frames_model.h5', compile=False)

        for layer_index, layer in enumerate(ST40_model.layers[:-2]):
          ST40_layer_weights = layer.get_weights()
          TVHI_frames.model.layers[layer_index].set_weights(ST40_layer_weights)

        TVHI_frames.model.trainable = False
        TVHI_frames.learning_rate = (1e-4, 1e-2)

        # Rebuild top
        output = TVHI_frames.model.layers[-2].output 
        output = Dense(units=tvhi_classes, 
                       activation='softmax',
                       kernel_initializer=TVHI_frames.conv_init,
                       name='probs')(output) 

        TVHI_frames.model = Model(inputs=TVHI_frames.model.inputs, outputs=output, name=TVHI_frames.model_name)
        TVHI_frames.run_mono(data=F_TVHI_data) 

        # Fine-tune the transfer-learning model
        TVHI_frames.learning_rate = (1e-6, 1e-4)
        # Unfreeze the convolution block 6 and the top layer while leaving BatchNorm layers frozen
        for layer in TVHI_frames.model.layers[-155:]:
          if not isinstance(layer, BatchNormalization):
              layer.trainable = True
        TVHI_frames.patience = 5 
        TVHI_frames.run_mono(data=F_TVHI_data) 
        backend.clear_session()

    # ######################### Build TV-HI Optical Flow model ##########################
    if not isfile(f'{getcwd()}/TVHI_Optical_Flow_model.h5'):
        
        #List of videos
        tv_hi_videos = [TVHI[0], TVHI[2]]

        # Compute optical frames
        of_train_tvhi, of_test_tvhi = get_optical_flow(tv_hi_videos)
        of_tvhi_data = of_train_tvhi, TVHI[1], of_test_tvhi, TVHI[3]
        of_final_of_tvhi_data = preprocess_TVHI(of_tvhi_data)

        # Validation split
        train_x, val_x, train_y, val_y = train_test_split(of_final_of_tvhi_data[0], of_final_of_tvhi_data[1],
                                                      test_size=0.10, random_state=42)
        OF_tvhi_classes = of_final_of_tvhi_data[6]
       
        OF_TVHI_data = [train_x, train_y, val_x, val_y, 
                      of_final_of_tvhi_data[3], of_final_of_tvhi_data[4], of_final_of_tvhi_data[5]
                       ]
        with open('OF_TVHI_data.pickle', 'wb') as f:
            pickle.dump([OF_TVHI_data], f)

        TVHI_Optical_Flow = DCNN_clfr(model_name='TVHI_Optical_Flow',
                                      epochs=75, 
                                      batch_size=8,
                                      early_stopping=25,
                                      num_classes=OF_tvhi_classes)
        TVHI_Optical_Flow.model = TVHI_Optical_Flow.define_model(input_shape=train_x[0].shape)
        TVHI_Optical_Flow.run_mono(data=OF_TVHI_data)
        backend.clear_session()
   
    # ######################### Build TwoStream Frame-Flow model ##########################
        # Load the optical flow and frame data
        with open('F_TVHI_data.pickle', 'rb') as f:
            F_TVHI_data = pickle.load(f)[0]
        with open('OF_TVHI_data.pickle', 'rb') as f:
            OF_TVHI_data = pickle.load(f)[0]

        TwoStream = DCNN_clfr(model_name='TwoStream',
                              epochs=75, 
                              batch_size=32,
                              early_stopping=10,
                              num_classes=OF_tvhi_classes)
        activation = get_swish()

        # Build the frame-stream
        TVHI_frames = load_model(f'{getcwd()}/TVHI_frames_model.h5', compile=False)

        # Build the optical-flow-stream
        TVHI_opt_flow = load_model(f'{getcwd()}/TVHI_Optical_Flow_model.h5', compile=False)

        # Conv2D Merge and output block of the two stream model
        x = Concatenate(name='concat')([TVHI_frames.layers[-4].output, TVHI_opt_flow.layers[-4].output])
        x = Conv2D(filters=1792, kernel_size=(1, 1),
                  kernel_regularizer=TwoStream.conv_reg,
                  kernel_initializer=TwoStream.conv_init,
                  padding='same',
                  name='connect_conv')(x)
        x = BatchNormalization( name='connect_bn')(x)
        
        x = Activation(activation, name='connect_activation')(x)
        x = GlobalAveragePooling2D(name=f'Two_stream_avg_pool')(x)
        x = Dropout(0.4, name=f'Two_stream_top_dropout')(x)


        x = Flatten()(x)
        print(f'x.shape   :     {x.shape}')          
        x = Dense(numpy.round(x.shape[1]/4,0),                  
                  kernel_initializer=TwoStream.conv_init,
                  #kernel_regularizer=TwoStream.kernel_regularization,
                  name='fully_connected') (x)
        x = BatchNormalization( name='fully_connected_bn')(x)
        x = Activation(activation, name='fully_connected_activation')(x)


        output = Dense(units=tvhi_classes, 
                       activation='softmax',
                       kernel_initializer=TwoStream.dense_init,
                       name='final_probs')(x)
        TwoStream.model = Model(inputs=[TVHI_frames.input, TVHI_opt_flow.input], outputs=output, name=TwoStream.model_name)

        
        TwoStream.run_dual(data1=OF_TVHI_data, data2=F_TVHI_data)
        backend.clear_session()

if __name__ == '__main__':
    main()

# from tensorflow.image import resize_with_crop_or_pad


# ## =================================================== Creating stacks of Optical flow frames ==============================================================================================

# ## FOR Dense Optical Flow in OpenCV 
# def get_optical_flow_stack(data, maxsize=(170, 170), verbose=0):
#   """ Computes the Dense Optical flow from video frames"""

#   # Get the videos from training and test set
#   for set_index, video_set in enumerate(data):
  
#     # Lists to store the generated stacks and batches of optical flow frames alongside their corresponding actual frame
#     stacks_optical_flow_images = []

#     #for each video in either the training or the test set
#     for index, filename in enumerate(video_set):
#       if filename.endswith("avi"): 
#         video_file = f'/content/TV-HI/tv_human_interactions_videos/{filename}'
#         cap = cv2.VideoCapture(video_file)

#         ret, frame1 = cap.read()
#         frame1 = resize_with_crop_or_pad(frame1, 170,170)
#         frame1 = frame1.numpy()
#         prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#         hsv = numpy.zeros_like(frame1)
#         hsv[..., 1] = 255
#         of_images = []

#         while(1):
#           ret, frame2 = cap.read()

#           if not ret:
#               #print('No frames grabbed!')
#               break
          
#           frame2 = resize_with_crop_or_pad(frame2, 170,170)
#           frame2 = frame2.numpy()
#           next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#           flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#           mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#           hsv[..., 0] = ang*180/numpy.pi/2
#           hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#           bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#           #add hsv to list of images
#           of_images.append(bgr)
#           if(len(of_images) >= 16):
#             final_images = numpy.array(of_images)
#             break
          
#           prvs = next

#       print(f'Next video, video numer {index}')

#     # #======================== STACKS AND BATCHES =====================================
#       stacks_optical_flow_images.append(final_images)
#       cv2.destroyAllWindows()

#   # ============================= Pickle BATCHES AND STACKS ===================================
#     if set_index == 0:   train_all_optical_flow_frames = numpy.array(stacks_optical_flow_images)
#     elif set_index == 1:  test_all_optical_flow_frames = numpy.array(stacks_optical_flow_images)


#   train_all_optical_flow_frames = open('train_all_optical_flow_frames.pickle', 'wb')
#   pickle.dump(final_stacks_of_images, train_all_optical_flow_frames)

#   test_optical_flow_file = open('test_optical_flow_file.pickle', 'wb')
#   pickle.dump(final_stacks_of_images, test_optical_flow_file)

#   print(f'final_stacks_of_images shape:   {final_stacks_of_images.shape}')