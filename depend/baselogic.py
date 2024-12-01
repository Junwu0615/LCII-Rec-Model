# -*- coding: utf-8 -*-
"""
@author: PC
Update Time: 2024-12-01
"""
import os, random
import numpy as np
from datetime import datetime
import tensorflow.compat.v1 as tf

class BaseLogic:
    def __init__(self):
        pass

    @staticmethod
    def check_folder(path: str):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)

    @staticmethod
    def time_now():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def settings_gpu_threshold():
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except IOError as e:
                print(e)

    @staticmethod
    def check_gpu_running():
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TensorFlow")

    @staticmethod
    def settings_seed(seed: int = 888):
        tf.set_random_seed(seed)
        random.seed(seed)
        np.random.seed(seed)