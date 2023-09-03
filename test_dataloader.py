from MAMLDataLoader import MAMLDataLoader
import glob
import tensorflow as tf
import numpy as np


import os
import PIL
import PIL.Image
import tensorflow_datasets as tfds

import pathlib


def main():
    dataloader = MAMLDataLoader(data_path = './omniglot/python/images_background/', batch_size = 2)
    support_image, support_label, query_image, query_label = dataloader.get_one_task_data()



    pass
    








if __name__ == '__main__':
    main()