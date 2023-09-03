
# code adapt from https://github.com/Runist/MAML-keras

import random
import os
import glob

import numpy as np
import tensorflow as tf


class MAMLDataLoader:

    def __init__(self, data_path, batch_size, n_way=5, k_shot=1, q_query=1):
        """
        MAML dataloader for omiglot dataset or other similar structured grayscale impage dataset

        :param data_path: path to the folder of the whole dataset.
        :param batch_size: number of tasks
        :param n_way: number of classes in one task
        :param k_shot: number of smaples per class
        :param q_query: number of samples in query set in one task
        """
        file_name_regex = os.path.join(data_path, "**/character*")
        self.file_list = [f for f in glob.glob(file_name_regex, recursive=True)]
        self.steps = len(self.file_list) // batch_size

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.meta_batch_size = batch_size

    def __len__(self):
        return self.steps

    def get_one_task_data(self):
        """
        Gather a shuffled dataset for one task
        There are n_way classes in one task.
        There are k_shot in one class.

        support_data are used for inner loop weights update.
        query_data are used for outter loop model weights update

        """
        img_dirs = random.sample(self.file_list, self.n_way)


        support_image = []
        support_label = []
        query_image = []
        query_label = []

        for label, img_dir in enumerate(img_dirs):
            img_list = [f for f in glob.glob(img_dir + "**/*.png", recursive=True)] # When recursive is set True “**” followed by path separator('./**/') will match any files or directories.

            images = random.sample(img_list, self.k_shot + self.q_query)

            # Read support set
            for img_path in images[:self.k_shot]:
                image = tf.keras.utils.img_to_array(tf.keras.utils.load_img(img_path, color_mode = "grayscale"))
                image /= 255. #normalize value to [0,1]

                support_image.append(image)
                support_label.append(label)

            # Read query set
            for img_path in images[self.k_shot:]:
                image = tf.keras.utils.img_to_array(tf.keras.utils.load_img(img_path, color_mode = "grayscale"))
                image /= 255. #normalize value to [0,1]

                query_image.append(image)
                query_label.append(label)

        # shuffle support and query set
        # support_data = list(zip(support_image, support_label))
        # random.shuffle(support_data)
        # support_image, support_label = zip(*support_data)

        # query_data = list(zip(query_image, query_label))
        # random.shuffle(query_data)
        # query_image, query_label = zip(*query_data)

        return np.array(support_image), np.array(support_label), np.array(query_image), np.array(query_label)

    def get_one_batch(self):
        """
        yield a batch a of data (yield tasks of data)
        """

        while True:
            batch_support_image = []
            batch_support_label = []
            batch_query_image = []
            batch_query_label = []

            for _ in range(self.meta_batch_size):
                support_image, support_label, query_image, query_label = self.get_one_task_data()
                batch_support_image.append(support_image)
                batch_support_label.append(support_label)
                batch_query_image.append(query_image)
                batch_query_label.append(query_label)

            yield np.array(batch_support_image), np.array(batch_support_label), \
                  np.array(batch_query_image), np.array(batch_query_label)
