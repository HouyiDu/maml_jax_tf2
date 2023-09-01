import sys
import os

import tensorflow as tf
import numpy as np

from sinusoid_data_generator import SinusoidDataGenerator


def main():
    #create a sinusoid data generator with tf.data with a python geenrator
    num_samples_per_class = 10
    num_classes_per_task = 5

    total_samples = num_samples_per_class * num_classes_per_task

    sinusoid_data_gen = SinusoidDataGenerator(num_classes_per_task=num_classes_per_task, num_samples_per_class=num_samples_per_class)
    #sinusoid_data_gen = SinusoidDataGenerator()

    ds_sinusoid = tf.data.Dataset.from_generator(
                        sinusoid_data_gen.generate_sinusoid_batch,
                        args = [num_samples_per_class],
                        output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
                        output_shapes=((), (), (), ())

    )

    print(ds_sinusoid.element_spec)

    for step, (init_inputs, outputs, amplitude, phase) in enumerate(ds_sinusoid.batch(num_samples_per_class).take(2)):
        print("step: ", step)
        # print('init_inputs.shape: ', init_inputs.shape)
        # print('outputs.shape: ', outputs.shape)
        # print('amplitude.shape: ', amplitude.shape)
        # print('phase.shape: ', phase.shape)
        print('phase:', phase)
    

if __name__ == '__main__':
    main()