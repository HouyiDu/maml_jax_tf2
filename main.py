import sys
import os

import tensorflow as tf
import numpy as np

from sinusoid_data_generator import SinusoidDataGenerator



def main():
    #create a sinusoid data generator with tf.data with a python geenrator
    num_samples_per_class = 2
    num_classes_per_task = 3
    num_tasks = 2

    total_samples = num_samples_per_class * num_classes_per_task

    sinusoid_data_gen = SinusoidDataGenerator(num_classes_per_task=num_classes_per_task, num_samples_per_class=num_samples_per_class)
    #sinusoid_data_gen = SinusoidDataGenerator()

    ds_sinusoid = tf.data.Dataset.from_generator(
                        sinusoid_data_gen.generate_sinusoid_batch,
                        args = [num_samples_per_class],
                        output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
                        output_shapes=((1), (1), (1), (1))

    )

    print(ds_sinusoid.element_spec)

    for task_id, (init_inputs, outputs, amplitude, phase) in enumerate(ds_sinusoid.batch(num_samples_per_class * num_classes_per_task).take(num_tasks)):
        #num_samples_per_class * num_classes_per_task for N-way and K-shot learning
        print("step: ", task_id)
        print('outputs: ', outputs)
        print('phase:', phase)
    

if __name__ == '__main__':
    main()