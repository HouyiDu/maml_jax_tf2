import sys
import os

from sinusoid_data_generator import SinusoidDataGenerator

def main():
    test_num_updates = 5

    update_batch_size = 10 #number of examples used for inner gradient update (K for K-shot learning).
    meta_batch_size = 25 #number of tasks sampled per meta-update
    data_generator = SinusoidDataGenerator(update_batch_size*2, meta_batch_size)

    output_dimension = data_generator.output_dimension

    baseline = None

    if baseline == 'oracle':
        pass
    input_dimension = data_generator.input_dimension



    tf_data_load = False
    input_tensors = None
    train =True

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)

    