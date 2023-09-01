import numpy as np

class SinusoidDataGenerator(object):

    def __init__(self, num_classes_per_task, num_samples_per_class, config ={}):
        self.num_samples_per_class = num_samples_per_class
        self.num_classes_per_task = num_classes_per_task
        
        self.amplitude_range = config.get('amp_range', [0.1, 5.0])
        self.phase_range = config.get('phase_range', [0, np.pi])
        self.input_range = config.get('input_range', [-5.0, 5.0])
    
    def generate_sinusoid_batch(self, batch_size):
        #batch_size shall be the same as num_samples_per_class
        amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1])

        i = 0
        while True:
            if i >= batch_size:
            #if i >= self.num_samples_per_class:
                amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1])
                phase = np.random.uniform(self.phase_range[0], self.phase_range[1])
                i = 0

            init_inputs = np.random.uniform(self.input_range[0], self.input_range[1])
            outputs = amplitude * np.sin(init_inputs - phase)
            
            yield [init_inputs], [outputs], [amplitude], [phase]
            i += 1

    