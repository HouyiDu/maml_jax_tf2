import numpy as np

class SinusoidDataGenerator(object):
    def __init__(self, num_samples_per_class, batch_size, config ={}):
        self.num_samples_per_class = num_samples_per_class
        self.batch_size = batch_size
        

        self.generator = self.generate_sinusoid_batch
        self.amplitude_range = config.get('amp_range', [0.1, 5.0])
        self.phase_range = config.get('phase_range', [0, np.pi])
        self.input_range = config.get('input_range', [-5.0, 5.0])
        self.input_dimension = 1
        self.output_dimension = 1
    

    def generate_sinusoid_batch(self, input_idx = None):
        amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.output_dimension])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.input_dimension])

        for sinusoid_func_idx in range(self.batch_size):
            init_inputs[sinusoid_func_idx] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, self.input_dimension])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx)

            outputs[sinusoid_func_idx] = amplitude[sinusoid_func_idx] * np.sin(init_inputs[sinusoid_func_idx] - phase[sinusoid_func_idx])
        
        return init_inputs, outputs, amplitude, phase