import numpy as np

# class SinusoidDataGenerator:

#     def __init__(self, config ={}):
        
#         self.amplitude_range = config.get('amp_range', [0.1, 5.0])
#         self.phase_range = config.get('phase_range', [0, np.pi])
#         self.input_range = config.get('input_range', [-5.0, 5.0])
    
#     def generate_sinusoid_batch(self):
#         #return the data for one class
#         #how many classes per task? Ans: self.num_classes_per_task
#         #how many samples per class? Ans: self.num_samples_per_class
#         while True:
#             amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1])
#             phase = np.random.uniform(self.phase_range[0], self.phase_range[1])

#             init_inputs = np.random.uniform(self.input_range[0], self.input_range[1])
#             outputs = amplitude * np.sin(init_inputs - phase)
                
#             yield init_inputs, outputs, amplitude, phase


class SinusoidDataGenerator(object):

    def __init__(self, num_classes_per_task, num_samples_per_class, config ={}):
        self.num_samples_per_class = num_samples_per_class
        self.num_classes_per_task = num_classes_per_task
        
        self.amplitude_range = config.get('amp_range', [0.1, 5.0])
        self.phase_range = config.get('phase_range', [0, np.pi])
        self.input_range = config.get('input_range', [-5.0, 5.0])
    
    def generate_sinusoid_batch(self, batch_size):
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
            
            yield init_inputs, outputs, amplitude, phase
            i += 1


# class SinusoidDataGenerator(object):
#     def __init__(self, num_classes_per_task, num_samples_per_class, config ={}):
#         self.num_samples_per_class = num_samples_per_class
#         self.num_classes_per_task = num_classes_per_task
        
#         self.amplitude_range = config.get('amp_range', [0.1, 5.0])
#         self.phase_range = config.get('phase_range', [0, np.pi])
#         self.input_range = config.get('input_range', [-5.0, 5.0])


#         self.phase = np.random.uniform(self.phase_range[0], self.phase_range[1])
#         self.amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1])
    
#     def generate_sinusoid_batch(self):
#         while True:
#             outputs = np.zeros([self.num_classes_per_task, self.num_samples_per_class ])
#             inputs = np.zeros([self.num_classes_per_task, self.num_samples_per_class])

#             for sinusoid_func_idx in range(self.num_classes_per_task):
#                 inputs[sinusoid_func_idx] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class])
#                 outputs[sinusoid_func_idx] = self.amplitude[sinusoid_func_idx] * np.sin(inputs[sinusoid_func_idx] - self.phase[sinusoid_func_idx])
            

#             yield inputs.flatten(), outputs.flatten(), self.amplitude, self.phase