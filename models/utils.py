import tensorflow as tf

# class Model(object):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.name = self.__class__.__name__
#         self.scope = None
    
#     def inputs(self, *args, **kwargs):
#         pass

#     def _build(self, input, output):
#         model = tf.keras.Model(input, output, name = self.name)
#         return model
#     def __call__(self, inputs = None):
#         pass


# class SinusoidModel(Model):
#     def __init__(self, dim_hidden = 40, dim_output = 1,  *args, **kwargs):
#         super(SinusoidModel, self).__init__(*args, **kwargs)
#         self.dim_hidden = dim_hidden
#         self.dim_output = dim_output
#         self.scope = 'sinusoid'
    
#     def inputs(self, input_dimension):
#         return [tf.keras.Input(shape=(input_dimension, ), name=self.scope)]

#     def __call__(self, inputs):
#         x = inputs[0]

#         initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01)
#         x = tf.keras.layers.Dense(self.dim_hidden, 
#                                   activation='relu', 
#                                   kernel_initializer=initializer,
#                                   bias_initializer='zeros')(x)

#         x = tf.keras.layers.Dense(self.dim_hidden, 
#                                   activation='relu', 
#                                   kernel_initializer=initializer,
#                                   bias_initializer='zeros')(x)
        
#         x = tf.keras.layers.Dense(self.dim_output, 
#                                   activation=None, 
#                                   kernel_initializer=initializer,
#                                   bias_initializer='zeros')(x)
        
#         return self._build(inputs , x)



class DenseNet(tf.keras.Model):
    def __init__(self, hidden_dim = [40, 40], out_dim = 1):
        super().__init__()

        self.initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01)
        self.dense1 = tf.keras.layers.Dense(hidden_dim, 
                                  activation='relu', 
                                  kernel_initializer=self.initializer,
                                  bias_initializer='zeros')
        self.dense2 = tf.keras.layers.Dense(hidden_dim, 
                                  activation='relu', 
                                  kernel_initializer=self.initializer,
                                  bias_initializer='zeros')
        
        self.classifier = tf.keras.layers.Dense(out_dim, 
                                  activation=None, 
                                  kernel_initializer=self.initializer,
                                  bias_initializer='zeros')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.classifier(x)

    def train_step(self, data):
        pass