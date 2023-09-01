import tensorflow as tf


class DenseNet(tf.keras.Model):
    def __init__(self, hidden_dim = [40, 40], out_dim = 1):
        super().__init__()

        self.initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01)
        self.dense1 = tf.keras.layers.Dense(hidden_dim[0],
                                  activation='relu', 
                                  kernel_initializer=self.initializer,
                                  bias_initializer='zeros')
        self.dense2 = tf.keras.layers.Dense(hidden_dim[1], 
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