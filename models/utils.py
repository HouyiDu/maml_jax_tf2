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

    #ABANDON THIS! cant implement with custom train_step. Start to implement custom training step in the future

    # def train_step(self, data):
    #     x_a, x_b, y_a, y_b = data

    #     task_output_b, task_losses_b = [], []

    #     with tf.GradientTape() as tape:
    #         task_output_a =  self(x_a) #forward pass
    #         task_loss_a = self.compute_loss(y = y_a, y_pred = task_output_a) #loss

    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(task_loss_a, trainable_vars)

    #     # fast_trainbale_vars = [v for v in trainable_vars] #if trainable_weights is a list
    #     # fast_weights = self.optimizer.apply_gradients(zip(gradients, fast_trainbale_vars))

    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #     output = self(x_b)
    #     task_output_b.append(output)
    #     task_losses_b.append(self.compute_loss(y = y_b, y_pred = output))

    #     self.save_weights('./checkpoints/my_checkpoint')

    #     for i in range(5 - 1): #change this 5 to a variable passsed to this class as a self.var in the future
    #         loss = self.compute_loss(y = y_a, y_pred = self(x_a))
    #         gradients = tape.gradient(loss, trainable_vars)
    #         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #         output = self(x_b)
    #         task_output_b.append(output)
    #         task_losses_b.append(self.compute_loss(y = y_b, y_pred = output))

    #     self.load_weights('./checkpoints/my_checkpoint')
    #     task_output = [task_output_a, task_output_b, task_loss_a, task_losses_b]
    #     pass