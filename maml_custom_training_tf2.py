from models.utils import DenseNet
from sinusoid_data_generator import SinusoidDataGenerator

import tensorflow as tf
import time

import matplotlib.pyplot as plt


#make the baseline

num_samples_per_class = 10
num_classes_per_task = 5
num_tasks_train = 10   #total number of tasks for training will be num_tasks_train * epoches
num_tasks_valid = 10

sinusoid_train_data = SinusoidDataGenerator(num_classes_per_task=num_classes_per_task, num_samples_per_class=num_samples_per_class)
sinusoid_valid_data = SinusoidDataGenerator(num_classes_per_task=num_classes_per_task, num_samples_per_class=num_samples_per_class)
sinusoid_test_data = SinusoidDataGenerator(num_classes_per_task=num_classes_per_task, num_samples_per_class=num_samples_per_class)


ds_sinusoid_train = tf.data.Dataset.from_generator(
                        sinusoid_train_data.generate_sinusoid_batch,
                        args = [num_samples_per_class],
                        output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
                        output_shapes=((1), (1), (1), (1))
                    )

ds_sinusoid_valid = tf.data.Dataset.from_generator(
                        sinusoid_valid_data.generate_sinusoid_batch,
                        args = [num_samples_per_class],
                        output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
                        output_shapes=((1), (1), (1), (1))
                    )

ds_sinusoid_test = tf.data.Dataset.from_generator(
                        sinusoid_test_data.generate_sinusoid_batch,
                        args = [num_samples_per_class],
                        output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
                        output_shapes=((1), (1), (1), (1))
                    )





model = DenseNet(out_dim = 1)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
meta_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

loss_fn = tf.keras.losses.MeanSquaredError()

train_mse_metric = tf.keras.metrics.MeanSquaredError()
val_mse_metric = tf.keras.metrics.MeanSquaredError()


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_mse_metric.update_state(y, logits)
    return loss_value




epochs = 10
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()


    # ensemble_models = [DenseNet(out_dim=1) for _ in range(num_tasks_train)]
    # ensemble_model_weights = [m.trainable_weights for m in ensemble_models]
    # losses = []

    with tf.GradientTape() as meta_train_tape:
        total_loss_over_tasks = []

        for task, (inputs, outputs, amplitude, phase) in enumerate(ds_sinusoid_train.batch(num_samples_per_class * num_classes_per_task).take(num_tasks_train)):
            #batch(num1) num1= N*K, one dataset of a task for N-way and K-shot learning
            #task(num2) num2 is number of tasks

            model_for_this_task = DenseNet(out_dim = 1)
            for i, l in enumerate(model_for_this_task.layers):
                l.set_weights(model.layers[i].get_weights())

            with tf.GradientTape() as train_tape:
                logits = model_for_this_task(inputs, training = True)
                loss_value = loss_fn(outputs, logits)
            
            grads = train_tape.gradient(loss_value, model_for_this_task.trainable_weights)
            optimizer.apply_gradients(zip(grads, model_for_this_task.trainable_weights))
            total_loss_over_tasks.append(loss_value)
        
        total_loss_summed = sum(total_loss_over_tasks)
    
    
    grads = meta_train_tape.gradient(total_loss_summed, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

        

            


