from models.utils import DenseNet
from sinusoid_data_generator import SinusoidDataGenerator

import tensorflow as tf
import time

import matplotlib.pyplot as plt


#make the baseline

num_samples_per_class = 10
num_classes_per_task = 5
num_tasks_train = 400   #total number of tasks for training will be num_tasks_train * epoches
num_tasks_valid = 10

sinusoid_train_data = SinusoidDataGenerator(num_classes_per_task=num_classes_per_task, num_samples_per_class=num_samples_per_class)
sinusoid_valid_data = SinusoidDataGenerator(num_classes_per_task=num_classes_per_task, num_samples_per_class=num_samples_per_class)
sinusoid_test_data = SinusoidDataGenerator(num_classes_per_task=num_classes_per_task, num_samples_per_class=num_samples_per_class)
# sinusoid_train_data = SinusoidDataGenerator()
# sinusoid_test_data = SinusoidDataGenerator()


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

    for task, (inputs, outputs, amplitude, phase) in enumerate(ds_sinusoid_train.batch(num_samples_per_class * num_classes_per_task).take(num_tasks_train)):
        #batch(num1) num1= N*K, one dataset of a task for N-way and K-shot learning
        #task(num2) num2 is number of tasks
        with tf.GradientTape() as tape:
            logits = model(inputs, training = True)
            loss_value = loss_fn(outputs, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_mse_metric.update_state(outputs, logits)


        if task % 100 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (task, float(loss_value))
            )
            print("Seen so far: %s samples" % ((task + 1) * num_samples_per_class))
        

    train_mse = train_mse_metric.result()
    print("Training mse over epoch: %.4f" % (float(train_mse),))

    train_mse_metric.reset_states()

    for task, (inputs, outputs, amplitude, phase) in enumerate(ds_sinusoid_valid.batch(num_samples_per_class * num_classes_per_task).take(num_tasks_valid)):
        val_logits = model(inputs, training = False)
        val_mse_metric.update_state(outputs, logits)
    val_mse = val_mse_metric.result()
    val_mse_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_mse),))
    print("Time taken: %.2fs" % (time.time() - start_time))

