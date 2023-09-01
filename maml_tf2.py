from models.utils import DenseNet
from sinusoid_data_generator import SinusoidDataGenerator

import tensorflow as tf
import time


#make the baseline

num_samples_per_class = 10
num_classes_per_task = 5

total_samples = num_samples_per_class * num_classes_per_task

sinusoid_train_data = SinusoidDataGenerator(num_classes_per_task=num_classes_per_task, num_samples_per_class=num_samples_per_class)
sinusoid_test_data = SinusoidDataGenerator(num_classes_per_task=num_classes_per_task, num_samples_per_class=num_samples_per_class)
# sinusoid_train_data = SinusoidDataGenerator()
# sinusoid_test_data = SinusoidDataGenerator()


ds_sinusoid_train = tf.data.Dataset.from_generator(
                        sinusoid_train_data.generate_sinusoid_batch,
                        output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
                        output_shapes=((total_samples, ), (total_samples, ), (num_classes_per_task, ), (num_classes_per_task, ))
                    )

ds_sinusoid_test = tf.data.Dataset.from_generator(
                        sinusoid_test_data.generate_sinusoid_batch,
                        output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
                        output_shapes=((total_samples, ), (total_samples, ), (num_classes_per_task, ), (num_classes_per_task, ))
                    )




model = DenseNet(out_dim = 1)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()



epochs = 10
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    #start_time = time.time()

    for step, (init_inputs, outputs, amplitude, phase) in enumerate(ds_sinusoid_train.batch(5).take(1)):
        with tf.GradientTape() as tape:
            logits = model(tf.transpose(init_inputs), training = True)
            loss_value = loss_fn(outputs, logits)
        

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if step % 2 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            #print("Seen so far: %s samples" % ((step + 1) * batch_size))

# construct a dataset
# num_samples_per_class = 10
# num_classes_per_task = 1


# sinusoid_train_data = SinusoidDataGenerator(num_samples_per_class=num_samples_per_class, num_classes_per_task=num_classes_per_task)
# sinusoid_test_data = SinusoidDataGenerator(num_samples_per_class=num_samples_per_class, num_classes_per_task=num_classes_per_task)


# ds_sinusoid_train = tf.data.Dataset.from_generator(
#                         sinusoid_train_data.generate_sinusoid_batch,
#                         output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
#                         output_shapes=((10,), (10,), (), ())
#                     )

# ds_sinusoid_test = tf.data.Dataset.from_generator(
#                         sinusoid_test_data.generate_sinusoid_batch,
#                         output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
#                         output_shapes=((10,), (10,), (), ())
#                     )




# model = DenseNet(out_dim = 10)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# loss_fn = tf.keras.losses.MeanSquaredError()



# epochs = 3
# for epoch in range(epochs):
#     print("\nStart of epoch %d" % (epoch,))
#     #start_time = time.time()

#     for step, (init_inputs, outputs, amplitude, phase) in enumerate(ds_sinusoid_train.batch(10).take(10)):
#         with tf.GradientTape() as tape:
#             print('init_inputs.shape:', init_inputs.shape)
#             logits = model(init_inputs, training = True)
#             print('logits.shape:', logits.shape)
#             loss_value = loss_fn(outputs, logits)
        

#         grads = tape.gradient(loss_value, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))

#         print(
#             "Training loss (for one batch) at step %d: %.4f"
#             % (step, float(loss_value))
#         )






    