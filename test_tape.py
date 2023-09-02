import tensorflow as tf

layer = tf.keras.layers.Dense(2, activation = 'relu', kernel_initializer=tf.keras.initializers.Constant(2.0))
support_data = tf.constant([[1. , 2. , 3.]])
query_data = tf.constant([[1. , 2. , 3.]])
layer(support_data)  #run forward to initialize weights


inner_steps = 1
num_tasks = 3
task_weights = []

optimizer = tf.keras.optimizers.Adam(0.01)
outer_optimizer = tf.keras.optimizers.Adam(0.01)



meta_weights = layer.get_weights()
for i in range(num_tasks):
    print("train task ", i)

    layer.set_weights(meta_weights) # in the start of every task, use the meta model weights as the beginning

    for _ in range(inner_steps): #inner steps usually be 1
        with tf.GradientTape() as tape:
            logits = layer(support_data, training = True)  #this x should be support set
            loss = tf.reduce_mean(logits**2)
        
        print('task ', i, ' with loss:', loss)
        grads = tape.gradient(loss, layer.trainable_variables)
        optimizer.apply_gradients(zip(grads, layer.trainable_variables))

    task_weights.append(layer.get_weights())

print('finish meta train')
print('')
for i in range(len(task_weights)):
    print('task ', i, 'with weights:', task_weights[i])

tasks_loss = []

with tf.GradientTape() as tape:
    for i in range(num_tasks):
        layer.set_weights(task_weights[i])
        logits = layer(query_data, training = True)
        loss = tf.reduce_mean(logits**2)
        
        tasks_loss.append(loss)
    
    mean_loss = tf.reduce_mean(tasks_loss)

layer.set_weights(meta_weights)
grads = tape.gradient(mean_loss, layer.trainable_weights)

print('grads:', grads)

outer_optimizer.apply_gradients(zip(grads, layer.trainable_variables))


for var, g in zip(layer.trainable_variables, grads):
  print(f'{var.name}, shape: {g.shape}')