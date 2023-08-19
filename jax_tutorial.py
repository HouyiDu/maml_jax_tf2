import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from jax.scipy.special import logsumexp
import time

import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')


