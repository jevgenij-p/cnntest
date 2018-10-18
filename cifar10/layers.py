import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input, shape):
    with tf.name_scope("weights"):
        W = weight_variable(shape)
        variable_summaries(W)
        kernel_visualization(W)
    with tf.name_scope("biases"):
        b = bias_variable([shape[3]])
        variable_summaries(b)
    activation = tf.nn.relu(conv2d(input, W) + b)
    conv_visualization(activation, "convolution")
    return activation

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    with tf.name_scope("weights"):
        W = weight_variable([in_size, size])
        variable_summaries(W)
    with tf.name_scope("biases"):
        b = bias_variable([size])
        variable_summaries(b)
    return tf.matmul(input, W) + b

def variable_summaries(var):
    """Attach summaries for TensorBoard visualization."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def kernel_visualization(kernel, name="filters"):
    channels = int(kernel.get_shape()[2])
    if (channels != 1 and channels != 3):
        return
    with tf.variable_scope("visualization"):
        # scale weights to [0 1]
        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        kernel_scaled = (kernel - x_min) / (x_max - x_min)

        # to tf.image_summary format [batch_size, height, width, channels]
        kernel_transposed = tf.transpose(kernel_scaled, [3, 0, 1, 2])

        # this will display random 3 filters
        tf.summary.image(name, kernel_transposed, max_outputs=3)

def conv_visualization(kernel, name="convolution"):
    shape = kernel.get_shape()
    channel = tf.slice(kernel, [0,0,0,0], [-1, int(shape[1]), int(shape[2]), 1])
    with tf.variable_scope("visualization"):
        # scale weights to [0 1]
        x_min = tf.reduce_min(channel)
        x_max = tf.reduce_max(channel)
        channel_scaled = (channel - x_min) / (x_max - x_min)

        # to tf.image_summary format [batch_size, height, width, channels]
        tf.summary.image(name, channel_scaled, max_outputs=3)
