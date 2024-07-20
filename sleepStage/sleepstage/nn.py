import tensorflow as tf


def lrelu(inputs, leak=0.1):
    with tf.variable_scope('relu') as scope:
        return tf.maximum(inputs, leak * inputs, name=scope)


def fc(
    name,
    inputs,
    n_hiddens,
    bias=None,
):
    # Weight initializer
    weight_initializer = tf.variance_scaling_initializer(
        scale=1.0,
        mode="fan_in",
        distribution="normal",
    )
    # # MSRA initialization
    # weight_initializer = tf.contrib.layers.variance_scaling_initializer(
    #     factor=2.0,
    #     mode='FAN_IN',
    #     uniform=False
    # )

    # Determine whether to use bias
    use_bias = False
    bias_initializer = tf.zeros_initializer()
    if bias is not None:
        use_bias = True
        bias_initializer = tf.constant_initializer(bias)

    # Dense
    with tf.variable_scope(name) as scope:
        outputs = tf.layers.dense(
            inputs=inputs,
            units=n_hiddens,
            use_bias=use_bias,
            kernel_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )

    return outputs

# 我改动了
def conv1d(
    name,
    inputs,
    n_filters,
    filter_size,
    stride_size,
    bias=None,
    padding="SAME",
):
    # Weight initializer
    weight_initializer = tf.variance_scaling_initializer(
        scale=1.0,
        mode="fan_in",
        distribution="normal",
    )
    # # MSRA initialization
    # weight_initializer = tf.contrib.layers.variance_scaling_initializer(
    #     factor=2.0,
    #     mode='FAN_IN',
    #     uniform=False
    # )

    # Determine whether to use bias
    use_bias = False
    bias_initializer = tf.zeros_initializer()
    if bias is not None:
        use_bias = True
        bias_initializer = tf.constant_initializer(bias)

    # Convolution
    with tf.variable_scope(name) as scope:
        outputs = tf.layers.conv2d(
            inputs=inputs,
            filters=n_filters,
            kernel_size=(filter_size, 1),
            strides=(stride_size, 1),
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )

    return outputs

# 我改动了
def max_pool1d(
    name,
    inputs,
    pool_size,
    stride_size,
    padding="SAME",
):
    # Max pooling
    with tf.variable_scope(name) as scope:
        outputs = tf.layers.max_pooling2d(
            inputs,
            pool_size=(pool_size, 1),
            strides=(stride_size, 1),
            padding=padding
        )

    return outputs


def batch_norm(
    name,
    inputs,
    is_training,
    momentum=0.99,
    epsilon=0.001
):
    # Batch normalization
    with tf.variable_scope(name) as scope:
        outputs = tf.layers.batch_normalization(
            inputs=inputs,
            momentum=momentum,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
        )

    return outputs


def adam_optimizer(
    loss,
    training_variables,
    global_step,
    learning_rate=1e-4,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    with tf.variable_scope("adam_optimizer") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=beta1,
                                           beta2=beta2,
                                           epsilon=epsilon)
        grads_and_vars_op = optimizer.compute_gradients(
            loss=loss,
            var_list=training_variables
        )
        apply_gradient_op = optimizer.apply_gradients(
            grads_and_vars=grads_and_vars_op,
            global_step=global_step
        )
        return apply_gradient_op, grads_and_vars_op


def adam_optimizer_clip(
    loss,
    training_variables,
    global_step,
    learning_rate=1e-4,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    clip_value=1.0,
):
    with tf.variable_scope("adam_optimizer") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=beta1,
                                           beta2=beta2,
                                           epsilon=epsilon)
        grads_and_vars_op = optimizer.compute_gradients(
            loss=loss,
            var_list=training_variables
        )
        grads_op, vars_op = zip(*grads_and_vars_op)
        grads_op, _ = tf.clip_by_global_norm(grads_op, clip_value)  # 进行梯度裁剪
        # 输入值 Tensor list 确定值 clip value 表示梯度值不会超过该value = 5
        # 返回值 剪裁后的tensor 剪裁前的global norm
        # print("#################################################################")
        # print(f"nn global check numerics : grads_op{grads_op},clip_value{clip_value}")
        # print(f"_ global norm : {_}")  # 0？
        apply_gradient_op = optimizer.apply_gradients(
            grads_and_vars=zip(grads_op, vars_op),  # 优化器将裁剪的梯度应用到所有的训练参数上，并将这个操作返回,
            global_step=global_step  # 在执行这个操作之后,进行下一轮梯度的裁剪过程
        )
        # print("#################################################################")
        # print(f"apply grad op : {apply_gradient_op}")
        # print(f"global step : {global_step}")

        return apply_gradient_op, grads_and_vars_op


def adam_optimizer_clip_lrs(
    loss,
    list_train_vars,
    list_lrs,
    global_step,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    clip_value=5.0,
):
    assert len(list_lrs) == len(list_train_vars)

    train_vars = []
    for v in list_train_vars:
        if len(train_vars) == 0:
            train_vars = list(v)
        else:
            train_vars.extend(v)

    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars),
                                      clip_value)

    offset = 0
    apply_gradient_ops = []
    grads_and_vars = []
    for i, v in enumerate(list_train_vars):
        g = grads[offset:offset+len(v)]
        opt = tf.train.AdamOptimizer(
            learning_rate=list_lrs[i],
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            name="Adam"
        )
        if i == 0:
            # Only increase global step once
            apply_gradient_op = opt.apply_gradients(
                grads_and_vars=zip(g, v),
                global_step=global_step
            )
        else:
            apply_gradient_op = opt.apply_gradients(
                grads_and_vars=zip(g, v)
            )

        apply_gradient_ops.append(apply_gradient_op)
        if len(grads_and_vars) == 0:
            grads_and_vars = list(zip(g, v))
        else:
            grads_and_vars.extend(list(zip(g, v)))
        offset += len(v)

    apply_gradient_ops = tf.group(*apply_gradient_ops)
    return apply_gradient_ops, grads_and_vars
