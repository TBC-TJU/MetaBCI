import os
from time import time

import numpy as np
import sklearn.metrics as skmetrics
import tensorflow as tf
import timeit
import tensorflow.contrib.metrics as contrib_metrics
import tensorflow.contrib.slim as contrib_slim
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

import nn
from tensorflow.python import keras
from tensorflow.python.keras import layers

import logging

logger = logging.getLogger("default_log")


class TinySleepNet(object):

    def __init__(
            self,
            config,
            output_dir="./output",
            use_rnn=False,
            testing=False,
            use_best=False,
            fine_tune=False,
            finetune_model_dir="./output"
    ):
        self.init_state = None
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")
        self.finetune_model_dir = os.path.join(finetune_model_dir, "best_ckpt")
        self.use_rnn = use_rnn

        # Placeholder
        with tf.variable_scope("placeholders") as scope:
            self.EEGsignals = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1),
                                             name='EEGsignals')
            self.EOGsignals = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1),
                                             name='EOGsignals')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

            if self.use_rnn:
                self.loss_weights = tf.placeholder(dtype=tf.float32, shape=(None,), name='loss_weights')

                self.seq_lengths = tf.placeholder(dtype=tf.int32, shape=(None,), name='seq_lengths')

        # Monitor global step update
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Monitor the number of epochs passed
        self.global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

        # Build a network that receives inputs from placeholders
        EEGnet = self.build_EEGcnn()  # 使用cnn
        EOGnet = self.build_EOGcnn()  # 使用cnn

        # 结合两个CNN的模型
        mul = layers.multiply([EEGnet, EOGnet])
        merge = layers.add([EEGnet, EOGnet, mul])

        # reshape to (batch_size, seq_length, input_dim) for multi
        input_dim = merge.shape[-1].value
        reshape_merge = tf.reshape(merge, shape=[-1, self.config["seq_length"], input_dim])

        # # attention MME
        # se = layers.Reshape((1, 1, 2048))(merge)
        # # excitation
        # se = layers.Dense(2048 // 4, activation='relu', name="dense_1")(se)
        # se = layers.Dense(2048, activation='sigmoid', name="dense_2")(se)
        #
        # re-weight
        # x = layers.multiply([merge, se])

        # block
        self.num_blocks = 2
        self.attention_dropout_rate = 0.5
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # self-attention
                se = self.multihead_attention(queries=reshape_merge,
                                              keys=reshape_merge,
                                              values=reshape_merge,
                                              attention_dropout_rate=self.attention_dropout_rate,
                                              training=self.is_training)
        # se = self.multihead_attention(queries=reshape_merge,
        #                               keys=reshape_merge,
        #                               values=reshape_merge,
        #                               attention_dropout_rate=self.attention_dropout_rate,
        #                               training=self.is_training)
        # residual connection
#         se += reshape_merge
        # ResidualBlock 残差
        rb = self.ln(se)
        sub = layers.Dense(2048, activation='relu', name="ResidualBlock_dense_1")(rb)
        sub = tf.layers.dropout(sub, rate=0.1, training=self.is_training)
        sub = layers.Dense(2048, activation='relu', name="ResidualBlock_dense_2")(sub)
        rb = tf.layers.dropout(sub, rate=0.1, training=self.is_training)
        rb = rb + se
        # rb = layers.Dense(2048, activation='relu', name="ResidualBlock_dense_1")(se)
        # sub = layers.Dense(2048, activation='relu', name="ResidualBlock_dense_2")(rb)
        # sub += reshape_merge
        # rb = self.ln(sub)
        # rb = tf.layers.dropout(rb, rate=0.1, training=self.is_training)
        # rb += rb
        reshape_rb = tf.reshape(rb, [-1, 1, 1, 2048])
        # conv
        reshape = layers.Conv2D(2048, (1, 1), activation='tanh',
                                padding='same', name="lastconv")(reshape_rb)
        pool = tf.layers.flatten(reshape, name="flatten")

        if self.use_rnn:
            # Check whether the corresponding config is given
            if "n_rnn_layers" not in self.config:
                raise Exception("Invalid config.")
            # Append the RNN if needed
            pool = self.append_rnn(pool)  # use rnn
            # pool = self.append_bilstm(pool)  # use BiLSTM

        pool = tf.layers.dropout(pool, rate=0.5, training=self.is_training, name="drop_2")  # 池化
        # Softmax linear
        net = nn.fc("softmax_linear", pool, self.config["n_classes"], bias=0.0)  # 0.0

        # Outputs
        self.logits = net
        self.preds = tf.argmax(self.logits, axis=1)

        # Cross-entropy loss
        # expanded_labels = tf.expand_dims(self.labels, axis=-1)  # BiLSTM
        # self.labels = expanded_labels  # BiLSTM
        self.loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=self.logits,
            name="loss_ce_per_sample"
        )
        with tf.name_scope("loss_ce_mean") as scope:
            if self.use_rnn:
                # Weight by sequence

                loss_w_seq = tf.multiply(self.loss_weights, self.loss_per_sample)

                # Weight by class
                sample_weights = tf.reduce_sum(
                    tf.multiply(
                        tf.one_hot(indices=self.labels, depth=self.config["n_classes"]),
                        np.asarray(self.config["class_weights"], dtype=np.float32)
                    ), 1
                )
                loss_w_class = tf.multiply(loss_w_seq, sample_weights)

                # Computer average loss scaled with the sequence length
                self.loss_ce = tf.reduce_sum(loss_w_class) / tf.reduce_sum(self.loss_weights)
            else:
                self.loss_ce = tf.reduce_mean(self.loss_per_sample)

        # Regularization loss
        self.reg_losses = self.regularization_loss()

        # Total loss
        self.loss = self.loss_ce + self.reg_losses

        # Metrics (used when we want to compute a metric from the output from minibatches)
        # self.labels = tf.reshape(self.labels, shape=[-1])  # BiLSTM
        # self.preds = tf.reshape(self.preds, shape=[-1])  # BiLSTM
        with tf.variable_scope("stream_metrics") as scope:
            self.metric_value_op, self.metric_update_op = contrib_metrics.aggregate_metric_map({
                "loss": tf.metrics.mean(values=self.loss),
                "accuracy": tf.metrics.accuracy(labels=self.labels, predictions=self.preds),
                "precision": tf.metrics.precision(labels=self.labels, predictions=self.preds),
                "recall": tf.metrics.recall(labels=self.labels, predictions=self.preds),
            })
            # Manually create reset operations of local vars
            metric_vars = contrib_slim.get_local_variables(scope=scope.name)
            self.metric_init_op = tf.variables_initializer(metric_vars)

        # Training outputs
        self.train_outputs = {
            "global_step": self.global_step,
            "train/loss": self.loss,
            "train/preds": self.preds,
            "train/stream_metrics": self.metric_update_op,
        }
        if self.use_rnn:
            self.train_outputs.update({
                "train/init_state": self.init_state,
                "train/final_state": self.final_state,
                # "train/init_state_fw": self.init_state_fw,  # BiLSTM
                # "train/init_state_bw": self.init_state_bw,  # BiLSTM
                # "train/final_state_fw": self.final_state[0],  # BiLSTM
                # "train/final_state_bw": self.final_state[1],  # BiLSTM
            })

        # Test outputs
        self.test_outputs = {
            "global_step": self.global_step,
            "test/loss": self.loss,
            "test/preds": self.preds,
        }
        if self.use_rnn:
            self.test_outputs.update({
                "test/init_state": self.init_state,
                "test/final_state": self.final_state,
                # "test/init_state_fw": self.init_state_fw,  # BiLSTM
                # "test/init_state_bw": self.init_state_bw,  # BiLSTM
                # "test/final_state_fw": self.final_state[0],  # BiLSTM
                # "test/final_state_bw": self.final_state[1],  # BiLSTM
            })

        # Tensorflow
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        if not testing:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"))
            self.train_writer.add_graph(self.sess.graph)
            logger.info("Saved tensorboard graph to {}".format(self.train_writer.get_logdir()))
        # Optimizer
        if not testing:
            # self.lr = tf.train.exponential_decay(
            #     learning_rate=self.config["learning_rate_decay"],
            #     global_step=self.global_step,
            #     decay_steps=self.config["decay_steps"],
            #     decay_rate=self.config["decay_rate"],
            #     staircase=False,
            #     name="learning_rate"
            # )
            self.lr = tf.constant(self.config["learning_rate"], dtype=tf.float32)

            # 要恢复的层
            restore_layers = ['EEGcnn', 'EOGcnn', 'dense_1', 'dense_2', 'lastconv', 'rnn', 'softmax_linear']
            # 要冻结的层
            frozen_layers = ['EEGcnn']
            # 全部可训练变量
            trainable_vars = tf.global_variables()
            trainable_vars_dict = {var.name: var for var in trainable_vars}
            # 恢复的参数
            restore_vars = [var for var in trainable_vars if any(layer in var.name for layer in restore_layers)]
            # 要冻结的参数
            frozen_vars = [var for var in trainable_vars if any(layer in var.name for layer in frozen_layers)]
            # 要训练的参数（把要冻结的参数去除了）
            var_list = [var for var in trainable_vars if var not in frozen_vars]
            with tf.variable_scope("optimizer") as scope:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Pretraining
                    if not self.use_rnn:
                        self.train_step_op, self.grad_op = nn.adam_optimizer(
                            loss=self.loss,
                            training_variables=tf.trainable_variables(),
                            global_step=self.global_step,
                            learning_rate=self.lr,
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                        )
                    # Fine-tuning
                    else:
                        # Use different learning rates for CNN and RNN
                        self.train_step_op, self.grad_op = nn.adam_optimizer_clip(
                            loss=self.loss,
                            training_variables=tf.trainable_variables(),
                            global_step=self.global_step,
                            # learning_rate=self.config["learning_rate"],
                            learning_rate=self.lr,
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                            clip_value=self.config["clip_grad_value"],
                        )

        # Initializer
        with tf.variable_scope("initializer") as scope:
            # tf.trainable_variables() or tf.global_variables()
            self.init_global_op = tf.variables_initializer(tf.global_variables())
            self.init_local_op = tf.variables_initializer(tf.local_variables())

        # Saver for storing variables
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Initialize variables
        self.run([self.init_global_op, self.init_local_op])

        # Restore variables (if possible)
        is_restore = False
        if use_best:
            if os.path.exists(self.best_ckpt_path):
                if os.path.isfile(os.path.join(self.best_ckpt_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.best_ckpt_path)
                    self.saver.restore(self.sess, latest_checkpoint)
                    logger.info("Best model restored from {}".format(latest_checkpoint))
                    is_restore = True
        else:
            if os.path.exists(self.checkpoint_path):
                if os.path.isfile(os.path.join(self.checkpoint_path, "checkpoint")):
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
                    self.saver.restore(self.sess, latest_checkpoint)
                    logger.info("Model restored from {}".format(latest_checkpoint))
                    is_restore = True

        # restore for finetune
        if fine_tune:
            if os.path.exists(self.finetune_model_dir):
                if os.path.isfile(os.path.join(self.finetune_model_dir, "checkpoint")):
                    saver_restore = tf.train.Saver(restore_vars)
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.finetune_model_dir)
                    saver_restore.restore(self.sess, latest_checkpoint)
                    graph = tf.get_default_graph()
                    logger.info("Best model for fine tuning restored from {}".format(latest_checkpoint))
                    is_restore = True

        if not is_restore:
            logger.info("Model started from random weights")

        # 查看lastconv、bilstm和softmax的参数
        detailLayers = ['lastconv', 'bilstm', 'softmax_linear']
        trainable_vars = tf.trainable_variables()
        self.detailVars = [var for var in trainable_vars if any(layer in var.name for layer in detailLayers)]

        # self.qq = self.sess.graph.get_tensor_by_name('multihead_attention/scaled_dot_product_attention/Softmax:0')
        # if use_best:
        #     global_vars = tf.global_variables()
        #     for var in global_vars:
        #         name = var.name
        #         value = self.run(var)
        #         print(f"var name:{name}, var shape:{var.shape}")
        #         print(f"{value}\n")

    def build_EEGcnn(self):
        first_filter_size = int(self.config["sampling_rate"] / 2.0)
        first_filter_stride = int(self.config["sampling_rate"] / 16.0)

        with tf.variable_scope("EEGcnn") as scope:
            net = nn.conv1d("EEGconv1d_1", self.EEGsignals, 128, first_filter_size, first_filter_stride)
            net = nn.batch_norm("EEGbn_1", net, self.is_training)
            net = tf.nn.relu(net, name="EEGrelu_1")

            net = nn.max_pool1d("EEGmaxpool1d_1", net, 8, 8)

            net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="EEGdrop_1")

            net = nn.conv1d("EEGconv1d_2_1", net, 128, 8, 1)
            net = nn.batch_norm("EEGbn_2_1", net, self.is_training)
            net = tf.nn.relu(net, name="EEGrelu_2_1")
            net = nn.conv1d("EEGconv1d_2_2", net, 128, 8, 1)
            net = nn.batch_norm("EEGbn_2_2", net, self.is_training)
            net = tf.nn.relu(net, name="EEGrelu_2_2")
            net = nn.conv1d("EEGconv1d_2_3", net, 128, 8, 1)
            net = nn.batch_norm("EEGbn_2_3", net, self.is_training)
            net = tf.nn.relu(net, name="EEGrelu_2_3")

            net = nn.max_pool1d("EEGmaxpool1d_2", net, 4, 4)
            net = tf.layers.flatten(net, name="EEGflatten_2")

        # net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="EEGdrop_2")

        return net

    def build_EOGcnn(self):
        first_filter_size = int(self.config["sampling_rate"] / 4.0)
        first_filter_stride = int(self.config["sampling_rate"] / 16.0)

        with tf.variable_scope("EOGcnn") as scope:
            # name, inputs, n_filters, filter_size, stride_size)
            net = nn.conv1d("EOGconv1d_1", self.EOGsignals, 128, first_filter_size, first_filter_stride)
            net = nn.batch_norm("EOGbn_1", net, self.is_training)
            net = tf.nn.relu(net, name="EOGrelu_1")

            net = nn.max_pool1d("EOGmaxpool1d_1", net, 8, 8)

            net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="EOGdrop_1")

            net = nn.conv1d("EOGconv1d_2_1", net, 128, 8, 1)
            net = nn.batch_norm("EOGbn_2_1", net, self.is_training)
            net = tf.nn.relu(net, name="EOGrelu_2_1")

            net = nn.conv1d("EOGconv1d_2_2", net, 128, 8, 1)
            net = nn.batch_norm("EOGbn_2_2", net, self.is_training)
            net = tf.nn.relu(net, name="EOGrelu_2_2")

            net = nn.conv1d("EOGconv1d_2_3", net, 128, 8, 1)
            net = nn.batch_norm("EOGbn_2_3", net, self.is_training)
            net = tf.nn.relu(net, name="EOGrelu_2_3")

            net = nn.max_pool1d("EOGmaxpool1d_2", net, 4, 4)
            # 我增加的
            # net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="EOGdrop_2")

            net = tf.layers.flatten(net, name="EOGflatten_2")

        # net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="EOGdrop_2")

        return net

    # rnn + atten
    def append_rnn(self, inputs):
        # RNN
        with tf.variable_scope("rnn") as scope:
            # Reshape the input from (batch_size * seq_length, input_dim) to
            # (batch_size, seq_length, input_dim)
            input_dim = inputs.shape[-1].value
            seq_inputs = tf.reshape(inputs, shape=[-1, self.config["seq_length"], input_dim], name="reshape_seq_inputs")

            def _create_rnn_cell(n_units):
                """A function to create a new rnn cell."""
                cell = tf.contrib.rnn.LSTMCell(
                    num_units=n_units,
                    use_peepholes=True,
                    forget_bias=1.0,
                    state_is_tuple=True,
                )
                # Dropout wrapper
                keep_prob = tf.cond(self.is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
                return cell

            # LSTM
            cells = []
            for l in range(self.config["n_rnn_layers"]):
                cells.append(_create_rnn_cell(self.config["n_rnn_units"]))
                # cells.append(_create_rnn_cell(64))

            # Multiple layers of forward and backward cells
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)

            # Initial states
            self.init_state = multi_cell.zero_state(self.config["batch_size"], tf.float32)

            # Create rnn
            outputs, states = tf.nn.dynamic_rnn(
                cell=multi_cell,
                inputs=seq_inputs,
                initial_state=self.init_state,
                sequence_length=self.seq_lengths,
            )
            # logger.info(outputs)
            # Final states
            self.final_state = states

            # Concatenate the output from forward and backward cells
            net = tf.reshape(outputs, shape=[-1, self.config["n_rnn_units"]], name="reshape_nonseq_input")

            #####################################
            # begin
#             d_model = outputs.get_shape().as_list()[-1]
#             # Linear projections
#             dense_Q = layers.Dense(d_model)
#             dense_K = layers.Dense(d_model)
#             dense_V = layers.Dense(d_model)

#             Q = dense_Q(outputs)  # Tensor("multihead_attention/dense/BiasAdd:0", shape=(?, 20, 2048), dtype=float32)
#             K = dense_K(outputs)
#             V = dense_V(outputs)

#             # Attention
#             scpa,weight = self.scaled_dot_product_attention(Q, K, V, 0.5)

#             outputs = outputs * scpa
#             net = tf.reshape(outputs, shape=[-1, self.config["n_rnn_units"]])
            # end
            #####################################
        return net

    #################################################
    # begin
    # BiLSTM
    def append_bilstm(self, inputs):
        # Reshape the input from (batch_size * seq_length, input_dim) to (batch_size, seq_length, input_dim)
        with tf.variable_scope("bilstm") as scope:
            input_dim = inputs.shape[-1].value
            seq_inputs = tf.reshape(inputs, shape=[-1, self.config["seq_length"], input_dim],
                                    name="reshape_seq_inputs")

            # assert self.config['batch_size'] == seq_inputs.shape[-1].value

            def _create_bilstm_cell(n_units):
                """A function to create a new bidirectional lstm cell."""
                forward_cell = tf.contrib.rnn.LSTMCell(
                    num_units=n_units,
                    use_peepholes=True,
                    forget_bias=1.0,
                    state_is_tuple=True,
                )
                backward_cell = tf.contrib.rnn.LSTMCell(
                    num_units=n_units,
                    use_peepholes=True,
                    forget_bias=1.0,
                    state_is_tuple=True,
                )

                # Apply dropout wrapper to both forward and backward cells
                keep_prob = tf.cond(self.is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))
                forward_cell = tf.contrib.rnn.DropoutWrapper(forward_cell, output_keep_prob=keep_prob)
                backward_cell = tf.contrib.rnn.DropoutWrapper(backward_cell, output_keep_prob=keep_prob)

                return forward_cell, backward_cell

            cells_fw = []
            cells_bw = []
            for l in range(self.config["n_rnn_layers"]):
                forward_cell, backward_cell = _create_bilstm_cell(self.config["n_rnn_units"])
                cells_fw.append(forward_cell)
                cells_bw.append(backward_cell)

            multi_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells=cells_fw, state_is_tuple=True)
            multi_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells=cells_bw, state_is_tuple=True)

            self.init_state_fw = multi_cell_fw.zero_state(self.config["batch_size"], tf.float32)
            self.init_state_bw = multi_cell_bw.zero_state(self.config["batch_size"], tf.float32)

            outputs, self.final_state = bidirectional_dynamic_rnn(
                cell_fw=multi_cell_fw,
                cell_bw=multi_cell_bw,
                # inputs=tf.unstack(seq_inputs, axis=1),
                inputs=seq_inputs,
                initial_state_fw=self.init_state_fw,
                initial_state_bw=self.init_state_bw,
                sequence_length=self.seq_lengths,
            )

            output = tf.concat(outputs, axis=-1)
            final_hidden_state = tf.concat([self.final_state[1][0].c, self.final_state[1][0].h], 1)

            final_hidden_state = tf.expand_dims(final_hidden_state, 2)

            # 计算每个时间步的输出与最后输出状态的相似度
            # [batch_size, step, hidden*2] * [batch_size, hidden*2, 1] = squeeze([batch_size, step, 1]) = [batch_size, step]
            # print(np.shape(output))
            attn_weights = tf.squeeze(tf.matmul(output, final_hidden_state), 2)
            # 在时间步维度上进行 softmax 得到权重向量
            soft_attn_weights = tf.nn.softmax(attn_weights, 1)
            # 各时间步输出和对应的权重想成得到上下文矩阵 [batch_size, hidden*2, step] * [batch_size, step, 1] = [batch_size, hidden*2, 1]
            self.context = tf.matmul(tf.transpose(output, [0, 2, 1]), tf.expand_dims(soft_attn_weights, 2))
            # print(np.shape(self.context))
            # squeeze([batch_size, hidden*2, 1]) = [batch_size, hidden*2]
            self.context = tf.squeeze(self.context, 2)
            # print(np.shape(self.context))
            # out = tf.Variable(tf.random_normal([self.config["n_rnn_units"] * 2, self.config["n_classes"]]))
            # output = tf.concat([output, tf.expand_dims(self.context, 1)], axis=1)
            # print(np.shape(self.context))

            output = tf.reshape(output, shape=[-1, self.config["n_rnn_units"] * 2], name="reshape_nonseq_input")

            # atten
            # begin
            # d_model = output.get_shape().as_list()[-1]
            # # Linear projections
            # Q = tf.layers.dense(output, d_model, use_bias=True)  # (N, T_q, d_model)
            # K = tf.layers.dense(output, d_model, use_bias=True)  # (N, T_k, d_model)
            # V = tf.layers.dense(output, d_model, use_bias=True)  # (N, T_k, d_model)
            # # Attention
            # scpa = self.scaled_dot_product_attention(Q, K, V, 0.5)
            # output = layers.multiply([output, scpa])
            # output = tf.reshape(output, shape=[-1, 2 * self.config["n_rnn_units"]])
            # end
        return output

    def ln(self, inputs, epsilon=1e-8, scope="ln"):
        '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
        inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        scope: Optional scope for `variable_scope`.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def scaled_dot_product_attention(self, q, k, v, dropout_rate=0.5,
                                     scope="scaled_dot_product_attention"):
        '''See 3.2.1.
        Q: Packed queries. 3d tensor. [N, T_q, d_k].
        K: Packed keys. 3d tensor. [N, T_k, d_k].
        V: Packed values. 3d tensor. [N, T_k, d_v].
        key_masks: A 2d tensor with shape of [N, key_seqlen]
        causality: If True, applies masking for future blinding
        dropout_rate: A floating point number of [0, 1].
        training: boolean for controlling droput
        scope: Optional scope for `variable_scope`.
        '''
        # with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        #     d_k = Q.get_shape().as_list()[-1]
        #
        #     # dot product
        #     outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)
        #
        #     # scale
        #     outputs /= d_k ** 0.5
        #
        #     # softmax
        #     outputs = tf.nn.softmax(outputs)
        #     attention = tf.transpose(outputs, [0, 2, 1])
        #     tf.summary.image("attention", tf.expand_dims(attention[:1], -1))
        #
        #     # attention dropout
        #     outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=self.is_training)
        #
        #     # weighted sum (context vectors)
        #     outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)
        # return outputs

        ##################BEGIN#########################
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            matmul_qk = tf.matmul(q, k, transpose_b=True)
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def causal_conv1d(self, inputs, filters, kernel_size, dilation_rate, activation=None, name=None):
        padding = (kernel_size - 1) * dilation_rate
        result = tf.layers.conv1d(
            inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='valid',
            dilation_rate=dilation_rate,
            activation=activation,
            name=name
        )
        if padding != 0:
            return result[:, :-padding, :]
        return result

    def multihead_attention(self, queries, keys, values,
                            # key_masks,
                            num_heads=8,
                            attention_dropout_rate=0.1,
                            training=True,
                            # causality=False,
                            scope="multihead_attention"):
        '''Applies multihead attention. See 3.2.2
        queries: A 3d tensor with shape of [N, T_q, d_model].
        keys: A 3d tensor with shape of [N, T_k, d_model].
        values: A 3d tensor with shape of [N, T_k, d_model].
        key_masks: A 2d tensor with shape of [N, key_seqlen]
        num_heads: An int. Number of heads.
        dropout_rate: A floating point number.
        training: Boolean. Controller of mechanism for dropout.
        causality: Boolean. If true, units that reference the future are masked.
        scope: Optional scope for `variable_scope`.

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        d_model = queries.get_shape().as_list()[-1]

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Linear projections
            dense_Q = layers.Dense(d_model)
            dense_K = layers.Dense(d_model)
            dense_V = layers.Dense(d_model)

            Q = dense_Q(queries)  # Tensor("multihead_attention/dense/BiasAdd:0", shape=(?, 20, 2048), dtype=float32)
            K = dense_K(keys)
            V = dense_V(values)

            # Split and concat
            # Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            # K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
            # V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_v, d_model/h)

            depth = d_model // num_heads
            Q_split = tf.reshape(Q, [self.config["batch_size"], self.config["seq_length"], num_heads, depth])
            K_split = tf.reshape(K, [self.config["batch_size"], self.config["seq_length"], num_heads, depth])
            V_split = tf.reshape(V, [self.config["batch_size"], self.config["seq_length"], num_heads, depth])

            # Q_ = tf.transpose(Q_split, perm=[0, 2, 1, 3])  # [batchsize, num_heads, sequence_length, d_model//num_heads]
            # K_ = tf.transpose(K_split, perm=[0, 2, 1, 3])  # [batchsize, num_heads, sequence_length, d_model//num_heads]
            # V_ = tf.transpose(V_split, perm=[0, 2, 1, 3])  # [batchsize, num_heads, sequence_length, d_model//num_heads]

            # Attention
            scaled_attention, attention_weights = self.scaled_dot_product_attention(Q_split, K_split, V_split, attention_dropout_rate)
            # scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

            # Restore shape
            # outputs = tf.concat(tf.split(Q_, num_heads, axis=0), axis=2)  # (N, T_q, d_model)
            concat_attention = tf.reshape(scaled_attention, (self.config["batch_size"], -1, d_model))

            concat_attention += queries
#             concat_attention = self.ln(concat_attention)
        return concat_attention

    # end
    #################################################

    def train(self, minibatches):
        self.run(self.metric_init_op)
        start = timeit.default_timer()
        preds = []
        trues = []

        if not self.use_rnn:
            for x, y, z, w, sl, re in minibatches:
                feed_dict = {
                    self.EEGsignals: x,
                    self.EOGsignals: y,
                    self.labels: z,
                    self.is_training: True
                }
                _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)

                #######################################
                tmp_preds = np.reshape(outputs["train/preds"], (self.config["batch_size"], self.config["seq_length"]))
                tmp_trues = np.reshape(z, (self.config["batch_size"], self.config["seq_length"]))

                for i in range(self.config["batch_size"]):
                    preds.extend(tmp_preds[i, :sl[i]])
                    trues.extend(tmp_trues[i, :sl[i]])
                #######################################
                # preds.extend(outputs["train/preds"])
                # trues.extend(z)
        else:
            for x, y, z, w, sl, re in minibatches:
                feed_dict = {
                    self.EEGsignals: x,
                    self.EOGsignals: y,
                    self.labels: z,
                    self.is_training: True,
                    self.loss_weights: w,
                    self.seq_lengths: sl,
                }
                if re:  # BiLSTM
                    # Initialize state of BiLSTM
                    # "test/init_state_fw": self.init_state_fw,  # BiLSTM
                    # "test/init_state_bw": self.init_state_bw,  # BiLSTM
                    # "train/final_state_fw": self.final_state[0],  # BiLSTM
                    # "train/final_state_bw": self.final_state[1],  # BiLSTM
                    # init_state_fw = self.run(self.init_state_fw)
                    # init_state_bw = self.run(self.init_state_bw)
                    state = self.run(self.init_state)

                    # Create a feed_dict for initial states
                    # for i, (c_fw, h_fw, c_bw, h_bw) in enumerate(
                    #         zip(self.init_state_fw, init_state_fw, self.init_state_bw, init_state_bw)):
                    #     feed_dict[c_fw] = init_state_fw[i].c
                    #     feed_dict[h_fw] = init_state_fw[i].h
                    #     feed_dict[c_bw] = init_state_bw[i].c
                    #     feed_dict[h_bw] = init_state_bw[i].h

                # for i, (c_fw, h_fw) in enumerate(self.init_state_fw):
                #     feed_dict[c_fw] = init_state_fw[i].c
                #     feed_dict[h_fw] = init_state_fw[i].h
                #
                # for i, (c_bw, h_bw) in enumerate(self.init_state_bw):
                #     feed_dict[c_bw] = init_state_bw[i].c
                #     feed_dict[h_bw] = init_state_bw[i].h

                for i, (c, h) in enumerate(self.init_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h
                # Run the training step and get outputs #BiLSTM
                _, outputs, = self.run(
                    [self.train_step_op, self.train_outputs],
                    feed_dict=feed_dict)
                # logger.info("train set Q_.shape: {}".format(qq.shape))
                # Buffer the final states #BiLSTM
                # final_state_fw = outputs["train/final_state_fw"]
                # final_state_bw = outputs["train/final_state_bw"]
                state = outputs["train/final_state"]

                tmp_preds = np.reshape(outputs["train/preds"], (self.config["batch_size"], self.config["seq_length"]))
                tmp_trues = np.reshape(z, (self.config["batch_size"], self.config["seq_length"]))

                for i in range(self.config["batch_size"]):
                    preds.extend(tmp_preds[i, :sl[i]])
                    trues.extend(tmp_trues[i, :sl[i]])

        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs.update({
            "train/trues": trues,
            "train/preds": preds,
            "train/accuracy": acc,
            "train/f1_score": f1_score,
            "train/cm": cm,
            "train/duration": duration,
        })
        #############
        # 获取所有可训练变量的权重
        # cnt = 0
        # # 打印每个变量的名称和值
        # with open('weight.txt', 'a') as f:
        #     f.write("-------------------------------------------")
        #     for var in self.detailVars:
        #         var_name = var.name
        #         f.write(f"Variable: {var_name}\n, Shape: {var.shape}\n, Value: {detailVars[cnt].shape}\n\n\n")
        #         cnt = cnt + 1
        ###############
        return outputs

    def evaluate(self, minibatches):
        start = timeit.default_timer()
        losses = []
        preds = []
        trues = []
        if not self.use_rnn:
            for x, y, z, w, sl, re in minibatches:
                feed_dict = {
                    self.EEGsignals: x,
                    self.EOGsignals: y,
                    self.labels: z,
                    self.is_training: False,
                }

            outputs = self.run(self.test_outputs, feed_dict=feed_dict)

            losses.append(outputs["test/loss"])

            tmp_preds = np.reshape(outputs["test/preds"], (self.config["batch_size"], self.config["seq_length"]))
            tmp_trues = np.reshape(z, (self.config["batch_size"], self.config["seq_length"]))

            for i in range(self.config["batch_size"]):
                preds.extend(tmp_preds[i, :sl[i]])
                trues.extend(tmp_trues[i, :sl[i]])
            # losses.append(outputs["test/loss"])
            # preds.extend(outputs["test/preds"])
            # trues.extend(z)

        else:
            for x, y, z, w, sl, re in minibatches:
                feed_dict = {
                    self.EEGsignals: x,
                    self.EOGsignals: y,
                    self.labels: z,
                    self.is_training: False,
                    self.loss_weights: w,
                    self.seq_lengths: sl,
                }
                # Bilstm##########
                # if re:
                #     # BiLSTM
                #     init_state_fw = self.run(self.init_state_fw)
                #     init_state_bw = self.run(self.init_state_bw)
                #
                # # Create a feed_dict for initial states
                # for i, (c_fw, h_fw) in enumerate(self.init_state_fw):
                #     feed_dict[c_fw] = init_state_fw[i].c
                #     feed_dict[h_fw] = init_state_fw[i].h
                #
                # for i, (c_bw, h_bw) in enumerate(self.init_state_bw):
                #     feed_dict[c_bw] = init_state_bw[i].c
                #     feed_dict[h_bw] = init_state_bw[i].h

                if re:
                    # Initialize state of RNN
                    state = self.run(self.init_state)
                # Carry the states from the previous batches through time
                for i, (c, h) in enumerate(self.init_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

                outputs = self.run(self.test_outputs, feed_dict=feed_dict)
                # logger.info("test set : {}".format(qq.shape))
                # Buffer the final states
                # BiLSTM
                state = outputs["test/final_state"]
                # final_state_fw = outputs["test/final_state_fw"]
                # final_state_bw = outputs["test/final_state_bw"]

                losses.append(outputs["test/loss"])

                tmp_preds = np.reshape(outputs["test/preds"], (self.config["batch_size"], self.config["seq_length"]))
                tmp_trues = np.reshape(z, (self.config["batch_size"], self.config["seq_length"]))

                for i in range(self.config["batch_size"]):
                    preds.extend(tmp_preds[i, :sl[i]])
                    trues.extend(tmp_trues[i, :sl[i]])

        loss = np.mean(losses)
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs = {
            "test/trues": trues,
            "test/preds": preds,
            "test/loss": loss,
            "test/accuracy": acc,
            "test/f1_score": f1_score,
            "test/cm": cm,
            "test/duration": duration,
        }
        return outputs

    def get_current_epoch(self):
        return self.run(self.global_epoch)

    def pass_one_epoch(self):
        self.run(tf.assign(self.global_epoch, self.global_epoch + 1))

    def run(self, *args, **kwargs):
        return self.sess.run(*args, **kwargs)

    def save_checkpoint(self, name):
        path = self.saver.save(
            self.sess,
            os.path.join(self.checkpoint_path, "{}.ckpt".format(name)),
            global_step=self.global_step
        )

        logger.info("Saved checkpoint to {}".format(path))

    def save_best_checkpoint(self, name):
        path = self.best_saver.save(
            self.sess,
            os.path.join(self.best_ckpt_path, "{}.ckpt".format(name)),
            global_step=self.global_step
        )

        logger.info("Saved best checkpoint to {}".format(path))

    def save_weights(self, scope, name, key_variables=tf.GraphKeys.TRAINABLE_VARIABLES):
        # Save weights
        path = os.path.join(self.weights_path, "{}.npz".format(name))
        logger.info("Saving weights in scope: {} to {}".format(scope, path))

        save_dict = {}
        cnn_vars = tf.get_collection(key_variables, scope=scope)
        for v in cnn_vars:
            save_dict[v.name] = self.sess.run(v)
            logger.info("  variable: {}".format(v.name))
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        np.savez(path, **save_dict)

    def load_weights(self, scope, weight_file, key_variables=tf.GraphKeys.TRAINABLE_VARIABLES):
        # Load weights
        logger.info("Loading weights in scope: {} from {}".format(scope, weight_file))
        cnn_vars = tf.get_collection(key_variables, scope=scope)
        with np.load(weight_file) as f:
            for v in cnn_vars:
                tensor = tf.get_default_graph().get_tensor_by_name(v.name)
                self.run(tf.assign(tensor, f[v.name]))
                logger.info("  variable: {}".format(v.name))

    def regularization_loss(self):
        reg_losses = []
        list_vars = [
            "EOGcnn/EOGconv1d_1/conv2d/kernel:0",
            "EOGcnn/EOGconv1d_2_1/conv2d/kernel:0",
            "EOGcnn/EOGconv1d_2_2/conv2d/kernel:0",
            "EOGcnn/EOGconv1d_2_3/conv2d/kernel:0",
            "EEGcnn/EEGconv1d_1/conv2d/kernel:0",
            "EEGcnn/EEGconv1d_2_1/conv2d/kernel:0",
            "EEGcnn/EEGconv1d_2_2/conv2d/kernel:0",
            "EEGcnn/EEGconv1d_2_3/conv2d/kernel:0",
            "lastconv/conv2d/kernel:0",
            # 我增加的
            # "rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0",
            # "softmax_linear/dense/kernel:0",
        ]
        for v in tf.trainable_variables():
            if any(v.name in s for s in list_vars):
                reg_losses.append(tf.nn.l2_loss(v))
        if len(reg_losses):
            reg_losses = tf.multiply(tf.add_n(reg_losses, name="l2_loss"), self.config["l2_weight_decay"])
        else:
            reg_losses = 0
        return reg_losses

    def kappa_cal(self, matrix):
        n = np.sum(matrix)
        sum_po = 0
        sum_pe = 0
        for i in range(len(matrix[0])):
            sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        return (po - pe) / (1 - pe)


if __name__ == "__main__":
    from config import pretrain

    model = Model(config=pretrain, output_dir="./output/test", use_rnn=False)
    tf.reset_default_graph()

    from config import finetune

    model = Model(config=finetune, output_dir="./output/test", use_rnn=True)
