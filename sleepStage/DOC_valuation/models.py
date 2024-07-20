import os
import numpy as np
import sklearn.metrics as skmetrics
import tensorflow as tf
import timeit
import tensorflow.contrib.metrics as contrib_metrics
import tensorflow.contrib.slim as contrib_slim
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
import nn
from tensorflow.python import keras
from tensorflow.python.keras import layers
from functools import partial

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
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")
        self.finetune_model_dir = os.path.join(finetune_model_dir, "best_ckpt")
        self.use_rnn = use_rnn

        # Placeholder 占位符
        with tf.variable_scope("placeholders") as scope:
            self.EEGsignals = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1),
                                             name='EEGsignals')
            self.EOGsignals = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1),
                                             name='EOGsignals')
            self.SleepStages = tf.placeholder(dtype=tf.float32, shape=(None, self.config["input_size"], 1, 1),
                                              name='SleepStages')  # 新增睡眠分期结果
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
        Sleepnet = self.build_Sleepcnn()

        # 结合两个CNN的模型
        mul = layers.multiply([EEGnet, EOGnet])
        merge = layers.add([EEGnet, EOGnet, mul])
        #
        # attention MME
        # se = layers.GlobalAveragePooling2D()(merge) 压缩
        se = layers.Reshape((1, 1, 2048))(merge)  # 卷积核大小 128
        # excitation 激励
        se = layers.Dense(2048 // 4, activation='relu', name="dense_1")(se)  # 修改使用leak-relu
        # se = tf.layers.dense(se, 192 // 4, activation=partial(tf.nn.leaky_relu, alpha=0.01))
        se = layers.Dense(2048, activation='sigmoid', name="dense_2")(se)
        # re-weight 通道权重分配
        x = layers.multiply([merge, se])
        x = tf.layers.dropout(x, rate=0.5, training=self.is_training, name="drop_1")
        # conv
        reshape = layers.Conv2D(2048, (1, 1), activation='tanh',  # 此处
                                padding='same', name="lastconv")(x)

        # reshape = nn.batch_norm("conv2Dbn", reshape, self.is_training) #新增
        # reshape = nn.lrelu(reshape, name="conv2Dlrelu") #新增
        # reshape = nn.max_pool1d("conv2Dmaxpool1d", reshape, 4, 4, 4, 4) #新增
        # reshape = tf.layers.dropout(reshape, rate=0.5, training=self.is_training, name="conv2Ddrop") #新增
        reshape = layers.multiply([reshape, Sleepnet])
        pool = tf.layers.flatten(reshape, name="flatten")  # 一维

        if self.use_rnn:
            # Check whether the corresponding config is given
            if "n_rnn_layers" not in self.config:
                raise Exception("Invalid config.")
            # Append the RNN if needed
            pool = self.append_rnn(pool)  # use rnn
            # pool = self.append_bilstm(pool)  # use BiLSTM

            # 使用 append_bilstm 并获取正向和反向传播的输出 #BiLSTM
            # pool_fw, pool_bw = self.append_bilstm(pool) #BiLSTM
            # 将正向和反向传播的输出连接起来 #BiLSTM
            # pool = tf.concat([pool_fw, pool_bw], axis=-1) #BiLSTM

        # pool = tf.layers.dropout(pool, rate=0.5, training=self.is_training, name="drop_2")  # 池化
        # Softmax linear
        net = nn.fc("softmax_linear", pool, self.config["n_classes"], bias=0.0)  # 0.0

        # net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="drop_2") #池化
        # Outputs
        # print(np.shape(net))
        self.logits = net
        self.preds = tf.argmax(self.logits, axis=1)

        # Cross-entropy loss
        # print(np.shape(self.logits)) #BiLSTM
        # expanded_labels = tf.expand_dims(self.labels, axis=-1) #BiLSTM
        # self.labels=expanded_labels #BiLSTM
        # print(np.shape(self.labels))
        # print(np.shape(expanded_labels)) #BiLSTM
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
        # self.labels=tf.reshape(self.labels, shape=[-1]) #BiLSTM
        # print(np.shape(self.labels))
        # self.preds=tf.reshape(self.preds, shape=[-1]) #BiLSTM
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

        # Traini ngoutputs
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
                # "train/init_state_fw": self.init_state_fw, #BiLSTM
                # "train/init_state_bw": self.init_state_bw, #BiLSTM
                # "train/final_state_fw": self.final_state[0], #BiLSTM
                # "train/final_state_bw": self.final_state[1], #BiLSTM
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
                # "test/init_state_fw": self.init_state_fw, #BiLSTM
                # "test/init_state_bw": self.init_state_bw, #BiLSTM
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
            # 这里解开注释了，学习率 指数衰减
            # self.lr = tf.train.exponential_decay(
            #     learning_rate=self.config["learning_rate_decay"],
            #     global_step=self.global_step,
            #     decay_steps=self.config["decay_steps"],
            #     decay_rate=self.config["decay_rate"],
            #     staircase=False,
            #     name="learning_rate"
            # )
            # self.lr = tf.constant(self.config["learning_rate"], dtype=tf.float32)
            # 冻结指定层的变量
            # 法1：
            # var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='softmax_linear')
            # 法2：
            frozen_layers = ['EEGcnn', 'EOGcnn', 'rnn', 'dense_1', 'dense_2', 'lastconv']
            # frozen_layers = ['EEGcnn']
            trainable_vars = tf.global_variables()
            trainable_vars_dict = {var.name: var for var in trainable_vars}
            frozen_vars = [var for var in trainable_vars if any(layer in var.name for layer in frozen_layers)]
            # var_list = [var for var in trainable_vars if var not in frozen_vars]
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='softmax_linear') #只训练softmax层

            with tf.variable_scope("optimizer") as scope:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Pretraining
                    if not self.use_rnn:
                        self.train_step_op, self.grad_op = nn.adam_optimizer(  # adam
                            loss=self.loss,
                            # training_variables=tf.trainable_variables(),
                            training_variables=var_list,
                            global_step=self.global_step,
                            # learning_rate=self.config["learning_rate"],
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
                            training_variables=var_list,
                            # training_variables=tf.trainable_variables(),  ##训练变量
                            global_step=self.global_step,
                            # learning_rate=self.config["learning_rate"],
                            learning_rate=self.lr,
                            beta1=self.config["adam_beta_1"],
                            beta2=self.config["adam_beta_2"],
                            epsilon=self.config["adam_epsilon"],
                            clip_value=self.config["clip_grad_value"],
                        )

                        # self.train_step_op, self.grad_op = nn.RMSprop_optimizer_clip(
                        #     loss=self.loss,
                        #     training_variables=tf.trainable_variables(),  ##训练变量
                        #     global_step=self.global_step,
                        #     # learning_rate=self.config["learning_rate"],
                        #     learning_rate=self.lr,
                        #     epsilon=self.config["adam_epsilon"],
                        #     clip_value=self.config["clip_grad_value"],
                        # )

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
                    saver_restore = tf.train.Saver(frozen_vars)
                    # Restore the last checkpoint
                    latest_checkpoint = tf.train.latest_checkpoint(self.finetune_model_dir)
                    saver_restore.restore(self.sess, latest_checkpoint)
                    graph = tf.get_default_graph()
                    logger.info("Best model for fine tuning restored from {}".format(latest_checkpoint))
                    is_restore = True

        if not is_restore:
            logger.info("Model started from random weights")

        ##########################
        # 用tsne画图
        # self.test = self.sess.graph.get_tensor_by_name('rnn/reshape_nonseq_input:0')
        self.test = self.EEGsignals
        self.test_lables = self.labels
        # 画图2
        self.draw2 = self.sess.graph.get_tensor_by_name('flatten/Reshape:0')
        ############################

        ############################
        # 绘制条形图
        self.se = self.sess.graph.get_tensor_by_name('dense_2/Sigmoid:0')

    def build_Sleepcnn(self):

        with tf.variable_scope("Sleepcnn") as scope:
        #     net = nn.conv1d("Sleepconv1d_1", self.SleepStages, 128, 3, 3)
        #     net = nn.batch_norm("Sleepbn_1_1", net, self.is_training)
        #     net = nn.lrelu(net, name="Sleeplrelu_1_1")
        #     net = nn.max_pool1d("Sleepmaxpool1d_1_1", net, 4, 4)
        #
        #     net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="Sleepdrop_1")
        #
        #     net = nn.conv1d("Sleepconv1d_1_3", net, 128, 3, 3)
        #     net = nn.batch_norm("Sleepbn_1_3", net, self.is_training)
        #     net = nn.lrelu(net, name="Sleeplrelu_1_3")
        #
        #     net = nn.max_pool1d("Sleepmaxpool1d_1_2", net, 4, 4)
        #
        #     net = tf.layers.flatten(net, name="Sleepflatten_1")
            first_filter_size = int(self.config["sampling_rate"] / 2.0)
            first_filter_stride = int(self.config["sampling_rate"] / 16.0)

            net = nn.conv1d("Sleepconv1d_1", self.EEGsignals, 128, first_filter_size, first_filter_stride)
            net = nn.batch_norm("Sleepbn_1", net, self.is_training)
            net = tf.nn.relu(net, name="Sleeprelu_1")

            net = nn.max_pool1d("EEGmaxpool1d_1", net, 8, 8)

            net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="EEGdrop_1")

            net = nn.conv1d("Sleepconv1d_2_1", net, 128, 8, 1)
            net = nn.batch_norm("Sleepbn_2_1", net, self.is_training)
            net = tf.nn.relu(net, name="Sleeprelu_2_1")
            net = nn.conv1d("Sleepconv1d_2_2", net, 128, 8, 1)
            net = nn.batch_norm("Sleepbn_2_2", net, self.is_training)
            net = tf.nn.relu(net, name="Sleeprelu_2_2")
            net = nn.conv1d("Sleepconv1d_2_3", net, 128, 8, 1)
            net = nn.batch_norm("Sleepbn_2_3", net, self.is_training)
            net = tf.nn.relu(net, name="Sleeprelu_2_3")

            net = nn.max_pool1d("Sleepmaxpool1d_2", net, 4, 4)

            net = tf.layers.flatten(net, name="Sleepflatten_2")

            # net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="Sleepdrop_2")
        return net

    def build_EEGcnn(self):
        # first_filter_size = int(self.config["sampling_rate"] / 2.0)
        # # first_filter_stride = int(self.config["sampling_rate"] / 16.0)
        # first_filter_stride = int(self.config["sampling_rate"] / 4.0)
        #
        # with tf.variable_scope("EEGcnn") as scope:
        #     net = nn.conv1d("EEGconv1d_1", self.EEGsignals, 8, 3, 3, 3, 3)
        #     net = nn.batch_norm("EEGbn_1_1", net, self.is_training)
        #     net = nn.lrelu(net, name="EEGlrelu_1_1")
        #     net = nn.max_pool1d("EEGmaxpool1d_1_1", net, 8, 8, 8, 8)
        #     net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="EEGdrop_1")
        #     net = nn.conv1d("EEGconv1d_1_3", net, 16, 3, 3, 3, 3)
        #     net = nn.batch_norm("EEGbn_1_3", net, self.is_training)
        #     net = nn.lrelu(net, name="EEGlrelu_1_3")
        #     net = nn.max_pool1d("EEGmaxpool1d_1_2", net, 8, 8, 8, 8)
        #     net = tf.layers.flatten(net, name="EEGflatten_1")
        #
        #     net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="EEGdrop_2")

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
        return net

    def build_EOGcnn(self):
        # first_filter_size = int(self.config["sampling_rate"] / 2.0)
        # # first_filter_stride = int(self.config["sampling_rate"] / 16.0)
        # first_filter_stride = int(self.config["sampling_rate"] / 4.0)
        #
        # with tf.variable_scope("EOGcnn") as scope:
        #     net = nn.conv1d("EOGconv1d_1", self.EOGsignals, 8, 3, 3, 3, 3)
        #     net = nn.batch_norm("EOGbn_1_1", net, self.is_training)
        #     net = nn.lrelu(net, name="EOGlrelu_1_1")
        #     net = nn.max_pool1d("EOGmaxpool1d_1_1", net, 8, 8, 8, 8)
        #     net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="EOGdrop_1")
        #     net = nn.conv1d("EOGconv1d_1_3", net, 16, 3, 3, 3, 3)
        #     net = nn.batch_norm("EOGbn_1_3", net, self.is_training)
        #     net = nn.lrelu(net, name="EOGlrelu_1_3")
        #     net = nn.max_pool1d("EOGmaxpool1d_1_2", net, 8, 8, 8, 8)
        #     net = tf.layers.flatten(net, name="EOGflatten_1")
        #
        #     net = tf.layers.dropout(net, rate=0.5, training=self.is_training, name="EOGdrop_3")

        first_filter_size = int(self.config["sampling_rate"] / 2.0)
        first_filter_stride = int(self.config["sampling_rate"] / 16.0)

        with tf.variable_scope("EOGcnn") as scope:
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

            net = tf.layers.flatten(net, name="EOGflatten_2")

        return net

    # BiLSTM
    def append_bilstm(self, inputs):
        with tf.variable_scope("bilstm") as scope:
            input_dim = inputs.shape[-1].value
            seq_inputs = tf.reshape(inputs, shape=[-1, self.config["seq_length"], input_dim], name="reshape_seq_inputs")

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

            output = tf.concat([outputs[0], outputs[1]], 2)
            final_hidden_state = tf.concat([self.final_state[1][0].c, self.final_state[1][0].h], 1)

            final_hidden_state = tf.expand_dims(final_hidden_state, 2)

            # 计算每个时间步的输出与最后输出状态的相似度
            # [batch_size, step, hidden*2] * [batch_size, hidden*2, 1] = squeeze([batch_size, step, 1]) = [batch_size, step]
            print(np.shape(output))
            attn_weights = tf.squeeze(tf.matmul(output, final_hidden_state), 2)
            # 在时间步维度上进行 softmax 得到权重向量
            soft_attn_weights = tf.nn.softmax(attn_weights, 1)
            # 各时间步输出和对应的权重想成得到上下文矩阵 [batch_size, hidden*2, step] * [batch_size, step, 1] = [batch_size, hidden*2, 1]
            self.context = tf.matmul(tf.transpose(output, [0, 2, 1]), tf.expand_dims(soft_attn_weights, 2))
            print(np.shape(self.context))
            # squeeze([batch_size, hidden*2, 1]) = [batch_size, hidden*2]
            self.context = tf.squeeze(self.context, 2)
            print(np.shape(self.context))
            out = tf.Variable(tf.random_normal([self.config["n_rnn_units"] * 2, self.config["n_classes"]]))
            # output = tf.concat([output, tf.expand_dims(self.context, 1)], axis=1)
            print(np.shape(self.context))

            output = tf.reshape(output, shape=[-1, self.config["n_rnn_units"] * 2], name="reshape_nonseq_input")
            print(np.shape(output))

        return output

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

            # Multiple layers of forward and backward cells
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)
            # Initial states
            self.init_state = multi_cell.zero_state(self.config["batch_size"], tf.float32)

            # Create rnn      tf.nn.dynamic_rnn
            outputs, states = dynamic_rnn(
                cell=multi_cell,
                inputs=seq_inputs,
                initial_state=self.init_state,
                sequence_length=self.seq_lengths,
            )
            logger.info(outputs)
            # Final states
            self.final_state = states

            # Concatenate the output from forward and backward cells
            net = tf.reshape(outputs, shape=[-1, self.config["n_rnn_units"]], name="reshape_nonseq_input")

            # net = tf.layers.dropout(net, rate=0.75, training=self.is_training, name="drop")

        return net

    def train(self, minibatches):
        self.run(self.metric_init_op)
        start = timeit.default_timer()
        preds = []
        trues = []

        if not self.use_rnn:
            for x, y, z, s in minibatches:
                feed_dict = {
                    self.EEGsignals: x,
                    self.EOGsignals: y,
                    self.SleepStages: s,   # 新增睡眠分期结果
                    self.labels: z,
                    self.is_training: True,
                }
                _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)
            preds.extend(outputs["train/preds"])
            trues.extend(z)
        else:
            for x, y, z, s, w, sl, re in minibatches:   # 新增睡眠分期结果  s
                # x = np.reshape(x, (-1, self.config["input_size"], 1, 1))
                # y = np.reshape(y, (-1, self.config["input_size"], 1, 1))
                feed_dict = {
                    self.EEGsignals: x,
                    self.EOGsignals: y,
                    self.SleepStages: s,   # 新增睡眠分期结果
                    self.labels: z,
                    self.is_training: True,
                    self.loss_weights: w,
                    self.seq_lengths: sl,
                }
                #                 if re: #BiLSTM

                #                     # Initialize state of BiLSTM
                #                 # "test/init_state_fw": self.init_state_fw,  # BiLSTM
                #                 # "test/init_state_bw": self.init_state_bw,  # BiLSTM
                #                 # "train/final_state_fw": self.final_state[0],  # BiLSTM
                #                 # "train/final_state_bw": self.final_state[1],  # BiLSTM
                #                     init_state_fw = self.run(self.init_state_fw)
                #                     init_state_bw = self.run(self.init_state_bw)

                #                     # Create a feed_dict for initial states
                #                     # for i, (c_fw, h_fw, c_bw, h_bw) in enumerate(
                #                     #         zip(self.init_state_fw, init_state_fw, self.init_state_bw, init_state_bw)):
                #                     #     feed_dict[c_fw] = init_state_fw[i].c
                #                     #     feed_dict[h_fw] = init_state_fw[i].h
                #                     #     feed_dict[c_bw] = init_state_bw[i].c
                #                     #     feed_dict[h_bw] = init_state_bw[i].h

                #                 for i, (c_fw, h_fw) in enumerate(self.init_state_fw):
                #                      feed_dict[c_fw] = init_state_fw[i].c
                #                      feed_dict[h_fw] = init_state_fw[i].h

                #                 for i, (c_bw, h_bw) in enumerate(self.init_state_bw):
                #                     feed_dict[c_bw] = init_state_bw[i].c
                #                     feed_dict[h_bw] = init_state_bw[i].h

                #                 # Run the training step and get outputs #BiLSTM

                #                 _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)

                #                 # Buffer the final states #BiLSTM
                #                 final_state_fw = outputs["train/final_state_fw"]
                #                 final_state_bw = outputs["train/final_state_bw"]

                if re:
                    # Initialize state of RNN
                    state = self.run(self.init_state)
                #
                # Carry the states from the previous batches through time
                for i, (c, h) in enumerate(self.init_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

                _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)
                # Buffer the final states
                state = outputs["train/final_state"]

                tmp_preds = np.reshape(outputs["train/preds"], (self.config["batch_size"], self.config["seq_length"]))
                tmp_trues = np.reshape(z, (self.config["batch_size"], self.config["seq_length"]))

                for i in range(self.config["batch_size"]):
                    preds.extend(tmp_preds[i, :sl[i]])
                    trues.extend(tmp_trues[i, :sl[i]])

        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1])  ##改动
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

        return outputs

    def evaluate(self, minibatches):
        start = timeit.default_timer()
        losses = []
        preds = []
        trues = []

        if not self.use_rnn:
            for x, y, z, s in minibatches:  # 新增睡眠分期结果
                feed_dict = {
                    self.EEGsignals: x,
                    self.EOGsignals: y,
                    self.SleepStages: s,  # 新增睡眠分期结果
                    self.labels: z,
                    self.is_training: False,
                }

                outputs = self.run(self.test_outputs, feed_dict=feed_dict)

                losses.append(outputs["test/loss"])
                preds.extend(outputs["test/preds"])
                trues.extend(z)
        else:
            for x, y, z, s, w, sl, re in minibatches:     # 新增睡眠分期结果
                feed_dict = {
                    self.EEGsignals: x,
                    self.EOGsignals: y,
                    self.SleepStages: s,  # 新增睡眠分期结果
                    self.labels: z,
                    self.is_training: False,
                    self.loss_weights: w,
                    self.seq_lengths: sl,
                }

                if re:
                    # BiLSTM
                    # init_state_fw = self.run(self.init_state_fw)
                    # init_state_bw = self.run(self.init_state_bw)

                    # Create a feed_dict for initial states
                    #                 for i, (c_fw, h_fw) in enumerate(self.init_state_fw):
                    #                     feed_dict[c_fw] = init_state_fw[i].c
                    #                     feed_dict[h_fw] = init_state_fw[i].h

                    #                 for i, (c_bw, h_bw) in enumerate(self.init_state_bw):
                    #                     feed_dict[c_bw] = init_state_bw[i].c
                    #                     feed_dict[h_bw] = init_state_bw[i].h

                    # Initialize state of RNN
                    state = self.run(self.init_state)

                # Carry the states from the previous batches through time
                for i, (c, h) in enumerate(self.init_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

                outputs = self.run(self.test_outputs, feed_dict=feed_dict)

                # Buffer the final states
                # BiLSTM
                # state = outputs["test/final_state"]
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
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1])  ##改动
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
        list_vars = [  ##变量
            "EOGcnn/EOGconv1d_1/conv2d/kernel:0",
            "EOGcnn/EOGconv1d_2_1/conv2d/kernel:0",
            "EOGcnn/EOGconv1d_2_2/conv2d/kernel:0",
            "EOGcnn/EOGconv1d_2_3/conv2d/kernel:0",
            "EEGcnn/EEGconv1d_1/conv2d/kernel:0",
            "EEGcnn/EEGconv1d_2_1/conv2d/kernel:0",
            "EEGcnn/EEGconv1d_2_2/conv2d/kernel:0",
            "EEGcnn/EEGconv1d_2_3/conv2d/kernel:0",
            "lastconv/conv2d/kernel:0",
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

