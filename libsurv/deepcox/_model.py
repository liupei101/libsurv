import os
import tensorflow as tf

from . import _check_config, _batch_gen, _check_surv_data, _safe_mkdir

class model(object):
    """docstring for model"""
    def __init__(self, data, batch_size, input_nodes, hidden_layers_nodes, config={}):
        """
        DeepCox Neural Network Class Constructor.

        Parameters
        ----------
        data: dict
            Survival dataset follows the format {"X": DataFrame, "Y": DataFrame}.
            It's suggested that you utilize `libsurv.datasets.survival_df` to obtain 
            the DataFrame object and then construct the target dict.
        batch_size: int
            The number of samples used in each iteration of training.
        input_nodes: int
            The number of input nodes. It's also equal to the number of features.
        hidden_layers_nodes:
            Number of nodes in hidden layers of neural network.
        config: dict
            Some configurations or hyper-parameters of neural network.
            Defalt settings is below:
            config = {
                "learning_rate": 0.001,
                "learning_rate_decay": 1.0,
                "activation": "tanh",
                "L2_reg": 0.0,
                "L1_reg": 0.0,
                "optimizer": "sgd",
                "dropout_keep_prob": 1.0,
                "seed": 42
            }
        """
        super(model, self).__init__()
        # dataset
        _check_surv_data(data)
        self.raw_data = data
        self.batch_size = batch_size
        # neural nodes
        self.input_nodes = input_nodes
        self.hidden_layers_nodes = hidden_layers_nodes
        # network hyper-parameters
        _check_config(config)
        self.config = config
        # graph level random seed
        tf.set_random_seed(config["seed"])
        # some gobal settings
        self.global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)
        self.keep_prob = tf.placeholder(tf.float32)

    def _gen(self):
        yield from _batch_gen(self.raw_data, self.batch_size)

    def _import_data(self):
        self.dataset = tf.data.Dataset.from_generator(
            self._gen, 
            (tf.float32, tf.float32), 
            (tf.TensorShape([self.batch_size, self.input_nodes]), 
             tf.TensorShape([self.batch_size]))
        )
        with tf.name_scope("data"):
            self.iterator = self.dataset.make_initializable_iterator()
            # self.X with shape of (batch_size, m), m = input_nodes
            # self.Y with shape of (batch_size, )
            self.X, self.Y = self.iterator.get_next()

    def _create_fc_layer(self, x, output_dim, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            w = tf.get_variable('weights', [x.shape[1], output_dim], 
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )

            b = tf.get_variable('biases', [output_dim], 
                initializer=tf.constant_initializer(0.0)
            )

            # add weights and bias to collections
            tf.add_to_collection("var_weight", w)
            tf.add_to_collection("var_bias", b)

            layer_out = tf.nn.dropout(tf.matmul(x, w) + b, self.keep_prob)

            if activation == 'relu':
                layer_out = tf.nn.relu(layer_out)
            elif activation == 'sigmoid':
                layer_out = tf.nn.sigmoid(layer_out)
            elif activation == 'tanh':
                layer_out = tf.nn.tanh(layer_out)
            else:
                raise NotImplementedError('activation not recognized')

            return layer_out

    def _create_network(self):
        """
        Define the neural network that only includes FC layers.
        """
        with tf.name_scope("hidden_layers"):
            cur_x = self.X
            for i, num_nodes in enumerate(self.hidden_layers_nodes):
                cur_x = self._create_fc_layer(cur_x, num_nodes, "layer"+str(i+1))
            # output of network
            # squeeze output from shape of (batch_size, 1) to (batch_size, )
            self.Y_hat = tf.squeeze(cur_x)

    def _create_loss(self):
        """
        Define the loss function.

        Notes
        -----
        The loss function definded here is negative log of Breslow Approximation partial 
        likelihood function. See more in "Breslow N., 'Covariance analysis of censored 
        survival data, ' Biometrics 30.1(1974):89-99.".
        """
        with tf.name_scope("loss"):
            # Obtain T and E from self.Y
            # NOTE: negtive value means E = 0
            Y_label_T = tf.abs(self.Y)
            Y_label_E = tf.cast(tf.greater(self.Y, 0), dtype=tf.float32)

            Y_hat_hr = tf.exp(self.Y_hat)
            Y_hat_cumsum = tf.log(tf.cumsum(Y_hat_hr))
            
            # Start Computation of Loss function

            # Get Segment from T
            unique_values, segment_ids = tf.unique(Y_label_T)
            # Get Segment_max
            loss_s2_v = tf.segment_max(Y_hat_cumsum, segment_ids)
            # Get Segment_count
            loss_s2_count = tf.segment_sum(Y_label_E, segment_ids)
            # Compute S2
            loss_s2 = tf.reduce_sum(tf.multiply(loss_s2_v, loss_s2_count))
            # Compute S1
            loss_s1 = tf.reduce_sum(tf.multiply(Y_hat, Y_label_E))
            # Compute Breslow Loss
            loss_breslow = tf.subtract(loss_s2, loss_s1)

            # Compute Regularization Term Loss
            reg_item = tf.contrib.layers.l1_l2_regularizer(self.config["L1_reg"], self.config["L2_reg"])
            loss_reg = tf.contrib.layers.apply_regularization(reg_item, tf.get_collection("var_weight"))

            # Loss function = Breslow Function + Regularization Term
            self.loss = tf.add(loss_breslow, loss_reg)

    def _create_optimizer(self):
        """
        Define optimizer
        """
        # SGD Optimizer
        if self.config["optimizer"] == 'sgd':
            lr = tf.train.exponential_decay(
                self.config["learning_rate"],
                self.global_step,
                1,
                self.config["learning_rate_decay"]
            )
            self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self.loss, global_step=self.global_step)
        # Adam Optimizer
        elif self.config["optimizer"] == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.config["learning_rate"]).minimize(self.loss, global_step=self.global_step)
        elif self.config["optimizer"] == 'rms':
            self.optimizer = tf.train.RMSPropOptimizer(self.config["learning_rate"]).minimize(self.loss, global_step=self.global_step)     
        else:
            raise NotImplementedError('Optimizer not recognized')

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """Build graph of DeepCox
        """
        self._import_data()
        self._create_network()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def train(self, num_steps, num_skip_steps):
        """
        Training DeepCox model.

        Parameters
        ----------
        num_steps: int
            The number of training steps.
        num_skip_steps: int
            The number of skipping training steps. Model would be saved after 
            each `num_skip_steps`.

        Notes
        -----
        This method will write graph definition to the directory `./graph` and save model 
        to the directory `./checkpoints`. Please use `tensorboard` tools to watch the model.
        """
        saver = tf.train.Saver() # defaults to saving all variables

        initial_step = 0
        _safe_mkdir('checkpoints')
        # Session Running 
        with tf.Session() as sess:
            sess.run(self.iterator.initializer)
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
            writer = tf.summary.FileWriter('graphs/lr' + str(self.config["learning_rate"]), sess.graph)
            # Get current global step
            initial_step = self.global_step.eval()

            for index in range(initial_step, initial_step + num_steps):
                try:
                    loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op]
                                                      feed_dict={self.keep_prob: self.config['dropout_keep_prob']})
                    writer.add_summary(summary, global_step=index)
                    total_loss += loss_batch
                    if (index + 1) % num_skip_steps == 0:
                        print('Average loss at step {}: {:5.1f}'.format(index, total_loss / num_skip_steps))
                        total_loss = 0.0
                        saver.save(sess, 'checkpoints/deepcox', index)
                except tf.errors.OutOfRangeError:
                    sess.run(self.iterator.initializer)
            writer.close()
            