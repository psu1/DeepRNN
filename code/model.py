from __future__ import print_function

import collections

import tensorflow as tf
import numpy as np

import process
import utils

TrainForwardOutputTuple = collections.namedtuple("TrainForwardOutputTuplee",
                                                 ("train_summary", "train_loss",  # need to change to train_forward_loss
                                                  "predict_beat_count",
                                                  "global_step", "batch_size", "grad_norm",
                                                  "learning_rate", "train_src_ref", "train_tgt_ref",
                                                  "train_src2tgt_output", "train_tgt2src_output",
                                                  "src_seq_len", "tgt_seq_len"))

TrainBackwardOutputTuple = collections.namedtuple("TrainBackwardOutputTuplee",
                                                  ("train_summary", "train_backward_loss",
                                                   "global_step", "batch_size", "grad_norm",
                                                   "learning_rate", "train_src_ref", "train_tgt_ref",
                                                   "train_src2tgt2src_output", "train_tgt2src2tgt_output"))

EvalOutputTuple = collections.namedtuple("EvalOutputTuple", ("eval_loss", "predict_beat_count", "batch_size"))

InferOutputTuple = collections.namedtuple("InferOutputTuple",
                                          ("infer_tgt_output", "infer_tgt_ref", "infer_loss", "predict_beat_count",
                                           "batch_size"))

class Model(object):
    """
    This class implements a multi-layer recurrent neural network as "lstm"
    """

    def __init__(self, hparams, iterator, mode):
        # Set params
        self.set_params(hparams, iterator, mode)

        # build graph
        self.build_graph(hparams)

    def set_params(self, hparams, iterator, mode):
        """Set various params for self and initialize."""
        self.architecture = hparams.architecture
        assert hparams.src_len == hparams.tgt_len
        self.length = hparams.src_len
        self.iterator = iterator
        self.mode = mode
        self.dtype = tf.float64
        self.unit_type = hparams.unit_type
        self.num_units = hparams.num_units  ## this is the size in each cell
        self.forget_bias = hparams.forget_bias
        self.dropout = hparams.dropout
        self.src_feature_size = hparams.src_feature_size
        self.tgt_feature_size = hparams.tgt_feature_size
        if self.architecture == "deepRNN":
            self.num_bi_layers = hparams.num_bi_layers
            self.num_uni_layers = hparams.num_uni_layers

        # Set num layers
        self.num_layers=hparams.num_layers
        # Set num residual layers
        self.num_residual_layers = hparams.num_residual_layers

        # Batch size
        self.batch_size = hparams.batch_size
        self.batch_size_tensor = tf.constant(self.batch_size)
        # train directions
        self.forward_directions = hparams.forward_directions
        self.backward_directions = hparams.backward_directions

        # Global step
        self.global_step = tf.Variable(0, trainable=False)
        self.epoch_num = tf.Variable(0, trainable=False)

    """--------------------------"""

    def build_graph(self, hparams):
        utils.print_out("# Creating %s graph ..." % self.mode)

        with tf.variable_scope("network", dtype=self.dtype, reuse=tf.AUTO_REUSE):
            self.top_scope = tf.get_variable_scope()

            # Initializer
            initializer = tf.random_uniform_initializer(-hparams.init_weight, hparams.init_weight,
                                                        seed=hparams.random_seed)
            self.top_scope.set_initializer(initializer)
            ## so the initializer will be the default initializer for the following variable in this variable scope
            "---------"
            # the variable scope specification is left for _encode function
            # common components in three mode
            if self.architecture == "deepRNN":
                # lstm (bi_lstm, then stacked with uni_lstm)
                self.src_bi_lstm, self.src_bi_lstm_condition = self._build_bi_lstm(hparams)
                self.src_uni_lstm, self.src_uni_lstm_condition = self._build_uni_lstm(hparams)
                self.tgt_bi_lstm, self.tgt_bi_lstm_condition = self._build_bi_lstm(hparams)
                self.tgt_uni_lstm, self.tgt_uni_lstm_condition = self._build_uni_lstm(hparams)

                # Projector
                self.src_projector = self._build_projector(hparams, field='src')
                self.tgt_projector = self._build_projector(hparams, field='tgt')
            else:
                raise ValueError("Unknown architecture_type %s" % hparams.lstm_type)

            "------------"

            # set mode-specific component
            self.set_mode_phase(hparams)

            # Saver
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def _single_cell(self, unit_type, num_units, forget_bias, dropout, mode, residual_connection=False):
        """Create an instance of a single RNN cell."""
        # dropout (= 1 - keep_prob) is set to 0 during eval and infer
        dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
        # Cell Type
        if unit_type == "lstm":
            utils.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
            single_cell = tf.nn.rnn_cell.LSTMCell(
                num_units,
                forget_bias=forget_bias)
        else:
            raise ValueError("Unknown unit type %s!" % unit_type)
        # Dropout (= 1 - keep_prob)
        if dropout > 0.0:
            single_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=single_cell, input_keep_prob=(1.0 - dropout))
            utils.print_out("  %s, dropout=%g " % (type(single_cell).__name__, dropout), new_line=False)
        # Residual
        if residual_connection:
            single_cell = tf.nn.rnn_cell.ResidualWrapper(single_cell)
            utils.print_out("  %s" % type(single_cell).__name__, new_line=False)

        return single_cell

    def _build_cell(self, num_layers, num_residual_layers):
        cell_list = []
        for i in range(num_layers):
            utils.print_out("  cell %d " % i, new_line=False)
            single_cell = self._single_cell(
                unit_type=self.unit_type,
                num_units=self.num_units,
                forget_bias=self.forget_bias,
                dropout=self.dropout,
                mode=self.mode,
                residual_connection=(i >= (num_layers - num_residual_layers)))
            utils.print_out("", new_line=True)
            cell_list.append(single_cell)

        if len(cell_list) == 1:  # Single layer.
            return cell_list[0]
        else:  # Multi layers
            return tf.nn.rnn_cell.MultiRNNCell(cell_list)

    def _build_bi_lstm(self, hparams):
        utils.print_out("# Build bidirectional lstm")
        num_bi_layers = self.num_bi_layers
        num_bi_residual_layers = 0
        utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" % (num_bi_layers, num_bi_residual_layers))
        # Construct forward and backward cells
        fw_cell = self._build_cell(num_bi_layers, num_bi_residual_layers)
        bw_cell = self._build_cell(num_bi_layers, num_bi_residual_layers)
        bi_lstm = (fw_cell, bw_cell)
        bi_lstm_condition = ("bi", num_bi_layers)
        return bi_lstm, bi_lstm_condition

    def _build_uni_lstm(self, hparams):
        utils.print_out("# Build unidirectional lstm")
        num_uni_layers = self.num_uni_layers
        num_uni_residual_layers = self.num_uni_layers - 1
        utils.print_out("  num_layers = %d, num_residual_layers=%d" % (num_uni_layers, num_uni_residual_layers))
        cell = self._build_cell(num_uni_layers, num_uni_residual_layers)
        uni_lstm = cell
        uni_lstm_condition = ("uni", None)
        return uni_lstm, uni_lstm_condition

    def _build_projector(self, hparams, field=None):
        assert field
        if field == "src":
            projector = tf.layers.Dense(self.src_feature_size, use_bias=True)
        elif field == "tgt":
            projector = tf.layers.Dense(self.tgt_feature_size, use_bias=True)
        else:
            raise ValueError("Unknown field type %s" % field)
        return projector

    """----------------------"""

    def _set_input_ref(self, direction, lstm_input_given, ref_given, seq_len_given):
        if direction == "src2tgt":
            lstm_input = self.iterator.source_ref
            ref = self.iterator.target_ref
            seq_len = self.iterator.target_sequence_length
        return lstm_input, ref, seq_len

    def _set_lstm_projector(self, direction):
        # lstm type is related to the input, while the projector used is related to output!!!
        if direction in ["src2tgt"]:
            bi_lstm, bi_lstm_condition = self.src_bi_lstm, self.src_bi_lstm_condition
            uni_lstm, uni_lstm_condition = self.src_uni_lstm, self.src_uni_lstm_condition
            lstm_scope = "src_lstm"

        lstm_list = [bi_lstm, uni_lstm]  # bi lstm then stacked with uni lstm
        lstm_condition_list = [bi_lstm_condition, uni_lstm_condition]

        if direction in ["src2tgt"]:
            projector = self.tgt_projector
            projector_scope = "tgt_projector"
        return lstm_list, lstm_condition_list, lstm_scope, projector, projector_scope

    def _encode(self, lstm_scope, input, seq_len, lstm_list, lstm_condition_list):
        # here input is rank-3, [batch size, seq_len, src_feature_size]
        assert len(lstm_list) == len(lstm_condition_list)
        lstm_outputs = None
        with tf.variable_scope(lstm_scope):
            for i in range(len(lstm_list)):
                if i == 0:
                    input = input
                else:
                    input = lstm_outputs
                lstm = lstm_list[i]
                lstm_condition = lstm_condition_list[i]
                lstm_type = lstm_condition[0]
                if lstm_type == "uni":
                    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                        cell=lstm, inputs=input, dtype=self.dtype, sequence_length=seq_len, swap_memory=True, scope="")
                elif lstm_type == "bi":
                    fw_cell, bw_cell = lstm
                    bi_outputs, bi_lstm_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input,
                                                                                dtype=self.dtype,
                                                                                sequence_length=seq_len,
                                                                                swap_memory=True, scope="")
                    lstm_outputs = tf.concat(bi_outputs, -1)
                    num_bi_layers = lstm_condition[1]
                    if num_bi_layers == 1:
                        lstm_state = bi_lstm_state
                    else:
                        # alternatively concat forward and backward states
                        lstm_state = []
                        for layer_id in range(num_bi_layers):
                            lstm_state.append(bi_lstm_state[0][layer_id])  # forward
                            lstm_state.append(bi_lstm_state[1][layer_id])  # backward
                        lstm_state = tuple(lstm_state)
                else:
                    raise ValueError("Unknown lstm or lstm_condition")

        return lstm_outputs, lstm_state

    def compute_loss(self, hparams, direction, lstm_input_given, ref_given, seq_len_given, feature_size):

        lstm_input, ref, seq_len = self._set_input_ref(direction, lstm_input_given, ref_given, seq_len_given)
        lstm_list, lstm_condition_list, lstm_scope, projector, projector_scope = self._set_lstm_projector(direction)
        lstm_output, lstm_state = self._encode(lstm_scope, lstm_input, seq_len, lstm_list, lstm_condition_list)
        output = projector(lstm_output)
        loss = tf.sqrt(
            tf.reduce_sum(tf.square(output - ref)) / (tf.to_double(self.batch_size * hparams.src_len * feature_size)))
        return loss, output

    def forward(self, hparams, directions=None):
        src2tgt_loss = 0
        src2tgt_output = None
        ## forward directions just include ["src2tgt", "tgt2src", "src2src", "tgt2tgt"] at most

        if "src2tgt" in directions:
            src2tgt_loss, src2tgt_output = self.compute_loss(hparams, direction="src2tgt", lstm_input_given=None,
                                                             ref_given=None, seq_len_given=None,
        forward_loss = src2tgt_loss
        return forward_loss, src2tgt_output

    def set_mode_phase(self, hparams):
        self.predict_beat_count = tf.math.reduce_sum(self.iterator.target_sequence_length)  # need modification
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:

            # set optimizer
            self.learning_rate = tf.constant(hparams.learning_rate)
            if hparams.optimizer == "adam":
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            else:
                raise ValueError("Unknown optimizer type %s" % hparams.optimizer)
            "-------------"
            # forward phase
            # self.train_forward_loss, self.src2tgt_output, self.tgt2src_output = self.forward(
            #    hparams, directions=["src2tgt", "tgt2src", "src2src", "tgt2tgt"])
            self.train_forward_loss, self.src2tgt_output, self.tgt2src_output = self.forward(
                hparams, directions=self.forward_directions)  # this is for test of basic unidirection seq2seq model

            # test variable
            utils.test_trainable_variables()

            # Gradients
            params = tf.trainable_variables()
            forward_gradients = tf.gradients(self.train_forward_loss, params)
            forward_clipped_grads, self.forward_grad_norm = tf.clip_by_global_norm(forward_gradients, 5.0)

            # key point: to update the parameter
            self.forward_update = self.optimizer.apply_gradients(zip(forward_clipped_grads, params),
                                                                 global_step=self.global_step)
            ## in this step, global step will increase by 1!!

            # Summary
            self.forward_train_summary = tf.summary.merge(
                [tf.summary.scalar("lr", self.learning_rate),
                 tf.summary.scalar("forward_train_loss", self.train_forward_loss),
                 tf.summary.scalar("forward_grad_norm", self.forward_grad_norm),
                 tf.summary.scalar("forward_clipped_gradient", tf.global_norm(forward_clipped_grads))])
            "---------------"

            # backward phase
            self.src_ref_given = tf.placeholder(self.dtype, shape=(self.batch_size, self.length, self.src_feature_size))
            self.tgt_ref_given = tf.placeholder(self.dtype, shape=(self.batch_size, self.length, self.tgt_feature_size))
            self.src2tgt_output_given = tf.placeholder(self.dtype,
                                                       shape=(self.batch_size, self.length, self.tgt_feature_size))
            self.tgt2src_output_given = tf.placeholder(self.dtype,
                                                       shape=(self.batch_size, self.length, self.src_feature_size))
            """
            self.src_sos_given = tf.placeholder(self.dtype, shape=(1, self.src_feature_size))
            self.tgt_sos_given = tf.placeholder(self.dtype, shape=(1, self.tgt_feature_size))
            """
            self.src_seq_len_given = tf.placeholder(tf.int32, shape=(self.batch_size))
            self.tgt_seq_len_given = tf.placeholder(tf.int32, shape=(self.batch_size))

            self.train_backward_loss, self.src2tgt2src_output, self.tgt2src2tgt_output = self.backward(
                hparams, directions=self.backward_directions,
                src2tgt_output=self.src2tgt_output_given, tgt2src_output=self.tgt2src_output_given,
                src_ref=self.src_ref_given, tgt_ref=self.tgt_ref_given,
                src_seq_len=self.src_seq_len_given, tgt_seq_len=self.tgt_seq_len_given)

            # test variable
            utils.test_trainable_variables()

            # Gradients
            params = tf.trainable_variables()
            backward_gradients = tf.gradients(self.train_backward_loss, params)
            backward_clipped_grads, self.backward_grad_norm = tf.clip_by_global_norm(backward_gradients, 5.0)

            # key point: to update the parameter
            self.backward_update = self.optimizer.apply_gradients(zip(backward_clipped_grads, params))
            # here we doesn't list the global step to avoid increment by 1

            # Summary
            self.backward_train_summary = tf.summary.merge(
                [tf.summary.scalar("lr", self.learning_rate),
                 tf.summary.scalar("backward_train_loss", self.train_backward_loss),
                 tf.summary.scalar("backward_grad_norm", self.backward_grad_norm),
                 tf.summary.scalar("backward_clipped_gradient", tf.global_norm(backward_clipped_grads))])

        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss, _, _ = self.forward(hparams, directions=["src2tgt"])

        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_loss, self.infer_target_output, _ = self.forward(hparams, directions=["src2tgt"])

        """"------------------------"""

    def increase_epoch_num(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run(tf.assign_add(self.epoch_num, 1))

    def train_forward(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        output_tuple = TrainForwardOutputTuple(train_summary=self.forward_train_summary,
                                               train_loss=self.train_forward_loss,
                                               predict_beat_count=self.predict_beat_count,
                                               global_step=self.global_step,
                                               batch_size=self.batch_size_tensor,
                                               grad_norm=self.forward_grad_norm,
                                               learning_rate=self.learning_rate,
                                               train_src_ref=self.iterator.source_ref,
                                               train_tgt_ref=self.iterator.target_ref,
                                               train_src2tgt_output=self.src2tgt_output,
                                               src_seq_len=self.iterator.source_sequence_length,
                                               tgt_seq_len=self.iterator.target_sequence_length)
        return sess.run([self.forward_update, output_tuple])

    def train_backward(self, sess, input):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        src_ref, tgt_ref, src2tgt_output, tgt2src_output, src_seq_len, tgt_seq_len = input
        feed_dict = {self.src_ref_given: src_ref, self.tgt_ref_given: tgt_ref,
                     self.src2tgt_output_given: src2tgt_output, self.tgt2src_output_given: tgt2src_output,
                     self.src_seq_len_given: src_seq_len, self.tgt_seq_len_given: tgt_seq_len}
        output_tuple = TrainBackwardOutputTuple(train_summary=self.backward_train_summary,
                                                train_backward_loss=self.train_backward_loss,
                                                global_step=self.global_step,
                                                batch_size=self.batch_size_tensor,
                                                grad_norm=self.backward_grad_norm,
                                                learning_rate=self.learning_rate,
                                                train_src_ref=self.src_ref_given,
                                                train_tgt_ref=self.tgt_ref_given)

        return sess.run([self.backward_update, output_tuple], feed_dict=feed_dict)

    def eval(self, sess):
        """Execute eval graph."""
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        output_tuple = EvalOutputTuple(eval_loss=self.eval_loss,
                                       predict_beat_count=self.predict_beat_count,
                                       batch_size=self.batch_size_tensor)
        return sess.run(output_tuple)

    def infer(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        output_tuple = InferOutputTuple(infer_tgt_output=self.infer_target_output,
                                        infer_tgt_ref=self.iterator.target_ref,
                                        infer_loss=self.infer_loss,
                                        predict_beat_count=self.predict_beat_count,
                                        batch_size=self.batch_size_tensor)
        return sess.run(output_tuple)
