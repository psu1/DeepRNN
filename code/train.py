from __future__ import print_function

import argparse
import os
import codecs
import json

import tensorflow as tf

import utils
import process
import preprocess


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")
    """
    # for ordinary lstm
    parser.add_argument("--num_layers", type=int, default=4, help="number of layers")
    parser.add_argument("--num_residual_layers", type=int, default=2,
                        help="number of residual layers of encoder, should be not greater than the number of encoder layers.")
    parser.add_argument("--lstm_type", type=str, default="uni", help="\
              uni | bi | gnmt.
              For bi, we build num_encoder_layers/2 bi-directional layers.
              For gnmt, we build 1 bi-directional layer, and (num_encoder_layers - 1)
                uni-directional layers.\
              ")
    """

    # network
    parser.add_argument("--architecture", type=str, default="deepRNN", help="""deepRNN""")
    ## layer(for peng's model)
    parser.add_argument("--num_bi_layers", type=int, default=1,
                        help="number of bidirectional layers, 1 layer acually represent 2 directions")
    parser.add_argument("--num_uni_layers", type=int, default=3,
                        help="number of uni layers stacked on the bi layers, "
                             "with uni_residual_layer=uni_layers-1 setting as default .")
    ## cell
    parser.add_argument("--unit_type", type=str, default="lstm", help="lstm")
    parser.add_argument("--num_units", type=int, default=128, help="the hidden size of each cell.")
    parser.add_argument("--forget_bias", type=float, default=1.0,
                        help="Forget bias for BasicLSTMCell.")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout rate (not keep_prob)")
    ## initializer (uniform initializer as default)
    parser.add_argument("--init_weight", type=float, default=0.1,
                        help=("for uniform init_op, initialize weights "
                              "between [-this, this]."))
    ## optimizer
    parser.add_argument("--optimizer", type=str, default="adam", help="the type of optimizer")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate. Adam: 0.001 | 0.0001")

    # data
    ## data source
    parser.add_argument('--data_source_dir', type=str,
                        default='/path/to/data/',
                        help='the path to load training data')

    ## data detail
    parser.add_argument("--src_len", type=int, default=64,
                        help="length of src sequences during training. "
                             "the input are all in same length")
    parser.add_argument("--tgt_len", type=int, default=64,
                        help="length of tgt sequences during training.")
    parser.add_argument("--src_feature_size", type=int, default=7,
                        help="feature size of src")
    parser.add_argument("--tgt_feature_size", type=int, default=3,
                        help="feature size of tgt")

    # expriment
    parser.add_argument("--exprs_dir", type=str,
                        default='/path/to/save/experiments',
                        help="the dir to store experiment.")
    parser.add_argument("--expr_name", type=str,
                        default='deepRNN',
                        help='the name of dir to store config, model, data, figure, result')
    parser.add_argument('--train_ratio', default=0.7, type=float,
                        help='the ratio of data for trainning')
    parser.add_argument('--eval_ratio', default=0.1, type=float,
                        help='the ratio of data for validation')
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--random_seed", type=int, default=1234,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--num_train_steps", type=int, default=100000, help="Num steps to train.")
    parser.add_argument('--backward_enable', default=False, type="bool",
                        help="if enable the backward training")
    parser.add_argument('--forward_directions', default=["src2tgt"], type=list,
                        help="the directions for forward phase of traning"
                             " can only be 'src2tgt'")
    parser.add_argument("--steps_per_stats", type=int, default=1,
                        help=("How many training steps to do per stats logging."
                              "Save checkpoint every epoch"))


def create_dirs(hparams):
    """create directories"""
    assert hparams.exprs_dir
    exprs_dir = hparams.exprs_dir
    if not tf.gfile.Exists(exprs_dir):
        tf.gfile.MakeDirs(exprs_dir)
    utils.print_out("# experiments directory: %s " % exprs_dir)
    expr_dir = os.path.join(exprs_dir, hparams.expr_name)
    if not tf.gfile.Exists(expr_dir):
        tf.gfile.MakeDirs(expr_dir)
    utils.print_out("# -experiment directory: %s " % expr_dir)
    config_dir = os.path.join(expr_dir, 'config')
    if not tf.gfile.Exists(config_dir):
        tf.gfile.MakeDirs(config_dir)
    utils.print_out("# --config directory: %s " % config_dir)
    log_dir = os.path.join(expr_dir, 'log')
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
    utils.print_out("# --log directory: %s " % log_dir)
    data_dir = os.path.join(expr_dir, 'data')
    if not tf.gfile.Exists(data_dir):
        tf.gfile.MakeDirs(data_dir)
    utils.print_out("# --data directory: %s " % data_dir)
    model_dir = os.path.join(expr_dir, 'model')
    if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)
    utils.print_out("# --model directory: %s " % model_dir)
    figure_dir = os.path.join(expr_dir, 'figure')
    if not tf.gfile.Exists(figure_dir):
        tf.gfile.MakeDirs(figure_dir)
    utils.print_out("# --figure directory: %s " % figure_dir)
    result_dir = os.path.join(expr_dir, 'result')
    if not tf.gfile.Exists(result_dir):
        tf.gfile.MakeDirs(result_dir)
    utils.print_out("# --result directory: %s " % result_dir)
    return expr_dir, config_dir, log_dir, data_dir, model_dir, figure_dir, result_dir


def check_and_save_hparams(out_dir, hparams):
    """Save hparams."""
    hparams_file = os.path.join(out_dir, "hparams")
    if tf.gfile.Exists(hparams_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
            origin_hparams = json.load(f)
            origin_hparams = tf.contrib.training.HParams(**origin_hparams)
        wrong_keys = []
        keys = set(list(hparams.values().keys()) + (list(origin_hparams.values().keys())))
        for key in keys:
            if (hparams.values().get(key, None) != origin_hparams.values().get(key, None) or
                    hparams.values().get(key, None) == None or
                    hparams.values().get(key, None) == None):
                wrong_keys.append(key)
        try:
            assert origin_hparams.values() == hparams.values()
            utils.print_out("using the same hparams of old %s" % hparams_file)
        except:
            utils.print_out('new hparams not the same with the existed one')
            for wrong_key in wrong_keys:
                utils.print_out(" keys: %s, \norigin_value: %s, \nnew_value: %s\n" % (
                wrong_key, origin_hparams.values()[wrong_key], hparams.values()[wrong_key]))
            raise ValueError
    else:
        utils.print_out("  not old hparams found, create new hparams to %s" % hparams_file)
        with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
            f.write(hparams.to_json(indent=4))


def main():
    parser = argparse.ArgumentParser(description='Deep BiLSTM with Residual')
    add_arguments(parser)
    args = parser.parse_args()
    print(args)
    hparams = tf.contrib.training.HParams(**vars(args))
    # check GPU device
    utils.print_out("# Devices visible to TensorFlow: %s" % repr(tf.Session().list_devices()))
    #  create dirs
    expr_dir, config_dir, log_dir, data_dir, model_dir, figure_dir, result_dir = create_dirs(hparams)
    # save hyperameter
    check_and_save_hparams(config_dir, hparams)

    stage = 'test'  # preprocess','train_eval', or 'test'
    assert stage in [, 'train_eval', 'test'], 'stage not recognized'
    utils.print_out('stage: %s' % stage)
    # if stage == 'preprocess':
    #     preprocess.preprocess(hparams, data_dir)
        # the data are stored in the data_dir for the training step
    if stage == 'train_eval':
        process.train_eval(hparams, data_dir, model_dir, log_dir)
    if stage == 'test':
        process.infer(hparams, data_dir, model_dir, result_dir)


if __name__ == '__main__':
    main()
