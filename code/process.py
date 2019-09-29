from __future__ import print_function

import collections
import time
import os

import numpy as np
import tensorflow as tf
import scipy.io as scio

import model
import utils

BatchedInput = collections.namedtuple("BatchedInput",
                                      ("initializer", "source_ref", "target_ref",
                                       "source_sequence_length", "target_sequence_length"))


def get_iterator(src_dataset, tgt_dataset, batch_size, random_seed, is_train):
    output_buffer_size = batch_size * 1000
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    # just try not to use random seed
    if is_train:
        src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size,# random_seed,
                                                  reshuffle_each_iteration=True)
    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src_ref, tgt_ref: tf.logical_and(tf.size(src_ref) > 0, tf.size(tgt_ref) > 0))

    # Add in sequence lengths.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src_ref, tgt_ref: (src_ref, tgt_ref, tf.shape(src_ref)[0], tf.shape(tgt_ref)[0]))

    batched_dataset = src_tgt_dataset.batch(batch_size)

    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ref, tgt_ref, src_seq_len, tgt_seq_len) = (batched_iter.get_next())
    return BatchedInput(
        initializer=batched_iter.initializer,
        source_ref=src_ref,  ## this should be rank-3
        target_ref=tgt_ref,  ## this should be rank-3
        source_sequence_length=src_seq_len,  ## this should be rank-1
        target_sequence_length=tgt_seq_len)  ## this should be rank-1


def get_model_creator(hparams):
    """Get the right model class depending on configuration."""
    if hparams.architecture == 'peng':
        model_creator = model.Model
    """vanilla lstm, seq2seq"""
    return model_creator


def load_data_mean_std(hparams, data_dir):
    if hparams.normalize_principle == "all":
        data_mean = scio.loadmat(os.path.join(data_dir, 'all_data_mean'))['data']
        data_std = scio.loadmat(os.path.join(data_dir, 'all_data_std'))['data']
    elif hparams.normalize_principle == "train":
        data_mean = scio.loadmat(os.path.join(data_dir, 'train_data_mean'))['data']
        data_std = scio.loadmat(os.path.join(data_dir, 'train_data_std'))['data']
    return data_mean, data_std


TrainModel = collections.namedtuple("TrainModel", ("graph", "model", "iterator"))


def create_train_model(model_creator, hparams, data_dir):
    """Create train graph, model, and iterator."""
    train_data_path = []
    for root, _, name in os.walk(os.path.join(data_dir, 'train_data')):
        for x in name:
            if x.split('.')[-1] == 'mat':
                train_data_path.append(os.path.join(root, x))
    assert len(train_data_path) == 1
    train_data = scio.loadmat(*train_data_path)['data']
    assert hparams.src_len == hparams.tgt_len == train_data.shape[1]
    graph = tf.Graph()

    with graph.as_default(), tf.container("train"):
        # channels: [features, SBP, DBP, MBP]
        train_src_data = train_data[:, :, 0:hparams.src_feature_size]
        train_tgt_data = train_data[:, :, hparams.src_feature_size:hparams.src_feature_size + hparams.tgt_feature_size]
        src_dataset = tf.data.Dataset.from_tensor_slices(train_src_data)
        tgt_dataset = tf.data.Dataset.from_tensor_slices(train_tgt_data)
        iterator = get_iterator(src_dataset, tgt_dataset, batch_size=hparams.batch_size,
                                random_seed=hparams.random_seed, is_train=True)
        model = model_creator(hparams, iterator=iterator, mode=tf.contrib.learn.ModeKeys.TRAIN)
    return TrainModel(graph=graph, model=model, iterator=iterator)


EvalModel = collections.namedtuple("EvalModel", ("graph", "model", "iterator", "data_mean", "data_std"))


def create_eval_model(model_creator, hparams, data_dir):
    """Create eval graph, model and iterator."""
    eval_data_path = []
    for root, _, name in os.walk(os.path.join(data_dir, 'eval_data')):
        for x in name:
            if x.split('.')[-1] == 'mat':
                eval_data_path.append(os.path.join(root, x))
    assert len(eval_data_path) == 1
    eval_data = scio.loadmat(*eval_data_path)['data']
    data_mean, data_std = load_data_mean_std(hparams, data_dir)
    batch_size = eval_data.shape[0]
    graph = tf.Graph()

    with graph.as_default(), tf.container("eval"):
        eval_src_data = eval_data[:, :, 0:hparams.src_feature_size]
        # channels: [features, SBP, DBP, MBP]
        eval_tgt_data = eval_data[:, :, hparams.src_feature_size:hparams.src_feature_size + hparams.tgt_feature_size]
        src_dataset = tf.data.Dataset.from_tensor_slices(eval_src_data)
        tgt_dataset = tf.data.Dataset.from_tensor_slices(eval_tgt_data)
        iterator = get_iterator(src_dataset, tgt_dataset, batch_size=batch_size,
                                random_seed=hparams.random_seed, is_train=False)
        model = model_creator(hparams, iterator=iterator, mode=tf.contrib.learn.ModeKeys.EVAL)
    return EvalModel(graph=graph, model=model, iterator=iterator, data_mean=data_mean, data_std=data_std)


InferModel = collections.namedtuple("InferModel", ("graph", "model", "iterator"))


def create_infer_model(model_creator, hparams, infer_data, batch_size):
    """Create inference model."""
    graph = tf.Graph()

    with graph.as_default(), tf.container("infer"):
        infer_src_data = infer_data[:, :, 0:hparams.src_feature_size]
        # channels:[features, SBP, SBP, MBP]
        infer_tgt_data = infer_data[:, :, hparams.src_feature_size:hparams.src_feature_size + hparams.tgt_feature_size]
        src_dataset = tf.data.Dataset.from_tensor_slices(infer_src_data)
        tgt_dataset = tf.data.Dataset.from_tensor_slices(infer_tgt_data)
        iterator = get_iterator(src_dataset, tgt_dataset, batch_size=batch_size,
                                random_seed=hparams.random_seed, is_train=False)
        model = model_creator(hparams, iterator=iterator, mode=tf.contrib.learn.ModeKeys.INFER)
    return InferModel(graph=graph, model=model, iterator=iterator)


def load_model(model, ckpt_path, session, name):
    """Load model from a checkpoint."""
    try:
        model.saver.restore(session, ckpt_path)
    except tf.errors.NotFoundError as e:
        utils.print_out("Can't load checkpoint")
        utils.print_out("%s" % str(e))
    # session.run(tf.tables_initializer())  ## why table still need to be initialized even model loaded??
    utils.print_out("  loaded %s model parameters from %s" % (name, ckpt_path))
    return model


def create_or_load_model(model, model_dir, session, name):
    """Create translation model and initialize or load parameters in session."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model._replace(model=load_model(model.model, latest_ckpt, session, name))
        utils.print_out("checkpoint found, load checkpoint\n %s" % latest_ckpt)
    else:
        utils.print_out("  checkpoint not found in %s" % (model_dir))
        utils.print_out("  created %s model with fresh parameters" % (name))
        session.run(tf.global_variables_initializer())
        # session.run(tf.tables_initializer())
    global_step = model.model.global_step.eval(session=session)
    epoch_num = model.model.epoch_num.eval(session=session)
    return model, global_step, epoch_num


def train_eval(hparams, data_dir, model_dir, log_dir):
    # Log and output files
    log_file = os.path.join(log_dir, "log_%d" % time.time())
    log_f = tf.gfile.GFile(log_file, mode="a")
    utils.print_out("# log_file=%s" % log_file, log_f)

    # create model and session
    model_creator = get_model_creator(hparams)  # model_creator is just a factory function
    train_model = create_train_model(model_creator, hparams, data_dir)
    eval_model = create_eval_model(model_creator, hparams, data_dir)
    config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True
    train_sess = tf.Session(config=config_proto, graph=train_model.graph)
    eval_sess = tf.Session(config=config_proto, graph=eval_model.graph)

    # Summary writer
    summary_writer = tf.summary.FileWriter(os.path.join(log_dir, "train_log"), train_model.graph)

    with train_model.graph.as_default():
        train_model, global_step, epoch_num = create_or_load_model(train_model, model_dir, train_sess, "train")

    # first time train
    # start_train_time = time.time()
    utils.print_out("# Start step %d, %s" % (global_step, time.ctime()), log_f)
    epoch_step = 0  # just to count the current step in this epoch
    stats = {"step_time": 0.0, "train_loss": 0.0,
             "predict_beat_count": 0.0,  # number of beat count
             "sequence_count": 0.0,  # number of training examples processed
             }
    info = {}
    # Initialize iterators
    train_sess.run(train_model.iterator.initializer)
    # This is the training loop.
    while global_step < hparams.num_train_steps:
        ### Run a step ###
        start_time = time.time()
        try:
            # forward train phase
            forward_result = train_model.model.train_forward(train_sess)[1]
            step_result = forward_result
            src_ref, tgt_ref, src2tgt_output, tgt2src_output, src_seq_len, tgt_seq_len = (
                forward_result.train_src_ref, forward_result.train_tgt_ref,
                forward_result.train_src2tgt_output, forward_result.train_tgt2src_output,
                forward_result.src_seq_len, forward_result.tgt_seq_len)
            # backward train phase
            if hparams.backward_enable:
                backward_input = [src_ref, tgt_ref, src2tgt_output, tgt2src_output, src_seq_len, tgt_seq_len]
                backward_result = train_model.model.train_backward(train_sess, backward_input)[1]
            epoch_step += 1
        except tf.errors.OutOfRangeError:
            # Finished going through the training dataset.  Go to next epoch.
            epoch_step = 0
            epoch_num = train_model.model.increase_epoch_num(train_sess)
            utils.print_out("# Finished an epoch, global step %d. Perform evaluation" % global_step)
            # save model for evaluation
            train_model.model.saver.save(train_sess, os.path.join(model_dir, "model.ckpt"), global_step=global_step)
            eval(eval_model, eval_sess, model_dir, hparams, summary_writer, log_f)
            train_sess.run(train_model.iterator.initializer)
            continue

        # Process step_result, accumulate stats, and write summary
        global_step = step_result.global_step
        summary_writer.add_summary(step_result.train_summary, global_step)
        # Update statistics
        stats["step_time"] += time.time() - start_time
        stats["train_loss"] += step_result.train_loss * step_result.predict_beat_count
        stats["predict_beat_count"] += step_result.predict_beat_count
        stats["sequence_count"] += step_result.batch_size
        if global_step % hparams.steps_per_stats == 0:
            info["avg_step_time"] = stats["step_time"] / hparams.steps_per_stats
            # the time to take for process one batch
            info["sample_per_second"] = stats["sequence_count"] / (stats["step_time"])
            info["avg_beat_loss"] = (stats["train_loss"] / stats["predict_beat_count"])
            info["epoch_num"] = epoch_num
            utils.print_out(
                "global step: %d, epoch_num: %d, avg step time: %.2fs, sample per second: %.2f, avg beat loss: %.2f, time；%s" %
                (
                    global_step, info['epoch_num'], info["avg_step_time"], info["sample_per_second"],
                    info["avg_beat_loss"],
                    time.ctime()), log_f)
            for key in info:
                summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=info[key])]),
                                           global_step)

            stats = {"step_time": 0.0, "train_loss": 0.0,
                     "predict_beat_count": 0.0,  # word count on the target side
                     "sequence_count": 0.0,  # number of training examples processed
                     }

    # Done training in one epoch
    train_model.model.saver.save(
        train_sess,
        os.path.join(model_dir, "model.ckpt"),
        global_step=global_step)

    # ...result summary

    summary_writer.close()


def eval(eval_model, eval_sess, model_dir, hparams, summary_writer, log_f):
    with eval_model.graph.as_default():
        eval_model, global_step, epoch_num = create_or_load_model(eval_model, model_dir, eval_sess, "eval")
        eval_sess.run(eval_model.model.iterator.initializer)
        eval_info = {}
        total_loss = 0
        total_predict_beat_count = 0
        total_sequence_count = 0
        while True:
            try:
                step_result = eval_model.model.eval(eval_sess)
                total_loss += step_result.eval_loss * step_result.predict_beat_count
                total_predict_beat_count += step_result.predict_beat_count
                total_sequence_count += step_result.batch_size
            except tf.errors.OutOfRangeError:
                eval_info['epoch_num'] = epoch_num
                eval_info['eval_avg_beat_loss'] = total_loss / total_predict_beat_count
                eval_info['eval_predict_beat_count'] = total_predict_beat_count
                eval_info['eval_sample_num'] = total_sequence_count
                utils.print_out(
                    "\neval: global step: %d, epoch_num: %d, eval_avg beat loss: %.2f, eval_predict_beat_count: %d, eval_sample_num: %d, time；%s\n" %
                    (global_step, eval_info['epoch_num'], eval_info["eval_avg_beat_loss"],
                     eval_info['eval_predict_beat_count'], eval_info['eval_sample_num'], time.ctime()), log_f)
                for key in eval_info:
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=eval_info[key])]),
                        global_step)
                break


def infer(hparams, data_dir, model_dir, result_dir):
    # infer mode is different from train and eval mode that we prepare data in advanced
    # becasue we may have various test dataset, but only one dataset in the case of train and eval
    # prepare data
    infer_data_path = []
    for root, _, name in os.walk(os.path.join(data_dir, 'test_data')):
        for x in name:
            if x.split('.')[-1] == 'mat':
                infer_data_path.append(os.path.join(root, x))

    for j in range(len(infer_data_path)):
        data_name = os.path.basename(infer_data_path[j]).split('.')[0]
        infer_data = scio.loadmat(infer_data_path[j])['data']
        data_mean, data_std = load_data_mean_std(hparams, data_dir)
        batch_size = infer_data.shape[0]

        # Create model
        model_creator = get_model_creator(hparams)
        infer_model = create_infer_model(model_creator, hparams, infer_data, batch_size)
        # TensorFlow model
        config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config_proto.gpu_options.allow_growth = True
        infer_sess = tf.Session(config=config_proto, graph=infer_model.graph)
        with infer_model.graph.as_default():
            infer_model, global_step, epoch_num = create_or_load_model(
                infer_model, model_dir, infer_sess, "infer")
            infer_sess.run(infer_model.model.iterator.initializer)
            # because we have already set batch size to be the size of all sample, so we dont use loop but just calculate the metrics
            # otherwise, we can implement similarly as eval mode

            step_result = infer_model.model.infer(infer_sess)
            infer_avg_beat_loss = step_result.infer_loss
            total_predict_beat_count = step_result.predict_beat_count
            total_sequence_count = step_result.batch_size
            infer_tgt_output = step_result.infer_tgt_output
            # scale back to the original size
            # (both the infered one and the input one needed to be scaled back, because they are both normalized)
            infer_real_tgt_feature = np.zeros(infer_tgt_output.shape)
            for i in range(hparams.tgt_feature_size):
                infer_real_tgt_feature[:, :, i] = (
                        infer_tgt_output[:, :, i] * data_std[:, hparams.src_feature_size + i]
                        + data_mean[:, hparams.src_feature_size + i])

            origin_tgt_data = infer_data[:, :,
                              hparams.src_feature_size: hparams.src_feature_size + hparams.tgt_feature_size]
            origin_real_tgt_data = np.zeros(origin_tgt_data.shape)
            for i in range(hparams.tgt_feature_size):
                origin_real_tgt_data[:, :, i] = (
                        origin_tgt_data[:, :, i] * data_std[:, hparams.src_feature_size + i]
                        + data_mean[:, hparams.src_feature_size + i])

            # infer_info['epoch_num'] = epoch_num
            infer_info = {}
            infer_info['infer_avg_beat_loss'] = infer_avg_beat_loss
            infer_info['infer_predict_beat_count'] = total_predict_beat_count
            infer_info['infer_sample_num'] = total_sequence_count
            utils.print_out(
                "\ninfer: global step: %d, infer_avg beat loss: %.2f, infer_predict_beat_count: %d, infer_sample_num: %d, time；%s\n" %
                (global_step, infer_info["infer_avg_beat_loss"], infer_info['infer_predict_beat_count'],
                 infer_info['infer_sample_num'], time.ctime()))
            """
            for key in eval_info:
                summary_writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=eval_info[key])]),
                    global_step)
            """

            # save_result()
            step_result_dir = os.path.join(result_dir, 'step-' + str(global_step))
            if not tf.gfile.Exists(step_result_dir):
                tf.gfile.MakeDirs(step_result_dir)
            scio.savemat(os.path.join(step_result_dir, data_name + '_real_origin.mat'), {'data': origin_real_tgt_data})
            utils.print_out("save origin data to %s" % os.path.join(step_result_dir, data_name + '_real_origin.mat'))
            scio.savemat(os.path.join(step_result_dir, data_name + '_real_prediction.mat'),
                         {'data': infer_real_tgt_feature})
            utils.print_out(
                "save prediction data to %s" % os.path.join(step_result_dir, data_name + '_real_prediction.mat'))

            # performance metrics calculation
            # RMSE: root mean square error
            real_test_loss_SBP_RMSE = np.sqrt(
                np.mean(np.square(origin_real_tgt_data[:, :, 0] - infer_real_tgt_feature[:, :, 0])))
            real_test_loss_DBP_RMSE = np.sqrt(
                np.mean(np.square(origin_real_tgt_data[:, :, 1] - infer_real_tgt_feature[:, :, 1])))
            real_test_loss_MBP_RMSE = np.sqrt(
                np.mean(np.square(origin_real_tgt_data[:, :, 2] - infer_real_tgt_feature[:, :, 2])))
            # MAD: mean absolute difference
            real_test_loss_SBP_MAD = np.mean(
                np.absolute(origin_real_tgt_data[:, :, 0] - infer_real_tgt_feature[:, :, 0]))
            real_test_loss_DBP_MAD = np.mean(
                np.absolute(origin_real_tgt_data[:, :, 1] - infer_real_tgt_feature[:, :, 1]))
            real_test_loss_MBP_MAD = np.mean(
                np.absolute(origin_real_tgt_data[:, :, 2] - infer_real_tgt_feature[:, :, 2]))
            # MD: mean difference (absolute is not used!)
            real_test_loss_SBP_MD = np.mean(origin_real_tgt_data[:, :, 0] - infer_real_tgt_feature[:, :, 0])
            real_test_loss_DBP_MD = np.mean(origin_real_tgt_data[:, :, 1] - infer_real_tgt_feature[:, :, 1])
            real_test_loss_MBP_MD = np.mean(origin_real_tgt_data[:, :, 2] - infer_real_tgt_feature[:, :, 2])
            # SD1: the standard deviation of delta
            real_test_loss_SBP_SD1 = np.sqrt(
                np.mean(np.square(
                    (origin_real_tgt_data[:, :, 0] - infer_real_tgt_feature[:, :, 0]) - real_test_loss_SBP_MD)))
            real_test_loss_DBP_SD1 = np.sqrt(
                np.mean(np.square(
                    (origin_real_tgt_data[:, :, 1] - infer_real_tgt_feature[:, :, 1]) - real_test_loss_DBP_MD)))
            real_test_loss_MBP_SD1 = np.sqrt(
                np.mean(np.square(
                    (origin_real_tgt_data[:, :, 2] - infer_real_tgt_feature[:, :, 2]) - real_test_loss_MBP_MD)))
            # SD2: the standard deviation of predicted value itself
            real_test_loss_SBP_SD2 = np.std(infer_real_tgt_feature[:, :, 0])
            real_test_loss_DBP_SD2 = np.std(infer_real_tgt_feature[:, :, 1])
            real_test_loss_MBP_SD2 = np.std(infer_real_tgt_feature[:, :, 2])
            name_collection = ['real_test_loss_SBP_RMSE', 'real_test_loss_DBP_RMSE', 'real_test_loss_MBP_RMSE',
                               'real_test_loss_SBP_MAD', 'real_test_loss_DBP_MAD', 'real_test_loss_MBP_MAD',
                               'real_test_loss_SBP_MD', 'real_test_loss_DBP_MD', 'real_test_loss_MBP_MD',
                               'real_test_loss_SBP_SD1', 'real_test_loss_DBP_SD1', 'real_test_loss_MBP_SD1',
                               'real_test_loss_SBP_SD2', 'real_test_loss_DBP_SD2', 'real_test_loss_MBP_SD2']
            metric_collection = [real_test_loss_SBP_RMSE, real_test_loss_DBP_RMSE, real_test_loss_MBP_RMSE,
                                 real_test_loss_SBP_MAD, real_test_loss_DBP_MAD, real_test_loss_MBP_MAD,
                                 real_test_loss_SBP_MD, real_test_loss_DBP_MD, real_test_loss_MBP_MD,
                                 real_test_loss_SBP_SD1, real_test_loss_DBP_SD1, real_test_loss_MBP_SD1,
                                 real_test_loss_SBP_SD2, real_test_loss_DBP_SD2, real_test_loss_MBP_SD2]

            with open(os.path.join(step_result_dir, 'step_' + str(global_step) + '_performance_metrics.txt'),
                      'a') as result_fo:
                for i in range(len(name_collection)):
                    if i % 3 == 0:
                        print('\n')
                        result_fo.write('\n')
                    print(data_name + '_' + name_collection[i], ':', metric_collection[i])
                    result_fo.write('%s_%s:%f\n' % (data_name, name_collection[i], metric_collection[i]))
