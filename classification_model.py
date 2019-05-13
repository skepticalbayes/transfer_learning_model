# coding=utf-8
"""BERT finetuning runner."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import collections
import pandas as pd
import numpy as np
import os
from encoder_transfer_model import universal_encoder
from encoder_transfer_model import optimization
from encoder_transfer_model import tokenization
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.externals import joblib
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "file_name", None,
    "Training data filename")

flags.DEFINE_string(
    "scope", None,
    "Variable scope name")

flags.DEFINE_string(
    "label_name", None,
    "Target variable identifier")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("is_multilabel", False, "Whether classification task is multiclass or multilabel.")


ts = np.arange(0, 1, 0.01, dtype=float)
ts = sorted(ts, reverse=True)

class Input(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a Input.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInput(object):
  """Fake input class so the num inputs is a multiple of the batch size.

  When running eval/predict, we need to pad the number of examples
  to be a multiple of the batch size, because the it requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""
  def __init__(self):
    self.mlb = None


  def get_train_inputs(self, data_dir, file_name, label_column):
    """Gets a collection of `Input`s for the train set."""
    raise NotImplementedError()

  def get_test_inputs(self, data_dir, file_name, label_column):
    """Gets a collection of `Input`s for prediction."""
    raise NotImplementedError()

  def get_labels(self, data_dir, file_name, label_column):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_data(cls, input_file):
    """Reads a input value file."""
    df = joblib.load(input_file)
    return df


class MultiLabelProcessor(DataProcessor):
  """Processor for the multilabel dataset."""

  def get_train_inputs(self, data_dir, file_name, label_column):
    """See base class."""
    return self._create_examples(
        self._read_data(os.path.join(data_dir, file_name)), 'train', label_column)

  def get_test_inputs(self, data_dir, file_name, label_column):
    """See base class."""
    return self._create_examples(
        self._read_data(os.path.join(data_dir, file_name)), 'test', label_column)

  def get_labels(self, data_dir, file_name, label_column):
    """See base class."""
    if self.mlb is None:
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(self._read_data(os.path.join(data_dir, file_name))[label_column])
        joblib.dump(self.mlb, 'mlb.pkl')
    return self.mlb.classes_

  def _create_examples(self, df, set_type, label_column=None):
    """Creates examples for the training and dev sets."""
    if self.mlb is not None:
        pass
    else:
        self.mlb = MultiLabelBinarizer()
    print ("num labels: {}".format(self.mlb.classes_.shape[0]))
    print ('converting kwds to labels')
    if set_type=='train':
      df[label_column] = self.mlb.fit_transform(df[label_column]).tolist()
    else:
      df[label_column] = self.mlb.transform(df[label_column]).tolist()
    if set_type == 'train':
        inputs = df.apply(lambda x: Input(guid=x['code'], text_a=x['questions'], text_b=None, label=x[label_column]), axis=1).tolist()
    else:
        inputs = df.apply(
            lambda x: Input(guid=x['code'], text_a=x['questions'], text_b=None, label=x[label_column]),
            axis=1).tolist()
    return inputs


class MultiClassProcessor(DataProcessor):
  """Processor for the multilabel dataset."""

  def get_train_inputs(self, data_dir, file_name, label_column):
    """See base class."""
    return self._create_examples(
        self._read_data(os.path.join(data_dir, file_name)), 'train', label_column)

  def get_test_inputs(self, data_dir, file_name, label_column):
    """See base class."""
    return self._create_examples(
        self._read_data(os.path.join(data_dir, file_name)), 'test', label_column)

  def get_labels(self, data_dir, file_name, label_column):
    """See base class."""
    if self.mlb is None:
        self.mlb = LabelBinarizer()
        self.mlb.fit(self._read_data(os.path.join(data_dir, file_name))[label_column])
        joblib.dump(self.mlb, 'lb.pkl')
    return self.mlb.classes_

  def _create_examples(self, df, set_type, label_column=None):
    """Creates examples for the training and dev sets."""
    if self.mlb is not None:
        pass
    else:
        self.mlb = LabelBinarizer()
    if set_type=='train':
      df[label_column] = self.mlb.fit_transform(df[label_column]).tolist()
    else:
      df[label_column] = self.mlb.transform(df[label_column]).tolist()
    if set_type == 'train':
        inputs = df.apply(lambda x: Input(guid=x['code'], text_a=x['questions'], text_b=None, label=x[label_column]), axis=1).tolist()
    else:
        inputs = df.apply(
            lambda x: Input(guid=x['code'], text_a=x['questions'], text_b=None, label=x[label_column]),
            axis=1).tolist()
    return inputs


def convert_single_input(ex_index, example, label_list, max_seq_length,
                         tokenizer):
  """Converts a single `Input` into a single `InputFeatures`."""

  if isinstance(example, PaddingInput):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=[0]* len(label_list),
        is_real_example=False)

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = example.label
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("labels: %s " % (label_list[np.where(np.array(example.label)>0)[0]]))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `Input`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_input(ex_index, example, label_list,
                                   max_seq_length, tokenizer)
    if ex_index % 10000 == 0:
      tf.logging.info("labels:{}".format(feature.label_id))

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_id)
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, num_labels):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([num_labels], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, is_multilabel):
  """Creates a classification model."""
  model = universal_encoder.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=False)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    if is_multilabel:
        probabilities = tf.nn.sigmoid(logits)
        per_input_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(per_input_loss)
        print("Sigmoid loss for multiple labels")
    else:
        print("softmax loss for single labels")
        probabilities = tf.nn.softmax(logits)
        log_probs = tf.nn.log_softmax(logits)
        # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_input_loss = -tf.reduce_sum(labels * log_probs, axis=-1)
        # per_input_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        loss = tf.reduce_mean(per_input_loss)
    return (loss, per_input_loss, logits, probabilities)

# num_labels, learning_rate, num_train_steps,
#                      num_warmup_steps
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, is_multilabel, scope):

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = tf.cast(features["label_ids"], dtype=tf.float32)
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, is_multilabel)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = universal_encoder.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics.
      def metric_fn(label_ids, predicted_labels):
          label_ids, predicted_labels = tf.argmax(label_ids, axis=1), tf.argmax(predicted_labels, axis=1)
          accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
          f1_score = tf.contrib.metrics.f1_score(
              label_ids,
              predicted_labels)
          auc = tf.metrics.auc(
              label_ids,
              predicted_labels)
          recall = tf.metrics.recall(
              label_ids,
              predicted_labels)
          precision = tf.metrics.precision(
              label_ids,
              predicted_labels)
          true_pos = tf.metrics.true_positives(
              label_ids,
              predicted_labels)
          true_neg = tf.metrics.true_negatives(
              label_ids,
              predicted_labels)
          false_pos = tf.metrics.false_positives(
              label_ids,
              predicted_labels)
          false_neg = tf.metrics.false_negatives(
              label_ids,
              predicted_labels)
          return {
              "eval_accuracy": accuracy,
              "f1_score": f1_score,
              "auc": auc,
              "precision": precision,
              "recall": recall,
              "true_positives": true_pos,
              "true_negatives": true_neg,
              "false_positives": false_pos,
              "false_negatives": false_neg
          }
      if not is_multilabel:
          # predicted_labels = tf.squeeze(tf.argmax(probabilities, axis=-1, output_type=tf.int32))
          eval_metrics = metric_fn(label_ids, probabilities)
          output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                   loss=total_loss,
                                                   train_op=train_op)
      else:
          output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                   loss=total_loss,
                                                   train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      def eval_th_metrics(label, probs):
        precisions = tf.metrics.precision_at_thresholds(labels=tf.reshape(label, shape=(-1,)),
                                                        predictions=tf.reshape(probs, shape=(-1,)),
                                                        thresholds=ts)
        recalls = tf.metrics.recall_at_thresholds(labels=tf.reshape(label, shape=(-1,)),
                                                  predictions=tf.reshape(probs, shape=(-1,)),
                                                  thresholds=ts)
        return {
          "precisions": precisions,
          "recalls": recalls
        }
      if not is_multilabel:
        eval_metrics = metric_fn(per_example_loss, label_ids, logits, is_real_example)
        output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                 loss=total_loss,
                                                 eval_metric_ops=eval_metrics)
      else:
        eval_metrics = eval_th_metrics(label_ids, probabilities)
        output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                 loss=total_loss,
                                                 eval_metric_ops=eval_metrics)
    else:
      output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                               # eval_metric_ops=eval_th_metrics,
                                               predictions={'probabilities': probabilities})
    return output_spec
  if scope:
      with tf.variable_scope(scope):
        return model_fn
  else:
      return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder, num_labels):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples, num_labels], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_inputs_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `Input`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_input(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "multilabel": MultiLabelProcessor,
      "multiclass": MultiClassProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = universal_encoder.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()
  if FLAGS.do_train:
      label_list = processor.get_labels(FLAGS.data_dir,
                                        FLAGS.file_name,
                                        FLAGS.label_name)
  elif FLAGS.do_predict or FLAGS.do_eval:
    if task_name == 'multilabel':
      processor.mlb = joblib.load('mlb.pkl')
      label_list = processor.mlb.classes_
    else:
      processor.mlb = joblib.load('lb.pkl')
      label_list = processor.mlb.classes_

  num_labels = label_list.shape[0]

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir,
      save_summary_steps=FLAGS.save_checkpoints_steps,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps)

  train_inputs = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_inputs = processor.get_train_inputs(FLAGS.data_dir,
                                              FLAGS.file_name,
                                              FLAGS.label_name)
    num_train_steps = int(
        len(train_inputs) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=num_labels,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      is_multilabel=FLAGS.is_multilabel,
      scope=FLAGS.scope
  )

  estimator = tf.estimator.Estimator(
                  model_fn=model_fn,
                  config=run_config,
                  params={"batch_size": FLAGS.train_batch_size})

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(train_inputs, label_list,
                                            FLAGS.max_seq_length,
                                            tokenizer,
                                            train_file)
    # train_features = convert_inputs_to_features(train_inputs, label_list, FLAGS.max_seq_length,
    #                                             tokenizer)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num inputs = %d", len(train_inputs))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        num_labels=num_labels)
    print ("total trainable variables: {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_test_inputs(FLAGS.data_dir, FLAGS.file_name, FLAGS.label_name)
    num_actual_eval_examples = len(eval_examples)
    # if FLAGS.use_tpu:
    #   # TPU requires a fixed batch size for all batches, therefore the number
    #   # of examples must be a multiple of the batch size, or else examples
    #   # will get dropped. So we pad with fake examples which are ignored
    #   # later on. These do NOT count towards the metric (all tf.metrics
    #   # support a per-instance weight, and these get a weight of 0.0).
    #   while len(eval_examples) % FLAGS.eval_batch_size != 0:
    #     eval_examples.append(PaddingInput())

    eval_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    # file_based_convert_examples_to_features(
    #     eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    # if FLAGS.use_tpu:
    #   assert len(eval_examples) % FLAGS.eval_batch_size == 0
    #   eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)
    #
    # eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False,
        num_labels=num_labels
    )
    if FLAGS.task_name=='multilabel':
      b1_Ts, b2_Ts, b3_Ts, p_Ts, r_Ts, P, R = [], [], [], [], [], [], []

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    print (result['global_step'], result)
    # output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    # with tf.gfile.GFile(output_eval_file, "w") as writer:
    tf.logging.info("***** Eval results *****")
    # for key in sorted(result.keys()):
    if FLAGS.task_name == 'multilabel':
        prec = result['precisions']
        recall = result['recalls']
        if prec[prec>=0.95].shape[0]>0:
            b1_Ts.append(ts[np.argmin(prec[prec >= .95])])
        if prec[prec >= .9].shape[0]>0:
            b2_Ts.append(ts[np.argmin(prec[prec >= .9])])
        if prec[prec>=0.8].shape[0]>0:
            b3_Ts.append(ts[np.argmin(prec[prec >= .8])])
        if prec.shape[0]>0:
            p_Ts.append(ts[np.argmax(prec)])
            P.append(np.max(prec))
        if recall.shape[0]>0:
            r_Ts.append(ts[np.argmax(recall)])
            R.append(np.max(recall))
        if result['global_step'] and result['global_step'] % 10000:
          eval_summ = "p:{p}, r:{r}, p_ts:{pt}, r_ts:{rt}, b1_ts:{b1}, b2_ts:{b2}, b3_ts:{b3}".format(
              p=np.mean(P), r=np.mean(R), pt=np.mean(p_Ts),
              rt=np.mean(r_Ts), b1=np.mean(b1_Ts), b2=np.mean(b2_Ts), b3=np.mean(b3_Ts))
          tf.logging.info('step:{}'.format(result['global_step']))
          tf.logging.info(eval_summ)
          print ('step:{}'.format(result['global_step']))
          print (eval_summ)
    else:
        for key in result.keys():
            eval_summ = "{k}: {r}".format(k=key, r=result[key])
            tf.logging.info(eval_summ)
            print (eval_summ)

  if FLAGS.do_predict:
    print ("predict mode")
    predict_examples = processor.get_test_inputs(FLAGS.data_dir, FLAGS.file_name, FLAGS.label_name)
    num_actual_predict_examples = len(predict_examples)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False,
        num_labels=num_labels
    )

    result = estimator.predict(input_fn=predict_input_fn)
    probs = []
    num_written_lines = 0
    tf.logging.info("***** Predict results *****")
    # print("result length: {}".format(len(result)))
    for (i, prediction) in enumerate(result):
      if i % 1000 == 0 and i:
          print("iteration:{}".format(i))
      probabilities = prediction["probabilities"]
      if i >= num_actual_predict_examples:
        break
      probs.append(probabilities)
      num_written_lines += 1
    try:
      assert num_written_lines == num_actual_predict_examples
    except:
      print("num_written_lines:{n} num_actual_predict_examples:{e}".format(n=num_written_lines,
                                                                           e=num_actual_predict_examples))
    print ("dumping probs of len:{}".format(len(probs)))
    joblib.dump(probs, 'probs_tip.pkl')


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("max_seq_length")
  flags.mark_flag_as_required("file_name")
  flags.mark_flag_as_required("label_name")
  flags.mark_flag_as_required("do_train")
  # flags.mark_flag_as_required("is_multilabel")
  tf.app.run()