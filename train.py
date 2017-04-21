
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from helpers import configuration
from helpers import skip_thoughts_model

inputfiles = "bookcorpus/train-?????-of-00100"
savedir = "trained"


tf.logging.set_verbosity(tf.logging.INFO)


def _setup_learning_rate(config, global_step):
  if config.learning_rate_decay_factor > 0:
    learning_rate = tf.train.exponential_decay(
        learning_rate=float(config.learning_rate),
        global_step=global_step,
        decay_steps=config.learning_rate_decay_steps,
        decay_rate=config.learning_rate_decay_factor,
        staircase=False)
  else:
    learning_rate = tf.constant(config.learning_rate)
  return learning_rate


def main(unused_argv):

  model_config = configuration.model_config(input_file_pattern=inputfiles)
  training_config = configuration.training_config()

  tf.logging.info("Building training graph.")
  g = tf.Graph()
  with g.as_default():
    model = skip_thoughts_model.SkipThoughtsModel(model_config, mode="train")
    model.build()

    learning_rate = _setup_learning_rate(training_config, model.global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_tensor = tf.contrib.slim.learning.create_train_op(
        total_loss=model.total_loss,
        optimizer=optimizer,
        global_step=model.global_step,
        clip_gradient_norm=training_config.clip_gradient_norm)

    saver = tf.train.Saver()

  tf.contrib.slim.learning.train(
      train_op=train_tensor,
      logdir=savedir,
      graph=g,
      global_step=model.global_step,
      number_of_steps=training_config.number_of_steps,
      save_summaries_secs=training_config.save_summaries_secs,
      saver=saver,
      save_interval_secs=training_config.save_model_secs)


tf.app.run()
