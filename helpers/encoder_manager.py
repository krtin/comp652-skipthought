from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import collections


import numpy as np
import tensorflow as tf

from helpers import skip_thoughts_encoder


class EncoderManager(object):
  

  def __init__(self):
    self.encoders = []
    self.sessions = []

  def load_model(self, model_config, vocabulary_file, embedding_matrix_file,
                 checkpoint_path):
    tf.logging.info("Reading vocabulary from %s", vocabulary_file)
    with tf.gfile.GFile(vocabulary_file, mode="r") as f:
      lines = list(f.readlines())
    reverse_vocab = [line.decode("utf-8").strip() for line in lines]
    tf.logging.info("Loaded vocabulary with %d words.", len(reverse_vocab))

    tf.logging.info("Loading embedding matrix from %s", embedding_matrix_file)
    # Note: tf.gfile.GFile doesn't work here because np.load() calls f.seek()
    # with 3 arguments.
    with open(embedding_matrix_file, "r") as f:
      embedding_matrix = np.load(f)
    tf.logging.info("Loaded embedding with shape %s",
                    embedding_matrix.shape)

    word_embeddings = collections.OrderedDict(
        zip(reverse_vocab, embedding_matrix))

    g = tf.Graph()
    with g.as_default():
      encoder = skip_thoughts_encoder.SkipThoughtsEncoder(word_embeddings)
      restore_model = encoder.build_graph_from_config(model_config, checkpoint_path)

    sess = tf.Session(graph=g)
    restore_model(sess)

    self.encoders.append(encoder)
    self.sessions.append(sess)

  def encode(self, data, use_norm=True, verbose=False, batch_size=128, use_eos=False):
 
    if not self.encoders:
      raise ValueError("Must load model first before before calling encode.")

    encoded = []
    for encoder, sess in zip(self.encoders, self.sessions):
      encoded.append(
          np.array(
              encoder.encode(sess, data, use_norm=use_norm, verbose=verbose, batch_size=batch_size, use_eos=use_eos)))

    return np.concatenate(encoded, axis=1)

  def close(self):
    
    for sess in self.sessions:
      sess.close()
