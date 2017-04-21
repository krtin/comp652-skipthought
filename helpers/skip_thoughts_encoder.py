
from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import os.path


import nltk
import nltk.tokenize
import numpy as np
import tensorflow as tf

from helpers import skip_thoughts_model
from helpers import special_words


def _pad(seq, target_len):
  
  seq_len = len(seq)
  if seq_len <= 0 or seq_len > target_len:
    raise ValueError("Expected 0 < len(seq) <= %d, got %d" % (target_len,
                                                              seq_len))

  emb_dim = seq[0].shape[0]
  padded_seq = np.zeros(shape=(target_len, emb_dim), dtype=seq[0].dtype)
  mask = np.zeros(shape=(target_len,), dtype=np.int8)
  for i in range(seq_len):
    padded_seq[i] = seq[i]
    mask[i] = 1
  return padded_seq, mask


def _batch_and_pad(sequences):
  
  batch_embeddings = []
  batch_mask = []
  batch_len = max([len(seq) for seq in sequences])
  for seq in sequences:
    embeddings, mask = _pad(seq, batch_len)
    batch_embeddings.append(embeddings)
    batch_mask.append(mask)
  return np.array(batch_embeddings), np.array(batch_mask)


class SkipThoughtsEncoder(object):
  

  def __init__(self, embeddings):
    
    self._sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")
    self._embeddings = embeddings

  def _create_restore_fn(self, checkpoint_path, saver):
    
    if tf.gfile.IsDirectory(checkpoint_path):
      latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
      if not latest_checkpoint:
        raise ValueError("No checkpoint file found in: %s" % checkpoint_path)
      checkpoint_path = latest_checkpoint

    def _restore_fn(sess):
      tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
      saver.restore(sess, checkpoint_path)
      tf.logging.info("Successfully loaded checkpoint: %s",
                      os.path.basename(checkpoint_path))

    return _restore_fn

  def build_graph_from_config(self, model_config, checkpoint_path):
    
    tf.logging.info("Building model.")
    model = skip_thoughts_model.SkipThoughtsModel(model_config, mode="encode")
    model.build()
    saver = tf.train.Saver()

    return self._create_restore_fn(checkpoint_path, saver)

  def build_graph_from_proto(self, graph_def_file, saver_def_file,
                             checkpoint_path):
    
    # Load the Graph.
    tf.logging.info("Loading GraphDef from file: %s", graph_def_file)
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(graph_def_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

    # Load the Saver.
    tf.logging.info("Loading SaverDef from file: %s", saver_def_file)
    saver_def = tf.train.SaverDef()
    with tf.gfile.FastGFile(saver_def_file, "rb") as f:
      saver_def.ParseFromString(f.read())
    saver = tf.train.Saver(saver_def=saver_def)

    return self._create_restore_fn(checkpoint_path, saver)

  def _tokenize(self, item):
    """Tokenizes an input string into a list of words."""
    tokenized = []
    for s in self._sentence_detector.tokenize(item):
      tokenized.extend(nltk.tokenize.word_tokenize(s))

    return tokenized

  def _word_to_embedding(self, w):
    """Returns the embedding of a word."""
    return self._embeddings.get(w, self._embeddings[special_words.UNK])

  def _preprocess(self, data, use_eos):
    
    preprocessed_data = []
    for item in data:
      tokenized = self._tokenize(item)
      if use_eos:
        tokenized.append(special_words.EOS)
      preprocessed_data.append([self._word_to_embedding(w) for w in tokenized])
    return preprocessed_data

  def encode(self, sess, data, use_norm=True, verbose=True, batch_size=128, use_eos=False):
    
    data = self._preprocess(data, use_eos)
    thought_vectors = []

    batch_indices = np.arange(0, len(data), batch_size)
    for batch, start_index in enumerate(batch_indices):
      if verbose:
        tf.logging.info("Batch %d / %d.", batch, len(batch_indices))

      embeddings, mask = _batch_and_pad(
          data[start_index:start_index + batch_size])
      feed_dict = {
          "encode_emb:0": embeddings,
          "encode_mask:0": mask,
      }
      thought_vectors.extend(
          sess.run("encoder/thought_vectors:0", feed_dict=feed_dict))

    if use_norm:
      thought_vectors = [v / np.linalg.norm(v) for v in thought_vectors]

    return thought_vectors
