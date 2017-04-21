from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from helpers import eval_sick
from helpers import configuration
from helpers import encoder_manager

vocabfile='trained/vocab.txt'
embeddingfile='trained/embeddings.npy'
datadir='eval_data'
checkpointdir='trained'


encoder = encoder_manager.EncoderManager()
config = configuration.model_config()
encoder.load_model(config, vocabfile,embeddingfile, checkpointdir)
eval_sick.evaluate(encoder, evaltest=True, loc=datadir)
encoder.close()


