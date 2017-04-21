
from helpers import paraphrase
from helpers import configuration
from helpers import encoder_manager

vocabfile='vocab/vocab.txt'
embeddingfile='vocab/embeddings.npy'
datadir='eval_data'
checkpointdir='trained'


encoder = encoder_manager.EncoderManager()
config = configuration.model_config()
encoder.load_model(config, vocabfile,embeddingfile, checkpointdir)
paraphrase.evaluate(encoder, evaltest=True, loc=datadir)
encoder.close()


