import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'

corpusFileName = 'corpusData'
wordsFileName = 'wordsData'
modelFileName = 'NMTmodel'

tokensFileName = 'tokenizer.json'

log_filename = "training_log.csv"

device = torch.device("cuda:0")
#device = torch.device("cpu")

d_model = 128
n_heads = 8
num_layers = 4

max_len = 2000

learning_rate = 0.001
batchSize = 32
clip_grad = 10.0

maxEpochs = 10000
log_every = 10
log_file_every = 200
test_every = 200
