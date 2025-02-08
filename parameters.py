import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'

corpusFileName = 'corpusData'
wordsFileName = 'wordsData'
modelFileName = 'NMTmodel'

device = torch.device("cuda:0")
#device = torch.device("cpu")

d_model = 128
n_heads = 16
num_layers = 2

max_len = 2000

learning_rate = 0.01
batchSize = 16
clip_grad = 10.0

maxEpochs = 10000
log_every = 10
test_every = 200
