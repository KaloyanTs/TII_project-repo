#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import torch
import parameters

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(1, max_len, d_model)

        position = torch.arange(max_len).unsqueeze(0).unsqueeze(2)
        div_term = (
            (10000.0 ** (torch.arange(0, d_model, 2) / d_model))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        pe[0, :, 0::2] = torch.sin(position / div_term)
        pe[0, :, 1::2] = torch.cos(position / div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = (
            x + self.pe[:, : x.shape[1], :]
        )  # x.shape = (batch_size, seq_len, embedding_dim)
        return x

class LanguageModel(torch.nn.Module):
    def __init__(self, n_head, d_model, num_layers, word2ind, unkToken, padToken, transToken, endToken, maxlen=5000):
        super(LanguageModel, self).__init__()
        self.word2ind = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.transTokenIdx = word2ind[transToken]
        self.endTokenIdx = word2ind[endToken]

        self.embed = torch.nn.Embedding(len(word2ind), d_model)
        self.pos_embed = PositionalEncoding(d_model)
        self.dropout1 = torch.nn.Dropout(0.1)

        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4, dropout=0.2, batch_first=True, device=parameters.device)
        self.Transformer = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.projection = torch.nn.Linear(d_model, len(word2ind))

        self.register_buffer("mask", torch.triu(torch.ones(parameters.max_len, parameters.max_len, dtype=torch.bool), diagonal=1))

    def preparePaddedBatch(self, source):
      device = next(self.parameters()).device
      m = max(len(s) for s in source)
      sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in source]
      return torch.tensor(sents_padded, dtype=torch.long, device=device)	# shape=(batch_size, seq_len)

    def forward(self, source):
        # id2word = {i: w for w, i in self.word2ind.items()}
        X = self.preparePaddedBatch(source)

        E = self.pos_embed(self.embed(X[:, :-1]))
        input = self.dropout1(E)

        src_key_padding_mask = (X[:, :-1] == self.padTokenIdx)
        seq_len = X.shape[1] - 1
        mask = self.mask[:seq_len, :seq_len]
        tr = self.Transformer(input, mask=mask, src_key_padding_mask=src_key_padding_mask)

        Z = self.projection(tr.flatten(0, 1))
        Y_bar = X[:, 1:].flatten(0, 1)
        H = torch.nn.functional.cross_entropy(Z, Y_bar, ignore_index=self.padTokenIdx)
        return H

    def save(self,fileName):
      torch.save(self.state_dict(), fileName)

    def load(self,fileName):
      self.load_state_dict(torch.load(fileName))

    def generate(self, prefix, limit=1000, temperature=0.75):
        self.eval()
        device = next(self.parameters()).device

        prefix_idx = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)

        generated = prefix_idx

        for _ in range(limit):
            with torch.no_grad():
                E = self.embed(generated)
                input = self.dropout1(self.pos_embed(E))

                seq_len = generated.shape[1]
                causal_mask = self.mask[:seq_len, :seq_len]

                tr = self.Transformer(input, mask=causal_mask)

                logits = self.projection(tr[:, -1, :]) / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)

                if next_token_idx.item() == self.endTokenIdx:
                    break

                generated = torch.cat((generated, next_token_idx), dim=1)

        result = generated[0]
        return result




