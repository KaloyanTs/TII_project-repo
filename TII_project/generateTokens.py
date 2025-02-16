from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os

def load_corpus(file_paths):
    all_sentences = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        all_sentences.extend([line.strip() for line in lines if line.strip()])

    return all_sentences

def train_bpe_tokenizer(corpus, vocab_size=8000, tokenizer_path="bpe_tokenizer.json"):
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[" ","<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>", "<TRANS>", "<S>", "</S>"],
        add_prefix_space=True
    )

    tokenizer.train_from_iterator(corpus, trainer)

    tokenizer.save(tokenizer_path)
    print(f"Tokenizer trained on both files and saved to {tokenizer_path}")

def load_and_tokenize(tokenizer_path, text):
    tokenizer = Tokenizer.from_file(tokenizer_path)

    encoding = tokenizer.encode(text)

    print(f"Original Text: {text}")
    print(f"Tokenized (Subwords): {encoding.tokens}")
    print(f"Token IDs: {encoding.ids}")

    return encoding.tokens, encoding.ids

def decode_tokens(tokenizer_path, token_ids):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    decoded_text = tokenizer.decode(token_ids)

    print(f"Decoded Text: {decoded_text}")
    return decoded_text

if __name__ == "__main__":
    file1 = "en_bg_data/train.en"
    file2 = "en_bg_data/train.bg"

    corpus = load_corpus([file1, file2])

    train_bpe_tokenizer(corpus, vocab_size=16000, tokenizer_path="bpe_tokenizer.json")

    sample_text = "The sun can be so burning that"

    tokens, ids = load_and_tokenize("bpe_tokenizer.json", sample_text)
    decode_tokens("bpe_tokenizer.json", ids)
