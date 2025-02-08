from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

# ====== STEP 1: Train a SentencePiece BPE Tokenizer ======
def train_sentencepiece_bpe(input_file, vocab_size=8000, model_prefix="bpe"):
    """
    Trains a SentencePiece BPE tokenizer on a given text file.

    Args:
    - input_file (str): Path to the text file.
    - vocab_size (int): Number of vocabulary tokens.
    - model_prefix (str): Prefix for the output model files.
    """
    SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",  # Byte Pair Encoding
        max_sentence_length=8192,  # Avoid truncation
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    print(f"✅ Tokenizer trained and saved as {model_prefix}.model & {model_prefix}.vocab")


# ====== STEP 2: Load the Trained Model and Tokenize Sentences ======
def load_and_tokenize(model_prefix, text):
    """
    Loads a trained SentencePiece tokenizer and tokenizes text.

    Args:
    - model_prefix (str): Prefix of the trained model.
    - text (str): Input sentence to tokenize.

    Returns:
    - tokenized_text (list): List of subwords.
    - token_ids (list): List of token IDs.
    """
    sp = SentencePieceProcessor(model_file=f"{model_prefix}.model")

    # Tokenize
    tokenized_text = sp.encode(text, out_type=str)
    token_ids = sp.encode(text, out_type=int)

    print(f"🔹 Original Text: {text}")
    print(f"📝 Tokenized (Subwords): {tokenized_text}")
    print(f"🔢 Token IDs: {token_ids}")

    return tokenized_text, token_ids


# ====== STEP 3: Decode Tokens Back to Text ======
def decode_tokens(model_prefix, token_ids):
    """
    Decodes a list of token IDs back to text.

    Args:
    - model_prefix (str): Prefix of the trained model.
    - token_ids (list): List of token IDs.

    Returns:
    - decoded_text (str): Reconstructed sentence.
    """
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    decoded_text = sp.decode(token_ids)
    print(f"🔄 Decoded Text: {decoded_text}")
    return decoded_text


# ====== RUN EVERYTHING ======
if __name__ == "__main__":
    # Set dataset path (replace with your file path)
    dataset_path = "./en_bg_data/train.en"

    # Train the tokenizer
    train_sentencepiece_bpe(dataset_path, vocab_size=8000, model_prefix="bpe")

    # Sample text to test tokenization
    sample_text = "This is an example sentence for BPE tokenization."

    # Tokenize & decode
    tokens, ids = load_and_tokenize("bpe", sample_text)
    decode_tokens("bpe", ids)
