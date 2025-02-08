from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os

# ====== STEP 1: Load Text Data from Two Files ======
def load_corpus(file_paths):
    """
    Reads multiple text files and combines them into a single list of sentences.

    Args:
        file_paths (list): List of file paths.

    Returns:
        list: List of text lines from both files.
    """
    all_sentences = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        all_sentences.extend([line.strip() for line in lines if line.strip()])

    return all_sentences

# ====== STEP 2: Train a BPE Tokenizer on Both Files ======
def train_bpe_tokenizer(corpus, vocab_size=8000, tokenizer_path="bpe_tokenizer.json"):
    """
    Trains a Hugging Face BPE tokenizer on a given text corpus.

    Args:
        corpus (list): List of text sentences.
        vocab_size (int): Number of vocabulary tokens.
        tokenizer_path (str): Path to save the trained tokenizer.
    """
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Pre-tokenization using whitespace
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Define a trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[" ","<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>", "<TRANS>", "<S>", "</S>"],
        add_prefix_space=True
    )

    # Train the tokenizer on both files' text
    tokenizer.train_from_iterator(corpus, trainer)

    # Save the tokenizer
    tokenizer.save(tokenizer_path)
    print(f"✅ Tokenizer trained on both files and saved to {tokenizer_path}")

# ====== STEP 3: Load the Tokenizer & Tokenize Sentences ======
def load_and_tokenize(tokenizer_path, text):
    """
    Loads a trained tokenizer and tokenizes a sentence.

    Args:
        tokenizer_path (str): Path to the trained tokenizer file.
        text (str): Input sentence.

    Returns:
        tuple: (tokenized_text, token_ids)
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Encode the sentence
    encoding = tokenizer.encode(text)

    print(f"🔹 Original Text: {text}")
    print(f"📝 Tokenized (Subwords): {encoding.tokens}")
    print(f"🔢 Token IDs: {encoding.ids}")

    return encoding.tokens, encoding.ids

# ====== STEP 4: Decode Tokens Back to Text ======
def decode_tokens(tokenizer_path, token_ids):
    """
    Decodes token IDs back to text.

    Args:
        tokenizer_path (str): Path to the trained tokenizer file.
        token_ids (list): List of token IDs.

    Returns:
        str: Decoded sentence.
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)
    decoded_text = tokenizer.decode(token_ids)

    print(f"🔄 Decoded Text: {decoded_text}")
    return decoded_text

# ====== RUN EVERYTHING ======
if __name__ == "__main__":
    # Define the paths to both text files
    file1 = "en_bg_data/train.en"  # Replace with your file path
    file2 = "en_bg_data/train.bg"  # Replace with your file path

    # Load combined text data
    corpus = load_corpus([file1, file2])

    # Train the BPE tokenizer on both files
    train_bpe_tokenizer(corpus, vocab_size=16000, tokenizer_path="bpe_tokenizer.json")

    # Sample text for tokenization
    sample_text = "Проверка дали всичко работи както тряба..."

    # Tokenize and decode
    tokens, ids = load_and_tokenize("bpe_tokenizer.json", sample_text)
    decode_tokens("bpe_tokenizer.json", ids)
