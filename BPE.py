from collections import Counter, defaultdict
import heapq

def get_stats(tokens):
    pairs = defaultdict(int)
    for word, freq in tokens.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def byte_pair_encoding(corpus, num_merges=10):
    tokens = {" ".join(word) + " </w>": freq for word, freq in Counter(corpus).items()}
    heap = []

    pair_freqs = get_stats(tokens)
  
    for pair, freq in pair_freqs.items():
        heapq.heappush(heap, (-freq, pair))

    for _ in range(num_merges):
        if not heap:
            break

        _, best_pair = heapq.heappop(heap)
        new_token = "".join(best_pair)

        tokens = {word.replace(" ".join(best_pair), new_token): freq for word, freq in tokens.items()}
        
        pair_freqs = get_stats(tokens)
        heap = [(-freq, pair) for pair, freq in pair_freqs.items()]
        heapq.heapify(heap)

    return tokens

# corpus = ["hello", "hell", "help", "held", "hero", "her"]
# bpe_vocab = byte_pair_encoding_optimized(corpus, num_merges=5)
# print("\nFinal Vocabulary:", bpe_vocab)
