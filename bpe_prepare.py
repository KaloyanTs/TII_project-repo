import re
import json
import os

specials = ["<PAD>", "<UNK>", " ", "<S>", "</S>", "<TRANS>"]


def split_text_keep_special(text, special=specials):
    special = sorted(special, key=len, reverse=True)

    special_pattern = "|".join(re.escape(s) for s in special)

    pattern = f"({special_pattern})"

    result = []

    parts = re.split(pattern, text)

    for part in parts:
        if part in special:
            result.append(part)
        else:
            result.extend(list(part))

    return result


def get_stats(text, tokens, id2token):
    pairs = {}
    specials_ids = [tokens[special] for special in specials]
    symbols = text
    for i in range(len(symbols) - 1):
        if (
            symbols[i] in specials_ids
            or symbols[i + 1] in specials_ids
            or symbols[i] == -1
            or symbols[i + 1] == -1
        ):
            continue
        if (symbols[i], symbols[i + 1]) not in pairs:
            pairs[(symbols[i], symbols[i + 1])] = 0
        pairs[(symbols[i], symbols[i + 1])] += 1
    return sorted(pairs.items(), key=lambda x: x[1], reverse=True)


def merge(text, v_in, to):
    v_out = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word in v_in:
        w_out = word.replace(bigram, replacement)
        v_out[w_out] = v_in[word]
    return v_out


def apply_rule(lst, a, b, c):
    result = []

    for i, num in enumerate(lst[:-1]):
        if i > 0 and lst[i - 1] == a and lst[i] == b:
            continue
        if lst[i] == a and lst[i + 1] == b:
            result.append(c)
        else:
            result.append(num)

    return result + [lst[-1]]


def byte_pair_encoding(text, limit):
    split_text = split_text_keep_special(text)
    unique_tokens = list(set(split_text + specials))
    # print(unique_tokens)

    tokens = {token: idx for idx, token in enumerate(unique_tokens)}
    tokens["\n"] = -1
    id2token = {idx: token for token, idx in tokens.items()}
    text = [tokens[c] for c in split_text]

    # print([f"{sp}: {tokens.get(sp)}" for sp in specials])
    # print("text", text[:10])

    while len(tokens) < limit:
        if(len(tokens) % 10 == 0):
            print(f"Tokens: {len(tokens)}/{limit}")
        pairs = get_stats(text, tokens, id2token)
        # print(pairs)
        # print(max(pairs, key=lambda x: x[1]))
        best_pair = max(pairs, key=lambda x: x[1])
        new_token = "".join((id2token[best_pair[0][0]], id2token[best_pair[0][1]]))
        # print("new token", "(" + new_token + ")", len(tokens))
        id2token[len(tokens)] = new_token
        tokens[new_token] = len(tokens)

        text = apply_rule(text, best_pair[0][0], best_pair[0][1], len(tokens) - 1)

        # print('-------------------')
        # print([f"{sp}: {tokens.get(sp)}" for sp in specials])
        # print(text[:10])

    tokens.pop("\n", None)
    id2token.pop(-1, None)

    return tokens, id2token


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


def encode(text, tokens):
    encoded_text = []
    i = 0
    while i < len(text):
        match = None
        for token in sorted(tokens.keys(), key=lambda x: len(x), reverse=True):
            if text[i : i + len(token)] == token:
                match = token
                break
        if match:
            encoded_text.append(tokens[match])
            i += len(match)
        elif text[i] == "\n":
            i += 1
        else:
            encoded_text.append(tokens["<UNK>"])
            i += 1
    return encoded_text

def decode(encoded_text, id2token):
    decoded_text = ""
    for token in encoded_text:
        decoded_text += id2token[token]
    return decoded_text


if __name__ == "__main__":
    file_path_en = "en_bg_data/dev.en"
    file_path_bg = "en_bg_data/dev.bg"
    limit = 1000
    text_en = read_text_file(file_path_en)
    text_bg = read_text_file(file_path_bg)
    token_dict, id2token = byte_pair_encoding(text_en + "\n" + text_bg, limit)
    # print(id2token)
    # print(token_dict)

    # text = read_text_file(file_path)
    # encoded = encode(text, token_dict)
    # print(encoded)
    # text = decode(encoded, id2token)
    # print(text)

    text = "A problem has been found with all the matters that were discussed in the Council.<TRANS>При всички теми, обсъждани в Съвета, се установи проблем."
    encoded = encode(text, token_dict)
    print([id2token[token] for token in encoded])
    text = decode(encoded, id2token)
    print(text)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    tokens_file_path = os.path.join(current_directory, "tokens.json")

    with open(tokens_file_path, "w", encoding="utf-8") as tokens_file:
        json.dump(token_dict, tokens_file, ensure_ascii=False, indent=4)
