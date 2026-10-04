from __future__ import annotations
import regex as re
import heapq
import multiprocessing

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Rev:
    def __init__(self, p):
        self.p = p

    def __lt__(self, other):
        return self.p > other.p

def pretokenize_doc(document: str):
    word_freq: dict[str, int] = {}
    word_tokens: dict[str, tuple[bytes, ...]] = {}
    for word in re.findall(PAT, document):
        raw = word.encode("utf-8")
        tokens = tuple(raw[i:i+1] for i in range(len(raw)))
        
        if word not in word_freq:
            word_freq[word] = 1
            word_tokens[word] = tokens
        else:
            word_freq[word] += 1
    return word_freq, word_tokens

def pretokenize(text:str, special_tokens:list[str]) -> tuple[dict[str, int], dict[str, tuple[bytes, ...]]]:
    if special_tokens and len(special_tokens) > 0:
        seperators  = "|".join(re.escape(st) for st in special_tokens)
        documents = re.split(seperators, text)
    else:
        documents = [text]

    print("documents:", len(documents))
    with multiprocessing.Pool() as pool:
        results = pool.map(pretokenize_doc, documents)

    # use dict to save space
    word_freq: dict[str, int] = {}
    word_tokens: dict[str, tuple[bytes, ...]] = {}

    for local_freq, local_tokens in results:
        for word, freq in local_freq.items():
            word_freq[word] = word_freq.get(word, 0) + freq

            if word not in word_tokens:
                word_tokens[word] = local_tokens[word]

    return word_freq, word_tokens


def find_max_pair(pair_freq: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    return max(pair_freq, key=lambda x: (pair_freq[x], x))


def train_bpe(
  input_path: str,
  vocab_size: int,
  special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. read the file from the input_path, run pre-processing
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    # 2. pretokenize the text, build word_freq, each word is split into tuple(byte, ...)
    word_freq, word_tokens = pretokenize(text, special_tokens)

    # 3. initialize vocab
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i]) 
    
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    # 4. initialize pair_freq and pair_occurrence
    pair_freq: dict[tuple[bytes, bytes], int] = {}
    pair_occurence: dict[tuple[bytes, bytes], set[str]] = {}
    for word, cnt in word_freq.items():
        old = word_tokens.get(word, ())
        for i in range(len(old)-1):
            pair = (old[i], old[i+1])
            if pair not in pair_freq:
                pair_freq[pair] = 0
            pair_freq[pair] += cnt
            if pair not in pair_occurence:
                pair_occurence[pair] = set()
            pair_occurence[pair].add(word)

    
    pair_freq_heap = [(-f, Rev(p)) for p, f in pair_freq.items()]
    heapq.heapify(pair_freq_heap)

    # 5. record merge order
    merged: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and len(pair_freq_heap) > 0:
        neg_pair_count, rev = heapq.heappop(pair_freq_heap)
        max_pair = rev.p
        if -neg_pair_count != pair_freq.get(max_pair, 0):
            continue

        l, r = max_pair
        new_token = l + r
        # update pair_freq and pair_occurence
        for word in pair_occurence[max_pair]:
            count = word_freq[word]
            old = word_tokens[word]
            if len(old) == 1:
                continue

            # remove all the old pairs
            deltas = {}
            for p in zip(old, old[1:]):
                deltas[p] = deltas.get(p, 0) - count
            
            # add new pairs to pair_freq and pair_occurence
            new, i = [], 0
            while i < len(old) - 1:
                if old[i] == l and old[i+1] == r:
                    new.append(new_token)
                    i+=2
                else:
                    new.append(old[i])
                    i+=1
            if(i == len(old) - 1):
                new.append(old[i])

            word_tokens[word] = tuple(new)
            for p in zip(new, new[1:]):
                deltas[p] = deltas.get(p, 0) + count

            for p, d in deltas.items():
                if d == 0:
                    continue

                pair_freq[p] = pair_freq.get(p, 0) + d
                if pair_freq[p]==0:
                    pair_freq.pop(p)
                    pair_occurence.pop(p)
                else:
                    heapq.heappush(pair_freq_heap, (-pair_freq[p], Rev(p)))
                    pair_occurence.setdefault(p, set()).add(word)
        
        # pair_freq.pop(max_pair)     
        # pair_occurence.pop(max_pair)

        # update vocab and merged
        vocab[len(vocab)] = new_token
        merged.append(max_pair)

        if len(vocab) % 100 == 0:
            print("Vocab size:", len(vocab))
    
    return vocab, merged

def main():
    # test your code here
    input_path = "/workspaces/cs336-assignment1-basics/data/test.txt"
    vocab_size=500
    special_tokens=["<|endoftext|>"]
    vocab, merged =train_bpe(input_path, vocab_size, special_tokens)
    print("Vocab size:", len(vocab))
    print("Merged pairs:", len(merged))

if __name__ == "__main__":
   main()   