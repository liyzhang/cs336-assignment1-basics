from collections.abc import Iterable
import heapq

import regex as re
from train_bpe import PAT

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.vocab = vocab
        self.vocab_rev = {t:id for id, t in vocab.items()}
        self.merges = {pair: i for i, pair in enumerate(merges)}

        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else:
            self.special_tokens = []

        # Add special tokens to vocab
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab_rev:
                token_id = len(self.vocab)
                self.vocab[token_id] = token_bytes
                self.vocab_rev[token_bytes] = token_id
    
    '''
    Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the
    same format that your BPE training code output) and (optionally) a list of special tokens.
    This method should accept the following additional parameters: 
    '''
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens_filepath=None):
        pass
 
    def _merge(self, tokens: list[bytes]) -> list[int]:
        queue = []
        for i, p in enumerate(zip(tokens, tokens[1:])):
            if p in self.merges:
                queue.append((self.merges[p], i, i+1, tokens[i], tokens[i+1]))

        heapq.heapify(queue)
        isValid = [True] * len(tokens)
        while len(queue) > 0:
            (_, i, j, ti, tj) = heapq.heappop(queue)
            if ti != tokens[i] or tj != tokens[j]:  # Check if the tokens still match
                continue

            tokens[i] = tokens[i] + tokens[j]
            isValid[j], tokens[j] = False, bytes(0)
            for k in reversed(range(0, i)):
                if isValid[k]: 
                    if (tokens[k], tokens[i]) in self.merges:
                        heapq.heappush(queue, (self.merges[(tokens[k], tokens[i])], k, i, tokens[k], tokens[i]))
                    break
            for k in range(i+1, len(tokens)):
                if isValid[k]:
                    if (tokens[i], tokens[k]) in self.merges:
                        heapq.heappush(queue, (self.merges[(tokens[i], tokens[k])], i, k, tokens[i], tokens[k]))
                    break 

        result = []
        for isValid, token in zip(isValid, tokens):
            if isValid:
                result.append(self.vocab_rev.get(token))
        return result

    '''
    Encode an input text into a sequence of token IDs.
    '''
    def encode(self, text: str) -> list[int]:
        # Placeholder for encoding logic
        documents = [text]
        if len(self.special_tokens) > 0:
            seperators = '(' + '|'.join(re.escape(st) for st in self.special_tokens) + ')'
            documents = re.split(seperators, text)

        results = []
        for doc in documents:
            if doc in self.special_tokens:
                results.append(self.vocab_rev.get(doc.encode("utf-8")))
                continue

            for word in re.findall(PAT, doc):
                raw = word.encode("utf-8")
                if raw in self.special_tokens:
                    results.append(self.vocab_rev.get(raw))
                    continue

                tokens = [raw[i:i+1] for i in range(len(raw))]
                results +=  self._merge(tokens)

        return results
        
    '''
    Given an iterable of
    strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
    required for memory-efficient tokenization of large files that we cannot directly load into
    memory.
    '''
    def encode_iterable(self, iteratable: Iterable[str]) -> Iterable[int]:
        for text in iteratable:
            yield from self.encode(text)

    '''
    Decode a sequence of token IDs into text.
    '''
    def decode(self, ids: list[int]) -> str:
        result = b''
        for id in ids:
            result += self.vocab[id]
        return result.decode("utf-8", errors="replace")