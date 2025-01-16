import unicodedata


def get_stats(ids, counts=None):
    """
    Count consecutive pairs of integers in a list.
    Example:
    >>> get_stats([1, 2, 3, 1, 2])
    {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    Combine two integers into a new index when they match a specified pair.
    Example:
    >>> ids=[1, 2, 3, 1, 2]
    >>> pair=(1, 2)
    >>> idx=4
    >>> merge(ids, pair, idx)
    [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def replace_control_characters(s: str) -> str:
    """
    Replace control characters with their Unicode escape sequences.
    Example:
    >>> replace_control_characters("Hello\bWorld")
    'Hello\\\\u0008World'
    """
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)


def render_token(t: bytes) -> str:
    """
    Example:
    >>> t = b"Hello, World! This\x01is a test.\x08"
    >>> render_token(t)
    'Hello, World! This\\\\u0001is a test.\\\\u0008'
    """
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s


class Tokenizer:

    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.pattern = ""
        self.special_tokens = {}  # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab()  # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        # int -> bytes
        # A dictionary that maps integers to bytes
        # based on the known merges and special_tokens dictionary
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]  # concatenate two bytes
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode('utf-8')
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for p0, p1 in self.merges:
                f.write(f"{p0} {p1}\n")
        # --- --- --- --- --- ---
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    p0, p1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[p0])
                    s1 = render_token(self.vocab[p1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding='utf-8') as f:
            version = f.readline().strip()
            assert version == "minbpe v1"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                p0, p1 = map(int, line.strip().split())
                merges[(p0, p1)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
