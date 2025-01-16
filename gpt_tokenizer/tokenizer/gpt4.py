import tiktoken
from .regext import RegexTokenizer


def bpe(mergeable_ranks, token, max_rank):
    # bytes -> bytes (firstly separated and then merged after referring to the dictionary mergeable_ranks)
    """
    Example:
    >>> mergeable_ranks = {b"he": 256, b"ll": 257, b"lo": 258, b"lol": 259, b"lolit": 260}
    >>> token = b"hello she is lolita"
    >>> bpe(mergeable_ranks, token, 299)
    [b'he', b'll', b'o', b' ', b's', b'he', b' ', b'i', b's', b' ', b'lol', b'i', b't', b'a']
    """
    parts = [bytes([i]) for i in token]  # split the token
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i  # the minimum index of the pair to be merged
                min_rank = rank  # the minimum rank in the mergeable_ranks dict
        # used to find the smallest mergeable index in the current pair and the corresponding position in the mergeable_ranks
        if min_rank is None or (min_rank is not None and min_rank >= max_rank):
            # Note that the "equal to" above is very important
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] +
                                   parts[min_idx+1]] + parts[min_idx+2:]
    return parts


def recover_merge(mergeable_ranks):
    # reveal the process by which the merge occurs
    """
    Example:
    >>> mergeable_ranks = {b"h": -1, b"e": -2, b"l": -3, b"o": -4, b"he": 1, b"ll": 2, b"lo": 3, b"hell": 4, b"hello": 5}
    >>> recover_merge(mergeable_ranks)
    {(-1, -2): 1, (-3, -3): 2, (-3, -4): 3, (1, 2): 4, (4, -4): 5}
    """
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        merges[(mergeable_ranks[pair[0]], mergeable_ranks[pair[1]])] = rank
    return merges


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}


class GPT4Tokenizer(RegexTokenizer):

    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        super().__init__(pattern)
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        self.merges = recover_merge(mergeable_ranks)
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        self.byte_shuffle = {
            i: mergeable_ranks[bytes([i])] for i in range(256)}
        # A mapping that associates the original (messy) rank with the standard rank
        self.reverse_byte_shuffle = {
            v: k for k, v in self.byte_shuffle.items()}
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, text_bytes):
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids = super()._encode_chunk(text_bytes)
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[i] for i in ids)
        text_bytes = bytes(self.reverse_byte_shuffle[b] for b in text_bytes)
        text = text_bytes.decode('utf-8', errors='replace')
        return text

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def save(self, file_prefix):
        raise NotImplementedError("GPT4Tokenizer cannot be saved.")

    def load(self, model_file):
        raise NotImplementedError("GPT4Tokenizer cannot be loaded.")

    def save_vocab(self, vocab_file):
        from .base import render_token
        vocab = {idx: bytes([self.reverse_byte_shuffle[idx]])
                 for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")
