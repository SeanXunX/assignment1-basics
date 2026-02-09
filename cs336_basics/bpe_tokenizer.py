import json
import math
import regex as re
from collections.abc import Iterable


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges

        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens: list[str] = special_tokens

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self._bytes_to_idx: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "BPETokenizer":
        # load vocab
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab: dict[str, int] = json.load(f)
        vocab: dict[int, bytes] = {idx: token.encode("utf-8") for token, idx in raw_vocab.items()}

        # load merges
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split(" ")
                merges.append((a.encode("utf-8"), b.encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def _encode_one_word(self, word: str) -> list[int]:
        byte_group = [bytes([b]) for b in word.encode("utf-8")]
        merge_rank = {bp: i for i, bp in enumerate(self.merges)}
        while True:
            merge_bp: tuple[bytes, bytes] | None = None  # merge idx and idx + 1
            rank: int = int(math.inf)
            # find merge idx
            for i, bp in enumerate(zip(byte_group[:-1], byte_group[1:])):
                if bp in merge_rank and merge_rank.get(bp, math.inf) < rank:
                    rank = merge_rank[bp]
                    merge_bp = bp
            if merge_bp is None:
                # done
                break
            else:
                # merge target bytes pair
                new_byte_group = []
                i = 0
                while i < len(byte_group):
                    # 1. try match pair
                    if i < len(byte_group) - 1 and merge_bp == (byte_group[i], byte_group[i + 1]):
                        # 2. matched
                        new_byte_group.append(merge_bp)
                        i += 2
                    else:
                        # 3. failed
                        new_byte_group.append(byte_group[i])
                        i += 1
                byte_group = new_byte_group
        return [self._bytes_to_idx[b] for b in byte_group]

    def encode(self, text: str) -> list[int]:
        """
        Encode input str.
        In pretokenized words, iterate over all byte tuples and merge by the ranking order.

        - text: str
        """
        pattern = re.compile(self.PAT)
        split_pattern = re.compile("|".join(self.special_tokens))

        res: list[int] = []

        for text in re.split(split_pattern, text):
            for match in re.finditer(pattern, text):
                word = match.group()
                encoded_word = self._encode_one_word(word)
                res += encoded_word
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_group: list[bytes] = []
        for id in ids:
            token_bytes = self.vocab.get(id)
            assert token_bytes is not None
            byte_group.append(token_bytes)
        all_bytes = b"".join(byte_group)
        text = all_bytes.decode(encoding="utf-8", errors="replace")
        return text
