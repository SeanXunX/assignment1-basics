import os
from multiprocessing import Pool
from typing import BinaryIO

import regex as re


class BPETokenizer:
    """
    Binary Pair Encoding tokenizer.
    """

    def __init__(self, num_processes: int) -> None:
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.num_processes = num_processes

        # vocabulary: mapping token id to bytes
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        # merges: used in the merging stage. Merge <t1, t2> into one.
        self.merges: list[tuple[bytes, bytes]] = []

    def _find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        """
        assert isinstance(split_special_token, bytes), "Must represetn special_tokens as a byte string"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        boundaries = [n * chunk_size for n in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size

        read_size = 4096

        for i in range(1, len(boundaries) - 1):
            cur_pos = boundaries[i]
            file.seek(cur_pos)
            while True:
                read_chunk = file.read(read_size)

                # Already at the end of the file
                if read_chunk == b"":
                    boundaries[i] = file_size
                    break

                found_at = read_chunk.find(split_special_token)
                if found_at != -1:
                    boundaries[i] = cur_pos + found_at
                    break
                cur_pos += read_size

        return sorted(set(boundaries))

    def _pre_tokenize_parallel(self):
        """
        Pre tokenize in parallel. Get word frequencies stored at `self.bytes_freq`.
        """
        assert self.special_tokens is not None
        split_pattern_str = "|".join(self.special_tokens)

        with open(self.input_path, "rb") as f:
            # Assuming split token is the first of special tokens
            split_special_token = self.special_tokens[0].encode("utf-8")
            boundaries = self._find_chunk_boundaries(f, self.num_processes, split_special_token)

            # if get fewer chunks
            self.num_processes = len(boundaries)

        word_freq: dict[str, int] = {}

        args = [
            (self.input_path, start, end, split_pattern_str, self.PAT)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        with Pool(processes=self.num_processes) as p:
            resutls = p.map(pre_tokenize, args)
        for wf in resutls:
            for word, count in wf.items():
                word_freq[word] = word_freq.get(word, 0) + count

        self.bytes_freq: dict[tuple[bytes, ...], int] = {
            tuple(bytes([b]) for b in word.encode("utf-8")): count for (word, count) in word_freq.items()
        }

    def _merge(self):
        """
        Merge adjacent bytes pair into one. Get `self.vocab` and `self.merges`.
        """
        pass

    def train(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
    ):
        """
        Trains bpe tokenizer.

        - input_path: str - Data to BPE tokenizer training data.
        - vocab_size: int - A positive integer defines the maximum final vaocabulary size.
        - special_tokens: list[str] - A list of strings to add to the vocabluary.
        """
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        self._pre_tokenize_parallel()
        self._merge()


def pre_tokenize(args: tuple[str, int, int, str, str]) -> dict[str, int]:
    input_path, start, end, split_pattern_str, PAT = args

    pattern = re.compile(PAT)
    split_pattern = re.compile(split_pattern_str)

    word_freq: dict[str, int] = {}

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
        for text in re.split(split_pattern, chunk):
            for match in re.finditer(pattern, text):
                word = match.group()
                word_freq[word] = word_freq.get(word, 0) + 1

    return word_freq
