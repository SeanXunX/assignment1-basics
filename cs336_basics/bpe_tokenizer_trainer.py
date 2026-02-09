import heapq
import os
from collections import defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

from cs336_basics.utils.max_heap import MaxHeapItem


class BPETokenizerTrainer:
    """
    Binary Pair Encoding tokenizer.
    """

    def __init__(self, num_processes: int = 4) -> None:
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.num_processes = num_processes

        # vocabulary: mapping token id to bytes. (id=i denotes the ith element)
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]

        # default vocabulary size
        self.vocab_size: int = 256

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

    def _pre_tokenize_parallel(self) -> tuple[list[list[bytes]], list[int]]:
        """
        Pre tokenize in parallel. Get freq of bytes in words.
        """
        assert self.special_tokens is not None
        split_pattern_str = "|".join(self.special_tokens)

        with open(self.input_path, "rb") as f:
            # Assuming split token is the first of special tokens
            split_special_token = self.special_tokens[0].encode("utf-8")
            boundaries = self._find_chunk_boundaries(f, self.num_processes, split_special_token)

            # if get fewer chunks
            self.num_processes = len(boundaries) - 1

        word_freq: dict[str, int] = {}

        args = [
            (self.input_path, start, end, split_pattern_str, self.PAT)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        with Pool(processes=self.num_processes) as p:
            results = p.map(pre_tokenize, args)
        for wf in results:
            for word, count in wf.items():
                word_freq[word] = word_freq.get(word, 0) + count

        word_bytes: list[list[bytes]] = []
        freqs: list[int] = []

        for word, count in word_freq.items():
            bytes_list = [bytes([b]) for b in word.encode("utf-8")]
            word_bytes.append(bytes_list)
            freqs.append(count)

        return word_bytes, freqs

    def _merge(self, word_bytes: list[list[bytes]], freqs: list[int]):
        """
        Merge adjacent bytes pair into one. Get `self.vocab` and `self.merges`.
        """
        # bytes_freq maintains the gloabal bytes pair frequency
        pair_freq: dict[tuple[bytes, bytes], int] = {}

        # indices pair postions (in which words)
        pair_position: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

        for i, (bs, f) in enumerate(zip(word_bytes, freqs)):
            for p in zip(bs[:-1], bs[1:]):
                pair_freq[p] = pair_freq.get(p, 0) + f
                pair_position[p].add(i)

        # max heap to get most frequent bytes pair.
        # ! Use lazy deletion. Check the pair_freq to find if exist.
        heap: list[MaxHeapItem] = []

        for pair, freq in pair_freq.items():
            heap.append(MaxHeapItem(bytes_pair=pair, frequency=freq))
        heapq.heapify(heap)

        while len(self.vocab) < self.vocab_size and len(heap) > 0:
            top = heapq.heappop(heap)
            bp = top.bytes_pair

            # Already merged, skip
            if bp not in pair_freq:
                continue

            # old record, update
            if top.frequency != pair_freq[bp]:
                heapq.heappush(heap, MaxHeapItem(bytes_pair=top.bytes_pair, frequency=pair_freq[bp]))
                continue

            # update self.merged and self.vocab
            self.merges.append(bp)
            new_bytes = bp[0] + bp[1]
            self.vocab.append(new_bytes)

            positions_to_check = list(pair_position.get(bp, set()))

            for pos in positions_to_check:
                word = word_bytes[pos]
                freq = freqs[pos]

                if len(word) < 2:
                    continue

                new_word: list[bytes] = []

                i = 0
                while i < len(word):
                    # check if bp found
                    if i < len(word) - 1 and word[i] == bp[0] and word[i + 1] == bp[1]:
                        # update self
                        pair_freq[bp] = pair_freq.get(bp, 0) - freq
                        pair_position[bp].discard(pos)

                        # update left
                        if new_word:
                            old_left = (new_word[-1], bp[0])
                            new_left = (new_word[-1], new_bytes)

                            pair_freq[old_left] = pair_freq.get(old_left, 0) - freq
                            pair_freq[new_left] = pair_freq.get(new_left, 0) + freq
                            pair_position[new_left].add(pos)
                            heapq.heappush(heap, MaxHeapItem(bytes_pair=new_left, frequency=pair_freq[new_left]))

                        # update right
                        if i + 2 < len(word):
                            old_right = (bp[1], word[i + 2])
                            new_right = (new_bytes, word[i + 2])

                            pair_freq[old_right] = pair_freq.get(old_right, 0) - freq
                            pair_freq[new_right] = pair_freq.get(new_right, 0) + freq
                            pair_position[new_right].add(pos)
                            heapq.heappush(heap, MaxHeapItem(bytes_pair=new_right, frequency=pair_freq[new_right]))

                        new_word.append(new_bytes)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                word_bytes[pos] = new_word

            # rm zero freq pairs in pair_freq
            pair_freq = {k: v for k, v in pair_freq.items() if v != 0}

            assert bp not in pair_freq

    def train(
        self,
        input_path: str | os.PathLike,
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

        # add special_tokens into vocab
        for st in special_tokens:
            self.vocab.append(st.encode("utf-8"))

        if len(self.vocab) > self.vocab_size:
            raise ValueError(
                f"vocab size exceeded: length of self.vocab={len(self.vocab)} > self.vocab_size={self.vocab_size}"
            )

        if len(special_tokens) == 0:
            pass

        word_freq_info = self._pre_tokenize_parallel()
        self._merge(*word_freq_info)


def pre_tokenize(args: tuple[str | os.PathLike, int, int, str, str]) -> dict[str, int]:
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
