import heapq
import os
from collections import defaultdict
from multiprocessing import Pool
from typing import BinaryIO
from loguru import logger

import regex as re

from cs336_basics.utils.max_heap import MaxHeapItem


class BPETokenizer:
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
            resutls = p.map(pre_tokenize, args)
        for wf in resutls:
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
            heap.append(MaxHeapItem(bytes_pair=pair, freqency=freq))
        heapq.heapify(heap)

        while len(self.vocab) < self.vocab_size and len(heap) > 0:
            top = heapq.heappop(heap)
            bp = top.bytes_pair

            # Already merged or old record
            if bp not in pair_freq or top.freqency != pair_freq[bp]:
                continue

            # logger.debug(f"f={top.freqency}, bp={top.bytes_pair}")

            # update self.merged and self.vocab
            self.merges.append(bp)
            new_bytes = bp[0] + bp[1]
            self.vocab.append(new_bytes)

            positions_to_check = list(pair_position.get(bp, set()))

            for pos in positions_to_check:
                word = word_bytes[pos]

            # new pairs to insert into heap
            new_pairs: list[tuple[bytes, bytes]] = []

            # search overlapped bytes in word_bytes and update
            for pos in pair_position[bp]:
                word_to_merge = word_bytes[pos]
                word_len = len(word_to_merge)
                word_freq = freqs[pos]

                indices = [
                    i
                    for i, (a, b) in enumerate(zip(word_to_merge[:-1], word_to_merge[1:]))
                    if a == bp[0] and b == bp[1]
                ]

                word_merged: list[bytes] = []
                indices_pos = 0
                i = 0

                while i < word_len:
                    if indices_pos < len(indices) and i == indices[indices_pos]:
                        word_merged.append(new_bytes)

                        # update pair_freq

                        # rm cur pair by word_freq
                        pair_freq[bp] -= word_freq
                        if i - 1 >= 0:
                            # prev
                            prev_cur_pair = (word_to_merge[i - 1], new_bytes)
                            if prev_cur_pair not in pair_freq:
                                new_pairs.append(prev_cur_pair)
                            pair_freq[prev_cur_pair] = pair_freq.get(prev_cur_pair, 0) + word_freq
                            pair_position[prev_cur_pair].add(pos)

                            # decrease old pair freq
                            old_prev_cur_pair = (word_to_merge[i - 1], bp[0])
                            pair_freq[old_prev_cur_pair] -= word_freq

                        if i + 2 < word_len:
                            # next
                            cur_next_pair = (new_bytes, word_to_merge[i + 2])
                            if cur_next_pair not in pair_freq:
                                new_pairs.append(cur_next_pair)
                            pair_freq[cur_next_pair] = pair_freq.get(cur_next_pair, 0) + word_freq
                            pair_position[cur_next_pair].add(pos)

                            # decrease old
                            old_cur_next_pair = (bp[1], word_to_merge[i + 2])
                            pair_freq[old_cur_next_pair] -= word_freq

                        indices_pos += 1
                        i += 2
                    else:
                        word_merged.append(word_to_merge[i])
                        i += 1

                # update word_bytes to merged one
                word_bytes[pos] = word_merged

            # rm zero freq pairs in pair_freq
            pair_freq = {k: v for k, v in pair_freq.items() if v != 0}

            # insert into heap
            for p in new_pairs:
                heapq.heappush(heap, MaxHeapItem(freqency=pair_freq[p], bytes_pair=p))

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
        logger.info("Start training~")
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # add special_tokens into vocab
        special_vocab = []
        for st in special_tokens:
            special_vocab.append(st.encode("utf-8"))
        self.vocab = special_vocab + self.vocab

        if len(self.vocab) > self.vocab_size:
            raise ValueError(
                f"vocab size exceeded: length of self.vocab={len(self.vocab)} > self.vocab_size={self.vocab_size}"
            )

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


"""
def _merge(self, word_bytes: list[list[bytes]], freqs: list[int]):
    pair_freq: dict[tuple[bytes, bytes], int] = {}
    pair_position: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for i, (bs, f) in enumerate(zip(word_bytes, freqs)):
        for p in zip(bs[:-1], bs[1:]):
            pair_freq[p] = pair_freq.get(p, 0) + f
            pair_position[p].add(i)

    heap: list[MaxHeapItem] = []
    for pair, freq in pair_freq.items():
        heap.append(MaxHeapItem(bytes_pair=pair, freqency=freq))
    heapq.heapify(heap)

    while len(self.vocab) < self.vocab_size:
        # 找到有效的最大频率 pair
        while heap:
            top = heapq.heappop(heap)
            bp = top.bytes_pair
            if bp in pair_freq and top.freqency == pair_freq[bp]:
                break
        else:
            break  # heap 空了

        self.merges.append(bp)
        new_bytes = bp[0] + bp[1]
        self.vocab.append(new_bytes)

        # 复制一份，因为迭代时会修改
        positions_to_check = list(pair_position.get(bp, set()))
        
        for pos in positions_to_check:
            word = word_bytes[pos]
            freq = freqs[pos]
            
            if len(word) < 2:
                continue
                
            new_word: list[bytes] = []
            i = 0
            
            while i < len(word):
                # 检查是否可以合并
                if i < len(word) - 1 and word[i] == bp[0] and word[i + 1] == bp[1]:
                    # 更新左边 pair 的频率
                    if new_word:
                        old_left = (new_word[-1], bp[0])
                        new_left = (new_word[-1], new_bytes)
                        
                        pair_freq[old_left] = pair_freq.get(old_left, 0) - freq
                        pair_freq[new_left] = pair_freq.get(new_left, 0) + freq
                        pair_position[new_left].add(pos)
                        heapq.heappush(heap, MaxHeapItem(freqency=pair_freq[new_left], bytes_pair=new_left))
                    
                    # 更新右边 pair 的频率
                    if i + 2 < len(word):
                        old_right = (bp[1], word[i + 2])
                        new_right = (new_bytes, word[i + 2])
                        
                        pair_freq[old_right] = pair_freq.get(old_right, 0) - freq
                        pair_freq[new_right] = pair_freq.get(new_right, 0) + freq
                        pair_position[new_right].add(pos)
                        heapq.heappush(heap, MaxHeapItem(freqency=pair_freq[new_right], bytes_pair=new_right))
                    
                    # 减少当前 pair 频率
                    pair_freq[bp] = pair_freq.get(bp, 0) - freq
                    
                    new_word.append(new_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word_bytes[pos] = new_word
        
        # 清理频率为 0 或负数的 pair
        if bp in pair_freq and pair_freq[bp] <= 0:
            del pair_freq[bp]

"""
