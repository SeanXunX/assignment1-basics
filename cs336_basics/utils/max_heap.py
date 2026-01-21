from dataclasses import dataclass


@dataclass
class MaxHeapItem:
    freqency: int
    bytes_pair: tuple[bytes, bytes]

    def __lt__(self, other):
        return self.freqency > other.freqency
