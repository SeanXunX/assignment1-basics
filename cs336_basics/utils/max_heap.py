from dataclasses import dataclass


@dataclass
class MaxHeapItem:
    frequency: int
    bytes_pair: tuple[bytes, bytes]

    def __lt__(self, other):
        if self.frequency != other.frequency:
            return self.frequency > other.frequency
        return self.bytes_pair > other.bytes_pair
