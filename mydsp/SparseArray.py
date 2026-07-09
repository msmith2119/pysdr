

from bisect import bisect_left
import pickle

class SparseArray:
    """
    Sparse array implemented with two sorted parallel lists.

        indices = [2, 10, 25]
        values  = ["a", "b", "c"]

    Lookup:      O(log N)
    Insert:      O(N)
    Delete:      O(N)
    Iteration:   O(N)
    """

    def __init__(self, default=0):
        self.default = default
        self._indices = []
        self._values = []

    def __len__(self):
        """Number of stored elements."""
        return len(self._indices)

    def __contains__(self, index):
        pos = bisect_left(self._indices, index)
        return pos < len(self._indices) and self._indices[pos] == index

    def __getitem__(self, index):
        pos = bisect_left(self._indices, index)

        if pos < len(self._indices) and self._indices[pos] == index:
            return self._values[pos]

        return self.default

    def __setitem__(self, index, value):
        pos = bisect_left(self._indices, index)

        if pos < len(self._indices) and self._indices[pos] == index:
            if value == self.default:
                del self._indices[pos]
                del self._values[pos]
            else:
                self._values[pos] = value
        else:
            if value != self.default:
                self._indices.insert(pos, index)
                self._values.insert(pos, value)

    def __delitem__(self, index):
        pos = bisect_left(self._indices, index)

        if pos < len(self._indices) and self._indices[pos] == index:
            del self._indices[pos]
            del self._values[pos]

    def clear(self):
        self._indices.clear()
        self._values.clear()

    def keys(self):
        return iter(self._indices)

    def values(self):
        return iter(self._values)

    def items(self):
        return zip(self._indices, self._values)

    def min_index(self):
        return self._indices[0] if self._indices else None

    def max_index(self):
        return self._indices[-1] if self._indices else None

    def dense(self):
        """
        Return a dense list spanning min_index..max_index.
        """
        if not self._indices:
            return []

        result = []
        j = 0

        for i in range(self._indices[0], self._indices[-1] + 1):
            if j < len(self._indices) and self._indices[j] == i:
                result.append(self._values[j])
                j += 1
            else:
                result.append(self.default)

        return result

    def __iter__(self):
        """Iterate over indices."""
        return iter(self._indices)

    def __repr__(self):
        items = ", ".join(
            f"{i}:{v}" for i, v in zip(self._indices, self._values)
        )
        return f"SparseArray({{{items}}})"

    def save(self, filename):
        """Save the SparseArray to a file."""
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "indices": self._indices,
                    "values": self._values,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, filename):
        """Load a SparseArray from a file."""
        with open(filename, "rb") as f:
            data = pickle.load(f)

        obj = cls()
        obj._indices = data["indices"]
        obj._values = data["values"]
        return obj