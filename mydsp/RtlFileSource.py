
import numpy as np


class RtlFileSource:
    """
    Reads unsigned 8-bit interleaved IQ files produced by rtl_sdr.

    Samples are returned as a NumPy complex64 array with values
    approximately in the range [-1.0, +1.0].

    File format:
        I0 Q0 I1 Q1 I2 Q2 ...

    Example
    -------
        src = RtlFileSource("capture.iq", 262144)

        while True:
            frame = src.getComplexFrame()
            if frame is None:
                break

            # Process frame...

        src.close()
    """

    def __init__(self, filename, frame_size):
        self.filename = filename
        self.frame_size = frame_size
        self.file = open(filename, "rb")
        self.num_channels = 2
    def getMultiFrame(self):
        return self.getFrame()

    def getFrame(self):
        """
        Returns the next block as an (N,2) float32 array.

        Column 0 : I samples
        Column 1 : Q samples

        Returns None on EOF.
        """

        raw = np.fromfile(
            self.file,
            dtype=np.uint8,
            count=2 * self.frame_size
        )

        if len(raw) == 0:
            return None

        if len(raw) & 1:
            raw = raw[:-1]

        raw = raw.astype(np.float32)

        i = (raw[0::2] - 128.0) / 128.0
        q = (raw[1::2] - 128.0) / 128.0

        return np.column_stack((i, q))

    def getComplexFrame(self):
        frame = self.getFrame()

        if frame is None:
            return None

        return (frame[:, 0] + 1j * frame[:, 1]).astype(np.complex64)


    def rewind(self):
        """Seek back to the beginning of the file."""
        self.file.seek(0)

    def close(self):
        """Close the IQ file."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()