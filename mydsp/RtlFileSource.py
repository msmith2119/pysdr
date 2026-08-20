
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
    description = "Source RtlFile <name> frame_size=<frame_size> path=<path>"
    def __init__(self, filename, frame_size):
        self.filename = filename
        self.frame_size = frame_size
        self.file = open(filename, "rb")
        self.num_channels = 1
        self.summary_text = f"Rtl Source frame_size={self.frame_size}"



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


        col = i.astype(np.complex64) + 1j * q.astype(np.complex64)

        f = np.column_stack([col])

        return f





    def rewind(self):
        """Seek back to the beginning of the file."""
        self.file.seek(0)

    def close(self):
        """Close the IQ file."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def get_num_channels(self):
        return self.num_channels

    def summary(self):
        return self.summary_text

        """Return a summary of the IQ file."""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()