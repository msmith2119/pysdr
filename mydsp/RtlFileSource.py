
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
    def __init__(self, filename, frame_size,loop=False):
        self.filename = filename
        self.frame_size = frame_size
        self.file = open(filename, "rb")
        self.num_channels = 1
        self.loop=loop



    def getFrame(self):

        raw = self.getRawFrame()

        if raw is None:
            return None

        if len(raw) & 1:
            raw = raw[:-1]


        samples = np.frombuffer(raw, dtype=np.uint8)


        i = (samples[0::2] - 128.0) / 128.0
        q = (samples[1::2] - 128.0) / 128.0

        col = i.astype(np.complex64) + 1j * q.astype(np.complex64)
        samples_read = len(col)
        if samples_read < self.frame_size:
            pad_rows = self.frame_size - samples_read
            padding = np.zeros(pad_rows,dtype=np.complex64)
            col = np.concatenate(( col,padding))

        return np.column_stack([col])


    def getRawFrame(self):
        samplesNeeded = self.frame_size
        chunks = []

        while samplesNeeded > 0:


            raw = np.fromfile(
                self.file,
                dtype=np.uint8,
                count=2 * self.frame_size
            )

            if len(raw) == 0:

                if not self.loop:

                    if len(chunks) == 0:
                        return None

                    break
                self.file.seek(0)

                continue

            samplesRead = len(raw) // 2

            chunks.append(raw)
            samplesNeeded -= samplesRead

        raw = b"".join(chunks)

        # convert raw -> float ndarray
        return raw

    def rewind(self):
        """Seek back to the beginning of the file."""
        self.file.seek(0)


    def start(self):
        return

    def close(self):
        """Close the IQ file."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def get_num_channels(self):
        return self.num_channels

    def summary(self):
        return f"Rtl Source frame_size={self.frame_size}, loop={self.loop}"

        """Return a summary of the IQ file."""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()