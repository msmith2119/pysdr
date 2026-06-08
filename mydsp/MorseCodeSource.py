
import re
from pathlib import Path
import numpy as np

from mydsp.Utils import create_ola_function

morse = {
    # Letters
    "a": [0, 1],
    "b": [1, 0, 0, 0],
    "c": [1, 0, 1, 0],
    "d": [1, 0, 0],
    "e": [0],
    "f": [0, 0, 1, 0],
    "g": [1, 1, 0],
    "h": [0, 0, 0, 0],
    "i": [0, 0],
    "j": [0, 1, 1, 1],
    "k": [1, 0, 1],
    "l": [0, 1, 0, 0],
    "m": [1, 1],
    "n": [1, 0],
    "o": [1, 1, 1],
    "p": [0, 1, 1, 0],
    "q": [1, 1, 0, 1],
    "r": [0, 1, 0],
    "s": [0, 0, 0],
    "t": [1],
    "u": [0, 0, 1],
    "v": [0, 0, 0, 1],
    "w": [0, 1, 1],
    "x": [1, 0, 0, 1],
    "y": [1, 0, 1, 1],
    "z": [1, 1, 0, 0],

    # Digits
    "0": [1, 1, 1, 1, 1],
    "1": [0, 1, 1, 1, 1],
    "2": [0, 0, 1, 1, 1],
    "3": [0, 0, 0, 1, 1],
    "4": [0, 0, 0, 0, 1],
    "5": [0, 0, 0, 0, 0],
    "6": [1, 0, 0, 0, 0],
    "7": [1, 1, 0, 0, 0],
    "8": [1, 1, 1, 0, 0],
    "9": [1, 1, 1, 1, 0],

    # Punctuation
    ".": [0, 1, 0, 1, 0, 1],
    ",": [1, 1, 0, 0, 1, 1],
    "?": [0, 0, 1, 1, 0, 0],
    "'": [0, 1, 1, 1, 1, 0],
    "!": [1, 0, 1, 0, 1, 1],
    "/": [1, 0, 0, 1, 0],
    "(": [1, 0, 1, 1, 0],
    ")": [1, 0, 1, 1, 0, 1],
    "&": [0, 1, 0, 0, 0],
    ":": [1, 1, 1, 0, 0, 0],
    ";": [1, 0, 1, 0, 1, 0],
    "=": [1, 0, 0, 0, 1],
    "+": [0, 1, 0, 1, 0],
    "-": [1, 0, 0, 0, 0, 1],
    "_": [0, 0, 1, 1, 0, 1],
    "\"": [0, 1, 0, 0, 1, 0],
    "$": [0, 0, 0, 1, 0, 0, 1],
    "@": [0, 1, 1, 0, 1, 0],

    # Prosign often used for "error"
    "#": [0, 0, 0, 0, 0, 0, 0, 0]  # eight dits (HH)
}

class MorseCodeSource:
    def __init__(self,sample_rate,frame_size,msgFile,amplitude,wpm,tone):

        self.frame_size = frame_size
        self.num_channels = 1
        self.msgFile = msgFile
        self.wpm = wpm
        self.tone = tone
        self.amplitude = amplitude
        self.sample_rate = sample_rate
        self.dit_ms = 1200.0 / wpm
        self.dah_ms = 3 * self.dit_ms
        self.dM = int(self.dit_ms*sample_rate/1000.0)
        self.dN = int(self.dah_ms*sample_rate/1000.0)
        self.W = 2*3.1415*tone/sample_rate

        env_dit = self.getEnvelope(self.dM,int(0.1*self.dM))
        env_dah = self.getEnvelope(self.dN,int(0.1*self.dN))
        self.dit_array = np.zeros(self.dM)
        self.dah_array = np.zeros(self.dN)
        i_dM = np.arange(self.dM)
        i_dN = np.arange(self.dN)
        self.dit_array = self.amplitude*np.sin(self.W * i_dM)*env_dit[i_dM]
        self.dah_array = self.amplitude*np.sin(self.W * i_dN)*env_dah[i_dN]
        self.dspace = np.zeros(self.dM)
        self.cspace = np.zeros(self.dN)
        self.wspace = np.zeros(self.dM*7)
        self.curpos = 0

        print(f"dit array type = {type(self.dit_array)}")

        self.loadMsg()
        self.curpos = 0

        self.summary_text = f"Morse Source : sample_rate = {self.sample_rate} frame_size = {self.frame_size} ,msg_file = {self.msgFile} ,wpm = {self.wpm} ,tone = {self.tone}"


    def loadMsg(self):

        if not Path(self.msgFile).exists():
            print(f"MorseCodeSource: {self.msgFile} not found")
            return 1

        with open(self.msgFile, "r", encoding="utf-8") as f:
            text = f.read()

        self.words = re.findall(r"\b\w+\b", text.lower())
        total_chars = sum(len(self.words) for w in self.words)
        t = 0
        for word in self.words:

            for c in word:

                for q in morse[c]:

                    if q == 0:
                        t+=self.dit_ms
                    else:
                        t+=self.dah_ms
                    t+=self.dit_ms
                t+=self.dah_ms
            t+=7*self.dit_ms


        maxGuess = int(t*self.sample_rate/1000.0)
        print(f"maxGuess={maxGuess}")
        self.y = np.empty(maxGuess,dtype=float)



        for word in self.words:
            for i  in range(len(word)):
                self.addChar(word[i])

            self.y[self.curpos:self.curpos + len(self.wspace)] = self.wspace
            self.curpos += len(self.wspace)

        print(f"len y = {len(self.y)}")
        print(f"curpos  is {self.curpos}")
        return 1


    def addChar(self,character):

        seq = morse[character]
        for i in range(len(seq)):
            if seq[i] == 0:
                self.y[self.curpos:self.curpos + len(self.dit_array)] = self.dit_array
                self.curpos += len(self.dit_array)
            elif seq[i] == 1:
                self.y[self.curpos:self.curpos + len(self.dah_array)] = self.dah_array
                self.curpos += len(self.dah_array)
            self.y[self.curpos:self.curpos + len(self.dspace)] = self.dspace
            self.curpos += len(self.dspace)
        self.y[self.curpos:self.curpos + len(self.cspace)] = self.cspace
        self.curpos += len(self.cspace)


    def getMultiFrame(self):


        frame = np.zeros(self.frame_size, dtype=self.y.dtype)
        available = len(self.y) - self.curpos
        if available > 0:
            n = min(available, self.frame_size)
            frame[:n] = self.y[self.curpos:self.curpos + n]
            self.curpos += n
            return frame[:, None]
        else:
            return None


    def getEnvelope(self,m,n):


        hwin = create_ola_function(m, n)
        arr = np.array([hwin(i) for i in range(m)])
        return arr

    def summary(self):
        return self.summary_text

    def close(self):
        print("close")



