import re, time, sys
from pathlib import Path
from collections import defaultdict
from typing import Iterator, Tuple, Dict, Optional, Union, List
import matplotlib.pyplot as plt 
import csv

import retico_core
from retico_core import abstract, UpdateMessage, UpdateType
from retico_core.text import TextIU, SpeechRecognitionIU

from nltk.stem import WordNetLemmatizer

# 1. Lexicon setup
LEXICON_DIR = Path(__file__).parent / "VAD"
FILES = {
    "valence":   "valence-NRC-VAD-Lexicon-v2.1.txt",
    "arousal":   "arousal-NRC-VAD-Lexicon-v2.1.txt",
    "dominance": "dominance-NRC-VAD-Lexicon-v2.1.txt",
}

def load_lexicon(fname: Path) -> dict[str, float]:
    with fname.open(encoding="utf-8") as f:
        next(f)
        return {w: float(v) for w,v in (line.split("\t") for line in f)}

lexicon = {m: load_lexicon(LEXICON_DIR / fn) for m,fn in FILES.items()}

# 2. Tokenizer + lemmatizer
WORD_RE = re.compile(r"[A-Za-z']+")
_lemmatizer = WordNetLemmatizer()

def tokenize(text: str) -> List[str]:
    raw = WORD_RE.findall(text)
    return [w.lower() for w in raw]

# 3. incremental VAD
def incremental_vad_scores(
    text: Union[str, List[str]],
    missing: Optional[float] = None
) -> Iterator[Tuple[str, Dict[str, Optional[float]]]]:
    tokens = tokenize(text) if isinstance(text, str) else [w.lower() for w in text]
    sums = defaultdict(float)
    cnts = defaultdict(int)

    for tok in tokens:
        lem = _lemmatizer.lemmatize(tok, pos="v")
        for metric, table in lexicon.items():
            if tok in table:
                score = table[tok]
            elif lem in table:
                score = table[lem]
            else:
                continue
            sums[metric] += score
            cnts[metric] += 1

        cur = {}
        for metric in FILES:
            if cnts[metric]:
                cur[metric] = sums[metric] / cnts[metric]
            elif missing is None:
                cur[metric] = None
            else:
                cur[metric] = missing

        yield tok, cur

# 4. Module
class EmotionTextIU(TextIU):
    @staticmethod
    def type():
        return TextIU.type()
    def __repr__(self):
        # show the full payload without truncation
        return f"{self.type()} - ({self.creator.name()}): {self.get_text()}"


class EmotionTrackingModule(abstract.AbstractModule):
    @staticmethod
    def name():
        return "Emotion Tracking Module"

    @staticmethod
    def description():
        return "Incrementally tracks valence/arousal/dominance."

    @staticmethod
    def input_ius():
        return [SpeechRecognitionIU]

    @staticmethod
    def output_iu():
        return EmotionTextIU

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_scores = {m: None for m in FILES}
        self.last_ts = 0.0
        self.csv_path = Path(__file__).parent / "vad_data.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp","valence","arousal","dominance"])

    def process_update(self, um):
        for iu, typ in um:
            if typ == UpdateType.ADD:
                return self.process_iu(iu)
        return UpdateMessage()

    def process_iu(self, input_iu):
        text = getattr(input_iu, "text", "")
        scores = None
        # only overwrite when the token actually had a lexicon match
        for _, cur in incremental_vad_scores(text, missing=0.0):
            # check if any dimension is non‐zero
            if any(val != 0.0 for val in cur.values()):
                scores = cur
        if not scores:
            return UpdateMessage()

        now = time.time()
        if scores == self.last_scores and now - self.last_ts < 1.0:
            return UpdateMessage()

        self.last_scores, self.last_ts = scores, now

        # write one row per new score
        now_ts = time.time()
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                f"{now_ts:.3f}",
                f"{scores['valence']:.3f}",
                f"{scores['arousal']:.3f}",
                f"{scores['dominance']:.3f}"
            ])

        payload_str = f"V/A/D = {scores['valence']:.2f}/" \
                      f"{scores['arousal']:.2f}/" \
                      f"{scores['dominance']:.2f}"
        print(f"Current Speaker Verbal Emotional States: {payload_str}")
        out_iu = self.create_iu(input_iu)
        out_iu.payload = payload_str
        # print(f"EmotionTrackingModule: Output IU: {out_iu.payload}")
        return abstract.UpdateMessage.from_iu(out_iu, abstract.UpdateType.ADD)
