import re, time, sys
from pathlib import Path
from collections import defaultdict
from typing import Iterator, Tuple, Dict, Optional, Union, List
import matplotlib.pyplot as plt        # add plotting import
import csv

import retico_core
from retico_core import abstract, UpdateMessage, UpdateType
from retico_core.text import TextIU, SpeechRecognitionIU
from retico_chatgpt.concatenation_module import ConcatTextIU

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
class ListenerVADTextIU(TextIU):
    @staticmethod
    def type():
        return TextIU.type()
    def __repr__(self):
        # show the full payload without truncation
        return f"{self.type()} - ({self.creator.name()}): {self.get_text()}"


class SentenceVADModule(abstract.AbstractModule):
    @staticmethod
    def name():
        return "Sentence VAD Module"

    @staticmethod
    def description():
        return "Compute V/A/D for each token in a full sentence."

    @staticmethod
    def input_ius():
        return [ConcatTextIU]

    @staticmethod
    def output_iu():
        return ListenerVADTextIU

    def process_update(self, update_message):
        for iu, typ in update_message:
            if typ == UpdateType.ADD:
                return self._on_sentence(iu)
        return UpdateMessage()

    def _on_sentence(self, input_iu: TextIU):
        sentence = input_iu.text or ""
        # collect per-token V/A/D
        vad_list: List[Tuple[str, Dict[str, Optional[float]]]] = list(
            incremental_vad_scores(sentence, missing=0.0)
        )
        # build a multiline string: "token: v,a,d"
        lines = [
            f"{tok}: "
            f"{cur['valence']:.2f},"
            f"{cur['arousal']:.2f},"
            f"{cur['dominance']:.2f}"
            for tok, cur in vad_list
        ]
        payload = "\n".join(lines)
        print(f"Computed VAD for sentence: {payload}")
        out_iu = self.create_iu(input_iu)
        out_iu.payload = payload
        return abstract.UpdateMessage.from_iu(out_iu, abstract.UpdateType.ADD)