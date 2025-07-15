import re, time, sys
from pathlib import Path
from collections import defaultdict
from typing import Iterator, Tuple, Dict, Optional, Union, List
import matplotlib.pyplot as plt        # add plotting import
import csv

import retico_core
from retico_core import abstract, UpdateMessage, UpdateType
from retico_core.text import TextIU, SpeechRecognitionIU

from nltk.stem import WordNetLemmatizer

# 1. Lexicon setup
LEXICON_DIR = Path(__file__).parent / "Intensity"
FILES = {
    "anger":   "anger-NRC-EmoIntv1-withZeroIntensityEntries.txt",
    "anticipation":   "anticipation-NRC-EmoIntv1-withZeroIntensityEntries.txt",
    "disgust": "disgust-NRC-EmoIntv1-withZeroIntensityEntries.txt",
    "fear": "fear-NRC-EmoIntv1-withZeroIntensityEntries.txt",
    "joy": "joy-NRC-EmoIntv1-withZeroIntensityEntries.txt",
    "sadness": "sadness-NRC-EmoIntv1-withZeroIntensityEntries.txt",
    "surprise": "surprise-NRC-EmoIntv1-withZeroIntensityEntries.txt",
    "trust": "trust-NRC-EmoIntv1-withZeroIntensityEntries.txt",
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
def incremental_intensity_scores(
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


class EmotionIntensityTrackingModule(abstract.AbstractModule):
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
        self.csv_path = Path(__file__).parent / "intensity_data.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp","anger","anticipation","disgust",
                                     "fear","joy","sadness","surprise","trust"])

    def process_update(self, um):
        for iu, typ in um:
            if typ == UpdateType.ADD:
                return self.process_iu(iu)
        return UpdateMessage()

    def process_iu(self, input_iu):
        text = getattr(input_iu, "text", "")
        scores = None
        # accept any token with ≥1 lexicon hit (0 → None)
        for _, cur in incremental_intensity_scores(text, missing=None):
            if any(v is not None for v in cur.values()):
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
                f"{scores['anger']:.3f}",
                f"{scores['anticipation']:.3f}",
                f"{scores['disgust']:.3f}",
                f"{scores['fear']:.3f}",
                f"{scores['joy']:.3f}",
                f"{scores['sadness']:.3f}",
                f"{scores['surprise']:.3f}",
                f"{scores['trust']:.3f}"
            ])

        payload_str = f"anger: {scores['anger']:.3f}, " \
                        f"anticipation: {scores['anticipation']:.3f}, " \
                        f"disgust: {scores['disgust']:.3f}, " \
                        f"fear: {scores['fear']:.3f}, " \
                        f"joy: {scores['joy']:.3f}, " \
                        f"sadness: {scores['sadness']:.3f}, " \
                        f"surprise: {scores['surprise']:.3f}, " \
                        f"trust: {scores['trust']:.3f}"
        print(f"Current Emotional States: {payload_str}")
        out_iu = self.create_iu(input_iu)
        out_iu.payload = payload_str
        return abstract.UpdateMessage.from_iu(out_iu, abstract.UpdateType.ADD)



# Uncomment this to run the manual text input module

# class ManualTextInputModule(abstract.AbstractModule):
#     @staticmethod
#     def name():
#         return "Manual Text Input Module"
#     @staticmethod
#     def description():
#         return "Read user lines and output EmotionTextIU for visualization."
#     @staticmethod
#     def input_ius():
#         return []
#     @staticmethod
#     def output_iu():
#         return EmotionTextIU

#     def run(self):
#         # loop in main thread
#         while True:
#             line = input("Enter text (empty to quit): ")
#             if not line:
#                 break
#             out_iu = self.create_iu(None)
#             out_iu.payload = line
#             um = UpdateMessage.from_iu(out_iu, UpdateType.ADD)
#             for sub in self.subscribers:
#                 sub.update(um)

# if __name__ == "__main__":
#     sentence = input("Enter sentence: ")
#     data = list(incremental_vad_scores(sentence, missing=0.0))
#     if not data:
#         print("No tokens to plot."); sys.exit()

#     tokens, scores = zip(*data)
#     idx = list(range(len(tokens)))
#     V = [s["valence"]   for s in scores]
#     A = [s["arousal"]   for s in scores]
#     D = [s["dominance"] for s in scores]

#     # incremental plotting
#     plt.ion()
#     fig, ax = plt.subplots()
#     for i in range(len(idx)):
#         ax.clear()
#         ax.plot(idx[:i+1], V[:i+1], "C0-o", label="Valence")
#         ax.plot(idx[:i+1], A[:i+1], "C1-o", label="Arousal")
#         ax.plot(idx[:i+1], D[:i+1], "C2-o", label="Dominance")
#         ax.set_xticks(idx[:i+1])
#         ax.set_xticklabels(tokens[:i+1], rotation=45)
#         ax.set_xlabel("Token Index")
#         ax.set_ylabel("Score")
#         ax.set_title("Incremental V/A/D")
#         ax.legend(loc="upper left")
#         fig.tight_layout()
#         fig.canvas.draw()
#         fig.canvas.flush_events()
#         plt.pause(0.5)      # pause between updates
#     plt.ioff()
#     plt.show()
