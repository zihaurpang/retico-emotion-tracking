import os, sys
import matplotlib.pyplot as plt

os.environ['CORE'] = 'retico-core'
os.environ['WHISPER'] = 'retico-whisperasr'
os.environ['EMOTION'] = 'retico-emotiontracking'

sys.path.append(os.environ['CORE'])
sys.path.append(os.environ['WHISPER'])
sys.path.append(os.environ['EMOTION'])

from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule
from retico_whisperasr.whisperasr import WhisperASRModule
from retico_emotion_track.emotion_module import EmotionTrackingModule
from retico_emotion_track.intensity_module import EmotionIntensityTrackingModule

mic = MicrophoneModule(rate=16000)
debug = DebugModule()
asr = WhisperASRModule(language='en')
emo = EmotionTrackingModule()
intensity = EmotionIntensityTrackingModule()

mic.subscribe(asr)
asr.subscribe(emo)
emo.subscribe(debug)

# If you prefer the intensity, please subscribe the following
# asr.subscribe(intensity)
# intensity.subscribe(debug)


mic.run()
asr.run()
emo.run()
debug.run()

input()

asr.stop()
emo.stop()
debug.stop()
