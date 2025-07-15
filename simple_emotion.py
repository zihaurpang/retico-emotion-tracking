import os, sys
import matplotlib.pyplot as plt

os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/Users/pang/.config/gcloud/application_default_credentials.json'

os.environ['CORE'] = 'retico-core'
os.environ['GASR'] = 'retico-googleasr'
os.environ['WHISPER'] = 'retico-whisperasr'
os.environ['EMOTION'] = 'retico-emotiontracking'

sys.path.append(os.environ['CORE'])
sys.path.append(os.environ['GASR'])
sys.path.append(os.environ['WHISPER'])
sys.path.append(os.environ['EMOTION'])

from retico_core.debug import DebugModule
from retico_core.audio import MicrophoneModule
from retico_googleasr.googleasr import GoogleASRModule
from retico_whisperasr.whisperasr import WhisperASRModule
from retico_emotion_track.emotion_module import EmotionTrackingModule
from retico_emotion_track.intensity_module import EmotionIntensityTrackingModule

# from retico_core.text import IncrementalizeASRModule # it's good to know about this module, but don't use it if you want to see revokes!

mic = MicrophoneModule(rate=16000)
debug = DebugModule()
#asr = GoogleASRModule()
asr = WhisperASRModule(language='en')
emo = EmotionTrackingModule()
intensity = EmotionIntensityTrackingModule()

mic.subscribe(asr)
asr.subscribe(emo)
emo.subscribe(debug)
# asr.subscribe(intensity)
# intensity.subscribe(debug)


mic.run()
asr.run()
emo.run()
# intensity.run()
debug.run()

input()

asr.stop()
emo.stop()
# intensity.stop()
debug.stop()