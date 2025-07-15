# Retico Emotion Tracking 🎭

Retico modules for **real-time user emotional state tracking** using the **VAD** (Valence, Arousal, Dominance) model or **Emotion Intensity** model. Powered by the [**NRC Valence, Arousal, and Dominance (NRC-VAD) Lexicon**](https://saifmohammad.com/WebPages/nrc-vad.html).

---

## 🚀 Features

- **Real-time emotion detection**  
  Analyze live input to estimate user emotional states using VAD or emotion intensity scores.

- **Dual tracking modes**  
  Supports both **VAD** and **Emotion Intensity** scoring models, switchable as needed.

- **Live visualization**  
  With a separate terminal, view a real-time line plot of emotion values updating as you speak or type.

---

## 🧰 Installation

```bash
git clone https://github.com/zihaurpang/retico-emotion-tracking.git
````

---

## 🛠️ Usage

### 1. Run real-time emotion tracking

Start live analysis from the microphone:

```bash
python simple_emotion.py
```

### 2. View real-time emotion visualization

In a **separate terminal**, run:

```bash
python realtime_vad_plot.py
```

This will display a continuously updating line chart of emotion values.

### 3. Manual text-based testing

Try a text input instead of live input:

```bash
python manual_emotion_module.py
```

Use this to quickly test or demo how the module responds to text.

---

## 💡 How It Works

1. **Input**: audio from microphone or user-entered text
2. **Processing**: Parse input into tokens and score each with NRC-VAD lexicon
3. **Output**:

   * In real-time module (`simple_emotion.py`): prints emotion values continuously
   * In plot module (`realtime_vad_plot.py`): streams values to a live line chart

---

## ✅ Requirements

* Python ≥ 3.7
* Simple microphone setup for real-time mode

