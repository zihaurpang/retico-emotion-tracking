import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []

index = count()


def animate(i):
    df = pd.read_csv('vad_data.csv')
    if df.empty:
        return                       # ← skip if no data yet
    t0 = df['timestamp'].iloc[0]
    t  = df['timestamp'] - t0
    data = df.assign(time=t)
    plt.cla()
    plt.plot(data['time'], data['valence'],   'C0-o', label='Valence')
    plt.plot(data['time'], data['arousal'],   'C1-o', label='Arousal')
    plt.plot(data['time'], data['dominance'], 'C2-o', label='Dominance')
    plt.xlabel("Time (s)")
    plt.ylabel("Score")
    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()