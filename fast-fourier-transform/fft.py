import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

dt = 0.001
t = np.linspace(0, 1, int(1/dt))
N = len(t)
noise = np.random.rand(N) * 10-5
y = np.sin(2*np.pi *50 * t) + np.sin(2*np.pi *120 * t)
signal_noisy = y + noise


f = np.fft.fft(signal_noisy)
PSD = (np.abs(f)**2)/N
PSD_filtered = PSD.copy()
PSD_filtered[PSD<100] = 0

f_filtered = f.copy()
f_filtered[PSD<100] = 0
signal_filterd = np.fft.ifft(f_filtered)

freq = 1/(dt*N) * np.arange(N)
L = np.int(N/2)

plt.figure("PSD")
plt.plot(freq[1:L], PSD[1:L])
plt.plot(freq[1:L], PSD_filtered[1:L])

plt.figure("Signal")

plt.plot(t, signal_noisy)
plt.plot(t, y)

plt.figure("Signal filtered")
plt.plot(t, signal_filterd, linewidth=1)
plt.plot(t, signal_filterd - y, linewidth=1)

sd.play(y, 1/dt, blocking=True)
sd.play(signal_noisy, 1/dt, blocking=True)
sd.play(signal_filterd.real, 1/dt, blocking=True)

plt.show()

