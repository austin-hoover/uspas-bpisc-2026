import numpy as np
import scipy.fft
import scipy.signal


def fft(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.copy(t)
    y = np.copy(y)
    
    dt = np.mean(np.diff(t))
    N = len(t)
    
    yf = scipy.fft.fft(y)
    xf = scipy.fft.fftfreq(N, dt)

    N = N // 2
    frequencies = xf[1:N]
    amplitudes = (2.0 / N) * np.abs(yf[1:N])
    return (frequencies, amplitudes)


def dominant_frequency(t: np.ndarray, y: np.ndarray) -> float:
    frequencies, amplitudes = fft(t, y)
    return frequencies[np.argmax(amplitudes)]