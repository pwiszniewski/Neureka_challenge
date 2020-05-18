import numpy as np
from scipy import signal
from scipy.signal import resample, hann
from sklearn import preprocessing
import mne
import torch
# from python_speech_features import mfcc
# optional modules for trying out different transforms
try:
    import pywt
except ImportError as e:
    pass

try:
    from scikits.talkbox.features import mfcc
except ImportError as e:
    pass


class ChannelTime2D:
    '''
    Spatial Temporal Respresentation
    '''
    def get_name(self):
        return 'chann_time_2d'

    def apply(self, data):
        data = data.reshape(1, data.shape[0], data.shape[1])
        return data


class MTSA:
    '''
    multitaper spectral analysis
    '''
    def get_name(self):
        return 'mtsa'

    def apply(self, data, fs=400):
        data = mne.time_frequency.psd_array_multitaper(data, fs, 0, 160, verbose=False)[0]
        return data


class STFT:
    '''
    Short-Term Fourier Transform using a sine window
    '''
    def __init__(self):
        self.fs = 400

    def get_name(self):
        return 'stft'

    def apply(self, data):
        f, t, Zxx = signal.stft(data, self.fs, nperseg=32)
        return np.abs(Zxx)


class CWTMorlet:
    '''
    Short-Term Fourier Transform using a sine window
    '''
    def __init__(self):
        self.fs = 400

    def get_name(self):
        return 'cwt_morl'

    def apply(self, data):
        freqs = np.arange(12, 160, 4)
        data = data.reshape(1, data.shape[0], data.shape[1])
        tfr = mne.time_frequency.tfr_array_morlet(data, self.fs, freqs, use_fft=True, n_cycles=7.0, zero_mean=False)[0]

        return np.abs(tfr)


class SignalImage:
    """
    Convert signal to image
    """
    def get_name(self):
        return "signal_img"

    def apply(self, data):
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        width, height = 250, 40
        imgs = np.zeros((data.shape[0], height, width), dtype=np.uint8)
        for i in range(data.shape[0]):
            my_dpi = 96
            fig = Figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
            canvas = FigureCanvas(fig)
            ax = fig.subplots()
            ax.plot(data[i], linewidth=1)
            ax.axis('off')
            fig.subplots_adjust(top=1, bottom=0, right=1.04, left=-0.04,
                                hspace=0, wspace=0)
            canvas.draw()
            buf = fig.canvas.buffer_rgba()
            X = np.frombuffer(buf, np.uint8).copy()
            X.shape = height, width, 4
            imgs[i, :, :] = X[:, :, 0]
        return imgs


class LPF:
    """
    Low-pass filter using FIR window
    """
    def __init__(self, f):
        self.f = f

    def get_name(self):
        return 'lpf%d' % self.f

    def apply(self, data):
        N = len(data[0])
        data = mne.filter.filter_data(data, N, None, self.f / 2)
        return data



class MFCC:
    """
    Mel-frequency cepstrum coefficients
    """
    def get_name(self):
        return "mfcc"

    def apply(self, data):
        all_ceps = []
        for ch in data:
            # ceps, mspec, spec = mfcc(ch)
            ceps = mfcc(ch)
            all_ceps.append(ceps.ravel())

        return np.array(all_ceps)


class Resample:
    """
    Resample time-series data.
    """
    def __init__(self, sample_rate):
        self.f = sample_rate

    def get_name(self):
        return "resample%d" % self.f

    def apply(self, data):
        axis = data.ndim - 1
        if data.shape[-1] > self.f:
            return resample(data, self.f, axis=axis)
        return data



