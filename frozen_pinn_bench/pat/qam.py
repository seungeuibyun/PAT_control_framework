"""Gray-coded 64-QAM with unit average energy and RRC pulse shaping."""
import numpy as np
from scipy.signal import upfirdn

# In increasing amplitude order: -7,-5,-3,-1,+1,+3,+5,+7.
GRAY = np.array([0, 1, 3, 2, 6, 7, 5, 4], dtype=np.uint8)
LEVELS = np.arange(-7, 8, 2, dtype=float)/np.sqrt(42.)
INVERSE_GRAY = np.argsort(GRAY)


def constellation():
    return (LEVELS[:, None]+1j*LEVELS[None, :]).ravel()


def modulate(bits):
    bits = np.asarray(bits)
    if bits.ndim != 2 or bits.shape[1] != 6 or np.any((bits != 0) & (bits != 1)):
        raise ValueError('64-QAM input must be an N x 6 binary array')
    labels = bits.reshape(-1, 2, 3)@np.array([4, 2, 1])
    return LEVELS[INVERSE_GRAY[labels[:, 0]]]+1j*LEVELS[INVERSE_GRAY[labels[:, 1]]]


def demodulate(symbols):
    symbols = np.asarray(symbols).ravel()
    # Only adjacent midpoint boundaries are used; no transmitted labels.
    midpoints = (LEVELS[:-1]+LEVELS[1:])/2
    indices = np.stack([np.searchsorted(midpoints, symbols.real),
                        np.searchsorted(midpoints, symbols.imag)], axis=1)
    labels = GRAY[indices]
    return ((labels[:, :, None] >> np.array([2, 1, 0])) & 1).reshape(-1, 6).astype(np.uint8)


def rrc_taps(samples_per_symbol, span_symbols, rolloff):
    if samples_per_symbol < 2 or span_symbols < 2 or span_symbols % 2 or not 0 < rolloff <= 1:
        raise ValueError('RRC requires sps >= 2, an even span and 0 < rolloff <= 1')
    t = np.arange(-span_symbols*samples_per_symbol//2,
                   span_symbols*samples_per_symbol//2+1)/samples_per_symbol
    taps = np.empty_like(t)
    zero = np.isclose(t, 0.)
    singular = np.isclose(abs(t), 1/(4*rolloff))
    general = ~(zero | singular)
    taps[zero] = 1+rolloff*(4/np.pi-1)
    taps[singular] = rolloff/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*rolloff))
                                     +(1-2/np.pi)*np.cos(np.pi/(4*rolloff)))
    u = t[general]
    taps[general] = (np.sin(np.pi*u*(1-rolloff))+4*rolloff*u*np.cos(np.pi*u*(1+rolloff)))/(np.pi*u*(1-(4*rolloff*u)**2))
    return taps/np.linalg.norm(taps)


def waveform_scale(taps, samples_per_symbol, peak_bound):
    # A fixed worst-case bound for every possible data sequence. No per-frame
    # automatic gain normalization conceals changes in optical power.
    l1 = max(abs(taps[p::samples_per_symbol]).sum() for p in range(samples_per_symbol))
    return peak_bound/(abs(constellation()).max()*l1*np.sqrt(samples_per_symbol))


def shape(symbols, taps, samples_per_symbol, scale):
    filtered = upfirdn(taps*np.sqrt(samples_per_symbol)*scale, symbols, up=samples_per_symbol)
    delay = (len(taps)-1)//2
    return filtered[delay:delay+len(symbols)*samples_per_symbol].astype(np.complex64)
