import numpy as np
from scipy import signal


class FIRFilterGenerator:
    """
    Generator for various Finite Impulse Response (FIR) filter coefficients.
    """

    def __init__(self, sampling_rate=10000):
        self.sampling_rate = sampling_rate

    def generate_low_pass(self, cutoff_freq, num_taps=101):
        """
        Generates FIR low-pass filter coefficients using the window method.
        """
        nyquist = self.sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        coeffs = signal.firwin(num_taps, normalized_cutoff, window="hamming")
        return coeffs

    def generate_high_pass(self, cutoff_freq, num_taps=101):
        """
        Generates FIR high-pass filter coefficients using the window method.
        """
        nyquist = self.sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        # High-pass requires odd num_taps for firwin
        if num_taps % 2 == 0:
            num_taps += 1
        coeffs = signal.firwin(
            num_taps, normalized_cutoff, pass_zero=False, window="hamming"
        )
        return coeffs

    def generate_band_pass(self, low_cutoff, high_cutoff, num_taps=101):
        """
        Generates FIR band-pass filter coefficients using the window method.
        """
        nyquist = self.sampling_rate / 2
        normalized_cutoffs = [low_cutoff / nyquist, high_cutoff / nyquist]
        coeffs = signal.firwin(
            num_taps, normalized_cutoffs, pass_zero=False, window="hamming"
        )
        return coeffs

    def generate_random_filter(self, num_taps=101, scale=1.0):
        """
        Generates a random FIR filter.
        """
        coeffs = np.random.uniform(-scale, scale, num_taps)
        # Normalize to prevent signal explosion
        coeffs /= np.sum(np.abs(coeffs))
        return coeffs

    def generate_weight_decay_filter(self, decay_rate=0.9, num_taps=101):
        """
        Generates a weight decay filter where coefficients decay exponentially.
        h[n] = exp(-decay_rate * n)
        """
        n = np.arange(num_taps)
        coeffs = np.exp(-decay_rate * n)  # Using exponential decay
        # Normalize
        coeffs /= np.sum(coeffs)
        return coeffs

    def apply_filter(self, data, coeffs):
        """
        Applies the FIR filter to the data using convolution.
        """
        return np.convolve(data, coeffs, mode="same")
