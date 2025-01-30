# Importing necessary libraries and modules
from torch.utils.data import DataLoader, Subset, IterableDataset
import tqdm
import wandb
import matplotlib.pyplot as plt
import torch
import numpy as np
from transformer_sim import Config, TSTransformer
from dataset import WHDataset
from pathlib import Path



def generate_input(batch_size=1, N=1000, p_low_pass=0.5, p_high_pass=0.05):
    """
    Generate input signals with specified filtering characteristics.

    Parameters:
    batch_size (int): Number of signal batches to generate. Each batch contains one signal,
                      enabling the simultaneous generation of multiple signals.

    N (int): Total number of data points in each signal. This defines both the length of the
             signal in the time domain and the resolution in the frequency domain.

    p_low_pass (float): Probability (range 0 to 1) that the minimum frequency will be set to zero,
                        effectively applying a low-pass filter.

    p_high_pass (float): Probability (range 0 to 1) that the maximum frequency will be set to the
                         Nyquist frequency, effectively applying a high-pass filter.

    Returns:
    u (ndarray): Array of generated time-domain signals. The array shape is (batch_size, N),
                 where each row represents a single time-domain signal.

    uf (ndarray): Array of frequency-domain representations of the signals. This complex array
                  corresponds to the frequency components of the signals in 'u'.

    fmin (int): Minimum frequency index that is active in the frequency domain representation,
                defining the lower bound of the filter's passband.

    fmax (int): Maximum frequency index that is active, defining the upper bound of the filter's
                passband.
    """

    # Randomly select two frequencies within the valid range
    f1, f2 = np.random.randint(low=1, high=N // 2 + 1, size=(2,))

    # Determine minimum and maximum frequencies based on random selection
    fmin, fmax = sorted([f1, f2])

    # Apply low-pass filter with the probability p_low_pass
    if np.random.uniform() < p_low_pass:
        fmin = 1
    # Apply high-pass filter with the probability p_high_pass
    if np.random.uniform() < p_high_pass:
        fmax = N // 2

    # Create an array of zeros for frequency components
    uf = np.zeros((batch_size, N // 2 + 1), dtype=np.complex64)
    # Assign random phase shifts to the frequency components within the passband
    uf[:, fmin:fmax + 1] = np.exp(1j * np.random.uniform(low=0, high=2 * np.pi, size=(batch_size, fmax - fmin + 1)))

    # Inverse real FFT to convert frequency domain to time domain
    u = np.fft.irfft(uf)
    # Normalize the signal
    u *= np.sqrt((N // 2 + 1) / (fmax - fmin + 1) * N)

    return u, uf, fmin, fmax


# Definition of the trajGenerator class, which generates trajectories for simulation
class trajGenerator(IterableDataset):
    def __init__(self, model, u_ctx, y_ctx, device, len_sim=100):
        super(trajGenerator, self).__init__()
        self.model = model  # GPT Model to be used for generating trajectories
        self.u_ctx = u_ctx  # Input context
        self.y_ctx = y_ctx  # Output context
        self.len_sim = len_sim  # Length of the simulation
        self.nu = self.u_ctx.shape[-1]  # Number of input features
        self.device = device

    # Iterator method to generate simulation data continuously
    def __iter__(self):
        while True:
            u_sim = torch.tensor(generate_input(batch_size=2, N=self.len_sim, p_low_pass=0.5, p_high_pass=0.05)[0]).unsqueeze(-1).float().to(self.device)#.reshape(-1, 1)
            y_sim, sigmay_sim, _ , _= self.model(torch.tensor(self.y_ctx).view(2,400,1).float(), torch.tensor(self.u_ctx).view(2,400,1).float(), u_sim.view(2,200,1).float(), torch.zeros([2,200,1]).float().to(self.device), 0)#,cfg.seq_len_n_in)
            yield u_sim, y_sim, sigmay_sim  # Yielding the generated data
