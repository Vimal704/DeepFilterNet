import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from df.modules import get_device
#from scipy.io import wavfile
#from scipy.signal import spectrogram
#from scipy.ndimage import gaussian_filter
#import librosa 
from mcra2_estimation import *
from scipy.signal import get_window

def literature_noise_estimation_wrapper(noisy_sig):
    # np.random.seed(1)
    # k = torch.from_numpy(np.random.random((481,96)))
    fs = 48000  # Sampling rate for processing
    # win_temp = 25.6
    # win_len = math.floor(win_temp * fs * 10**-3)
    fft_len = 960
    fft_half=int(fft_len/2+1)
    overlap = 0.5
    # window = get_window('rectangular', win_len)
    # total_frames = int(sig_len / (win_len * (1 - overlap)))
    noisy_sig_shape=noisy_sig.shape
    # print("spp_modified l-25 noisy_sig.shape",noisy_sig_shape)
    total_frames = noisy_sig_shape[2]
    lower_limit = 0
    frame_no = 0

    spp_mat = torch.zeros((noisy_sig.shape[0],1,total_frames, 481))

    noisy_mag = torch.abs(torch.squeeze(noisy_sig))
    for b in range(noisy_sig.shape[0]): 
        noise_mu_mcra2 = torch.zeros((total_frames, fft_half))
        for frame_no in range(noisy_sig.shape[2]-1):
            ns_ps = noisy_mag[frame_no,:] ** 2
            Srate = fs
            len_val = len(ns_ps)
            freq_res = Srate / len_val
            k_1khz = int(np.floor(1000 / freq_res))
            k_3khz = int(np.floor(3000 / freq_res))
            delta_val = torch.from_numpy(np.concatenate((2 * np.ones(k_1khz), 2 * np.ones(k_3khz - k_1khz),
                                        5 * np.ones(len_val // 2 - k_3khz), 5 * np.ones(len_val // 2 - k_3khz),
                                        2 * np.ones(k_3khz - k_1khz), 2 * np.ones(k_1khz), np.ones(1)))).to(device=get_device())
            
            
            if frame_no == 0:
                ns_ps1 = 0.1 * torch.ones((len_val)).to(device=get_device())
                parameters = {
                    'n': 2,
                    'len': len_val,
                    'ad': 0.95,
                    'as': 0.8,
                    'ap': 0.2,
                    'beta': 0.8,
                    'beta1': 0.98,
                    'gamma': 0.998,
                    'alpha': 0.7,
                    'delta': delta_val,
                    'pk': torch.zeros((len_val,1)).to(device=get_device()),
                    'noise_ps': ns_ps1,
                    'pxk_old': ns_ps1,
                    'pxk': ns_ps1,
                    'pnk_old': ns_ps1,
                    'pnk': ns_ps1
                }
            else:
                parameters = mcra2_estimation(ns_ps, parameters)
            
            pk = parameters['pk']
            noise_mu_mcra2[frame_no+1 , :] = torch.squeeze(torch.sqrt(pk[:fft_half]))
            # frame_no += 1
        spp_mat[b,0,:] = noise_mu_mcra2.float()

    spp_mat
    return spp_mat
