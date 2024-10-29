# Based on MATLAB Imaging Algorithms 
# This script has three imaging algorithms: MFA, RMA, and LIA (SPL2021)
# and two dataset to test the imaging algorithms 

import torch
import scipy.io as sio
from scipy.constants import speed_of_light
from sar_imaging_pytorch.imaging.mf import MF, create_matched_filer
from sar_imaging_pytorch.imaging.rma import RMA
from sar_imaging_pytorch.imaging.lia import LIA 

# Choose Imaging Algorithm and Dataset to Use.
imaging_alg = 3  # algorithm to image 1: MFA, 2:RMA or 3:LIA  
dataset = 2  # dataset: 1: yanik (different shaped rectangular metal),  or 2: ours (scissor)

def main():
    if dataset == 1:  # yanik (different shaped rectangular metal)
        data_name = 'rawData3D_simple2D'  # Dataset with rectangular metal objects
        data = sio.loadmat(data_name)
        raw_data = torch.tensor(data[data_name], dtype=torch.complex64)  # Convert to PyTorch tensor
        
        im_size = [-90, 60, -90, 60]  # Size of image area in mm
        
        # Define data parameters for Texas Dataset
        n_fft_time = 1024  # Number of FFT points for Range-FFT
        z0 = 280  # Range of target (range of corresponding image slice)
        dx = 200 / 406  # Sampling distance at x (horizontal) axis in mm
        dy = 2  # Sampling distance at y (vertical) axis in mm
        n_fft_space = 1024  # Number of FFT points for Spatial-FFT

        c = speed_of_light  # Speed of light (m/s)
        f_s = 9121e3  # Sampling rate (sps)
        t_s = 1 / f_s  # Sampling period
        k = 63.343e12  # Slope constant (Hz/sec)
        
        # Take Range-FFT of rawData3D
        raw_data_fft = torch.fft.fft(raw_data, n=n_fft_time, dim=0)

        # Range focusing to z0
        t_i = 4.5225e-10  # Instrument delay for range calibration
        k_bin = round(k * t_s * (2 * z0 * 1e-3 / c + t_i) * n_fft_time)  # Corresponding range bin
        sar_data = raw_data_fft[k_bin, :, :]  # Focused SAR data

    elif dataset == 2:
        data_name = 'adcDataCube'  # Dataset with scissor object
        data = sio.loadmat(data_name)
        raw_data = torch.tensor(data[data_name], dtype=torch.complex64)  # Convert to PyTorch tensor

        # Define data parameters for our Dataset
        n_fft_time = 1024  # Number of FFT points for Range-FFT
        z0 = 185  # Range of target (range of corresponding image slice)
        dx = 1  # Sampling distance at x (horizontal) axis in mm
        dy = 1  # Sampling distance at y (vertical) axis in mm
        n_fft_space = 1024  # Number of FFT points for Spatial-FFT

        c = speed_of_light  # Speed of light (m/s)
        f_s = 5000e3  # Sampling rate (sps)
        t_s = 1 / f_s  # Sampling period
        k = 70.295e12  # Slope constant (Hz/sec)

        # Take Range-FFT of rawData3D
        raw_data_fft = torch.fft.fft(raw_data, n=n_fft_time, dim=0)
        
        # Range focusing to z0
        t_i = 4.5225e-10  # Instrument delay for range calibration
        k_bin = round(k * t_s * (2 * z0 * 1e-3 / c + t_i) * n_fft_time)  # Corresponding range bin

        im_size = [-100, 70, -100, 70]  # Size of image area in mm
        sar_data = raw_data_fft[k_bin, :, :]  # Focused SAR data
        
        # (LD) Note: to fix the mirror image issue in MFA.
        for ii in range(1, sar_data.shape[0], 2):  # Start at 1 (2nd row), step by 2
            sar_data[ii, :] = sar_data[ii, torch.arange(sar_data.shape[1] - 1, -1, -1)]  # Flip row horizontally

    else:
        raise ValueError("Invalid dataset selection")

    # Imaging Algorithms: 1: MFA, 2:RMA or 3:LIA  
    if imaging_alg == 1:  # 1: MFA 
        
        # Create Matched Filter
        matched_filter = create_matched_filer(n_fft_space, dx, n_fft_space, dy, z0)

        # Create SAR Image
        mf = MF(x_step_m=dx, y_step_m=dy, xy_size_t=im_size)  # or xy_size_t=[x0, x1, y0, y1] for bounding box
        sar_image = mf(sar_data, matched_filter)
        
        # Plot the results
        mf.plot_result(sar_image)
    
    elif imaging_alg == 2:  # 2: RMA 
        # Rearrange the data for RMA
        raw_data = raw_data.permute(2, 1, 0)  # [horizontal, vertical, samples]
        num_sample = raw_data.shape[2]  # 256
        n_fft_t = num_sample

        # Perform FFT along the third axis
        Sr = torch.fft.fft(raw_data, n=n_fft_t, dim=2)
        
        # Calculate the average energy per range bin
        Sr_energy = torch.mean(torch.abs(Sr)**2, dim=(0, 1))

        # Find the range bin with maximum energy
        ID_select = torch.argmax(Sr_energy)
        print(f'ID_select = {ID_select}')

        # Select SAR data after pulse compression using ID_select
        sar_data = Sr[:, :, ID_select].T  # Transpose for consistency

        # Compute z0 based on the selected range bin
        z0_1 = c / 2 * (ID_select / (k * (1 / f_s) * n_fft_t) - t_i)

        # Flip every second row in sar_data horizontally
        for ii in range(1, sar_data.shape[0], 2):  # Start at 1 (second row), step by 2
            sar_data[ii, :] = sar_data[ii, torch.arange(sar_data.shape[1] - 1, -1, -1)]

        rma = RMA(nFFTspace=n_fft_space, z0=z0_1, dx=dx, dy=dy, bbox=im_size)
        sar_image = rma(sar_data)

        # Plot the results
        rma.plot_result(sar_image)

    elif imaging_alg == 3:  # 3: LIA

        # Calculate kk as a percentage of total number of elements in sar_image_0
        percentage = 1  # % of total elements
        N, M = sar_data.shape
        kk = int((N * M) * percentage)  # Calculate kk as 40% of total elements

        lia = LIA(z0=z0, dx=dx, dy=dy, im_size=im_size, kk=kk)
        sar_image = lia(sar_data)

        # Plot the results
        lia.plot_result(sar_image)

if __name__ == "__main__":
    main()
