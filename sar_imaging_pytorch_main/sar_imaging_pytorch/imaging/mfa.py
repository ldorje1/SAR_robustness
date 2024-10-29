import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class MFA(nn.Module):
    def __init__(self, x_step_m, y_step_m, xy_size_t):
        super().__init__()
        self.x_step_m = x_step_m
        self.y_step_m = y_step_m
        self.xy_size_t = xy_size_t
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, sar_data, matched_filter):
        sar_data = sar_data.to(self.device)
        matched_filter = matched_filter.to(self.device)

        # Get the dimensions of SAR data and matched filter
        y_point_m, x_point_m = sar_data.shape
        y_point_f, x_point_f = matched_filter.shape

        # Equalize dimensions of SAR data and matched filter with zero padding
        if x_point_f > x_point_m:
            pad_x_before = (x_point_f - x_point_m) // 2
            pad_x_after = x_point_f - x_point_m - pad_x_before
            sar_data = torch.nn.functional.pad(sar_data, (pad_x_before, pad_x_after, 0, 0))
        else:
            pad_x_before = (x_point_m - x_point_f) // 2
            pad_x_after = x_point_m - x_point_f - pad_x_before
            matched_filter = torch.nn.functional.pad(matched_filter, (pad_x_before, pad_x_after, 0, 0))

        if y_point_f > y_point_m:
            pad_y_before = (y_point_f - y_point_m) // 2
            pad_y_after = y_point_f - y_point_m - pad_y_before
            sar_data = torch.nn.functional.pad(sar_data, (0, 0, pad_y_before, pad_y_after))
        else:
            pad_y_before = (y_point_m - y_point_f) // 2
            pad_y_after = y_point_m - y_point_f - pad_y_before
            matched_filter = torch.nn.functional.pad(matched_filter, (0, 0, pad_y_before, pad_y_after))

        # Create SAR image
        sar_data_fft = torch.fft.fft2(sar_data)
        matched_filter_fft = torch.fft.fft2(matched_filter)
        sar_image = torch.fft.fftshift(torch.fft.ifft2(sar_data_fft * matched_filter_fft))

        return sar_image

    def plot_result(self, sar_image):
        sar_image = sar_image.detach().cpu()  # Move to CPU for plotting

        if isinstance(self.xy_size_t, (int, float)):  # If scalar: original approach
            y_point_t, x_point_t = sar_image.shape
            x_range_t = self.x_step_m * torch.arange(-(x_point_t - 1) / 2, (x_point_t - 1) / 2 + 1)
            y_range_t = self.y_step_m * torch.arange(-(y_point_t - 1) / 2, (y_point_t - 1) / 2 + 1)

            # Crop the image for the related region
            ind_x_part_t = (x_range_t > -self.xy_size_t / 2) & (x_range_t < self.xy_size_t / 2)
            ind_y_part_t = (y_range_t > -self.xy_size_t / 2) & (y_range_t < self.xy_size_t / 2)

            x_range_t = x_range_t[ind_x_part_t]
            y_range_t = y_range_t[ind_y_part_t]
            sar_image = sar_image[ind_y_part_t][:, ind_x_part_t]

            plt.figure()
            plt.pcolormesh(x_range_t, y_range_t, torch.abs(torch.fliplr(sar_image)), shading='auto', cmap='jet')
            plt.colorbar()
            plt.xlabel('Horizontal (mm)')
            plt.ylabel('Vertical (mm)')
            plt.title('SAR Image - Matched Filter Response')
            plt.show()

        elif isinstance(self.xy_size_t, (list, tuple)) and len(self.xy_size_t) == 4:  # Bounding box case
            xij = torch.round(torch.tensor(self.xy_size_t[0:2]) / self.x_step_m - 0.5 + sar_image.shape[1] / 2).int()
            ykl = torch.round(torch.tensor(self.xy_size_t[2:4]) / self.y_step_m - 0.5 + sar_image.shape[0] / 2).int()

            xij = torch.clamp(xij, 0, sar_image.shape[1])
            ykl = torch.clamp(ykl, 0, sar_image.shape[0])

            sar_image = torch.fliplr(torch.abs(sar_image[ykl[0]:ykl[1], xij[0]:xij[1]]))

            plt.figure()
            plt.pcolormesh(self.xy_size_t[0] + torch.arange(sar_image.shape[1]) * self.x_step_m,
                           self.xy_size_t[2] + torch.arange(sar_image.shape[0]) * self.y_step_m,
                           sar_image, shading='auto', cmap='jet')
            plt.xlabel('Horizontal (mm)')
            plt.ylabel('Vertical (mm)')
            plt.title('SAR Image - Matched Filter Response')
            plt.colorbar()
            plt.show()

def create_matched_filter(x_point_m, x_step_m, y_point_m, y_step_m, z_target):
    """
    Create a matched filter for a given set of parameters.

    Parameters:
    x_point_m (int): Number of measurement points at x (horizontal) axis.
    x_step_m (float): Sampling distance at x (horizontal) axis in mm.
    y_point_m (int): Number of measurement points at y (vertical) axis.
    y_step_m (float): Sampling distance at y (vertical) axis in mm.
    z_target (float): z distance of target in mm.

    Returns:
    torch.Tensor: The created matched filter.
    """
    # Define fixed parameters
    f0 = 77e9  # start frequency in Hz
    c = 3e8    # speed of light in m/s

    # Define measurement locations at linear rail
    x = x_step_m * (torch.arange(-(x_point_m - 1) / 2, (x_point_m - 1) / 2 + 1)) * 1e-3
    y = (y_step_m * (torch.arange(-(y_point_m - 1) / 2, (y_point_m - 1) / 2 + 1))).view(-1) * 1e-3

    # Define target location
    z0 = z_target * 1e-3  # convert to meters

    # Create single tone matched filter
    k = 2 * torch.pi * f0 / c
    X, Y = torch.meshgrid(x, y, indexing='ij')
    matched_filter = torch.exp(-1j * 2 * k * torch.sqrt(X**2 + Y**2 + z0**2))

    return matched_filter