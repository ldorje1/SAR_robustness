import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class RMA(nn.Module):
    def __init__(self, nFFTspace, z0, dx, dy, bbox):
        super().__init__()
        self.nFFTspace = nFFTspace
        self.z0 = z0
        self.dx = dx
        self.dy = dy
        self.bbox = bbox
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Constants
        self.F0 = 78.8 * 1e9  # Start frequency
        self.c = 3e8  # Speed of light in m/s
        self.k = 2 * torch.pi * self.F0 / self.c

        # Spatial frequencies
        self.wSx = 2 * torch.pi / (self.dx * 1e-3)  # Sampling frequency for x
        self.wSy = 2 * torch.pi / (self.dy * 1e-3)  # Sampling frequency for y

        self.initialize_phase_factor()

    def initialize_phase_factor(self):
        kX = torch.linspace(-self.wSx / 2, self.wSx / 2, self.nFFTspace, device=self.device)
        kY = torch.linspace(-self.wSy / 2, self.wSy / 2, self.nFFTspace, device=self.device)

        kX_grid, kY_grid = torch.meshgrid(kX, kY, indexing='ij')
        K_squared = (2 * self.k) ** 2 - kX_grid ** 2 - kY_grid ** 2
        K_squared = torch.clamp(K_squared, min=0)

        K = torch.sqrt(K_squared + 1e-10)

        phase_factor0 = torch.exp(-1j * self.z0 * K)
        phase_factor0[(kX_grid ** 2 + kY_grid ** 2) > (2 * self.k) ** 2] = 0
        self.phase_factor = K * phase_factor0
        self.phase_factor = torch.fft.fftshift(torch.fft.fftshift(self.phase_factor, dim=0), dim=1)

    def forward(self, sar_data):
        sar_data = sar_data.to(self.device)

        # Padding matrix with 0
        y_point_m, x_point_m = sar_data.shape
        y_point_f, x_point_f = self.phase_factor.shape

        # Equalize dimensions of sar_data and phase_factor with zero padding
        if x_point_f > x_point_m:
            pad_x_before = (x_point_f - x_point_m) // 2
            pad_x_after = x_point_f - x_point_m - pad_x_before
            sar_data = torch.nn.functional.pad(sar_data, (pad_x_before, pad_x_after, 0, 0))
        else:
            pad_x_before = (x_point_m - x_point_f) // 2
            pad_x_after = x_point_m - x_point_f - pad_x_before
            self.phase_factor = torch.nn.functional.pad(self.phase_factor, (pad_x_before, pad_x_after, 0, 0))

        if y_point_f > y_point_m:
            pad_y_before = (y_point_f - y_point_m) // 2
            pad_y_after = y_point_f - y_point_m - pad_y_before
            sar_data = torch.nn.functional.pad(sar_data, (0, 0, pad_y_before, pad_y_after))
        else:
            pad_y_before = (y_point_m - y_point_f) // 2
            pad_y_after = y_point_m - y_point_f - pad_y_before
            self.phase_factor = torch.nn.functional.pad(self.phase_factor, (0, 0, pad_y_before, pad_y_after))

        # Create SAR image
        sar_data_fft = torch.fft.fft2(sar_data, s=(self.nFFTspace, self.nFFTspace))
        sar_image = torch.fft.ifft2(sar_data_fft * self.phase_factor)

        return sar_image

    def plot_result(self, sar_image):
        sar_image = sar_image.detach().cpu()  # Move to CPU for plotting

        y_point_t, x_point_t = sar_image.shape
        x_range_t = self.dx * torch.arange(-(x_point_t - 1) / 2, (x_point_t - 1) / 2 + 1)
        y_range_t = self.dy * torch.arange(-(y_point_t - 1) / 2, (y_point_t - 1) / 2 + 1)

        if isinstance(self.bbox, (int, float)):  # Single cropped image size
            ind_x_part_t = (x_range_t > -self.bbox / 2) & (x_range_t < self.bbox / 2)
            ind_y_part_t = (y_range_t > -self.bbox / 2) & (y_range_t < self.bbox / 2)

            x_range_t = x_range_t[ind_x_part_t]
            y_range_t = y_range_t[ind_y_part_t]
            sar_image = torch.abs(sar_image[ind_y_part_t][:, ind_x_part_t])

            plt.figure()
            plt.pcolormesh(x_range_t, y_range_t, torch.abs(torch.fliplr(sar_image)), shading='auto', cmap='jet')
            plt.colorbar()
            plt.xlabel('Horizontal (mm)')
            plt.ylabel('Vertical (mm)')
            plt.title('2D SAR Imaging - RMA')
            plt.show()

        elif isinstance(self.bbox, (list, tuple)) and len(self.bbox) == 4:  # Bounding box [x0, x1, y0, y1]
            xij = torch.round(torch.tensor(self.bbox[0:2]) / self.dx - 0.5 + x_point_t / 2).int()
            ykl = torch.round(torch.tensor(self.bbox[2:4]) / self.dy - 0.5 + y_point_t / 2).int()

            xij = torch.clamp(xij, 0, sar_image.shape[1] - 1)
            ykl = torch.clamp(ykl, 0, sar_image.shape[0] - 1)

            sar_image = torch.fliplr(torch.abs(sar_image[ykl[0]:ykl[1], xij[0]:xij[1]]))

            sar_image_x = self.bbox[0] + torch.arange(sar_image.shape[1]) * self.dx
            sar_image_y = self.bbox[2] + torch.arange(sar_image.shape[0]) * self.dy

            plt.figure()
            plt.pcolormesh(sar_image_x, sar_image_y, sar_image, shading='auto', cmap='jet')
            plt.xlabel('Horizontal (mm)')
            plt.ylabel('Vertical (mm)')
            plt.title('2D SAR Imaging - RMA')
            plt.xlim(self.bbox[0], self.bbox[1])
            plt.ylim(self.bbox[2], self.bbox[3])
            plt.show()