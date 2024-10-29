import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.constants import speed_of_light

class LIA(nn.Module):
    def __init__(self, z0, dx, dy, im_size, kk):
        super().__init__()
        self.z0 = z0
        self.dx = dx
        self.dy = dy
        self.im_size = im_size
        self.kk = kk
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, sar_data):
        N, M = sar_data.shape
        rd = sar_data.reshape(-1, 1).to(self.device)

        cst = 1j * 2 * torch.pi * 77e9 * 2 / speed_of_light
        dxm = self.dx * 1e-3
        dym = self.dy * 1e-3
        z2 = (self.z0 * 1e-3) ** 2

        A, B = 50, 40
        wh1 = torch.linspace(self.im_size[0], self.im_size[1], A, device=self.device) * 1e-3
        wh2 = torch.linspace(self.im_size[2], self.im_size[3], B, device=self.device) * 1e-3

        py = torch.sort(torch.randperm(N * M, device=self.device)[:self.kk])[0]
        NM = len(py)
        BA = A * B
        Hp = torch.zeros((NM, BA), dtype=torch.complex64, device=self.device)
        for i in tqdm(range(NM), desc='Computing Hp'):
            iy = py[i] % N
            ix = (py[i] - iy) // N
            for j in range(BA):
                jy = j % B
                jx = (j - jy) // B
                dist2 = ((iy + 0.5 - N / 2) * dym - wh2[jy]) ** 2 + \
                        ((ix + 0.5 - M / 2) * dxm - wh1[jx]) ** 2 + z2
                Hp[i, j] = torch.exp(cst * torch.sqrt(dist2))

        di = 0.01
        G = di * torch.mm(Hp.T.conj(), Hp)
        xd = di * torch.mm(Hp.T.conj(), rd[py].to(torch.complex64))
        for j in tqdm(range(BA), desc='Iteration Progress'):
            temp = G[:, j] / (1 + G[j, j])
            temp = temp.reshape(-1, 1)
            xd -= temp * xd[j]
            G -= torch.outer(temp.view(-1), G[j, :].view(-1))  
        xd /= torch.diag(G).reshape(-1, 1)
        sar_image = torch.flip(xd.reshape(B, A), [1])

        return sar_image

    def plot_result(self, sar_image):
        sar_image = sar_image.detach().cpu()

        wh1 = torch.linspace(self.im_size[0], self.im_size[1], 50) * 1e-3
        wh2 = torch.linspace(self.im_size[2], self.im_size[3], 40) * 1e-3
        wh1_mm = wh1 * 1000
        wh2_mm = wh2 * 1000

        plt.figure()
        plt.imshow(torch.abs(sar_image).cpu().numpy(), extent=[wh1_mm[0].item(), wh1_mm[-1].item(), 
                                                         wh2_mm[0].item(), wh2_mm[-1].item()], 
                   cmap='jet', aspect='auto')
        plt.xlabel('Horizontal (mm)')
        plt.ylabel('Vertical (mm)')
        plt.title(f'2D SAR Imaging - LIA kk= {round(self.kk / (sar_image.shape[0] * sar_image.shape[1]) * 100)}%')
        plt.colorbar(label='Magnitude')
        plt.gca().invert_yaxis()
        plt.show()