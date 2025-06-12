# Second_paper
An Enhanced Convolutional Blind Denoiser Network with Augmented Noise Estimation For Medical Image Denoising

# CBD-Net with Augmented Noise Estimation for Biomedical Image Denoising

![MRI Denoising Example](https://example.com/mri-denoising-comparison.png)  
*Example of denoising performance on T2-weighted MRI (Left: Noisy, Right: Denoised)*

## 🔍 Project Overview
This repository implements an enhanced **Convolutional Blind Denoising Network (CBD-Net)** with an **augmented noise estimation channel** for biomedical images. The model specifically targets:
- **Non-IID noise**: T-Student and Laplace distributions (common in MRI artifacts)
- **IID noise**: Beta, Poisson, and Binomial distributions  
Validated on 7,023 grayscale MRI images with state-of-the-art metrics (SSIM↑, PSNR↑, MSE↓).

## 🚀 Key Features
- **Dual-path architecture**: Combines CNN's feature extraction with autoencoder's reconstruction capability
- **Augmented noise estimation**: Dedicated subnetwork for dynamic noise-level prediction
- **Realistic noise modeling**: 5 statistical distributions simulating clinical scenarios
- **Multi-metric evaluation**: SSIM, PSNR, MSE with cross-distribution benchmarking

## 📊 Performance Highlights
| Noise Type   | Distribution | SSIM    | PSNR   | MSE    |
|--------------|-------------|---------|--------|--------|
| **Non-IID**  | T-Student   | 0.901   | 30.724 | 0.008  |
|              | Laplace     | 0.897   | 31.091 | 0.007  |
| **IID**      | Beta        | 0.920   | 32.912 | 0.005  |
|              | Poisson     | 0.922   | 32.863 | 0.005  |
|              | Binomial    | 0.906   | 31.005 | 0.008  |

## 🛠️ Architecture
```python
class AugmentedCBDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_estimator = NoiseEstimationCNN()  # Augmented channel
        self.denoiser = CBDNetBackbone()
        
    def forward(self, x):
        noise_level = self.noise_estimator(x)
        return self.denoiser(x, noise_level)
