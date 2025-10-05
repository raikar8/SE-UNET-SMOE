"""
Deep Complex U-Net for Phase-Aware Speech Enhancement

Implementation of "Phase-Aware Speech Enhancement with Deep Complex U-Net"
(Choi et al., ICLR 2019)

This module provides a production-ready implementation of DCUnet with:
- Complex-valued operations (convolution, batch normalization, activation)
- Polar coordinate masking
- Weighted SDR loss function
- Configurable architecture depths (10, 16, 20 layers)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np

# ==============================================================================
# Complex-Valued Operations
# ==============================================================================

class ComplexConv2d(nn.Module):
    """Complex-valued 2D convolution.

    Implements: (A + iB) * (x + iy) = (A*x - B*y) + i(B*x + A*y)"""
    def __init__(self,in_channels:int,out_channels:int,kernel_size: Tuple[int, int],stride: Tuple[int, int]=(1,1),padding: Tuple[int, int]=(0,0)):
        super().__init__()

        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

            Args:
                x_real: Real part (B, C, F, T)
                x_imag: Imaginary part (B, C, F, T)

            Returns:
                Tuple of (real_out, imag_out)
            """
        # (A*x - B*y) + i(B*x + A*y)
        real_out = self.conv_real(x_real) - self.conv_imag(x_imag)
        imag_out = self.conv_real(x_imag) + self.conv_imag(x_real)

        return real_out, imag_out


class ComplexConvTranspose2d(nn.Module):
    """Complex-valued 2D transposed convolution (deconvolution).

        Implements the same complex multiplication as ComplexConv2d,
        but with transposed convolution for upsampling in the decoder.

        Args:
            ... (same as ComplexConv2d)
        """

    def __init__(self,in_channels: int, out_channels: int, kernel_size: Tuple[int, int], stride: Tuple[int, int]=(1,1),padding: Tuple[int, int]=(0,0), output_padding: Tuple[int, int]=(0,0)):
        super().__init__()

        self.deconv_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.deconv_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

            Args:
                x_real: Real part (B, C, F, T)
                x_imag: Imaginary part (B, C, F, T)

            Returns:
                Tuple of (real_out, imag_out)
            """

        real_out = self.deconv_real(x_real) - self.deconv_imag(x_imag)
        imag_out = self.deconv_real(x_imag) + self.deconv_imag(x_real)

        return real_out, imag_out


class ComplexBatchNorm2d(nn.Module):
    """Complex-valued batch normalization.
    Implements the complex batch-normalization."""

    def __init__(self,num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters (scale and shift)
        # TODO: What shape should these be?
        self.gamma_real = nn.Parameter(torch.ones(num_features))
        self.gamma_imag = nn.Parameter(torch.zeros(num_features))
        self.beta_real = nn.Parameter(torch.zeros(num_features))
        self.beta_imag = nn.Parameter(torch.zeros(num_features))

        # Running statistics (for inference)
        # TODO: What statistics do we need to track?
        self.register_buffer('running_mean_real', torch.zeros(num_features))
        self.register_buffer('running_mean_imag', torch.zeros(num_features))
        self.register_buffer('running_var_real', torch.ones(num_features))
        self.register_buffer('running_var_imag', torch.ones(num_features))
        self.register_buffer('running_covar', torch.zeros(num_features))  # New!

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with shape (B, C, F, T)."""

        if self.training:
            mean_real = x_real.mean(dim=[0,2,3])
            mean_imag = x_imag.mean(dim=[0,2,3])

            centered_real = x_real - mean_real.view(1, -1, 1, 1)
            centered_imag = x_imag - mean_imag.view(1, -1, 1, 1)

            var_real = (centered_real ** 2).mean(dim=[0,2,3])
            var_imag = (centered_imag ** 2).mean(dim=[0,2,3])

            covar = (centered_real * centered_imag).mean(dim=[0,2,3])

            with torch.no_grad():
                self.running_mean_real = (1-self.momentum)*self.running_mean_real + self.momentum*mean_real
                self.running_mean_imag = (1-self.momentum)*self.running_mean_imag + self.momentum*mean_imag

                self.running_var_real = (1-self.momentum)*self.running_var_real + self.momentum*var_real
                self.running_var_imag = (1-self.momentum)*self.running_var_imag + self.momentum*var_imag

                self.running_covar = (1-self.momentum)*self.running_covar + self.momentum*covar

        else:
            mean_real = self.running_mean_real
            mean_imag = self.running_mean_imag

            var_real = self.running_var_real
            var_imag = self.running_var_imag

            covar = self.running_covar

            centered_real = x_real - mean_real.view(1, -1, 1, 1)
            centered_imag = x_imag - mean_imag.view(1, -1, 1, 1)

        # ====================================================================
        # NORMALIZATION: Apply V^(-1/2) where V is covariance matrix
        # ====================================================================

        # For a 2x2 matrix: V = [[a, b], [b, c]]
        # tau = trace(V) = a + c
        # delta = det(V) = ac - b²


        tau = var_real + var_imag + self.eps
        delta = var_real*var_imag - covar**2 + self.eps

        s = torch.sqrt(delta/tau)
        t = torch.sqrt(tau) + self.eps

        normalized_real = (centered_real * (var_imag / t).view(1, -1, 1, 1) +
                           centered_imag * (-covar / t).view(1, -1, 1, 1)) / s.view(1, -1, 1, 1)

        normalized_imag = (centered_real * (-covar / t).view(1, -1, 1, 1) +
                           centered_imag * (var_real / t).view(1, -1, 1, 1)) / s.view(1, -1, 1, 1)

        #print(f"normalized_real shape {normalized_real.shape}")
        #print(f"normalized_imag shape {normalized_imag.shape}")

        #print(f"gamma_real shape {self.gamma_real.shape}")
        #print(f"gamma_imag shape {self.gamma_imag.shape}")

        gamma_real = self.gamma_real.view(1, -1, 1, 1)
        gamma_imag = self.gamma_imag.view(1, -1, 1, 1)
        beta_real = self.beta_real.view(1, -1, 1, 1)
        beta_imag = self.beta_imag.view(1, -1, 1, 1)

        # Apply complex multiplication: gamma * normalized
        real_scaled = gamma_real * normalized_real - gamma_imag * normalized_imag
        imag_scaled = gamma_real * normalized_imag + gamma_imag * normalized_real

        # Add shift: gamma * normalized + beta
        out_real = real_scaled + beta_real
        out_imag = imag_scaled + beta_imag

        return out_real, out_imag


class ComplexLeakyRelU(nn.Module):
    """
        Complex leaky ReLU activation.

        Applies leaky ReLU independently to real and imaginary parts.
        This is called "CReLU" in the paper (Trabelsi et al. 2018).

        Args:
            negative_slope: Slope for negative values (default: 0.01)
        """
    def __init__(self, negative_slope : float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Apply LeakyReLU to both components independently.

                Args:
                    x_real: Real part (B, C, F, T)
                    x_imag: Imaginary part (B, C, F, T)

                Returns:
                    Tuple of (activated_real, activated_imag)
                """

        activated_real = F.leaky_relu(x_real, self.negative_slope)
        activated_imag = F.leaky_relu(x_imag, self.negative_slope)

        return activated_real, activated_imag


class EncoderBlock(nn.Module):
    """
    Encoder block for DCUNet.

    Applies: ComplexConv2d → ComplexBatchNorm2d → ComplexLeakyReLU

    This downsamples spatial dimensions while increasing channels,
    extracting hierarchical features.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (freq, time)
        stride: Convolution stride (freq, time)
        padding: Convolution padding (freq, time)
        negative_slope: Slope for LeakyReLU (default: 0.01)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int]):

        super().__init__()

        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.activation = ComplexLeakyRelU(negative_slope=0.01)


    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x_real, x_imag = self.conv(x_real,x_imag)
        x_real, x_imag = self.bn(x_real,x_imag)
        x_real, x_imag = self.activation(x_real, x_imag)

        return x_real, x_imag


class DecoderBlock(nn.Module):
    """Decoder block with complex deconvolution, batch norm, and activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int], output_padding: Tuple[int, int] = (0, 0)):
        super().__init__()

        self.deconv = ComplexConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.activation = ComplexLeakyRelU(negative_slope=0.01)


    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_real, x_imag = self.deconv(x_real,x_imag)
        x_real, x_imag = self.bn(x_real,x_imag)
        x_real, x_imag = self.activation(x_real, x_imag)

        return x_real, x_imag



# ==============================================================================
# Deep Complex U-Net Architecture
# ==============================================================================

class DCUNet(nn.Module):
    """Deep Complex U-Net for speech enhancement.

      Args:
          architecture: One of '10', '16', '20' specifying the number of layers
          fix_length_mode: 'pad' or 'trim' for handling variable-length inputs
      """

    ARCHITECTURES = {
        '10': [
            # Encoder: (kernel, stride, channels)
            ((7, 5), (2, 2), 32),
            ((5, 3), (2, 2), 64),
            ((5, 3), (2, 2), 64),
            ((5, 3), (2, 2), 64),
            ((5, 3), (2, 1), 64),
        ],
        '16': [
            ((7, 5), (2, 2), 32),
            ((7, 5), (2, 1), 32),
            ((5, 3), (2, 2), 64),
            ((5, 3), (2, 1), 64),
            ((5, 3), (2, 2), 64),
            ((5, 3), (2, 1), 64),
            ((5, 3), (2, 2), 64),
            ((5, 3), (2, 1), 64),
        ],
        '20': [
            ((1, 7), (1, 1), 32),
            ((7, 1), (1, 1), 32),
            ((7, 5), (2, 2), 64),
            ((7, 5), (2, 1), 64),
            ((5, 3), (2, 2), 64),
            ((5, 3), (2, 1), 64),
            ((5, 3), (2, 2), 64),
            ((5, 3), (2, 1), 64),
            ((5, 3), (2, 2), 64),
            ((5, 3), (2, 1), 90),
        ],
    }

    def __init__(self, architecture: str = '20'):
        super().__init__()

        if architecture != '20':
            raise NotImplementedError("Only DCUNet-20 is implemented for now")

        self.architecture = architecture
        #self.fix_length_mode = fix_length_mode

        # ================================================================
        # ENCODER CONFIGURATION (from paper Figure 7)
        # ================================================================
        encoder_config = [
            # (in_channels, out_channels, kernel_size, stride, padding)
            (1, 32, (1, 7), (1, 1), (0, 3)),  # E1: No downsample
            (32, 32, (7, 1), (1, 1), (3, 0)),  # E2: No downsample
            (32, 64, (7, 5), (2, 2), (3, 2)),  # E3: Downsample 2×
            (64, 64, (7, 5), (2, 1), (3, 2)),  # E4: Downsample freq
            (64, 64, (5, 3), (2, 2), (2, 1)),  # E5: Downsample 2×
            (64, 64, (5, 3), (2, 1), (2, 1)),  # E6: Downsample freq
            (64, 64, (5, 3), (2, 2), (2, 1)),  # E7: Downsample 2×
            (64, 64, (5, 3), (2, 1), (2, 1)),  # E8: Downsample freq
            (64, 64, (5, 3), (2, 2), (2, 1)),  # E9: Downsample 2×
            (64, 90, (5, 3), (2, 1), (2, 1)),  # E10: Bottleneck
        ]



        # Build Encoders

        self.encoders = nn.ModuleList()

        for in_ch, out_ch, kernel_size, stride, padding in encoder_config:
            self.encoders.append(EncoderBlock(in_ch, out_ch, kernel_size, stride, padding))

        # ================================================================
        # DECODER CONFIGURATION (from paper Figure 7)
        # ================================================================
        # Note: in_channels accounts for concatenated skip connections
        # Formula: in_ch = previous_decoder_output + skip_connection_channels

        decoder_config = [
            # (in_ch, out_ch, kernel, stride, padding, output_padding)
            (90, 64, (5, 3), (2, 1), (2, 1), (0, 0)),  # D1: No skip (bottleneck)
            (128, 64, (5, 3), (2, 2), (2, 1), (1, 1)),  # D2: 64+64 from E9
            (128, 64, (5, 3), (2, 1), (2, 1), (0, 0)),  # D3: 64+64 from E8
            (128, 64, (5, 3), (2, 2), (2, 1), (1, 1)),  # D4: 64+64 from E7
            (128, 64, (5, 3), (2, 1), (2, 1), (0, 0)),  # D5: 64+64 from E6
            (128, 64, (5, 3), (2, 2), (2, 1), (1, 1)),  # D6: 64+64 from E5
            (128, 64, (7, 5), (2, 1), (3, 2), (0, 0)),  # D7: 64+64 from E4
            (128, 32, (7, 5), (2, 2), (3, 2), (1, 1)),  # D8: 64+64 from E3
            (64, 32, (7, 1), (1, 1), (3, 0), (0, 0)),  # D9: 32+32 from E2
            (64, 1, (1, 7), (1, 1), (0, 3), (0, 0)),  # D10: 32+32 from E1
        ]

        self.decoders = nn.ModuleList()

        for in_ch, out_ch, kernel_size, stride, padding, output_padding in decoder_config:
            self.decoders.append(DecoderBlock(in_ch, out_ch, kernel_size, stride, padding, output_padding))

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Forward pass through U-Net.

                Args:
                    x_real: Real part of input spectrogram (B, 1, F, T)
                    x_imag: Imaginary part of input spectrogram (B, 1, F, T)

                Returns:
                    Tuple of (output_real, output_imag) spectrograms

                Architecture flow:
                    Input → Encoders (save outputs) → Bottleneck →
                    Decoders (with skip connections) → Output
        """

        # ================================================================
        # ENCODER PATH: Downsample and extract features
        # ================================================================

        encoder_outputs = []
        h_real, h_imag = x_real, x_imag

        for i, encoder in enumerate(self.encoders):
            h_real, h_imag = encoder(h_real, h_imag)
            encoder_outputs.append((h_real, h_imag))

            print(f"Encoder {i + 1} output: {h_real.shape}")
            #print(f"Encoder {i + 1} output: {h_imag.shape}")

        # ================================================================
        # DECODER PATH: Upsample with skip connections
        # ================================================================

        # Start with bottleneck (last encoder output)
        # Note: We DON'T pop from encoder_outputs, we index into it

        for i, decoder in enumerate(self.decoders):
            if i == 0:
                # First decoder: No skip connection (process bottleneck only)
                # h_real, h_imag already contain bottleneck output
                pass

            else:
                # Subsequent decoders: Concatenate with skip connection
                # Skip connection index: mirrors encoder path
                # Dec2 connects to Enc9, Dec3 to Enc8, etc

                skip_idx = len(self.encoders) - i - 1  # -1 because we skipped bottleneck
                skip_real, skip_imag = encoder_outputs[skip_idx]

                # **FIX: Match shapes before concatenation**
                if h_real.shape[2:] != skip_real.shape[2:]:
                    # Resize decoder output to match skip connection size
                    h_real = F.interpolate(
                        h_real,
                        size=skip_real.shape[2:],  # Match (H, W)
                        mode='nearest'
                    )
                    h_imag = F.interpolate(
                        h_imag,
                        size=skip_imag.shape[2:],
                        mode='nearest'
                    )

                #print(f"Decoder {i + 1} input (before concat): {h_real.shape}")

                # Concatenate along channel dimension (dim=1)
                h_real = torch.cat([h_real, skip_real], dim=1)
                h_imag = torch.cat([h_imag, skip_imag], dim=1)

                #print(f"Decoder {i + 1} input (after concat): {h_real.shape}")
                #print(f"Decoder {i + 1} input (after concat): {h_imag.shape}")

            # Pass through decoder block
            h_real, h_imag = decoder(h_real, h_imag)

            print(f"Decoder {i + 1} output: {h_real.shape}")
            #print(f"Decoder {i + 1} output: {h_imag.shape}")

        return h_real, h_imag


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class STFT(nn.Module):
    """
    Short-Time Fourier Transform as a differentiable layer.

    Implements STFT using 1D convolution with fixed DFT basis functions.
    This allows gradients to flow through for end-to-end training.

    Args:
        n_fft: FFT size (default: 1024, gives 513 freq bins)
        hop_length: Hop size between frames (default: 256)
        win_length: Window length (default: 1024, same as n_fft)
        window: Window type (default: 'hann')
        center: Whether to pad signal for centered frames (default: True)
        normalized: Whether to normalize STFT (default: False)

    Input:
        audio: (B, T) or (B, 1, T) waveform

    Output:
        real: (B, 1, F, T) - Real part of STFT
        imag: (B, 1, F, T) - Imaginary part of STFT
        where F = n_fft//2 + 1 = 513 for n_fft=1024
    """

    def __init__(self, n_fft: int=1024, hop_length: int=256, win_length: Optional[int] = None, window: str='hann', center: bool = True, normalized: bool = False):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        #self.window = window
        self.center = center
        self.normalized = normalized

        # Number of frequency bins (positive frequencies only)
        self.n_bins = self.n_fft // 2 + 1

        if window == 'hann':
            window_fn = torch.hann_window(self.win_length, periodic=True)
        elif window == 'hamming':
            window_fn = torch.hamming_window(self.win_length, periodic=True)
        else:
            window_fn = torch.ones(self.win_length)

        # Pad window to n_fft if needed

        if self.win_length < self.n_fft:
            pad_left = (self.n_fft - self.win_length) // 2
            pad_right = self.n_fft - self.win_length - pad_left
            window_fn = F.pad(window_fn, (pad_left, pad_right))

        # ================================================================
        # Create DFT basis (Fourier basis functions)
        # ================================================================
        # For n_fft=1024, we create 513 cos and 513 sin basis functions

        # Frequency indices: [0, 1, 2, ..., n_fft//2]
        freq_bins = torch.arange(0, self.n_bins)

        # Time indices: [0, 1, 2, ..., n_fft-1]
        time_bins = torch.arange(0, self.n_fft)

        # DFT matrix: exp(-2πi * freq * time / n_fft)
        # Real part: cos(-2π * freq * time / n_fft)
        # Imag part: sin(-2π * freq * time / n_fft)

        # Shape: (n_bins, n_fft)
        dft_real = torch.cos(
            -2 * np.pi * freq_bins.unsqueeze(1) * time_bins.unsqueeze(0) / self.n_fft
        )
        dft_imag = torch.sin(
            -2 * np.pi * freq_bins.unsqueeze(1) * time_bins.unsqueeze(0) / self.n_fft
        )

        # Apply window to each basis function
        dft_real = dft_real * window_fn.unsqueeze(0)
        dft_imag = dft_imag * window_fn.unsqueeze(0)

        # Normalize if requested (not used in paper)
        if normalized:
            dft_real = dft_real / self.n_fft
            dft_imag = dft_imag / self.n_fft

        # Register as non-trainable parameters (buffers)
        # Shape: (n_bins, 1, n_fft) for conv1d
        self.register_buffer('dft_real', dft_real.unsqueeze(1))
        self.register_buffer('dft_imag', dft_imag.unsqueeze(1))

        # Store window for ISTFT compatibility
        self.register_buffer('window', window_fn)

    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Apply STFT to audio.

                Args:
                    audio: (B, T) or (B, 1, T) waveform

                Returns:
                    real: (B, 1, F, T_frames) - Real part
                    imag: (B, 1, F, T_frames) - Imaginary part
        """

        # Handle input shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # (B, T) -> (B, 1, T)

        # Center padding (like librosa.stft with center=True)
        if self.center:
            pad_amount = self.n_fft // 2
            audio = F.pad(audio, (pad_amount, pad_amount), mode='reflect')

        # Apply convolution with DFT basis
        # Conv1d expects (B, C_in, L)
        # Our audio is (B, 1, T)
        # Our filters are (C_out, C_in, kernel_size) = (n_bins, 1, n_fft)

        real = F.conv1d(audio, self.dft_real, stride=self.hop_length)
        imag = F.conv1d(audio, self.dft_imag, stride=self.hop_length)

        # Output shape: (B, n_bins, T_frames)
        # We want: (B, 1, F, T) to match DCUNet input
        # So we unsqueeze at dim=1

        real = real.unsqueeze(1)  # (B, 1, F, T)
        imag = imag.unsqueeze(1)  # (B, 1, F, T)

        return real, imag


class ISTFT(nn.Module):
    def __init__(
            self,
            n_fft: int = 1024,
            hop_length: int = 256,
            win_length: Optional[int] = None,
            window: str = 'hann',
            center: bool = True,
            normalized: bool = False
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.center = center
        self.normalized = normalized
        self.n_bins = n_fft // 2 + 1

        # Window
        if window == 'hann':
            window_fn = torch.hann_window(self.win_length, periodic=True)
        elif window == 'hamming':
            window_fn = torch.hamming_window(self.win_length, periodic=True)
        else:
            window_fn = torch.ones(self.win_length)

        if self.win_length < n_fft:
            pad_left = (n_fft - self.win_length) // 2
            pad_right = n_fft - self.win_length - pad_left
            window_fn = F.pad(window_fn, (pad_left, pad_right))

        # IDFT basis
        freq_bins = torch.arange(0, self.n_bins).float()
        time_bins = torch.arange(0, n_fft).float()

        # Shape: (n_bins, n_fft)
        idft_real = torch.cos(2 * np.pi * freq_bins.unsqueeze(1) * time_bins.unsqueeze(0) / n_fft)
        idft_imag = torch.sin(2 * np.pi * freq_bins.unsqueeze(1) * time_bins.unsqueeze(0) / n_fft)

        # Apply window: (n_bins, n_fft) * (n_fft,) -> (n_bins, n_fft)
        idft_real = idft_real * window_fn.unsqueeze(0)
        idft_imag = idft_imag * window_fn.unsqueeze(0)

        # Scale: (n_bins, 1)
        scale = torch.ones(self.n_bins, 1) * 2.0 / n_fft
        scale[0] = 1.0 / n_fft
        if n_fft % 2 == 0:
            scale[-1] = 1.0 / n_fft

        if not normalized:
            idft_real = idft_real * scale
            idft_imag = idft_imag * scale

        # Register: final shape MUST be (n_bins, n_fft)
        self.register_buffer('idft_real', idft_real)
        self.register_buffer('idft_imag', idft_imag)

    def forward(self, real: torch.Tensor, imag: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        B, _, n_freq, n_frames = real.shape

        real = real.squeeze(1)  # (B, F, T)
        imag = imag.squeeze(1)

        print(f"  Real part: {real.shape}")

        audio_length = self.hop_length * (n_frames - 1) + self.n_fft
        audio = torch.zeros(B, audio_length, device=real.device, dtype=real.dtype)

        for b in range(B):
            for t in range(n_frames):
                real_frame = real[b, :, t]  # (F,)
                imag_frame = imag[b, :, t]

                # (F,) @ (F, n_fft) -> (n_fft,)
                frame_audio = torch.matmul(real_frame, self.idft_real) - torch.matmul(imag_frame, self.idft_imag)

                start = t * self.hop_length
                audio[b, start:start + self.n_fft] += frame_audio

        if self.center:
            pad_amount = self.n_fft // 2
            audio = audio[:, pad_amount:-pad_amount] if length is None else audio[:, pad_amount:pad_amount + length]
        elif length is not None:
            audio = audio[:, :length]

        return audio



# ==============================================================================
# Convenience Functions
# ==============================================================================

def create_stft_layer(
        sr: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None
) -> STFT:
    """
    Create STFT layer with paper's default parameters.

    Paper settings (Section 4.1):
    - Sample rate: 16kHz
    - Window: 64ms Hann window = 1024 samples at 16kHz
    - Hop: 16ms = 256 samples at 16kHz
    - Gives 513 frequency bins

    Args:
        sr: Sample rate (Hz)
        n_fft: FFT size
        hop_length: Hop size in samples
        win_length: Window length in samples

    Returns:
        STFT layer configured for DCUNet
    """
    return STFT(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True,
        normalized=False
    )


def create_istft_layer(
        sr: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None
) -> ISTFT:
    """
    Create ISTFT layer matching STFT parameters.

    Args:
        sr: Sample rate (Hz)
        n_fft: FFT size
        hop_length: Hop size in samples
        win_length: Window length in samples

    Returns:
        ISTFT layer configured for DCUNet
    """
    return ISTFT(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True,
        normalized=False
    )






























