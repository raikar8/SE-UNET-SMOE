from baseBlocks_simple_UNET import ComplexConv2d
from baseBlocks_simple_UNET import ComplexBatchNorm2d
from baseBlocks_simple_UNET import ComplexLeakyRelU
from baseBlocks_simple_UNET import EncoderBlock
from baseBlocks_simple_UNET import DecoderBlock
from baseBlocks_simple_UNET import count_parameters
from baseBlocks_simple_UNET import DCUNet

from baseBlocks_simple_UNET import create_stft_layer
from baseBlocks_simple_UNET import create_istft_layer
import torch.nn.functional as F




import torch

import torch.nn as nn
import numpy as np


def verify_shape_network():


    conv = ComplexConv2d(in_channels=64, out_channels=128, kernel_size=(7,5), stride=(2,2), padding=(3,2))

    x_real = torch.randn(2,64,129,100)
    x_imag = torch.randn(2,64,129,100)


    out_real,out_imag = conv(x_real,x_imag)
    #out_imag = conv(x_imag)

    print(out_real.shape)
    print(out_imag.shape)


def calculate_parameters():

    conv = ComplexConv2d(in_channels=1, out_channels=64, kernel_size=(3,3))

    conv_orig = nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(3,3))

    total_parameters = sum(p.numel() for p in conv.parameters())
    print(f"Number of conv parameters: {total_parameters}")

    print(f"Number of conv parameters original implementation: {sum(p.numel() for p in conv_orig.parameters())}")

    # Break it down
    print(f"conv_real params: {sum(p.numel() for p in conv.conv_real.parameters()):,}")
    print(f"conv_imag params: {sum(p.numel() for p in conv.conv_imag.parameters()):,}")


def understanding_complexnorm():

    B,C,F,T = 2,64,4,5

    x_real = torch.randn(B,C,F,T)
    x_imag = torch.randn(B,C,F,T)

    print(x_real.shape)
    print(x_imag.shape)

    mean_real = x_real.mean(dim=[0, 2, 3])
    mean_imag = x_imag.mean(dim=[0, 2, 3])

    print(mean_real)
    print(mean_imag)

    print(mean_real.shape)
    print(mean_imag.shape)


def test_complex_leaky_relu():
    print("=" * 70)
    print("TEST: ComplexLeakyReLU")
    print("=" * 70)

    activation = ComplexLeakyRelU(negative_slope=0.01)

    # Test with positive and negative values
    x_real = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    x_imag = torch.tensor([[-1.5, -0.5, 0.0, 0.5, 1.5]])

    out_real, out_imag = activation(x_real, x_imag)

    print("Input real:", x_real)
    print("Output real:", out_real)
    print()
    print("Input imag:", x_imag)
    print("Output imag:", out_imag)
    print()

    # Check negative values are scaled
    assert out_real[0, 0] == -2.0 * 0.01, "Negative slope not applied!"
    assert out_real[0, 4] == 2.0, "Positive values should pass through!"

    print("✅ All tests passed!")


def test_encoder_decoder_blocks():
    print("=" * 70)
    print("TEST: EncoderBlock and DecoderBlock")
    print("=" * 70)

    # Test EncoderBlock
    encoder = EncoderBlock(
        in_channels=1,
        out_channels=64,
        kernel_size=(7, 5),
        stride=(2, 2),
        padding=(3, 2)
    )

    x_real = torch.randn(2, 1, 256, 100)
    x_imag = torch.randn(2, 1, 256, 100)

    out_real, out_imag = encoder(x_real, x_imag)

    print(f"Encoder Input:  {x_real.shape}")
    print(f"Encoder Output: {out_real.shape}")
    print(f"Expected: (2, 64, 128, 50)")
    assert out_real.shape == (2, 64, 128, 50), "Encoder output shape wrong!"
    print("✅ EncoderBlock shape correct!\n")

    # Test DecoderBlock
    decoder = DecoderBlock(
        in_channels=64,
        out_channels=1,
        kernel_size=(7, 5),
        stride=(2, 2),
        padding=(3, 2),
        output_padding=(1, 1)
    )

    dec_out_real, dec_out_imag = decoder(out_real, out_imag)

    print(f"Decoder Input:  {out_real.shape}")
    print(f"Decoder Output: {dec_out_real.shape}")
    print("✅ DecoderBlock works!\n")

    # Test gradient flow
    loss = dec_out_real.sum() + dec_out_imag.sum()
    loss.backward()

    print("✅ Gradients flow through entire encoder-decoder chain!")


def test_dcunet_architecture():
    """Test complete DCUNet-20 architecture."""

    print("=" * 70)
    print("TESTING: DCUNet-20 Complete Architecture")
    print("=" * 70)
    print()

    # Initialize model
    model = DCUNet(architecture='20')

    # Count parameters
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Expected from paper: ~3,500,000")
    print(f"Match: {'✅' if 3_400_000 < total_params < 3_600_000 else '❌'}")
    print()

    # Test with realistic input size (from paper)
    # Paper uses: 64ms window, 16ms hop, 16kHz sampling rate
    # This gives: 257 freq bins, ~100 time frames for 1 second
    B, F, T = 2, 513, 100

    print(f"Input shape: ({B}, 1, {F}, {T})")

    x_real = torch.randn(B, 1, F, T)
    x_imag = torch.randn(B, 1, F, T)

    # Forward pass
    print("\nRunning forward pass...")
    out_real, out_imag = model(x_real, x_imag)

    print(f"Output shape: {out_real.shape}")
    print(f"Expected: torch.Size([{B}, 1, {F}, {T}])")

    # Verify shape preservation
    assert out_real.shape == x_real.shape, f"Shape mismatch! {out_real.shape} != {x_real.shape}"
    assert out_imag.shape == x_imag.shape, f"Shape mismatch! {out_imag.shape} != {x_imag.shape}"

    print("✅ Shape preserved through U-Net!")
    print()

    # Test gradient flow
    print("Testing gradient flow...")
    loss = out_real.sum() + out_imag.sum()
    loss.backward()

    # Check gradients exist
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "Some parameters don't have gradients!"

    print("✅ Gradients flow through entire network!")
    print()

    # Test with different batch sizes
    print("Testing different batch sizes...")
    for batch_size in [1, 4, 8]:
        x_real_test = torch.randn(batch_size, 1, F, T)
        x_imag_test = torch.randn(batch_size, 1, F, T)

        with torch.no_grad():
            out_r, out_i = model(x_real_test, x_imag_test)

        assert out_r.shape == x_real_test.shape
        print(f"  Batch size {batch_size}: ✅")

    print()
    print("=" * 70)
    print("ALL TESTS PASSED! 🎉")
    print("=" * 70)
    print()
    print("Your DCUNet-20 implementation is complete and working!")
    print("Next steps:")
    print("  1. Add masking layer (bounded_tanh)")
    print("  2. Implement weighted SDR loss")
    print("  3. Add STFT/ISTFT for end-to-end processing")


def test_skip_connections():
    """Verify skip connections are working correctly."""

    print("=" * 70)
    print("TESTING: Skip Connection Verification")
    print("=" * 70)
    print()

    model = DCUNet(architecture='20')
    model.eval()

    # Create two different inputs
    x1_real = torch.ones(1, 1, 513, 100)
    x1_imag = torch.zeros(1, 1, 513, 100)

    x2_real = torch.zeros(1, 1, 513, 100)
    x2_imag = torch.ones(1, 1, 513, 100)

    with torch.no_grad():
        out1_real, out1_imag = model(x1_real, x1_imag)
        out2_real, out2_imag = model(x2_real, x2_imag)

    # Outputs should be different (skip connections preserve input info)
    diff_real = (out1_real - out2_real).abs().mean()
    diff_imag = (out1_imag - out2_imag).abs().mean()

    print(f"Output difference (real): {diff_real:.6f}")
    print(f"Output difference (imag): {diff_imag:.6f}")

    assert diff_real > 0.01, "Outputs too similar - skip connections may not be working!"
    assert diff_imag > 0.01, "Outputs too similar - skip connections may not be working!"

    print("✅ Skip connections are working correctly!")
    print()




if __name__ == '__main__':
    #verify_shape_network()
    #calculate_parameters()
    #understanding_complexnorm()
    #test_complex_leaky_relu()
    #test_encoder_decoder_blocks()
    #test_dcunet_architecture()
    #test_skip_connections()

    print("=" * 70)
    print("Testing STFT/ISTFT Modules")
    print("=" * 70)

    # Test parameters (from paper)
    sr = 16000
    duration = 1.0  # 1 second
    n_samples = int(sr * duration)

    # Create test signal (sine wave at 440 Hz)
    t = torch.linspace(0, duration, n_samples)
    audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # (1, T)

    print(f"\nInput audio shape: {audio.shape}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration} s")

    # Create STFT/ISTFT layers
    stft = create_stft_layer(sr=sr)
    istft = create_istft_layer(sr=sr)

    print(f"\nSTFT parameters:")
    print(f"  n_fft: {stft.n_fft}")
    print(f"  hop_length: {stft.hop_length}")
    print(f"  n_bins: {stft.n_bins}")

    # Forward STFT
    real, imag = stft(audio)
    print(f"\nSTFT output:")
    print(f"  Real shape: {real.shape}")
    print(f"  Imag shape: {imag.shape}")
    print(f"  Expected: (1, 1, 513, ~62) for 1s audio")

    # Compute magnitude for visualization
    magnitude = torch.sqrt(real ** 2 + imag ** 2)
    print(f"  Magnitude range: [{magnitude.min():.4f}, {magnitude.max():.4f}]")

    # Inverse ISTFT
    reconstructed = istft(real, imag, length=n_samples)
    print(f"\nISTFT output:")
    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  Expected: {audio.shape}")

    # Check reconstruction quality
    mse = F.mse_loss(audio, reconstructed)
    print(f"\nReconstruction quality:")
    print(f"  MSE: {mse.item():.6f}")
    print(f"  {'✅ Perfect reconstruction!' if mse < 1e-5 else '⚠️ Some error (expected with windowing)'}")

    # Test gradient flow
    print(f"\nTesting gradient flow...")
    real_grad = real.clone().requires_grad_(True)
    imag_grad = imag.clone().requires_grad_(True)

    reconstructed_grad = istft(real_grad, imag_grad, length=n_samples)
    loss = reconstructed_grad.sum()
    loss.backward()

    assert real_grad.grad is not None, "Gradients not flowing through ISTFT!"
    print(f"  ✅ Gradients flow through STFT/ISTFT!")

    print("\n" + "=" * 70)
    print("All tests passed! STFT/ISTFT ready for DCUNet integration.")
    print("=" * 70)




