"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function ConvolutionPyTorchComparison() {
  const code = `import torch
import torch.nn as nn

# Create a 2D convolution layer
# in_channels: number of input channels (1 for grayscale)
# out_channels: number of filters/kernels
# kernel_size: size of the convolution kernel
conv = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=False
)

# Input image: (batch_size, channels, height, width)
image = torch.randn(1, 1, 5, 5)

# Apply convolution
output = conv(image)

print(f"Input shape: {image.shape}")
print(f"Output shape: {output.shape}")
print(f"Kernel weights shape: {conv.weight.shape}")

# Access the kernel weights
kernel = conv.weight.data
print(f"Kernel:\\n{kernel}")

# Manual convolution (for understanding)
import torch.nn.functional as F

custom_kernel = torch.tensor([
    [[-1, -1, -1],
     [-1,  8, -1],
     [-1, -1, -1]]
], dtype=torch.float32).unsqueeze(0)

output = F.conv2d(image, custom_kernel)
print(f"Convolution output:\\n{output}")

# Common operations in CNNs
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3),  # 16 filters
    nn.ReLU(),                         # Activation
    nn.MaxPool2d(2),                   # Downsampling
    nn.Conv2d(16, 32, kernel_size=3),  # 32 filters
    nn.ReLU()
)

# Process image through network
features = model(torch.randn(1, 1, 28, 28))
print(f"Final feature shape: {features.shape}")`;

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">PyTorch Implementation</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex justify-between items-start mb-2">
          <p className="text-sm text-muted-foreground">
            Convolution with PyTorch:
          </p>
          <button
            onClick={handleCopy}
            className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded shrink-0"
            type="button"
          >
            Copy Code
          </button>
        </div>
        <pre className="rounded-lg bg-slate-950 text-slate-50 p-4 overflow-x-auto text-xs font-mono whitespace-pre-wrap break-words">
          <code>{code}</code>
        </pre>
      </CardContent>
    </Card>
  );
}
