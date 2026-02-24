"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function BackpropPyTorchComparison() {
  const code = `import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)  # Input to hidden
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)  # Hidden to output
    
    def forward(self, x):
        x = self.relu(self.fc1(x))  # Hidden with ReLU
        x = self.fc2(x)             # Output (no activation)
        return x

# Create model
model = TwoLayerNet()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Sample data
x = torch.randn(10, 2)  # 10 samples, 2 features
y = torch.randn(10, 1)  # 10 targets

# Training loop
for epoch in range(100):
    # Forward pass
    predictions = model(x)
    loss = criterion(predictions, y)
    
    # Backward pass
    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

# Inspect gradients
for name, param in model.named_parameters():
    print(f'{name}: gradient = {param.grad}')`;

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
            Backpropagation with PyTorch:
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
