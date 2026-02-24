"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function PyTorchComparison() {
  const code = `import torch
import torch.nn as nn

# Sample data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Create linear model (y = w*x + b)
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=0.01
)

# Training loop
for epoch in range(100):
    # Forward pass
    pred = model(x)
    loss = criterion(pred, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

# Get learned parameters
w = model.weight.item()
b = model.bias.item()
print(f'Learned: w = {w:.4f}, b = {b:.4f}')`;

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
            Gradient descent with PyTorch:
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
