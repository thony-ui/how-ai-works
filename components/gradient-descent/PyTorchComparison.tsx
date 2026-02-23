"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function PyTorchComparison() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">PyTorch Implementation</CardTitle>
      </CardHeader>
      <CardContent>
        <pre className="text-xs overflow-x-auto text-black rounded-lg min-w-0 max-w-full whitespace-pre-wrap break-words">
          <code className="language-python">{`import torch
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
print(f'Learned: w = {w:.4f}, b = {b:.4f}')`}</code>
        </pre>
      </CardContent>
    </Card>
  );
}
