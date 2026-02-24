"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export function MultiClassPyTorchComparison() {
  const modelCode = `import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the network architecture
class MultiClassNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=8, num_classes=3):
        super().__init__()
        # Layer 1: Input → Hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Layer 2: Hidden → Output
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Forward pass
        z1 = self.fc1(x)        # Linear transformation
        a1 = F.relu(z1)         # ReLU activation
        z2 = self.fc2(a1)       # Linear transformation
        # No softmax here - done by loss function
        return z2

# Create model
model = MultiClassNetwork(input_size=2, hidden_size=8, num_classes=3)

# Example forward pass
x = torch.tensor([[1.5, 2.0]])
logits = model(x)
print(f"Logits: {logits}")

# Get probabilities (for inference)
probabilities = F.softmax(logits, dim=1)
print(f"Probabilities: {probabilities}")

# Get prediction
predicted_class = torch.argmax(probabilities, dim=1)
print(f"Predicted class: {predicted_class.item()}")`;

  const trainingCode = `import torch
import torch.nn as nn
import torch.optim as optim

# Prepare data
# X: (N, 2) - N points with (x, y) coordinates
# y: (N,) - N target class labels
X_train = torch.tensor([
    [2.5, 3.0],
    [-1.0, 2.5],
    [1.5, -2.0],
    # ... more points
], dtype=torch.float32)

y_train = torch.tensor([0, 1, 2])  # Class labels

# Create model, loss, optimizer
model = MultiClassNetwork(input_size=2, hidden_size=8, num_classes=3)
criterion = nn.CrossEntropyLoss()  # Combines softmax + cross-entropy
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    logits = model(X_train)
    loss = criterion(logits, y_train)
    
    # Backward pass
    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
    
    # Compute accuracy
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == y_train).float().mean()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={accuracy:.2%}")`;

  const predictionCode = `import torch
import torch.nn.functional as F

# Set model to evaluation mode
model.eval()

# New data point
x_new = torch.tensor([[1.0, 1.5]])

# Make prediction
with torch.no_grad():  # Disable gradient computation
    logits = model(x_new)
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    confidence = probabilities[0, predicted_class]

print(f"Predicted class: {predicted_class.item()}")
print(f"Confidence: {confidence.item():.2%}")
print(f"All probabilities: {probabilities}")

# Visualize decision boundary
import numpy as np
import matplotlib.pyplot as plt

# Create grid
x_min, x_max = -5, 5
y_min, y_max = -5, 5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

# Predict for each point in grid
grid_points = torch.tensor(
    np.c_[xx.ravel(), yy.ravel()],
    dtype=torch.float32
)

with torch.no_grad():
    logits = model(grid_points)
    predictions = torch.argmax(logits, dim=1)
    predictions = predictions.numpy().reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, predictions, alpha=0.3, cmap='viridis')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Decision Boundary')
plt.show()`;

  const handleCopy = (code: string) => {
    navigator.clipboard.writeText(code);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>PyTorch Implementation</CardTitle>
        <CardDescription>
          Build a multi-class classifier with PyTorch
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="model">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="model">Model</TabsTrigger>
            <TabsTrigger value="training">Training</TabsTrigger>
            <TabsTrigger value="prediction">Prediction</TabsTrigger>
          </TabsList>

          <TabsContent value="model" className="space-y-3">
            <div className="flex justify-between items-start mb-2">
              <p className="text-sm text-muted-foreground">
                Multi-class network with PyTorch:
              </p>
              <button
                onClick={() => handleCopy(modelCode)}
                className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded shrink-0"
                type="button"
              >
                Copy Code
              </button>
            </div>
            <pre className="rounded-lg bg-slate-950 text-slate-50 p-4 overflow-x-auto text-xs font-mono whitespace-pre-wrap break-words">
              <code>{modelCode}</code>
            </pre>
            <p className="text-sm text-muted-foreground">
              The network outputs raw logits (unnormalized scores). Softmax is applied separately 
              for inference, but the loss function combines both for numerical stability during training.
            </p>
          </TabsContent>

          <TabsContent value="training" className="space-y-3">
            <div className="flex justify-between items-start mb-2">
              <p className="text-sm text-muted-foreground">
                Training loop with PyTorch:
              </p>
              <button
                onClick={() => handleCopy(trainingCode)}
                className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded shrink-0"
                type="button"
              >
                Copy Code
              </button>
            </div>
            <pre className="rounded-lg bg-slate-950 text-slate-50 p-4 overflow-x-auto text-xs font-mono whitespace-pre-wrap break-words">
              <code>{trainingCode}</code>
            </pre>
            <p className="text-sm text-muted-foreground">
              PyTorch&apos;s <code>CrossEntropyLoss</code> expects raw logits, not probabilities. 
              It internally applies log_softmax for numerical stability. The optimizer automatically 
              updates all network parameters based on computed gradients.
            </p>
          </TabsContent>

          <TabsContent value="prediction" className="space-y-3">
            <div className="flex justify-between items-start mb-2">
              <p className="text-sm text-muted-foreground">
                Making predictions with PyTorch:
              </p>
              <button
                onClick={() => handleCopy(predictionCode)}
                className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded shrink-0"
                type="button"
              >
                Copy Code
              </button>
            </div>
            <pre className="rounded-lg bg-slate-950 text-slate-50 p-4 overflow-x-auto text-xs font-mono whitespace-pre-wrap break-words">
              <code>{predictionCode}</code>
            </pre>
            <p className="text-sm text-muted-foreground">
              During inference, we use <code>model.eval()</code> and <code>torch.no_grad()</code> 
              to disable dropout/batch norm training behavior and gradient computation. Decision 
              boundaries visualize regions where the model predicts each class.
            </p>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
