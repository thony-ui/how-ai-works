"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export function SoftmaxPyTorchComparison() {
  const softmaxCode = `import torch
import torch.nn.functional as F

# Raw scores (logits)
logits = torch.tensor([2.0, 1.0, 0.1])

# Apply softmax
probabilities = F.softmax(logits, dim=0)
print(probabilities)
# tensor([0.6590, 0.2424, 0.0986])

# Manual computation (for understanding)
exp_logits = torch.exp(logits)
probabilities = exp_logits / exp_logits.sum()

# Numerical stability version
max_logit = logits.max()
exp_logits = torch.exp(logits - max_logit)
probabilities = exp_logits / exp_logits.sum()`;

  const lossCode = `import torch
import torch.nn.functional as F

# Logits and target
logits = torch.tensor([2.0, 1.0, 0.1])
target = 0  # Target is class 0

# Method 1: Separate softmax + cross-entropy
probabilities = F.softmax(logits, dim=0)
loss = -torch.log(probabilities[target])
print(f"Loss: {loss.item():.4f}")
# Loss: 0.4170

# Method 2: Combined (more efficient & stable)
loss = F.cross_entropy(
    logits.unsqueeze(0),  # Add batch dimension
    torch.tensor([target])
)
print(f"Loss: {loss.item():.4f}")
# Loss: 0.4170

# Method 3: Using log_softmax (also stable)
log_probs = F.log_softmax(logits, dim=0)
loss = -log_probs[target]
print(f"Loss: {loss.item():.4f}")
# Loss: 0.4170`;

  const gradientCode = `import torch
import torch.nn.functional as F

# Logits and target (with gradient tracking)
logits = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)
target = torch.tensor([0])

# Compute loss
loss = F.cross_entropy(logits.unsqueeze(0), target)

# Compute gradients
loss.backward()
print(f"Gradients: {logits.grad}")
# Gradients: tensor([-0.3410,  0.2424,  0.0986])

# Manual gradient computation
# For softmax + cross-entropy: ∇L = p - y
probabilities = F.softmax(logits, dim=0)
target_onehot = torch.zeros_like(probabilities)
target_onehot[target] = 1.0
gradients = probabilities - target_onehot
print(f"Manual gradients: {gradients}")
# Manual gradients: tensor([-0.3410,  0.2424,  0.0986])`;

  const handleCopy = (code: string) => {
    navigator.clipboard.writeText(code);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>PyTorch Implementation</CardTitle>
        <CardDescription>
          Compare with industry-standard deep learning framework
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="softmax">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="softmax">Softmax</TabsTrigger>
            <TabsTrigger value="loss">Cross-Entropy</TabsTrigger>
            <TabsTrigger value="gradient">Gradient</TabsTrigger>
          </TabsList>

          <TabsContent value="softmax" className="space-y-3">
            <div className="flex justify-between items-start mb-2">
              <p className="text-sm text-muted-foreground">
                Softmax with PyTorch:
              </p>
              <button
                onClick={() => handleCopy(softmaxCode)}
                className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded shrink-0"
                type="button"
              >
                Copy Code
              </button>
            </div>
            <pre className="rounded-lg bg-slate-950 text-slate-50 p-4 overflow-x-auto text-xs font-mono whitespace-pre-wrap break-words">
              <code>{softmaxCode}</code>
            </pre>
            <p className="text-sm text-muted-foreground">
              PyTorch&apos;s <code>F.softmax()</code> automatically handles numerical stability 
              by subtracting the maximum logit before computing exponentials.
            </p>
          </TabsContent>

          <TabsContent value="loss" className="space-y-3">
            <div className="flex justify-between items-start mb-2">
              <p className="text-sm text-muted-foreground">
                Cross-entropy loss with PyTorch:
              </p>
              <button
                onClick={() => handleCopy(lossCode)}
                className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded shrink-0"
                type="button"
              >
                Copy Code
              </button>
            </div>
            <pre className="rounded-lg bg-slate-950 text-slate-50 p-4 overflow-x-auto text-xs font-mono whitespace-pre-wrap break-words">
              <code>{lossCode}</code>
            </pre>
            <p className="text-sm text-muted-foreground">
              PyTorch provides <code>F.cross_entropy()</code> which combines softmax and 
              cross-entropy in a numerically stable way. This is preferred in practice.
            </p>
          </TabsContent>

          <TabsContent value="gradient" className="space-y-3">
            <div className="flex justify-between items-start mb-2">
              <p className="text-sm text-muted-foreground">
                Gradient computation with PyTorch:
              </p>
              <button
                onClick={() => handleCopy(gradientCode)}
                className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded shrink-0"
                type="button"
              >
                Copy Code
              </button>
            </div>
            <pre className="rounded-lg bg-slate-950 text-slate-50 p-4 overflow-x-auto text-xs font-mono whitespace-pre-wrap break-words">
              <code>{gradientCode}</code>
            </pre>
            <p className="text-sm text-muted-foreground">
              The gradient of softmax + cross-entropy has an elegant form: <strong>∇L = p - y</strong>
              <br />
              Where <strong>p</strong> is the probability vector and <strong>y</strong> is the one-hot target vector.
              <br />
              This simplicity makes backpropagation through classification layers very efficient.
            </p>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
