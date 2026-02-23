"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export function SoftmaxPyTorchComparison() {
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
            <div className="text-black rounded-lg overflow-x-auto">
              <pre className="font-mono text-sm min-w-0 max-w-full whitespace-pre-wrap break-words">
{`import torch
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
probabilities = exp_logits / exp_logits.sum()`}
              </pre>
            </div>
            <p className="text-sm text-muted-foreground">
              PyTorch&apos;s <code>F.softmax()</code> automatically handles numerical stability 
              by subtracting the maximum logit before computing exponentials.
            </p>
          </TabsContent>

          <TabsContent value="loss" className="space-y-3">
            <div className="text-black rounded-lg overflow-x-auto">
              <pre className="font-mono text-sm min-w-0 max-w-full whitespace-pre-wrap break-words">
{`import torch
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
# Loss: 0.4170`}
              </pre>
            </div>
            <p className="text-sm text-muted-foreground">
              PyTorch provides <code>F.cross_entropy()</code> which combines softmax and 
              cross-entropy in a numerically stable way. This is preferred in practice.
            </p>
          </TabsContent>

          <TabsContent value="gradient" className="space-y-3">
            <div className="text-black rounded-lg overflow-x-auto">
              <pre className="font-mono text-sm min-w-0 max-w-full whitespace-pre-wrap break-words">
{`import torch
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
# Manual gradients: tensor([-0.3410,  0.2424,  0.0986])`}
              </pre>
            </div>
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
