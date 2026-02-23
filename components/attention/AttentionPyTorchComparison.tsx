"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export function AttentionPyTorchComparison() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>PyTorch Implementation</CardTitle>
        <CardDescription>
          Implement scaled dot-product attention with PyTorch
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="basic">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="basic">Basic Attention</TabsTrigger>
            <TabsTrigger value="multihead">Multi-Head</TabsTrigger>
            <TabsTrigger value="module">nn.Module</TabsTrigger>
          </TabsList>

          <TabsContent value="basic" className="space-y-3">
            <div className="bg-slate-950 text-slate-50 p-4 rounded-lg overflow-x-auto">
              <pre className="font-mono text-sm min-w-0 max-w-full whitespace-pre-wrap break-words">
{`import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention
    
    Args:
        Q: Query tensor (batch_size, seq_len, d_k)
        K: Key tensor (batch_size, seq_len, d_k)
        V: Value tensor (batch_size, seq_len, d_v)
        mask: Optional mask tensor
        
    Returns:
        output: Attention output (batch_size, seq_len, d_v)
        attention_weights: Attention weights (batch_size, seq_len, seq_len)
    """
    # Get dimension of keys
    d_k = K.shape[-1]
    
    # Compute attention scores: Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Scale by sqrt(d_k)
    scores = scores / math.sqrt(d_k)
    
    # Apply mask if provided (for padding or causality)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Compute weighted sum of values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# Example usage
batch_size = 1
seq_len = 5
d_model = 4

# Create input embeddings
x = torch.randn(batch_size, seq_len, d_model)

# For self-attention, Q=K=V=x (in practice, use projections)
Q = K = V = x

# Compute attention
output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"\\nAttention weights:\\n{weights[0]}")`}
              </pre>
            </div>
            <p className="text-sm text-muted-foreground">
              The core attention mechanism: compute similarity (Q·Kᵀ), scale, apply softmax, 
              and create weighted sum of values. This is the foundation of all Transformer models.
            </p>
          </TabsContent>

          <TabsContent value="multihead" className="space-y-3">
            <div className="bg-slate-950 text-slate-50 p-4 rounded-lg overflow-x-auto">
              <pre className="font-mono text-sm min-w-0 max-w-full whitespace-pre-wrap break-words">
{`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x):
        """Split the last dimension into (num_heads, d_k)"""
        batch_size, seq_len, d_model = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # Linear projections
        Q = self.W_q(Q)  # (batch, seq_len, d_model)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attention_weights

# Example usage
d_model = 512
num_heads = 8
seq_len = 10
batch_size = 2

mha = MultiHeadAttention(d_model, num_heads)

x = torch.randn(batch_size, seq_len, d_model)
output, weights = mha(x, x, x)

print(f"Output shape: {output.shape}")  # (2, 10, 512)
print(f"Weights shape: {weights.shape}")  # (2, 8, 10, 10)`}
              </pre>
            </div>
            <p className="text-sm text-muted-foreground">
              Multi-head attention allows the model to jointly attend to information from different 
              representation subspaces. Each head learns to focus on different aspects of the input.
            </p>
          </TabsContent>

          <TabsContent value="module" className="space-y-3">
            <div className="bg-slate-950 text-slate-50 p-4 rounded-lg overflow-x-auto">
              <pre className="font-mono text-sm min-w-0 max-w-full whitespace-pre-wrap break-words">
{`import torch
import torch.nn as nn

# PyTorch provides built-in multi-head attention!
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Use (batch, seq, feature) format
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(
            x, x, x,
            attn_mask=mask,
            need_weights=True
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

# Example: Create a simple Transformer
d_model = 512
num_heads = 8
d_ff = 2048
seq_len = 20
batch_size = 4

block = TransformerBlock(d_model, num_heads, d_ff)

x = torch.randn(batch_size, seq_len, d_model)
output, weights = block(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")  # Same as input
print(f"Attention weights: {weights.shape}")

# For a full Transformer, stack multiple blocks
class SimpleTransformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x

model = SimpleTransformer(d_model=256, num_heads=8, d_ff=1024, num_layers=6)
out = model(torch.randn(2, 10, 256))
print(f"\\nTransformer output: {out.shape}")`}
              </pre>
            </div>
            <p className="text-sm text-muted-foreground">
              PyTorch provides <code>nn.MultiheadAttention</code> and <code>nn.Transformer</code> 
              modules out of the box. These are production-ready implementations used in state-of-the-art 
              models like BERT, GPT, and T5.
            </p>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
