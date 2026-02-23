"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';

export function TransformerPyTorchComparison() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>PyTorch Implementation</CardTitle>
        <CardDescription>
          Build a complete Transformer block with PyTorch
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className=" text-black rounded-lg overflow-x-auto">
          <pre className="font-mono text-sm min-w-0 max-w-full whitespace-pre-wrap break-words">
{`import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        
        return x, attn_weights

# Stack multiple blocks for a full Transformer
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        super().__init__()
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Embeddings
        tok_emb = self.token_embedding(x)  # (B, T, d_model)
        pos = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(pos)  # (T, d_model)
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x)
        
        # Output projection
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        
        return logits

# Example usage
model = Transformer(
    vocab_size=50257,  # GPT-2 vocab size
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    max_seq_len=1024
)

# Input token IDs
x = torch.randint(0, 50257, (2, 10))  # (batch=2, seq_len=10)

# Forward pass
logits = model(x)
print(f"Input: {x.shape}")
print(f"Output logits: {logits.shape}")  # (2, 10, 50257)

# Get predictions
predictions = logits.argmax(dim=-1)
print(f"Predicted tokens: {predictions.shape}")  # (2, 10)`}
          </pre>
        </div>
        <p className="text-sm text-muted-foreground mt-3">
          This is the core architecture used in GPT, BERT, and other Transformer models. The key 
          innovation is the combination of self-attention (for global context) and feed-forward 
          networks (for local processing), connected via residual pathways and layer normalization.
        </p>
      </CardContent>
    </Card>
  );
}
