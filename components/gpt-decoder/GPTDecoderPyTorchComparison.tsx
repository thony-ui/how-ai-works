"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';

export function GPTDecoderPyTorchComparison() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>PyTorch Implementation</CardTitle>
        <CardDescription>
          Build a GPT-style decoder from scratch with PyTorch
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="bg-slate-950 text-slate-50 p-4 rounded-lg overflow-x-auto">
          <pre className="font-mono text-sm">
{`import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer decoder blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',  # GPT uses GELU
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Language model head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (common in GPT)
        self.lm_head.weight = self.token_embedding.weight
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Generate causal mask (prevents attending to future tokens)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len), diagonal=1
        ).bool().to(input_ids.device)
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        pos = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.position_embedding(pos)
        x = token_emb + pos_emb
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, src_mask=causal_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=50):
        """Autoregressive text generation"""
        for _ in range(max_new_tokens):
            # Get predictions for current sequence
            logits = self(input_ids)
            
            # Focus on last token
            logits = logits[:, -1, :] / temperature
            
            # Top-K filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

# Create a small GPT model
model = GPTDecoder(
    vocab_size=50_257,  # GPT-2 vocab size
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    max_seq_len=1024,
    dropout=0.1
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # ~43M parameters

# Example: Generate text
input_text = "The cat sat on"
# (Assume we have a tokenizer)
# input_ids = tokenizer.encode(input_text, return_tensors='pt')
input_ids = torch.randint(0, 50257, (1, 5))  # Mock input

# Generate
generated_ids = model.generate(
    input_ids,
    max_new_tokens=20,
    temperature=0.8,
    top_k=50
)

# Decode
# generated_text = tokenizer.decode(generated_ids[0])
# print(generated_text)

# Using Hugging Face's GPT-2 (fully trained)
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

input_text = "The cat sat on"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate
output = model.generate(
    input_ids,
    max_new_tokens=20,
    temperature=0.8,
    top_k=50,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output[0])
print(generated_text)`}
          </pre>
        </div>
        <p className="text-sm text-muted-foreground mt-3">
          GPT models are decoder-only Transformers trained on massive text corpora to predict the 
          next token. The architecture shown here is the foundation of models like GPT-2, GPT-3, 
          GPT-4, and many other large language models.
        </p>
      </CardContent>
    </Card>
  );
}
