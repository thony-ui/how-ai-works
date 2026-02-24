"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';

export function TokenPredictionPyTorchComparison() {
  const tokenPredictionCode = `import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()

# Input text
text = "The cat"
input_ids = tokenizer.encode(text, return_tensors='pt')

# Generate next token predictions
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]  # Last token logits

# Apply temperature
temperature = 1.0
logits = logits / temperature

# Get probabilities
probs = F.softmax(logits, dim=-1)

# Get top-K predictions
top_k = 5
top_probs, top_indices = torch.topk(probs, top_k)

for prob, idx in zip(top_probs, top_indices):
    token = tokenizer.decode([idx])
    print(f"{token}: {prob.item():.2%}")

# Generate multiple tokens (autoregressive)
generated_ids = input_ids.clone()
max_new_tokens = 10

for _ in range(max_new_tokens):
    with torch.no_grad():
        outputs = model(generated_ids)
        logits = outputs.logits[0, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        
        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

generated_text = tokenizer.decode(generated_ids[0])
print(f"Generated: {generated_text}")

# Using the built-in generate method
output = model.generate(
    input_ids,
    max_new_tokens=10,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    do_sample=True
)
print(tokenizer.decode(output[0]))`;

  const handleCopy = (code: string) => {
    navigator.clipboard.writeText(code);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>PyTorch Implementation</CardTitle>
        <CardDescription>
          Generate text with a language model in PyTorch
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="flex justify-between items-start mb-2">
            <p className="text-sm text-muted-foreground">
              Text generation with PyTorch and Hugging Face:
            </p>
            <button
              onClick={() => handleCopy(tokenPredictionCode)}
              className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded shrink-0"
              type="button"
            >
              Copy Code
            </button>
          </div>
          <pre className="rounded-lg bg-slate-950 text-slate-50 p-4 overflow-x-auto text-xs font-mono whitespace-pre-wrap break-words">
            <code>{tokenPredictionCode}</code>
          </pre>
          <p className="text-sm text-muted-foreground">
            Hugging Face Transformers library provides easy access to pre-trained language models
            like GPT-2, GPT-3, and LLaMA. The <code>generate()</code> method handles all the
            complexity of autoregressive generation with various sampling strategies.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
