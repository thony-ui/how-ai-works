"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';

export function TokenPredictionPyTorchComparison() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>PyTorch Implementation</CardTitle>
        <CardDescription>
          Generate text with a language model in PyTorch
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-black rounded-lg overflow-x-auto">
          <pre className="font-mono text-sm min-w-0 max-w-full whitespace-pre-wrap break-words">
{`import torch
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
print(tokenizer.decode(output[0]))`}
          </pre>
        </div>
        <p className="text-sm text-muted-foreground mt-3">
          Hugging Face Transformers library provides easy access to pre-trained language models
          like GPT-2, GPT-3, and LLaMA. The <code>generate()</code> method handles all the
          complexity of autoregressive generation with various sampling strategies.
        </p>
      </CardContent>
    </Card>
  );
}
