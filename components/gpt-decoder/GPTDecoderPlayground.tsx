"use client";

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { GPTDecoderPyTorchComparison } from './GPTDecoderPyTorchComparison';
import { PlayIcon, RotateCcwIcon, Layers3Icon } from 'lucide-react';

export default function GPTDecoderPlayground() {
  const [inputText, setInputText] = useState<string>('The cat sat on');
  const [numLayers, setNumLayers] = useState<number>(3);
  const [numHeads, setNumHeads] = useState<number>(4);
  const [d_model, setD_model] = useState<number>(128);
  const [generatedText, setGeneratedText] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  const tokens = inputText.split(' ').filter(Boolean);

  const handleGenerate = () => {
    setIsGenerating(true);
    // Simulate generation
    const mockNext = [' the', ' mat', '.', ' It', ' was'];
    let result = inputText;
    
    // Simulate streaming generation
    let i = 0;
    const interval = setInterval(() => {
      if (i < mockNext.length) {
        result += mockNext[i];
        setGeneratedText(result);
        i++;
      } else {
        clearInterval(interval);
        setIsGenerating(false);
      }
    }, 300);
  };

  const reset = () => {
    setInputText('The cat sat on');
    setGeneratedText('');
    setIsGenerating(false);
    setNumLayers(3);
    setNumHeads(4);
    setD_model(128);
  };

  const parameters = useMemo(() => {
    // Simplified parameter counting
    const vocab_size = 50000;
    const embedding_params = vocab_size * d_model * 2; // Token + position embeddings
    const attention_params_per_layer = 4 * d_model * d_model; // QKV + output projection
    const ff_params_per_layer = 2 * d_model * (4 * d_model); // Two linear layers
    const layer_params = attention_params_per_layer + ff_params_per_layer;
    const total = embedding_params + (layer_params * numLayers) + (vocab_size * d_model); // + output head
    
    return {
      total,
      embedding: embedding_params,
      perLayer: layer_params,
      allLayers: layer_params * numLayers,
    };
  }, [numLayers, d_model]);

  const formatNumber = (num: number) => {
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num.toString();
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">GPT Decoder</h1>
        <p className="text-lg text-muted-foreground">
          Complete GPT-style autoregressive language model with stacked Transformer blocks.
        </p>
      </div>

      <Card>
        <CardContent className="">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-6">
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Layers</p>
                <p className="text-2xl font-bold">{numLayers}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Heads</p>
                <p className="text-2xl font-bold">{numHeads}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">d_model</p>
                <p className="text-2xl font-bold">{d_model}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Parameters</p>
                <p className="text-2xl font-bold text-purple-600">{formatNumber(parameters.total)}</p>
              </div>
            </div>
            <Badge variant={isGenerating ? "default" : "secondary"}>
              {isGenerating ? "Generating..." : "Ready"}
            </Badge>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Input/Output */}
          <Card>
            <CardHeader>
              <CardTitle>Text Generation</CardTitle>
              <CardDescription>
                Enter a prompt and let the GPT decoder generate completions
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Input Prompt</Label>
                <Input
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Enter text..."
                  disabled={isGenerating}
                  className="font-mono"
                />
              </div>

              {generatedText && (
                <div className="space-y-2">
                  <Label>Generated Text</Label>
                  <div className="p-4 bg-muted rounded font-mono text-sm">
                    {generatedText}
                    {isGenerating && <span className="animate-pulse">|</span>}
                  </div>
                </div>
              )}

              <Button
                onClick={handleGenerate}
                disabled={isGenerating || !inputText.trim()}
                className="w-full"
              >
                <PlayIcon className="w-4 h-4 mr-2" />
                {isGenerating ? 'Generating...' : 'Generate Text'}
              </Button>
            </CardContent>
          </Card>

          {/* Architecture Visualization */}
          <Card>
            <CardHeader>
              <CardTitle>Model Architecture</CardTitle>
              <CardDescription>
                GPT Decoder = Embeddings + {numLayers}× Transformer Blocks + Output Head
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 font-mono text-sm">
                {/* Input */}
                <div className="p-3 bg-blue-50 rounded">
                  <p className="font-semibold text-blue-800">📥 Input</p>
                  <p className="text-xs text-blue-600">Token IDs: {tokens.length} tokens</p>
                </div>

                {/* Embeddings */}
                <div className="flex items-center justify-center">
                  <div className="w-0.5 h-4 bg-gray-300" />
                </div>
                <div className="p-3 bg-purple-50 rounded">
                  <p className="font-semibold text-purple-800">🔤 Token + Positional Embeddings</p>
                  <p className="text-xs text-purple-600">Shape: ({tokens.length}, {d_model})</p>
                  <p className="text-xs text-purple-600">Params: {formatNumber(parameters.embedding)}</p>
                </div>

                {/* Transformer Blocks */}
                {Array.from({ length: numLayers }).map((_, i) => (
                  <React.Fragment key={i}>
                    <div className="flex items-center justify-center">
                      <div className="w-0.5 h-4 bg-gray-300" />
                    </div>
                    <div className="p-3 bg-green-50 rounded border-2 border-green-200">
                      <p className="font-semibold text-green-800">
                        <Layers3Icon className="w-4 h-4 inline mr-1" />
                        Transformer Block {i + 1}
                      </p>
                      <div className="ml-4 mt-2 space-y-1 text-xs text-green-600">
                        <p>• Multi-Head Attention ({numHeads} heads)</p>
                        <p>• Add & LayerNorm</p>
                        <p>• Feed-Forward ({d_model} → {d_model * 4} → {d_model})</p>
                        <p>• Add & LayerNorm</p>
                        <p className="font-semibold mt-1">Params: {formatNumber(parameters.perLayer)}</p>
                      </div>
                    </div>
                  </React.Fragment>
                ))}

                {/* Output Head */}
                <div className="flex items-center justify-center">
                  <div className="w-0.5 h-4 bg-gray-300" />
                </div>
                <div className="p-3 bg-orange-50 rounded">
                  <p className="font-semibold text-orange-800">🎯 Output Head (LM Head)</p>
                  <p className="text-xs text-orange-600">Linear: {d_model} → 50000 (vocab size)</p>
                  <p className="text-xs text-orange-600">Softmax → Probabilities</p>
                </div>

                {/* Output */}
                <div className="flex items-center justify-center">
                  <div className="w-0.5 h-4 bg-gray-300" />
                </div>
                <div className="p-3 bg-red-50 rounded">
                  <p className="font-semibold text-red-800">📤 Output</p>
                  <p className="text-xs text-red-600">Next Token Probabilities: 50000 classes</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Model Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs">Number of Layers: {numLayers}</Label>
                  <Slider
                    value={[numLayers]}
                    onValueChange={([val]) => setNumLayers(val)}
                    min={1}
                    max={6}
                    step={1}
                    disabled={isGenerating}
                  />
                  <p className="text-xs text-muted-foreground">
                    GPT-2 Small: 12, GPT-3: 96
                  </p>
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Attention Heads: {numHeads}</Label>
                  <Slider
                    value={[numHeads]}
                    onValueChange={([val]) => setNumHeads(val)}
                    min={2}
                    max={8}
                    step={2}
                    disabled={isGenerating}
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Model Dimension: {d_model}</Label>
                  <Slider
                    value={[d_model]}
                    onValueChange={([val]) => setD_model(val)}
                    min={64}
                    max={512}
                    step={64}
                    disabled={isGenerating}
                  />
                  <p className="text-xs text-muted-foreground">
                    GPT-2 Small: 768, GPT-3: 12288
                  </p>
                </div>
              </div>

              <Separator />

              <div className="space-y-2">
                <Button onClick={reset} variant="outline" className="w-full">
                  <RotateCcwIcon className="w-4 h-4 mr-2" />
                  Reset
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Parameter Breakdown</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 font-mono text-xs">
              <div className="flex justify-between">
                <span>Embeddings:</span>
                <span>{formatNumber(parameters.embedding)}</span>
              </div>
              <div className="flex justify-between">
                <span>Per Layer:</span>
                <span>{formatNumber(parameters.perLayer)}</span>
              </div>
              <div className="flex justify-between">
                <span>All Layers:</span>
                <span>{formatNumber(parameters.allLayers)}</span>
              </div>
              <Separator />
              <div className="flex justify-between font-bold text-sm">
                <span>Total:</span>
                <span className="text-purple-600">{formatNumber(parameters.total)}</span>
              </div>
              
              <div className="pt-3 border-t text-muted-foreground">
                <p className="text-xs mb-2">For Reference:</p>
                <p>GPT-2 Small: 124M</p>
                <p>GPT-3: 175B</p>
                <p>GPT-4: ~1.8T (estimated)</p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">What You&apos;re Learning</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-2 text-muted-foreground">
              <p>• GPT is a stack of Transformer decoder blocks</p>
              <p>• Autoregressive generation: one token at a time</p>
              <p>• Causal attention: can only see previous tokens</p>
              <p>• Model size scales with layers and dimensions</p>
              <p>• Real GPT models have billions of parameters</p>
            </CardContent>
          </Card>
        </div>
      </div>

      <GPTDecoderPyTorchComparison />
    </div>
  );
}
