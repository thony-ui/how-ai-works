"use client";

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  getTransformerBlockSteps,
  initializeTransformerWeights,
  transformerBlock,
  type Matrix,
} from '@/lib/math/transformer';
import { AttentionHeatmap } from '../attention/AttentionHeatmap';
import { TransformerPyTorchComparison } from './TransformerPyTorchComparison';
import { RotateCcwIcon, BrainIcon } from 'lucide-react';

export default function TransformerPlayground() {
  const [tokens, setTokens] = useState<string[]>(['The', 'cat', 'sat']);
  const [d_model, setD_model] = useState<number>(4);
  const [d_ff, setD_ff] = useState<number>(8);
  const [showSteps, setShowSteps] = useState<boolean>(false);

  // Generate simple embeddings
  const embeddings = useMemo<Matrix>(() => {
    return tokens.map((token, idx) => {
      const baseValue = token.length * 0.15;
      return Array(d_model)
        .fill(0)
        .map((_, dim) => baseValue + Math.sin(idx + dim) * 0.3);
    });
  }, [tokens, d_model]);

  // Initialize weights
  const weights = useMemo(
    () => initializeTransformerWeights(d_model, d_ff),
    [d_model, d_ff]
  );

  // Run Transformer block
  const result = useMemo(() => {
    return transformerBlock(
      embeddings,
      weights.W1_ff,
      weights.b1_ff,
      weights.W2_ff,
      weights.b2_ff
    );
  }, [embeddings, weights]);

  // Get step-by-step
  const steps = useMemo(() => {
    return getTransformerBlockSteps(
      embeddings,
      weights.W1_ff,
      weights.b1_ff,
      weights.W2_ff,
      weights.b2_ff
    );
  }, [embeddings, weights]);

  const reset = () => {
    setTokens(['The', 'cat', 'sat']);
    setD_model(4);
    setD_ff(8);
    setShowSteps(false);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Transformer Block</h1>
        <p className="text-lg text-muted-foreground">
          The building block of modern language models: self-attention + feed-forward with residual connections.
        </p>
      </div>

      <Card>
        <CardContent className="">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-6">
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Tokens</p>
                <p className="text-2xl font-bold">{tokens.length}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">d_model</p>
                <p className="text-2xl font-bold">{d_model}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">d_ff</p>
                <p className="text-2xl font-bold">{d_ff}</p>
              </div>
            </div>
            <Badge>Single Block</Badge>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Self-Attention Weights Heatmap</CardTitle>
              <CardDescription>
                Each row shows one word&apos;s attention distribution across all words (adds to 100%)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex justify-center overflow-x-auto">
                <AttentionHeatmap
                  weights={result.attentionWeights}
                  queryLabels={tokens}
                  keyLabels={tokens}
                  selectedQueryIdx={0}
                  onQuerySelect={() => {}}
                />
              </div>
              <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border-2 border-blue-300 space-y-3">
                <p className="font-bold text-blue-800 flex items-center gap-2">
                  <span className="text-xl">📖</span> Reading Guide:
                </p>
                <div className="text-sm space-y-2">
                  <div className="bg-white p-2 rounded">
                    <span className="font-semibold text-purple-600">Rows (vertical):</span> Each row is one word
                  </div>
                  <div className="bg-white p-2 rounded">
                    <span className="font-semibold text-blue-600">Columns (horizontal):</span> What that word attends to
                  </div>
                  <div className="bg-white p-2 rounded">
                    <span className="font-semibold text-green-600">Color intensity:</span> Darker = more attention
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {showSteps && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Transformer Block - Step by Step</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4 font-mono text-sm">
                  {steps.map((step, idx) => (
                    <div key={idx} className="space-y-2">
                      <p className="font-semibold text-blue-600">
                        Step {step.step}: {step.description}
                      </p>
                      <div className="bg-blue-50 p-3 rounded overflow-x-auto">
                        <p>{step.formula}</p>
                        {step.values && (
                          <div className="mt-2 space-y-0.5 text-xs">
                            {step.values.slice(0, 2).map((row, i) => (
                              <div key={i}>
                                [{row.map(v => v.toFixed(3)).join(', ')}]
                                {i === 1 && step.values!.length > 2 && (
                                  <span className="text-muted-foreground ml-2">
                                    (+ {step.values!.length - 2} more rows)
                                  </span>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                      {idx < steps.length - 1 && <Separator />}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Controls</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-xs">Model Dimension (d_model): {d_model}</Label>
                  <Slider
                    value={[d_model]}
                    onValueChange={([val]) => setD_model(val)}
                    min={2}
                    max={8}
                    step={2}
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">FF Dimension (d_ff): {d_ff}</Label>
                  <Slider
                    value={[d_ff]}
                    onValueChange={([val]) => setD_ff(val)}
                    min={4}
                    max={16}
                    step={4}
                  />
                  <p className="text-xs text-muted-foreground">
                    Typically 4× the model dimension
                  </p>
                </div>
              </div>

              <Separator />

              <div className="space-y-2">
                <Button
                  onClick={() => setShowSteps(!showSteps)}
                  variant="secondary"
                  className="w-full"
                >
                  <BrainIcon className="w-4 h-4 mr-2" />
                  {showSteps ? "Hide" : "Show"} Steps
                </Button>
                <Button onClick={reset} variant="outline" className="w-full">
                  <RotateCcwIcon className="w-4 h-4 mr-2" />
                  Reset
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Architecture</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-3 font-mono">
              <div className="bg-muted p-3 rounded space-y-1 text-xs">
                <p>Input: {tokens.length} × {d_model}</p>
                <p>↓</p>
                <p className="text-blue-600">Multi-Head Attention</p>
                <p>↓</p>
                <p>Add & LayerNorm</p>
                <p>↓</p>
                <p className="text-green-600">Feed-Forward ({d_model} → {d_ff} → {d_model})</p>
                <p>↓</p>
                <p>Add & LayerNorm</p>
                <p>↓</p>
                <p>Output: {tokens.length} × {d_model}</p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">What You&apos;re Learning</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-2 text-muted-foreground">
              <p>• Self-attention lets each token attend to all other tokens</p>
              <p>• Feed-forward networks transform token representations</p>
              <p>• Residual connections help information flow through layers</p>
              <p>• Layer normalization keeps values stable during training</p>
              <p>• Multiple transformer blocks can be stacked to build deep models</p>
            </CardContent>
          </Card>
        </div>
      </div>

      <TransformerPyTorchComparison />
    </div>
  );
}
