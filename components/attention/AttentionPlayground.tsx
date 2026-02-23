"use client";

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  getAttentionSteps,
  computeQueryAttention,
  type Matrix,
} from '@/lib/math/attention';
import { AttentionHeatmap } from './AttentionHeatmap';
import { AttentionVisualization } from './AttentionVisualization';
import { AttentionPyTorchComparison } from './AttentionPyTorchComparison';
import { RotateCcwIcon, BrainIcon } from 'lucide-react';

export default function AttentionPlayground() {
  // Sequence of tokens
  const [tokens, setTokens] = useState<string[]>(['The', 'cat', 'sat', 'on', 'mat']);
  const [selectedQueryIdx, setSelectedQueryIdx] = useState<number>(2); // "sat"
  const [embeddingDim, setEmbeddingDim] = useState<number>(4);
  const [showSteps, setShowSteps] = useState<boolean>(true);

  // Generate simple embeddings (in practice, these would be learned)
  const embeddings = useMemo<Matrix>(() => {
    return tokens.map((token, idx) => {
      // Create a simple embedding based on token and position
      const baseValue = token.length * 0.1;
      return Array(embeddingDim).fill(0).map((_, dim) => {
        // Mix of token identity and position
        return baseValue + Math.sin(idx + dim) * 0.3 + Math.cos(dim) * 0.2;
      });
    });
  }, [tokens, embeddingDim]);

  // For simplicity, use embeddings as Q, K, V (identity projection)
  // In real transformers, these would be learned projections
  const Q = embeddings;
  const K = embeddings;
  const V = embeddings;

  // Compute attention
  const attentionResult = useMemo(() => {
    return getAttentionSteps(Q, K, V);
  }, [Q, K, V]);

  // Compute attention for selected query
  const queryAttention = useMemo(() => {
    return computeQueryAttention(selectedQueryIdx, Q, K, V);
  }, [selectedQueryIdx, Q, K, V]);

  const handleTokenChange = (index: number, value: string) => {
    const newTokens = [...tokens];
    newTokens[index] = value;
    setTokens(newTokens);
  };

  const addToken = () => {
    setTokens([...tokens, 'new']);
  };

  const removeToken = (index: number) => {
    if (tokens.length > 2) {
      const newTokens = tokens.filter((_, i) => i !== index);
      setTokens(newTokens);
      if (selectedQueryIdx >= newTokens.length) {
        setSelectedQueryIdx(newTokens.length - 1);
      }
    }
  };

  const reset = () => {
    setTokens(['The', 'cat', 'sat', 'on', 'mat']);
    setSelectedQueryIdx(2);
    setEmbeddingDim(4);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Attention Mechanism</h1>
        <p className="text-lg text-muted-foreground">
          Understand how attention allows models to focus on relevant parts of the input sequence.
        </p>
      </div>

      {/* Status Bar */}
      <Card>
        <CardContent className="">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-6">
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Sequence Length</p>
                <p className="text-2xl font-bold">{tokens.length}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Embedding Dim</p>
                <p className="text-2xl font-bold">{embeddingDim}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Selected Token</p>
                <p className="text-2xl font-bold text-blue-600">&quot;{tokens[selectedQueryIdx]}&quot;</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={showSteps ? "default" : "secondary"}>
                {showSteps ? "Showing Steps" : "Steps Hidden"}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Visualization */}
        <div className="lg:col-span-2 space-y-6">
          {/* Attention Heatmap */}
          <Card>
            <CardHeader>
              <CardTitle>Attention Weights Matrix</CardTitle>
              <CardDescription>
                Each row shows how much each query token attends to all key tokens
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex justify-center overflow-x-auto">
                <AttentionHeatmap
                  weights={attentionResult.attentionWeights}
                  queryLabels={tokens}
                  keyLabels={tokens}
                  selectedQueryIdx={selectedQueryIdx}
                  onQuerySelect={setSelectedQueryIdx}
                />
              </div>
            </CardContent>
          </Card>

          {/* Attention Flow Visualization */}
          <Card>
            <CardHeader>
              <CardTitle>Attention Flow for &quot;{tokens[selectedQueryIdx]}&quot;</CardTitle>
              <CardDescription>
                Visualize how the selected token attends to other tokens
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex justify-center overflow-x-auto">
                <AttentionVisualization
                  tokens={tokens}
                  weights={queryAttention.weights}
                  queryIdx={selectedQueryIdx}
                />
              </div>
            </CardContent>
          </Card>

          {/* Step-by-Step Calculations */}
          {showSteps && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Attention Computation - Step by Step</CardTitle>
                <CardDescription>
                  Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4 font-mono text-sm">
                  {attentionResult.steps.map((step, idx) => (
                    <div key={idx} className="space-y-2">
                      <p className="font-semibold text-blue-600">
                        Step {step.step}: {step.description}
                      </p>
                      <div className="bg-blue-50 p-3 rounded overflow-x-auto">
                        <p className="mb-2">{step.formula}</p>
                        {step.values && Array.isArray(step.values) && step.values.length > 0 && (
                          <div className="mt-2 space-y-1">
                            {Array.isArray(step.values[0]) ? (
                              // Matrix display
                              <div className="space-y-0.5">
                                {(step.values as number[][]).map((row, i) => (
                                  <div key={i} className="flex gap-2">
                                    <span className="text-muted-foreground w-16">
                                      {i === 0 ? 'Matrix:' : ''}
                                    </span>
                                    <span>
                                      [{row.map(v => v.toFixed(3)).join(', ')}]
                                    </span>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              // Vector display
                              <p>[{(step.values as number[]).map(v => v.toFixed(3)).join(', ')}]</p>
                            )}
                          </div>
                        )}
                      </div>
                      {idx < attentionResult.steps.length - 1 && <Separator />}
                    </div>
                  ))}

                  {/* Detailed Math Breakdown */}
                  <div className="border-t-2 border-purple-300 pt-4 mt-4">
                    <p className="font-bold text-lg text-purple-700 mb-3">
                      🔍 Detailed Math Breakdown for Query: &quot;{tokens[selectedQueryIdx]}&quot;
                    </p>

                    {/* Show Q, K, V Matrices */}
                    <div className="space-y-3 mb-4">
                      <div className="bg-indigo-50 p-3 rounded">
                        <p className="font-semibold text-indigo-700 mb-2">📊 Input Matrices (Q, K, V)</p>
                        <p className="text-xs text-muted-foreground mb-2">
                          In this demo, Q = K = V = embeddings (identity projection). In real transformers, these are learned linear projections.
                        </p>
                        
                        <div className="space-y-2 text-xs">
                          <p className="font-semibold">Query Matrix Q ({tokens.length} × {embeddingDim}):</p>
                          {Q.map((row, i) => (
                            <div key={i} className={`${i === selectedQueryIdx ? 'bg-yellow-100 border border-yellow-400' : ''} p-1 rounded`}>
                              <span className="text-muted-foreground">{tokens[i]}:</span> [{row.map(v => v.toFixed(3)).join(', ')}]
                            </div>
                          ))}
                          
                          <p className="font-semibold mt-3">Key Matrix K ({tokens.length} × {embeddingDim}):</p>
                          {K.map((row, i) => (
                            <div key={i} className="p-1">
                              <span className="text-muted-foreground">{tokens[i]}:</span> [{row.map(v => v.toFixed(3)).join(', ')}]
                            </div>
                          ))}
                          
                          <p className="font-semibold mt-3">Value Matrix V ({tokens.length} × {embeddingDim}):</p>
                          {V.map((row, i) => (
                            <div key={i} className="p-1">
                              <span className="text-muted-foreground">{tokens[i]}:</span> [{row.map(v => v.toFixed(3)).join(', ')}]
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Selected Query Vector */}
                      <div className="bg-yellow-50 p-3 rounded border-2 border-yellow-400">
                        <p className="font-semibold text-yellow-800 mb-2">
                          🎯 Selected Query Vector (from token: &quot;{tokens[selectedQueryIdx]}&quot;)
                        </p>
                        <p className="text-xs font-mono">
                          q = [{Q[selectedQueryIdx].map(v => v.toFixed(3)).join(', ')}]
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          This is row {selectedQueryIdx} from the Query matrix Q
                        </p>
                      </div>

                      {/* Dot Product Computation */}
                      <div className="bg-blue-50 p-3 rounded">
                        <p className="font-semibold text-blue-700 mb-2">
                          Step 1: Compute Dot Products (q · k_i)
                        </p>
                        <p className="text-xs text-muted-foreground mb-2">
                          Calculate similarity between query and each key using dot product
                        </p>
                        {K.map((key, i) => {
                          const dotProduct = Q[selectedQueryIdx].reduce((sum, val, idx) => sum + val * key[idx], 0);
                          return (
                            <div key={i} className="mb-2 text-xs bg-white p-2 rounded">
                              <p className="font-semibold">q · k_{i} (key for &quot;{tokens[i]}&quot;):</p>
                              <p className="font-mono text-xs">
                                = {Q[selectedQueryIdx].map((v, idx) => `(${v.toFixed(2)} × ${key[idx].toFixed(2)})`).join(' + ')}
                              </p>
                              <p className="font-mono text-xs">
                                = {Q[selectedQueryIdx].map((v, idx) => (v * key[idx]).toFixed(3)).join(' + ')}
                              </p>
                              <p className="font-bold text-blue-600">= {dotProduct.toFixed(4)}</p>
                            </div>
                          );
                        })}
                      </div>

                      {/* Scaling */}
                      <div className="bg-green-50 p-3 rounded">
                        <p className="font-semibold text-green-700 mb-2">
                          Step 2: Scale by √d_k
                        </p>
                        <p className="text-xs text-muted-foreground mb-2">
                          Scale factor = √{embeddingDim} = {Math.sqrt(embeddingDim).toFixed(4)}
                        </p>
                        <p className="text-xs mb-2">
                          This prevents dot products from becoming too large, which would cause softmax to saturate.
                        </p>
                        {queryAttention.scores.map((score, i) => {
                          const rawDot = Q[selectedQueryIdx].reduce((sum, val, idx) => sum + val * K[i][idx], 0);
                          return (
                            <div key={i} className="text-xs mb-1">
                              <span className="text-muted-foreground">{tokens[i]}:</span>{' '}
                              {rawDot.toFixed(4)} / {Math.sqrt(embeddingDim).toFixed(4)} = <span className="font-bold text-green-600">{score.toFixed(4)}</span>
                            </div>
                          );
                        })}
                      </div>

                      {/* Softmax Computation */}
                      <div className="bg-orange-50 p-3 rounded">
                        <p className="font-semibold text-orange-700 mb-2">
                          Step 3: Apply Softmax to Get Attention Weights
                        </p>
                        <p className="text-xs text-muted-foreground mb-2">
                          Softmax converts scores to probabilities that sum to 1
                        </p>
                        
                        {/* Show exponentials */}
                        <div className="text-xs mb-3 bg-white p-2 rounded">
                          <p className="font-semibold mb-1">3a. Compute exponentials:</p>
                          {queryAttention.scores.map((score, i) => (
                            <div key={i}>
                              <span className="text-muted-foreground">{tokens[i]}:</span>{' '}
                              e^{score.toFixed(4)} = <span className="font-bold">{Math.exp(score).toFixed(6)}</span>
                            </div>
                          ))}
                        </div>

                        {/* Show sum */}
                        <div className="text-xs mb-3 bg-white p-2 rounded">
                          <p className="font-semibold mb-1">3b. Sum of all exponentials:</p>
                          <p className="font-mono">
                            sum = {queryAttention.scores.map((score, i) => Math.exp(score).toFixed(4)).join(' + ')}
                          </p>
                          <p className="font-bold text-orange-600">
                            = {queryAttention.scores.reduce((sum, score) => sum + Math.exp(score), 0).toFixed(6)}
                          </p>
                        </div>

                        {/* Show division */}
                        <div className="text-xs bg-white p-2 rounded">
                          <p className="font-semibold mb-1">3c. Divide each exp by sum to get probabilities:</p>
                          {queryAttention.scores.map((score, i) => {
                            const expValue = Math.exp(score);
                            const sumExp = queryAttention.scores.reduce((sum, s) => sum + Math.exp(s), 0);
                            return (
                              <div key={i}>
                                <span className="text-muted-foreground">{tokens[i]}:</span>{' '}
                                {expValue.toFixed(6)} / {sumExp.toFixed(6)} = <span className="font-bold text-orange-600">{(expValue / sumExp).toFixed(6)}</span>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      {/* Final Result */}
                      <div className="bg-purple-50 p-3 rounded border-2 border-purple-400">
                        <p className="font-semibold text-purple-700 mb-2">
                          ✅ Final Attention Weights (Percentages)
                        </p>
                        <p className="text-xs text-muted-foreground mb-2">
                          These weights determine how much the query token &quot;{tokens[selectedQueryIdx]}&quot; should attend to each token
                        </p>
                        {tokens.map((token, i) => (
                          <div key={i} className="flex justify-between items-center mb-1">
                            <span className="text-xs font-semibold">{token}:</span>
                            <div className="flex items-center gap-2">
                              <div className="w-32 h-4 bg-gray-200 rounded overflow-hidden">
                                <div
                                  className="h-full bg-purple-500"
                                  style={{ width: `${queryAttention.weights[i] * 100}%` }}
                                />
                              </div>
                              <span className="w-20 text-right font-bold text-purple-600">
                                {(queryAttention.weights[i] * 100).toFixed(3)}%
                              </span>
                            </div>
                          </div>
                        ))}
                        <div className="mt-2 pt-2 border-t border-purple-200">
                          <p className="text-xs">
                            <span className="font-semibold">Verification:</span> Sum = {(queryAttention.weights.reduce((sum, w) => sum + w, 0) * 100).toFixed(3)}% 
                            {Math.abs(queryAttention.weights.reduce((sum, w) => sum + w, 0) - 1) < 0.0001 ? ' ✓' : ' ⚠️'}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Column - Controls */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Controls</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Token Sequence */}
              <div className="space-y-3">
                <h3 className="font-semibold text-sm">Token Sequence</h3>
                <div className="space-y-2">
                  {tokens.map((token, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <Badge variant={i === selectedQueryIdx ? "default" : "outline"} className="w-8">
                        {i}
                      </Badge>
                      <Input
                        type="text"
                        value={token}
                        onChange={(e) => handleTokenChange(i, e.target.value)}
                        className="font-mono"
                      />
                      {tokens.length > 2 && (
                        <Button
                          onClick={() => removeToken(i)}
                          variant="ghost"
                          size="sm"
                        >
                          ×
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
                <Button
                  onClick={addToken}
                  variant="outline"
                  size="sm"
                  className="w-full"
                  disabled={tokens.length >= 8}
                >
                  + Add Token
                </Button>
              </div>

              <Separator />

              {/* Select Query Token */}
              <div className="space-y-3">
                <h3 className="font-semibold text-sm">Select Query Token</h3>
                <div className="grid grid-cols-2 gap-2">
                  {tokens.map((token, i) => (
                    <Button
                      key={i}
                      onClick={() => setSelectedQueryIdx(i)}
                      variant={selectedQueryIdx === i ? "default" : "outline"}
                      size="sm"
                    >
                      {i}: {token}
                    </Button>
                  ))}
                </div>
              </div>

              <Separator />

              {/* Embedding Dimension */}
              <div className="space-y-3">
                <h3 className="font-semibold text-sm">Configuration</h3>
                
                <div className="space-y-2">
                  <Label className="text-xs">
                    Embedding Dimension: {embeddingDim}
                  </Label>
                  <Slider
                    value={[embeddingDim]}
                    onValueChange={([val]) => setEmbeddingDim(val)}
                    min={2}
                    max={8}
                    step={2}
                  />
                  <p className="text-xs text-muted-foreground">
                    Higher dimensions capture more information
                  </p>
                </div>
              </div>

              <Separator />

              {/* Action Buttons */}
              <div className="space-y-2">
                <Button
                  onClick={() => setShowSteps(!showSteps)}
                  variant="secondary"
                  className="w-full"
                >
                  <BrainIcon className="w-4 h-4 mr-2" />
                  {showSteps ? "Hide" : "Show"} Steps
                </Button>
                
                <Button
                  onClick={reset}
                  variant="outline"
                  className="w-full"
                >
                  <RotateCcwIcon className="w-4 h-4 mr-2" />
                  Reset to Default
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* What You're Learning */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">What You&apos;re Learning</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-2 text-muted-foreground">
              <p>• How attention computes relevance between tokens</p>
              <p>• Query, Key, Value concept in attention</p>
              <p>• Why we scale by √d_k (prevents saturation)</p>
              <p>• Softmax creates a probability distribution</p>
              <p>• Output is a weighted sum of values</p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* PyTorch Code Comparison */}
      <AttentionPyTorchComparison />
    </div>
  );
}
