"use client";

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  createVocabulary,
  tokenize,
  predictNextToken,
  mockLanguageModel,
  applyTemperature,
  generateText,
} from '@/lib/math/token-prediction';
import { TokenPredictionViz } from './TokenPredictionViz';
import { TokenPredictionPyTorchComparison } from './TokenPredictionPyTorchComparison';
import { PlayIcon, RotateCcwIcon } from 'lucide-react';

const DEFAULT_VOCAB = [
  'The', 'cat', 'dog', 'sat', 'on', 'the', 'mat', 'quick',
  'brown', 'fox', 'jumped', 'over', 'lazy', 'ran', 'slept',
  'barked', 'a', 'top', 'down', 'still', 'red'
];

export default function TokenPredictionPlayground() {
  const vocab = useMemo(() => createVocabulary(DEFAULT_VOCAB), []);
  
  const [inputTokens, setInputTokens] = useState<string[]>(['The', 'cat']);
  const [temperature, setTemperature] = useState<number>(1.0);
  const [topK, setTopK] = useState<number>(5);
  const [generatedText, setGeneratedText] = useState<string[]>([]);

  const tokenIds = useMemo(() => tokenize(inputTokens, vocab), [inputTokens, vocab]);

  // Get model predictions
  const logits = useMemo(() => mockLanguageModel(tokenIds, vocab), [tokenIds, vocab]);
  
  const adjustedLogits = useMemo(
    () => applyTemperature(logits, temperature),
    [logits, temperature]
  );

  const prediction = useMemo(
    () => predictNextToken(adjustedLogits, vocab, topK),
    [adjustedLogits, vocab, topK]
  );

  const handleGenerate = () => {
    const generated = generateText(
      tokenIds,
      (ids) => mockLanguageModel(ids, vocab),
      vocab,
      5,
      temperature,
      topK
    );
    setGeneratedText(generated);
  };

  const handleAddToken = (token: string) => {
    setInputTokens([...inputTokens, token]);
    setGeneratedText([]);
  };

  const handleRemoveLastToken = () => {
    if (inputTokens.length > 1) {
      setInputTokens(inputTokens.slice(0, -1));
      setGeneratedText([]);
    }
  };

  const reset = () => {
    setInputTokens(['The', 'cat']);
    setTemperature(1.0);
    setTopK(5);
    setGeneratedText([]);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Next-Token Prediction</h1>
        <p className="text-lg text-muted-foreground">
          See how language models predict the next word by computing probability distributions over vocabulary.
        </p>
      </div>

      <Card>
        <CardContent className="">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-6">
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Input Length</p>
                <p className="text-2xl font-bold">{inputTokens.length}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Vocabulary Size</p>
                <p className="text-2xl font-bold">{vocab.tokens.length}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Top Prediction</p>
                <p className="text-2xl font-bold text-blue-600">&quot;{prediction.topK[0]?.token}&quot;</p>
              </div>
            </div>
            <Badge>Autoregressive</Badge>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Current Text */}
          <Card>
            <CardHeader>
              <CardTitle>Current Sequence</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap items-center gap-2">
                {inputTokens.map((token, i) => (
                  <Badge key={i} variant="outline" className="text-lg px-3 py-1">
                    {token}
                  </Badge>
                ))}
                <Badge variant="secondary" className="text-lg px-3 py-1">
                  ?
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* Prediction Visualization */}
          <Card>
            <CardHeader>
              <CardTitle>Next Token Probability Distribution</CardTitle>
              <CardDescription>
                Top {topK} most likely next tokens
              </CardDescription>
            </CardHeader>
            <CardContent>
              <TokenPredictionViz predictions={prediction.topK} />
              
              {/* Add predicted token */}
              <div className="mt-4 flex flex-wrap gap-2">
                {prediction.topK.slice(0, 5).map((pred, i) => (
                  <Button
                    key={i}
                    onClick={() => handleAddToken(pred.token)}
                    variant="outline"
                    size="sm"
                  >
                    Add &quot;{pred.token}&quot; ({(pred.probability * 100).toFixed(1)}%)
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Generated Text */}
          {generatedText.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Auto-Generated Sequence</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap items-center gap-2">
                  {generatedText.map((token, i) => (
                    <Badge
                      key={i}
                      variant={i < inputTokens.length ? "outline" : "default"}
                      className="text-base px-3 py-1"
                    >
                      {token}
                    </Badge>
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
                  <Label className="text-xs">Temperature: {temperature.toFixed(2)}</Label>
                  <Slider
                    value={[temperature * 100]}
                    onValueChange={([val]) => setTemperature(val / 100)}
                    min={1}
                    max={200}
                    step={10}
                  />
                  <p className="text-xs text-muted-foreground">
                    Lower = more confident, Higher = more random
                  </p>
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">Top-K: {topK}</Label>
                  <Slider
                    value={[topK]}
                    onValueChange={([val]) => setTopK(val)}
                    min={3}
                    max={10}
                    step={1}
                  />
                  <p className="text-xs text-muted-foreground">
                    Number of predictions to show
                  </p>
                </div>
              </div>

              <Separator />

              <div className="space-y-2">
                <Button onClick={handleGenerate} className="w-full">
                  <PlayIcon className="w-4 h-4 mr-2" />
                  Generate 5 Tokens
                </Button>
                <Button
                  onClick={handleRemoveLastToken}
                  variant="secondary"
                  className="w-full"
                  disabled={inputTokens.length <= 1}
                >
                  Remove Last Token
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
              <CardTitle className="text-base">What You&apos;re Learning</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-2 text-muted-foreground">
              <p>• Language models predict probability distributions</p>
              <p>• Temperature controls randomness vs. confidence</p>
              <p>• Autoregressive generation: one token at a time</p>
              <p>• Top-K sampling balances quality and diversity</p>
              <p>• Context (previous tokens) influences predictions</p>
            </CardContent>
          </Card>
        </div>
      </div>

      <TokenPredictionPyTorchComparison />
    </div>
  );
}
