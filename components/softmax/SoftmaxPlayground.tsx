"use client";

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  softmax,
  crossEntropyLoss,
  getSoftmaxSteps,
  getCrossEntropySteps,
  softmaxCrossEntropyGradient
} from '@/lib/math/softmax';
import { ProbabilityChart } from './ProbabilityChart';
import { SoftmaxPyTorchComparison } from './SoftmaxPyTorchComparison';
import { RotateCcwIcon, BrainIcon } from 'lucide-react';

export default function SoftmaxPlayground() {
  // Example: 3-class classification
  const [logits, setLogits] = useState<number[]>([2.0, 1.0, 0.1]);
  const [targetClass, setTargetClass] = useState<number>(0);
  const [showSteps, setShowSteps] = useState<boolean>(true);

  // Compute softmax probabilities
  const probabilities = useMemo(() => softmax(logits), [logits]);
  
  // Compute loss
  const loss = useMemo(() => crossEntropyLoss(probabilities, targetClass), [probabilities, targetClass]);
  
  // Get step-by-step calculations
  const softmaxSteps = useMemo(() => getSoftmaxSteps(logits), [logits]);
  const lossSteps = useMemo(() => getCrossEntropySteps(probabilities, targetClass), [probabilities, targetClass]);
  
  // Compute gradients
  const gradients = useMemo(() => softmaxCrossEntropyGradient(probabilities, targetClass), [probabilities, targetClass]);

  const handleLogitChange = (index: number, value: string) => {
    const newLogits = [...logits];
    newLogits[index] = parseFloat(value) || 0;
    setLogits(newLogits);
  };

  const addLogit = () => {
    setLogits([...logits, 0]);
  };

  const removeLogit = (index: number) => {
    if (logits.length > 2) {
      const newLogits = logits.filter((_, i) => i !== index);
      setLogits(newLogits);
      if (targetClass >= newLogits.length) {
        setTargetClass(newLogits.length - 1);
      }
    }
  };

  const reset = () => {
    setLogits([2.0, 1.0, 0.1]);
    setTargetClass(0);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Softmax & Cross-Entropy</h1>
        <p className="text-lg text-muted-foreground">
          Interactive visualization of softmax activation and cross-entropy loss for multi-class classification.
        </p>
      </div>

      {/* Status Bar */}
      <Card>
        <CardContent className="">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-6">
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Number of Classes</p>
                <p className="text-2xl font-bold">{logits.length}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Target Class</p>
                <p className="text-2xl font-bold">{targetClass}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Loss</p>
                <p className="text-2xl font-bold text-red-600">{loss.toFixed(4)}</p>
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
          {/* Probability Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Probability Distribution</CardTitle>
              <CardDescription>
                Softmax converts logits (raw scores) into probabilities that sum to 1
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex justify-center">
                <ProbabilityChart 
                  probabilities={probabilities}
                  targetClass={targetClass}
                  logits={logits}
                />
              </div>
            </CardContent>
          </Card>

          {/* Step-by-step Calculations */}
          {showSteps && (
            <>
              {/* Softmax Steps */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Softmax Computation - Step by Step</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4 font-mono text-sm">
                    {softmaxSteps.map((step, idx) => (
                      <div key={idx} className="space-y-2">
                        <p className="font-semibold text-blue-600">Step {step.step}: {step.description}</p>
                        <div className="bg-blue-50 p-3 rounded overflow-x-auto">
                          <p className="whitespace-nowrap">{step.formula}</p>
                          {step.values.length > 1 && step.values.length === logits.length && (
                            <div className="mt-2 space-y-1">
                              {step.values.map((val, i) => (
                                <p key={i} className="ml-2">
                                  Class {i}: {val.toFixed(4)}
                                </p>
                              ))}
                            </div>
                          )}
                        </div>
                        {idx < softmaxSteps.length - 1 && <Separator />}
                      </div>
                    ))}

                    {/* Verification */}
                    <div className="border-t border-blue-200 pt-3">
                      <p className="font-semibold text-blue-600">Verification:</p>
                      <div className="bg-blue-50 p-3 rounded mt-2">
                        <p>Sum of probabilities = {probabilities.reduce((a, b) => a + b, 0).toFixed(6)}</p>
                        <p className="mt-1">✓ Probabilities sum to 1.0</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Cross-Entropy Steps */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Cross-Entropy Loss - Step by Step</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4 font-mono text-sm">
                    {lossSteps.map((step, idx) => (
                      <div key={idx} className="space-y-2">
                        <p className="font-semibold text-red-600">Step {step.step}: {step.description}</p>
                        <div className="bg-red-50 p-3 rounded overflow-x-auto">
                          <p className="whitespace-nowrap">{step.formula}</p>
                        </div>
                        {idx < lossSteps.length - 1 && <Separator />}
                      </div>
                    ))}

                    {/* Gradient Information */}
                    <div className="border-t-2 border-purple-300 pt-4 mt-4">
                      <p className="font-bold text-lg text-purple-700 mb-3">
                        🎯 Gradients: How to Fix the Prediction
                      </p>
                      
                      {/* What are gradients */}
                      <div className="bg-purple-50 p-4 rounded mb-4 border-2 border-purple-200">
                        <p className="font-semibold text-purple-800 mb-2">📚 What Are Gradients?</p>
                        <p className="text-sm text-gray-700 mb-2">
                          Gradients tell us <strong>how much to adjust each logit</strong> to reduce the loss (make better predictions).
                        </p>
                        <p className="text-xs text-gray-600 italic">
                          Think of it like: &quot;If I change logit z₍{targetClass}₎ by +1, how much does the loss change?&quot;
                        </p>
                      </div>

                      {/* The magic formula */}
                      <div className="bg-blue-50 p-4 rounded mb-4 border-2 border-blue-200">
                        <p className="font-semibold text-blue-800 mb-2">✨ The Magic Formula</p>
                        <p className="text-sm font-mono mb-2">∇L = p - y</p>
                        <p className="text-xs text-gray-700">
                          <strong>Gradient = Predicted Probability - Target Probability</strong>
                        </p>
                        <p className="text-xs text-gray-600 mt-2">
                          • Target probability (y): 1 for the correct class, 0 for others<br/>
                          • Predicted probability (p): What softmax gave us
                        </p>
                      </div>

                      {/* Actual gradients with explanations */}
                      <div className="space-y-3">
                        <p className="font-semibold text-purple-700">📊 Gradient Values:</p>
                        {gradients.map((grad, i) => (
                          <div key={i} className={`p-3 rounded border-2 ${
                            i === targetClass 
                              ? 'bg-green-50 border-green-300' 
                              : 'bg-orange-50 border-orange-300'
                          }`}>
                            <div className="flex justify-between items-start mb-2">
                              <span className="font-mono font-bold">
                                ∂L/∂z₍{i}₎ = {grad.toFixed(4)}
                              </span>
                              {i === targetClass && (
                                <Badge className="bg-green-600">Target Class ✓</Badge>
                              )}
                            </div>
                            
                            <div className="text-xs space-y-1">
                              <p className="font-semibold">
                                {i === targetClass ? '🎯 This is the CORRECT class' : '❌ This is a WRONG class'}
                              </p>
                              
                              {i === targetClass ? (
                                <>
                                  <p className="text-gray-700">
                                    Predicted: {(probabilities[i] * 100).toFixed(1)}% | Target: 100%
                                  </p>
                                  <p className="text-gray-700">
                                    Calculation: {probabilities[i].toFixed(4)} - 1 = {grad.toFixed(4)}
                                  </p>
                                  <p className="font-semibold text-green-700 mt-2">
                                    📈 Interpretation: <span className="font-normal">
                                      Gradient is <strong>negative ({grad.toFixed(2)})</strong>, meaning we need to 
                                      <strong> INCREASE</strong> this logit to make the model more confident in the correct answer.
                                    </span>
                                  </p>
                                  <p className="text-green-700 mt-1">
                                    💡 Action: Add {Math.abs(grad * 0.1).toFixed(3)} × learning_rate to z₍{i}₎
                                  </p>
                                </>
                              ) : (
                                <>
                                  <p className="text-gray-700">
                                    Predicted: {(probabilities[i] * 100).toFixed(1)}% | Target: 0%
                                  </p>
                                  <p className="text-gray-700">
                                    Calculation: {probabilities[i].toFixed(4)} - 0 = {grad.toFixed(4)}
                                  </p>
                                  <p className="font-semibold text-orange-700 mt-2">
                                    📉 Interpretation: <span className="font-normal">
                                      Gradient is <strong>positive ({grad.toFixed(2)})</strong>, meaning we need to 
                                      <strong> DECREASE</strong> this logit to reduce confidence in the wrong answer.
                                    </span>
                                  </p>
                                  <p className="text-orange-700 mt-1">
                                    💡 Action: Subtract {Math.abs(grad * 0.1).toFixed(3)} × learning_rate from z₍{i}₎
                                  </p>
                                </>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Summary */}
                      <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded mt-4 border-2 border-blue-300">
                        <p className="font-bold text-blue-800 mb-2">🔑 Key Takeaway:</p>
                        <p className="text-sm text-gray-700 leading-relaxed">
                          During training, we use these gradients to update the network weights. 
                          The <strong className="text-green-600">negative gradient on the target class</strong> pushes 
                          the model to be MORE confident in the correct answer, while <strong className="text-orange-600">positive 
                          gradients on other classes</strong> push the model to be LESS confident in wrong answers.
                        </p>
                        <p className="text-xs text-gray-600 mt-2 italic">
                          This is gradient descent in action! Over many training steps, the model learns to 
                          predict the correct class with high confidence.
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>

        {/* Right Column - Controls */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Controls</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Logits (Raw Scores) */}
              <div className="space-y-3">
                <h3 className="font-semibold text-sm">Logits (Raw Scores)</h3>
                <div className="space-y-2">
                  {logits.map((logit, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <Label className="text-xs w-16">Class {i}</Label>
                      <Input
                        type="number"
                        step="0.1"
                        value={logit}
                        onChange={(e) => handleLogitChange(i, e.target.value)}
                        className="font-mono"
                      />
                      {logits.length > 2 && (
                        <Button
                          onClick={() => removeLogit(i)}
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
                  onClick={addLogit}
                  variant="outline"
                  size="sm"
                  className="w-full"
                >
                  + Add Class
                </Button>
              </div>

              <Separator />

              {/* Target Class */}
              <div className="space-y-3">
                <h3 className="font-semibold text-sm">Target Class (Ground Truth)</h3>
                <div className="grid grid-cols-3 gap-2">
                  {logits.map((_, i) => (
                    <Button
                      key={i}
                      onClick={() => setTargetClass(i)}
                      variant={targetClass === i ? "default" : "outline"}
                      size="sm"
                    >
                      Class {i}
                    </Button>
                  ))}
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

              <Separator />

              {/* Quick Stats */}
              <div className="space-y-2">
                <h3 className="font-semibold text-sm">Quick Stats</h3>
                <div className="space-y-1 font-mono text-xs">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Highest Prob:</span>
                    <span>Class {probabilities.indexOf(Math.max(...probabilities))}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Confidence:</span>
                    <span>{(Math.max(...probabilities) * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Prediction:</span>
                    <span className={probabilities.indexOf(Math.max(...probabilities)) === targetClass ? "text-green-600" : "text-red-600"}>
                      {probabilities.indexOf(Math.max(...probabilities)) === targetClass ? "✓ Correct" : "✗ Wrong"}
                    </span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* What You're Learning */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">What You&apos;re Learning</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-2 text-muted-foreground">
              <p>• How softmax converts scores to probabilities</p>
              <p>• Why probabilities must sum to 1</p>
              <p>• How cross-entropy measures prediction error</p>
              <p>• Why lower loss means better predictions</p>
              <p>• How gradients flow through softmax</p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* PyTorch Code Comparison */}
      <SoftmaxPyTorchComparison />
    </div>
  );
}
