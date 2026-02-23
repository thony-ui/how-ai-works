"use client";

import React, { useState, useMemo, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  generateDataset,
  initializeWeights,
  trainEpoch,
  computeAccuracy,
  forwardPass,
  getForwardPassSteps,
  getBackwardPassSteps,
  type MultiClassDataPoint,
  type NetworkWeights,
} from '@/lib/math/multiclass';
import { DecisionBoundaryCanvas } from './DecisionBoundaryCanvas';
import { TrainingChart } from './TrainingChart';
import { MultiClassPyTorchComparison } from './MultiClassPyTorchComparison';
import { PlayIcon, PauseIcon, RotateCcwIcon, SkipForwardIcon } from 'lucide-react';

export default function MultiClassPlayground() {
  // Dataset configuration
  const [numClasses, setNumClasses] = useState<number>(3);
  const [pointsPerClass, setPointsPerClass] = useState<number>(20);
  const [hiddenSize, setHiddenSize] = useState<number>(8);
  const [learningRate, setLearningRate] = useState<number>(0.01);

  // Dataset and weights
  const [dataset, setDataset] = useState<MultiClassDataPoint[]>(() =>
    generateDataset(3, 20)
  );
  const [weights, setWeights] = useState<NetworkWeights>(() =>
    initializeWeights(2, 8, 3)
  );

  // Training state
  const [epoch, setEpoch] = useState<number>(0);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [accuracyHistory, setAccuracyHistory] = useState<number[]>([]);
  const [showSteps, setShowSteps] = useState<boolean>(false);
  const [selectedPoint, setSelectedPoint] = useState<MultiClassDataPoint | null>(null);

  // Current metrics
  const accuracy = useMemo(() => computeAccuracy(dataset, weights), [dataset, weights]);
  const currentLoss = useMemo(() => {
    if (lossHistory.length === 0) return 0;
    return lossHistory[lossHistory.length - 1];
  }, [lossHistory]);

  // Step-by-step for selected point
  const forwardSteps = useMemo(() => {
    if (!selectedPoint) return [];
    return getForwardPassSteps([selectedPoint.x, selectedPoint.y], weights);
  }, [selectedPoint, weights]);

  const backwardSteps = useMemo(() => {
    if (!selectedPoint) return [];
    const forward = forwardPass([selectedPoint.x, selectedPoint.y], weights);
    return getBackwardPassSteps([selectedPoint.x, selectedPoint.y], forward, selectedPoint.class, weights);
  }, [selectedPoint, weights]);

  // Reset everything
  const handleReset = useCallback(() => {
    setIsTraining(false);
    setEpoch(0);
    setLossHistory([]);
    setAccuracyHistory([]);
    setDataset(generateDataset(numClasses, pointsPerClass));
    setWeights(initializeWeights(2, hiddenSize, numClasses));
    setSelectedPoint(null);
  }, [numClasses, pointsPerClass, hiddenSize]);

  // Train one epoch
  const trainOneEpoch = useCallback(() => {
    const result = trainEpoch(dataset, weights, learningRate);
    setWeights(result.weights);
    setEpoch(prev => prev + 1);
    setLossHistory(prev => [...prev, result.loss]);
    setAccuracyHistory(prev => [...prev, computeAccuracy(dataset, result.weights)]);
  }, [dataset, weights, learningRate]);

  // Training loop
  React.useEffect(() => {
    if (!isTraining) return;

    const interval = setInterval(() => {
      trainOneEpoch();
    }, 100);

    return () => clearInterval(interval);
  }, [isTraining, trainOneEpoch]);

  // Stop training if accuracy reaches 100%
  React.useEffect(() => {
    if (accuracy >= 0.99 && isTraining) {
      setIsTraining(false);
    }
  }, [accuracy, isTraining]);

  // Regenerate dataset and weights when configuration changes (but only if not training and epoch = 0)
  React.useEffect(() => {
    if (epoch === 0 && !isTraining) {
      setDataset(generateDataset(numClasses, pointsPerClass));
      setWeights(initializeWeights(2, hiddenSize, numClasses));
      setSelectedPoint(null);
      setLossHistory([]);
      setAccuracyHistory([]);
    }
  }, [numClasses, pointsPerClass, hiddenSize, epoch, isTraining]);

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Multi-Class Classification</h1>
        <p className="text-lg text-muted-foreground">
          Train a neural network to classify data points into multiple categories using softmax output.
        </p>
      </div>

      {/* Status Bar */}
      <Card>
        <CardContent className="">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-6">
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Epoch</p>
                <p className="text-2xl font-bold">{epoch}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Accuracy</p>
                <p className="text-2xl font-bold text-green-600">{(accuracy * 100).toFixed(1)}%</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Loss</p>
                <p className="text-2xl font-bold text-red-600">{currentLoss.toFixed(4)}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Data Points</p>
                <p className="text-2xl font-bold">{dataset.length}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={isTraining ? "default" : "secondary"}>
                {isTraining ? "Training..." : "Paused"}
              </Badge>
              {accuracy >= 0.99 && <Badge variant="default">✓ Converged</Badge>}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Visualization */}
        <div className="lg:col-span-2 space-y-6">
          {/* Decision Boundary */}
          <Card>
            <CardHeader>
              <CardTitle>Decision Boundary</CardTitle>
              <CardDescription>
                Click a point to see its forward and backward pass calculations
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex justify-center overflow-x-auto">
                <DecisionBoundaryCanvas
                  dataset={dataset}
                  weights={weights}
                  onPointClick={setSelectedPoint}
                  selectedPoint={selectedPoint}
                />
              </div>
            </CardContent>
          </Card>

          {/* Training Progress */}
          <Card>
            <CardHeader>
              <CardTitle>Training Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <TrainingChart
                lossHistory={lossHistory}
                accuracyHistory={accuracyHistory}
              />
            </CardContent>
          </Card>

          {/* Step-by-Step Calculations */}
          {showSteps && selectedPoint && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">
                    Forward Pass - Point ({selectedPoint.x.toFixed(2)}, {selectedPoint.y.toFixed(2)})
                  </CardTitle>
                  <CardDescription>Target class: {selectedPoint.class}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4 font-mono text-sm">
                    {forwardSteps.map((step, idx) => (
                      <div key={idx} className="space-y-2">
                        <p className="font-semibold text-blue-600">
                          Step {step.step}: {step.description}
                        </p>
                        <div className="bg-blue-50 p-3 rounded overflow-x-auto">
                          <p className="whitespace-nowrap">{step.formula}</p>
                          {step.values && Array.isArray(step.values) && step.values.length > 0 && (
                            <div className="mt-2">
                              {typeof step.values[0] === 'number' ? (
                                <p>
                                  [{(step.values as number[]).map(v => v.toFixed(4)).join(', ')}]
                                </p>
                              ) : null}
                            </div>
                          )}
                        </div>
                        {idx < forwardSteps.length - 1 && <Separator />}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Backward Pass - Gradients</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4 font-mono text-sm">
                    {backwardSteps.map((step, idx) => (
                      <div key={idx} className="space-y-2">
                        <p className="font-semibold text-purple-600">
                          Step {step.step}: {step.description}
                        </p>
                        <div className="bg-purple-50 p-3 rounded overflow-x-auto">
                          <p className="whitespace-nowrap">{step.formula}</p>
                        </div>
                        {idx < backwardSteps.length - 1 && <Separator />}
                      </div>
                    ))}
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
              {/* Training Controls */}
              <div className="space-y-3">
                <h3 className="font-semibold text-sm">Training</h3>
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    onClick={() => setIsTraining(!isTraining)}
                    disabled={accuracy >= 0.99}
                    className="w-full"
                  >
                    {isTraining ? (
                      <>
                        <PauseIcon className="w-4 h-4 mr-2" />
                        Pause
                      </>
                    ) : (
                      <>
                        <PlayIcon className="w-4 h-4 mr-2" />
                        Train
                      </>
                    )}
                  </Button>
                  <Button onClick={trainOneEpoch} variant="secondary" className="w-full">
                    <SkipForwardIcon className="w-4 h-4 mr-2" />
                    Step
                  </Button>
                </div>
                <Button onClick={handleReset} variant="outline" className="w-full">
                  <RotateCcwIcon className="w-4 h-4 mr-2" />
                  Reset
                </Button>
              </div>

              <Separator />

              {/* Dataset Configuration */}
              <div className="space-y-4">
                <h3 className="font-semibold text-sm">Dataset Configuration</h3>
                
                <div className="space-y-2">
                  <Label className="text-xs">
                    Number of Classes: {numClasses}
                  </Label>
                  <Slider
                    value={[numClasses]}
                    onValueChange={([val]) => setNumClasses(val)}
                    min={2}
                    max={5}
                    step={1}
                    disabled={isTraining || epoch > 0}
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-xs">
                    Points per Class: {pointsPerClass}
                  </Label>
                  <Slider
                    value={[pointsPerClass]}
                    onValueChange={([val]) => setPointsPerClass(val)}
                    min={10}
                    max={50}
                    step={5}
                    disabled={isTraining || epoch > 0}
                  />
                </div>
              </div>

              <Separator />

              {/* Network Configuration */}
              <div className="space-y-4">
                <h3 className="font-semibold text-sm">Network Architecture</h3>
                
                <div className="space-y-2">
                  <Label className="text-xs">
                    Hidden Layer Size: {hiddenSize}
                  </Label>
                  <Slider
                    value={[hiddenSize]}
                    onValueChange={([val]) => setHiddenSize(val)}
                    min={4}
                    max={16}
                    step={2}
                    disabled={isTraining || epoch > 0}
                  />
                </div>

                <div className="bg-muted p-3 rounded text-xs font-mono">
                  <p>Input: 2 features (x, y)</p>
                  <p>Hidden: {hiddenSize} neurons (ReLU)</p>
                  <p>Output: {numClasses} classes (Softmax)</p>
                </div>
              </div>

              <Separator />

              {/* Training Parameters */}
              <div className="space-y-4">
                <h3 className="font-semibold text-sm">Training Parameters</h3>
                
                <div className="space-y-2">
                  <Label className="text-xs">
                    Learning Rate: {learningRate.toFixed(3)}
                  </Label>
                  <Slider
                    value={[learningRate * 1000]}
                    onValueChange={([val]) => setLearningRate(val / 1000)}
                    min={1}
                    max={50}
                    step={1}
                  />
                </div>
              </div>

              <Separator />

              {/* Visualization */}
              <div className="space-y-2">
                <Button
                  onClick={() => setShowSteps(!showSteps)}
                  variant="secondary"
                  className="w-full"
                  disabled={!selectedPoint}
                >
                  {showSteps ? "Hide" : "Show"} Calculation Steps
                </Button>
                {!selectedPoint && (
                  <p className="text-xs text-muted-foreground text-center">
                    Click a point on the chart to see calculations
                  </p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* What You're Learning */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">What You&apos;re Learning</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-2 text-muted-foreground">
              <p>• How neural networks classify data into categories</p>
              <p>• Role of hidden layers in learning complex patterns</p>
              <p>• Softmax activation for probability distributions</p>
              <p>• Decision boundaries between classes</p>
              <p>• How training improves accuracy over time</p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* PyTorch Code Comparison */}
      <MultiClassPyTorchComparison />
    </div>
  );
}
