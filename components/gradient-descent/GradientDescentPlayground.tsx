"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Label } from '@/components/ui/label';
import { 
  linearRegressionGradient, 
  calculateMSE
} from '@/lib/math/gradient';
import { CanvasVisualizer } from './CanvasVisualizer';
import { LossChart } from './LossChart';
import { PyTorchComparison } from './PyTorchComparison';
import { PlayIcon, PauseIcon, RotateCcwIcon, SkipForwardIcon } from 'lucide-react';

interface Point {
  x: number;
  y: number;
}

export default function GradientDescentPlayground() {
  // Initial data points
  const [points, setPoints] = useState<Point[]>([
    { x: 1, y: 2 },
    { x: 2, y: 3.5 },
    { x: 3, y: 5 },
    { x: 4, y: 7 },
    { x: 5, y: 8.5 }
  ]);

  // Model parameters
  const [w, setW] = useState<number>(0.5);
  const [b, setB] = useState<number>(0);

  // Training state
  const [learningRate, setLearningRate] = useState<number>(0.01);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [iteration, setIteration] = useState<number>(0);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [gradients, setGradients] = useState<{ dw: number; db: number }>({ dw: 0, db: 0 });

  // Reset everything
  const handleReset = () => {
    setW(0.5);
    setB(0);
    setLossHistory([]);
    setIteration(0);
    setIsTraining(false);
    setGradients({ dw: 0, db: 0 });
  };

  // Single step of gradient descent
  const performStep = React.useCallback(() => {
    const xValues = points.map(p => p.x);
    const yValues = points.map(p => p.y);

    // Calculate gradients
    const grads = linearRegressionGradient(xValues, yValues, w, b);
    setGradients(grads);

    // Update parameters
    const newW = w - learningRate * grads.dw;
    const newB = b - learningRate * grads.db;

    setW(newW);
    setB(newB);

    // Calculate and store loss
    const loss = calculateMSE(xValues, yValues, newW, newB);
    setLossHistory(prev => [...prev, loss]);
    setIteration(prev => prev + 1);
  }, [points, w, b, learningRate]);

  // Auto training
  React.useEffect(() => {
    if (!isTraining) return;

    const interval = setInterval(() => {
      performStep();
    }, 100);

    return () => clearInterval(interval);
  }, [isTraining, performStep]);

  // Handle point dragging
  const handlePointDrag = (index: number, x: number, y: number) => {
    const newPoints = [...points];
    newPoints[index] = { 
      x: Math.max(0, Math.min(10, x)), 
      y: Math.max(0, Math.min(10, y)) 
    };
    setPoints(newPoints);
    
    // Recalculate loss if we have history
    if (lossHistory.length > 0) {
      const xValues = newPoints.map(p => p.x);
      const yValues = newPoints.map(p => p.y);
      const newLoss = calculateMSE(xValues, yValues, w, b);
      setLossHistory(prev => [...prev.slice(0, -1), newLoss]);
    }
  };

  // Calculate current loss
  const currentLoss = calculateMSE(
    points.map(p => p.x),
    points.map(p => p.y),
    w,
    b
  );

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Gradient Descent Playground</h1>
        <p className="text-lg text-muted-foreground">
          Interactive visualization of linear regression with gradient descent. Drag points to see how the algorithm adapts in real-time.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Visualization */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Linear Regression Visualization</CardTitle>
              <CardDescription>
                Drag data points to change the dataset. The blue line shows the current prediction (y = w×x + b).
                Red dashed lines show prediction errors.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <CanvasVisualizer
                  points={points}
                  w={w}
                  b={b}
                  onPointDrag={handlePointDrag}
                />
              </div>
            </CardContent>
          </Card>

          <LossChart lossHistory={lossHistory} width={600} height={250} />
        </div>

        {/* Right Column - Controls */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Training Controls</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Learning Rate */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label>Learning Rate</Label>
                  <Badge variant="secondary">{learningRate.toFixed(4)}</Badge>
                </div>
                <Slider
                  value={[learningRate]}
                  onValueChange={(values) => setLearningRate(values[0])}
                  min={0.001}
                  max={0.1}
                  step={0.001}
                  disabled={isTraining}
                />
              </div>

              <Separator />

              {/* Control Buttons */}
              <div className="grid grid-cols-2 gap-2">
                <Button
                  onClick={() => setIsTraining(!isTraining)}
                  variant={isTraining ? "destructive" : "default"}
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
                
                <Button
                  onClick={performStep}
                  disabled={isTraining}
                  variant="outline"
                  className="w-full"
                >
                  <SkipForwardIcon className="w-4 h-4 mr-2" />
                  Step
                </Button>
              </div>

              <Button
                onClick={handleReset}
                variant="outline"
                className="w-full"
              >
                <RotateCcwIcon className="w-4 h-4 mr-2" />
                Reset
              </Button>

              <Separator />

              {/* Parameters Display */}
              <div className="space-y-3">
                <h3 className="font-semibold text-sm">Model Parameters</h3>
                <div className="space-y-2 font-mono text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Weight (w):</span>
                    <span className="font-medium">{w.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Bias (b):</span>
                    <span className="font-medium">{b.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Iteration:</span>
                    <span className="font-medium">{iteration}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Current Loss:</span>
                    <span className="font-medium text-red-600">{currentLoss.toFixed(4)}</span>
                  </div>
                </div>
              </div>

              <Separator />

              {/* Gradients Display */}
              <div className="space-y-3">
                <h3 className="font-semibold text-sm">Current Gradients</h3>
                <div className="space-y-2 font-mono text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">∂Loss/∂w:</span>
                    <span className="font-medium">{gradients.dw.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">∂Loss/∂b:</span>
                    <span className="font-medium">{gradients.db.toFixed(4)}</span>
                  </div>
                </div>
              </div>

              {/* Equation */}
              <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                <p className="text-sm font-medium text-blue-900 text-center font-mono">
                  y = {w.toFixed(2)}x + {b.toFixed(2)}
                </p>
              </div>
            </CardContent>
          </Card>

          {/* What You're Learning */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">What You&apos;re Learning</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-2 text-muted-foreground">
              <p>• How gradients point toward optimal parameters</p>
              <p>• Why learning rate affects convergence speed</p>
              <p>• How loss decreases with each iteration</p>
              <p>• What ${"fitting"} data really means</p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* PyTorch Code Comparison */}
      <PyTorchComparison />
    </div>
  );
}
