"use client";

import React, { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { TwoLayerNetwork, type NetworkState } from '@/lib/math/network';
import { NetworkVisualizer } from './NetworkVisualizer';
import { BackpropPyTorchComparison } from './BackpropPyTorchComparison';
import { 
  PlayIcon, 
  RotateCcwIcon, 
  ArrowRightIcon,
  BrainIcon,
  TrendingDownIcon
} from 'lucide-react';

type TrainingPhase = 'idle' | 'forward' | 'loss' | 'backward' | 'update';

export default function BackpropagationPlayground() {
  const networkRef = useRef(new TwoLayerNetwork());
  // eslint-disable-next-line react-hooks/refs
  const [networkState, setNetworkState] = useState<NetworkState>(networkRef.current.getState());
  
  // Training inputs
  const [inputX1, setInputX1] = useState<number>(0.5);
  const [inputX2, setInputX2] = useState<number>(-0.3);
  const [targetY, setTargetY] = useState<number>(0.7);
  const [learningRate, setLearningRate] = useState<number>(0.01);
  
  // Training state
  const [phase, setPhase] = useState<TrainingPhase>('idle');
  const [iteration, setIteration] = useState<number>(0);
  const [isAutoTraining, setIsAutoTraining] = useState<boolean>(false);

  // Reset network
  const handleReset = () => {
    networkRef.current.reset();
    setNetworkState(networkRef.current.getState());
    setPhase('idle');
    setIteration(0);
    setIsAutoTraining(false);
  };

  // Step through phases
  const handleNextStep = () => {
    const network = networkRef.current;
    const input = [[inputX1, inputX2]];
    const target = [[targetY]];

    switch (phase) {
      case 'idle':
      case 'update':
        // Forward pass
        network.forward(input);
        setNetworkState(network.getState());
        setPhase('forward');
        break;

      case 'forward':
        // Compute loss
        network.computeLoss(target);
        setNetworkState(network.getState());
        setPhase('loss');
        break;

      case 'loss':
        // Backward pass
        network.backward();
        setNetworkState(network.getState());
        setPhase('backward');
        break;

      case 'backward':
        // Update weights
        network.updateWeights(learningRate);
        const newState = network.getState();
        setNetworkState(newState);
        setPhase('update');
        setIteration(prev => prev + 1);
        break;
    }
  };

  // Full training step
  const handleFullStep = React.useCallback(() => {
    const network = networkRef.current;
    const input = [[inputX1, inputX2]];
    const target = [[targetY]];
    
    network.trainStep(input, target, learningRate);
    setNetworkState(network.getState());
    setPhase('update');
    setIteration(prev => prev + 1);
  }, [inputX1, inputX2, targetY, learningRate]);

  // Auto training
  React.useEffect(() => {
    if (!isAutoTraining) return;

    const interval = setInterval(() => {
      handleFullStep();
    }, 200);

    return () => clearInterval(interval);
  }, [isAutoTraining, handleFullStep]);

  const getPhaseDescription = (): string => {
    switch (phase) {
      case 'idle':
        return 'Ready to start. Click "Next Step" to begin forward pass.';
      case 'forward':
        return 'Forward pass complete. Activations computed. Click "Next Step" to calculate loss.';
      case 'loss':
        return 'Loss computed. Click "Next Step" to start backpropagation.';
      case 'backward':
        return 'Gradients computed. Click "Next Step" to update weights.';
      case 'update':
        return 'Weights updated. Click "Next Step" for another iteration.';
      default:
        return '';
    }
  };

  const showGradients = phase === 'backward' || phase === 'update';

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Backpropagation Playground</h1>
        <p className="text-lg text-muted-foreground">
          Step through forward and backward passes in a 2-layer neural network. Watch how gradients flow backward to update weights.
        </p>
      </div>

      {/* Phase indicator */}
      <Card className="bg-linear-to-r from-blue-50 to-purple-50 border-blue-200">
        <CardContent className="">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Badge 
                variant={phase === 'idle' ? 'secondary' : 'default'}
                className="text-sm"
              >
                Phase: {phase.toUpperCase()}
              </Badge>
              <span className="text-sm text-muted-foreground">
                {getPhaseDescription()}
              </span>
            </div>
            {networkState.loss !== undefined && (
              <div className="flex items-center gap-2">
                <TrendingDownIcon className="w-4 h-4 text-red-600" />
                <span className="text-sm font-mono font-medium">
                  Loss: {networkState.loss.toFixed(6)}
                </span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Visualization */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>
                {showGradients ? 'Gradient Flow (Backward Pass)' : 'Network Architecture (Forward Pass)'}
              </CardTitle>
              <CardDescription>
                {showGradients 
                  ? 'Red/green connections show gradient direction. Node values show gradients.'
                  : 'Blue/red nodes show activation values. Edge labels show weights.'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <NetworkVisualizer 
                  state={networkState}
                  showGradients={showGradients}
                />
              </div>
            </CardContent>
          </Card>

          {/* Computation Details */}
          {phase !== 'idle' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Current Computation Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4 font-mono text-sm">
                  {phase === 'forward' && networkState.a1 && (
                    <div className="space-y-3">
                      <p className="font-semibold text-blue-600">Forward Pass - Step by Step:</p>
                      <div className="bg-blue-50 p-4 rounded space-y-3">
                        {/* Input */}
                        <div>
                          <p className="font-medium text-blue-700">Input:</p>
                          <p>x = [{networkState.input?.[0].map(v => v.toFixed(3)).join(', ')}]</p>
                        </div>

                        {/* Layer 1 - Hidden Layer */}
                        <div className="border-t border-blue-200 pt-2">
                          <p className="font-medium text-blue-700">Layer 1 → Hidden Layer (ReLU):</p>
                          
                          {/* Show W1 matrix */}
                          <div className="mt-1">
                            <p className="text-blue-600">Weights W₁ (2×4):</p>
                            {networkState.w1?.map((row, i) => (
                              <p key={i}>W₁[{i}] = [{row.map(v => v.toFixed(3)).join(', ')}]</p>
                            ))}
                          </div>
                          
                          {/* Show b1 */}
                          <div className="mt-1">
                            <p className="text-blue-600">Biases b₁:</p>
                            <p>b₁ = [{networkState.b1?.[0].map(v => v.toFixed(3)).join(', ')}]</p>
                          </div>

                          {/* Compute z1 */}
                          <div className="mt-2">
                            <p className="font-medium">z₁ = x · W₁ + b₁</p>
                            {networkState.z1 && (
                              <p>z₁ = [{networkState.z1[0].map(v => v.toFixed(3)).join(', ')}]</p>
                            )}
                          </div>

                          {/* Apply ReLU */}
                          <div className="mt-2">
                            <p className="font-medium">a₁ = ReLU(z₁) = max(0, z₁)</p>
                            <p>a₁ = [{networkState.a1[0].map(v => v.toFixed(3)).join(', ')}]</p>
                          </div>
                        </div>

                        {/* Layer 2 - Output Layer */}
                        {networkState.a2 && (
                          <div className="border-t border-blue-200 pt-2">
                            <p className="font-medium text-blue-700">Layer 2 → Output:</p>
                            
                            {/* Show W2 */}
                            <div className="mt-1">
                              <p className="text-blue-600">Weights W₂ (4×1):</p>
                              <p>W₂ = [{networkState.w2?.map(row => row[0].toFixed(3)).join(', ')}]</p>
                            </div>
                            
                            {/* Show b2 */}
                            <div className="mt-1">
                              <p className="text-blue-600">Bias b₂:</p>
                              <p>b₂ = {networkState.b2?.[0][0].toFixed(3)}</p>
                            </div>

                            {/* Compute output */}
                            <div className="mt-2">
                              <p className="font-medium">output = a₁ · W₂ + b₂</p>
                              <p>output = <span className="font-bold text-blue-800">{networkState.a2[0][0].toFixed(4)}</span></p>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {phase === 'loss' && networkState.loss !== undefined && (
                    <div className="space-y-2">
                      <p className="font-semibold text-red-600">Loss Computation - Step by Step:</p>
                      <div className="bg-red-50 p-4 rounded space-y-3">
                        <div>
                          <p className="font-medium text-red-700">Mean Squared Error (MSE) Loss Function:</p>
                          <div className="bg-white p-3 rounded border border-red-200 my-2">
                            <p className="font-mono text-sm">L = ½(ŷ - y)²</p>
                            <p className="text-xs text-muted-foreground mt-1">
                              where ŷ = prediction, y = target
                            </p>
                            <p className="text-xs text-muted-foreground mt-1">
                              Note: We use ½ so the derivative is cleaner: ∂L/∂ŷ = (ŷ - y)
                            </p>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <p className="font-medium text-red-700">Step 1: Get prediction and target</p>
                          <div className="ml-3">
                            <p>ŷ (prediction) = {networkState.a2![0][0].toFixed(4)}</p>
                            <p>y (target) = {targetY}</p>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <p className="font-medium text-red-700">Step 2: Calculate error</p>
                          <div className="ml-3">
                            <p>error = ŷ - y</p>
                            <p>error = {networkState.a2![0][0].toFixed(4)} - {targetY}</p>
                            <p className="font-bold">error = {(networkState.a2![0][0] - targetY).toFixed(4)}</p>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <p className="font-medium text-red-700">Step 3: Square the error and divide by 2</p>
                          <div className="ml-3">
                            <p>L = ½ × error²</p>
                            <p>L = ½ × ({(networkState.a2![0][0] - targetY).toFixed(4)})²</p>
                            <p>L = ½ × {Math.pow(networkState.a2![0][0] - targetY, 2).toFixed(6)}</p>
                            <p className="font-bold text-red-800 text-lg mt-1">L = {networkState.loss.toFixed(6)}</p>
                          </div>
                        </div>
                        <div className="bg-red-100 p-3 rounded border border-red-300 mt-2">
                          <p className="font-semibold text-red-900">Why MSE?</p>
                          <ul className="text-sm mt-1 space-y-1 list-disc list-inside">
                            <li>Penalizes larger errors more heavily (quadratic)</li>
                            <li>Always positive (squared term)</li>
                            <li>Differentiable everywhere (smooth gradient)</li>
                            <li>The ½ factor simplifies backpropagation math</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {phase === 'backward' && networkState.dw2 && (
                    <div className="space-y-3">
                      <p className="font-semibold text-purple-600">Backward Pass - Step by Step:</p>
                      <div className="bg-purple-50 p-4 rounded space-y-3">
                        
                        {/* Step 1: Output Gradient */}
                        <div>
                          <p className="font-medium text-purple-700">Step 1: Output Gradient</p>
                          <div className="mt-1 space-y-1">
                            <p>Given: output = {networkState.a2?.[0][0].toFixed(4)}, target = {targetY}</p>
                            <p>∂L/∂output = output - target</p>
                            <p>∂L/∂output = {networkState.a2?.[0][0].toFixed(4)} - {targetY}</p>
                            <p className="font-medium">∂L/∂output = {networkState.dz2?.[0][0].toFixed(4)}</p>
                          </div>
                        </div>
                        
                        {/* Step 2: Layer 2 Weight Gradients */}
                        <div className="border-t border-purple-200 pt-2">
                          <p className="font-medium text-purple-700">Step 2: Weight Gradients (Layer 2)</p>
                          <div className="mt-1 space-y-1">
                            <p>∂L/∂W₂ = a₁ᵀ · ∂L/∂output</p>
                            <p>a₁ = [{networkState.a1?.[0].map(v => v.toFixed(3)).join(', ')}]</p>
                            <p>∂L/∂output = {networkState.dz2?.[0][0].toFixed(4)}</p>
                            <p className="mt-1">Computing element-wise:</p>
                            {networkState.dw2?.map((row, i) => (
                              <p key={i}>
                                ∂W₂[{i}] = {networkState.a1?.[0][i].toFixed(3)} × {networkState.dz2?.[0][0].toFixed(4)} = {row[0].toFixed(4)}
                              </p>
                            ))}
                            <p className="font-medium mt-1">∂W₂ = [{networkState.dw2?.map(row => row[0].toFixed(4)).join(', ')}]</p>
                          </div>
                          
                          {networkState.db2 && (
                            <div className="mt-2 space-y-1">
                              <p>∂L/∂b₂ = ∂L/∂output</p>
                              <p className="font-medium">∂b₂ = {networkState.db2[0][0].toFixed(4)}</p>
                            </div>
                          )}
                        </div>
                        
                        {/* Step 3: Hidden Layer Gradient */}
                        <div className="border-t border-purple-200 pt-2">
                          <p className="font-medium text-purple-700">Step 3: Hidden Layer Gradient</p>
                          <div className="mt-1 space-y-1">
                            <p>∂L/∂a₁ = ∂L/∂output · W₂ᵀ</p>
                            <p>W₂ = [{networkState.w2?.map(row => row[0].toFixed(3)).join(', ')}]</p>
                            <p>∂L/∂output = {networkState.dz2?.[0][0].toFixed(4)}</p>
                            <p className="mt-1">Computing element-wise:</p>
                            {networkState.da1?.[0].map((val, i) => (
                              <p key={i}>
                                ∂a₁[{i}] = {networkState.dz2?.[0][0].toFixed(4)} × {networkState.w2?.[i][0].toFixed(3)} = {val.toFixed(4)}
                              </p>
                            ))}
                            <p className="font-medium mt-1">∂a₁ = [{networkState.da1?.[0].map(v => v.toFixed(4)).join(', ')}]</p>
                          </div>
                        </div>
                        
                        {/* Step 4: Pre-activation Gradient */}
                        <div className="border-t border-purple-200 pt-2">
                          <p className="font-medium text-purple-700">Step 4: Pre-activation Gradient (ReLU Derivative)</p>
                          <div className="mt-1 space-y-1">
                            <p>∂L/∂z₁ = ∂L/∂a₁ ⊙ ReLU&apos;(z₁)</p>
                            <p>ReLU&apos;(z) = 1 if z &gt; 0, else 0</p>
                            <p>z₁ = [{networkState.z1?.[0].map(v => v.toFixed(3)).join(', ')}]</p>
                            <p>∂L/∂a₁ = [{networkState.da1?.[0].map(v => v.toFixed(4)).join(', ')}]</p>
                            <p className="mt-1">Computing element-wise:</p>
                            {networkState.dz1?.[0].map((val, i) => {
                              const z_val = networkState.z1?.[0][i] || 0;
                              const relu_deriv = z_val > 0 ? 1 : 0;
                              return (
                                <p key={i}>
                                  ∂z₁[{i}] = {networkState.da1?.[0][i].toFixed(4)} × {relu_deriv} = {val.toFixed(4)}
                                </p>
                              );
                            })}
                            <p className="font-medium mt-1">∂z₁ = [{networkState.dz1?.[0].map(v => v.toFixed(4)).join(', ')}]</p>
                          </div>
                        </div>
                        
                        {/* Step 5: Layer 1 Weight Gradients */}
                        <div className="border-t border-purple-200 pt-2">
                          <p className="font-medium text-purple-700">Step 5: Weight Gradients (Layer 1)</p>
                          <div className="ml-2 mt-1 space-y-1">
                            <p>∂L/∂W₁ = xᵀ · ∂L/∂z₁</p>
                            <p>x = [{networkState.input?.[0].map(v => v.toFixed(3)).join(', ')}]</p>
                            <p>∂z₁ = [{networkState.dz1?.[0].map(v => v.toFixed(4)).join(', ')}]</p>
                            <p className="mt-1">Computing matrix (2×4):</p>
                            {networkState.dw1?.map((row, i) => (
                              <div key={i}>
                                <p>Row {i} (x[{i}] × ∂z₁):</p>
                                {row.map((val, j) => (
                                  <p key={j} className="ml-2">
                                    ∂W₁[{i}][{j}] = {networkState.input?.[0][i].toFixed(3)} × {networkState.dz1?.[0][j].toFixed(4)} = {val.toFixed(4)}
                                  </p>
                                ))}
                              </div>
                            ))}
                          </div>
                          
                          {networkState.db1 && (
                            <div className="ml-2 mt-2 space-y-1">
                              <p>∂L/∂b₁ = ∂L/∂z₁</p>
                              <p className="font-medium">∂b₁ = [{networkState.db1[0].map(v => v.toFixed(4)).join(', ')}]</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {phase === 'update' && networkState.dw1 && (
                    <div className="space-y-3">
                      <p className="font-semibold text-green-600">Weight Update - Step by Step:</p>
                      <div className="bg-green-50 p-4 rounded space-y-3">
                        <div>
                          <p className="font-medium text-green-700">Update Rule:</p>
                          <div className="space-y-1">
                            <p>W_new = W_old - α × ∂L/∂W</p>
                            <p>b_new = b_old - α × ∂L/∂b</p>
                            <p className="mt-1">where α (learning rate) = {learningRate}</p>
                          </div>
                        </div>
                        
                        <div className="border-t border-green-200 pt-2">
                          <p className="font-medium text-green-700">Layer 2 Updates:</p>
                          <div className="space-y-1">
                            <p>W₂: Each element updated by subtracting α × ∂W₂</p>
                            {networkState.dw2?.map((row, i) => (
                              <p key={i}>
                                W₂[{i}] ← W₂[{i}] - {learningRate} × {row[0].toFixed(4)}
                              </p>
                            ))}
                            {networkState.db2 && (
                              <p className="mt-1">
                                b₂ ← b₂ - {learningRate} × {networkState.db2[0][0].toFixed(4)}
                              </p>
                            )}
                          </div>
                        </div>
                        
                        <div className="border-t border-green-200 pt-2">
                          <p className="font-medium text-green-700">Layer 1 Updates:</p>
                          <div className="space-y-1">
                            <p>W₁: Each element updated by subtracting α × ∂W₁</p>
                            {networkState.dw1?.map((row, i) => (
                              <div key={i}>
                                <p>Row {i}:</p>
                                {row.map((grad, j) => (
                                  <p key={j}>
                                    W₁[{i}][{j}] ← W₁[{i}][{j}] - {learningRate} × {grad.toFixed(4)}
                                  </p>
                                ))}
                              </div>
                            ))}
                            {networkState.db1 && (
                              <>
                                <p className="mt-1">b₁:</p>
                                {networkState.db1[0].map((grad, i) => (
                                  <p key={i}>
                                    b₁[{i}] ← b₁[{i}] - {learningRate} × {grad.toFixed(4)}
                                  </p>
                                ))}
                              </>
                            )}
                          </div>
                        </div>
                        
                        <div className="border-t border-green-200 pt-2">
                          <p className="font-bold text-green-800">✓ All weights and biases updated!</p>
                          <p className="mt-1">Network is now ready for next iteration</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Column - Controls */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Training Controls</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Input values */}
              <div className="space-y-3">
                <h3 className="font-semibold text-sm">Input Values</h3>
                <div className="space-y-2">
                  <div>
                    <Label className="text-xs">x₁</Label>
                    <Input
                      type="number"
                      step="0.1"
                      value={inputX1}
                      onChange={(e) => setInputX1(parseFloat(e.target.value) || 0)}
                      disabled={isAutoTraining}
                      className="font-mono"
                    />
                  </div>
                  <div>
                    <Label className="text-xs">x₂</Label>
                    <Input
                      type="number"
                      step="0.1"
                      value={inputX2}
                      onChange={(e) => setInputX2(parseFloat(e.target.value) || 0)}
                      disabled={isAutoTraining}
                      className="font-mono"
                    />
                  </div>
                  <div>
                    <Label className="text-xs">Target (y)</Label>
                    <Input
                      type="number"
                      step="0.1"
                      value={targetY}
                      onChange={(e) => setTargetY(parseFloat(e.target.value) || 0)}
                      disabled={isAutoTraining}
                      className="font-mono"
                    />
                  </div>
                </div>
              </div>

              <Separator />

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
                  disabled={isAutoTraining}
                />
              </div>

              <Separator />

              {/* Control Buttons */}
              <div className="space-y-2">
                <Button
                  onClick={handleNextStep}
                  disabled={isAutoTraining}
                  variant="default"
                  className="w-full"
                >
                  <ArrowRightIcon className="w-4 h-4 mr-2" />
                  Next Step
                </Button>
                
                <Button
                  onClick={handleFullStep}
                  disabled={isAutoTraining}
                  variant="outline"
                  className="w-full"
                >
                  <BrainIcon className="w-4 h-4 mr-2" />
                  Full Step (Forward + Backward)
                </Button>

                <Button
                  onClick={() => setIsAutoTraining(!isAutoTraining)}
                  variant={isAutoTraining ? "destructive" : "secondary"}
                  className="w-full"
                >
                  <PlayIcon className="w-4 h-4 mr-2" />
                  {isAutoTraining ? 'Stop Auto-Training' : 'Start Auto-Training'}
                </Button>

                <Button
                  onClick={handleReset}
                  variant="outline"
                  className="w-full"
                >
                  <RotateCcwIcon className="w-4 h-4 mr-2" />
                  Reset Network
                </Button>
              </div>

              <Separator />

              {/* Stats */}
              <div className="space-y-2">
                <h3 className="font-semibold text-sm">Training Stats</h3>
                <div className="space-y-1 font-mono text-xs">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Iterations:</span>
                    <span>{iteration}</span>
                  </div>
                  {networkState.a2 && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Prediction:</span>
                      <span>{networkState.a2[0][0].toFixed(4)}</span>
                    </div>
                  )}
                  {networkState.loss !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Current Loss:</span>
                      <span className="text-red-600">{networkState.loss.toFixed(6)}</span>
                    </div>
                  )}
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
              <p>• How forward pass computes predictions</p>
              <p>• How loss measures prediction error</p>
              <p>• How chain rule enables backpropagation</p>
              <p>• How gradients flow backward through layers</p>
              <p>• How weights update to minimize loss</p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* PyTorch Code Comparison */}
      <BackpropPyTorchComparison />
    </div>
  );
}
