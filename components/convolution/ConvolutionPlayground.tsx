"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  convolve2D, 
  getConvolutionSteps, 
  KERNELS,
  type Image2D,
  type Kernel
} from '@/lib/math/convolution';
import { ImageEditor } from './ImageEditor';
import { KernelEditor } from './KernelEditor';
import { ConvolutionVisualizer } from './ConvolutionVisualizer';
import { ConvolutionPyTorchComparison } from './ConvolutionPyTorchComparison';
import { 
  PlayIcon, 
  PauseIcon, 
  RotateCcwIcon, 
  SkipForwardIcon,
  SkipBackIcon,
  WandSparklesIcon
} from 'lucide-react';

export default function ConvolutionPlayground() {
  // Image state (5x5 grid)
  const [image, setImage] = useState<Image2D>([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
  ]);

  // Kernel state (3x3)
  const [kernel, setKernel] = useState<Kernel>(KERNELS.edgeDetection);

  // Convolution state
  const [featureMap, setFeatureMap] = useState<Image2D>([]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [steps, setSteps] = useState<any[]>([]);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  // Compute convolution whenever image or kernel changes
  useEffect(() => {
    const result = convolve2D(image, kernel);
    const convSteps = getConvolutionSteps(image, kernel);
    
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setFeatureMap(result);
    setSteps(convSteps);
    setCurrentStep(0);
    setIsPlaying(false);
  }, [image, kernel]);

  // Auto-play through steps
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= steps.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 500);

    return () => clearInterval(interval);
  }, [isPlaying, steps.length]);

  const handleNextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(prev => prev + 1);
    }
  };

  const handlePrevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  };

  const handleReset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  const applyPresetKernel = (kernelName: keyof typeof KERNELS) => {
    setKernel(KERNELS[kernelName]);
  };

  const clearImage = () => {
    setImage(Array(5).fill(0).map(() => Array(5).fill(0)));
  };

  const createSampleImage = (pattern: string) => {
    switch (pattern) {
      case 'vertical':
        setImage([
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0]
        ]);
        break;
      case 'horizontal':
        setImage([
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]
        ]);
        break;
      case 'diagonal':
        setImage([
          [1, 0, 0, 0, 0],
          [0, 1, 0, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 0, 0, 1]
        ]);
        break;
      case 'cross':
        setImage([
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [1, 1, 1, 1, 1],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0]
        ]);
        break;
      case 'square':
        setImage([
          [0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0]
        ]);
        break;
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Convolution Playground</h1>
        <p className="text-lg text-muted-foreground">
          Explore how 2D convolution works. Draw on the image, modify the kernel, and watch the convolution operation step-by-step.
        </p>
      </div>

      {/* Main Visualization */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <div>
              <CardTitle>Convolution Process</CardTitle>
              <CardDescription>
                Step {currentStep + 1} of {steps.length}
              </CardDescription>
            </div>
            <Badge variant="secondary" className="text-sm">
              Output: {featureMap.length}×{featureMap[0]?.length || 0}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto flex justify-center">
            <ConvolutionVisualizer
              image={image}
              kernel={kernel}
              currentStep={currentStep}
              steps={steps}
              featureMap={featureMap}
            />
          </div>

          {/* Playback Controls */}
          <div className="flex items-center justify-center gap-2 mt-4">
            <Button
              onClick={handlePrevStep}
              disabled={currentStep === 0 || isPlaying}
              variant="outline"
              size="sm"
            >
              <SkipBackIcon className="w-4 h-4" />
            </Button>

            <Button
              onClick={() => setIsPlaying(!isPlaying)}
              disabled={currentStep >= steps.length - 1}
              variant={isPlaying ? "destructive" : "default"}
              size="sm"
            >
              {isPlaying ? (
                <>
                  <PauseIcon className="w-4 h-4 mr-2" />
                  Pause
                </>
              ) : (
                <>
                  <PlayIcon className="w-4 h-4 mr-2" />
                  Play
                </>
              )}
            </Button>

            <Button
              onClick={handleNextStep}
              disabled={currentStep >= steps.length - 1 || isPlaying}
              variant="outline"
              size="sm"
            >
              <SkipForwardIcon className="w-4 h-4" />
            </Button>

            <Button
              onClick={handleReset}
              variant="outline"
              size="sm"
            >
              <RotateCcwIcon className="w-4 h-4" />
            </Button>
          </div>

          {/* Current Calculation */}
          {currentStep < steps.length && (
            <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <h3 className="font-semibold text-sm mb-2 text-blue-900">
                Calculation at position ({steps[currentStep].row}, {steps[currentStep].col}):
              </h3>
              <p className="text-sm font-mono text-blue-800">
                {steps[currentStep].calculation}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Image Editor */}
        <Card>
          <CardHeader>
            <CardTitle>Input Image (5×5)</CardTitle>
            <CardDescription>
              Draw your own pattern or use presets
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <ImageEditor image={image} onImageChange={setImage} />
            
            {/* Legend */}
            <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground">
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 bg-black border border-gray-300 rounded" />
                <span>= 1</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-4 bg-white border border-gray-300 rounded" />
                <span>= 0</span>
              </div>
            </div>
            
            <Separator />
            
            <div className="space-y-2">
              <h3 className="font-semibold text-sm">Sample Patterns</h3>
              <div className="grid grid-cols-3 gap-2">
                <Button
                  onClick={() => createSampleImage('vertical')}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                >
                  Vertical
                </Button>
                <Button
                  onClick={() => createSampleImage('horizontal')}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                >
                  Horizontal
                </Button>
                <Button
                  onClick={() => createSampleImage('diagonal')}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                >
                  Diagonal
                </Button>
                <Button
                  onClick={() => createSampleImage('cross')}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                >
                  Cross
                </Button>
                <Button
                  onClick={() => createSampleImage('square')}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                >
                  Square
                </Button>
                <Button
                  onClick={clearImage}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                >
                  Clear
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Kernel Editor */}
        <Card>
          <CardHeader>
            <CardTitle>Convolution Kernel (3×3)</CardTitle>
            <CardDescription>
              Edit the filter or use common presets
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <KernelEditor kernel={kernel} onKernelChange={setKernel} />
            
            <Separator />
            
            <div className="space-y-2">
              <h3 className="font-semibold text-sm">Preset Filters</h3>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  onClick={() => applyPresetKernel('edgeDetection')}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                >
                  <WandSparklesIcon className="w-3 h-3 mr-1" />
                  Edge Detect
                </Button>
                <Button
                  onClick={() => applyPresetKernel('sharpen')}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                >
                  Sharpen
                </Button>
                <Button
                  onClick={() => applyPresetKernel('blur')}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                >
                  Blur
                </Button>
                <Button
                  onClick={() => applyPresetKernel('emboss')}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                >
                  Emboss
                </Button>
                <Button
                  onClick={() => applyPresetKernel('horizontalEdge')}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                >
                  H-Edge
                </Button>
                <Button
                  onClick={() => applyPresetKernel('verticalEdge')}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                >
                  V-Edge
                </Button>
                <Button
                  onClick={() => applyPresetKernel('identity')}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                >
                  Identity
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* What You're Learning */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">What You&apos;re Learning</CardTitle>
        </CardHeader>
        <CardContent className="grid md:grid-cols-2 gap-4 text-sm text-muted-foreground">
          <div className="space-y-2">
            <p>• How convolution slides a kernel over an image</p>
            <p>• What dot products compute at each position</p>
            <p>• How different kernels detect different features</p>
          </div>
          <div className="space-y-2">
            <p>• Why edge detection kernels work</p>
            <p>• How feature maps are generated</p>
            <p>• The foundation of convolutional neural networks</p>
          </div>
        </CardContent>
      </Card>

      {/* PyTorch Code Comparison */}
      <ConvolutionPyTorchComparison />
    </div>
  );
}
