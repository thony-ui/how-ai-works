"use client";

import React from 'react';
import type { ConvolutionStep } from '@/lib/math/convolution';
import { setupHDCanvas, clearCanvas } from '@/lib/canvas-utils';

interface ConvolutionVisualizerProps {
  image: number[][];
  kernel: number[][];
  currentStep: number;
  steps: ConvolutionStep[];
  featureMap: number[][];
  width?: number;
  height?: number;
}

export function ConvolutionVisualizer({
  image,
  kernel,
  currentStep,
  steps,
  featureMap,
  width = 900,
  height = 350
}: ConvolutionVisualizerProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Setup HD canvas
    setupHDCanvas(canvas, width, height);
    clearCanvas(ctx);

    const padding = 30;
    const imageSize = 220;
    const kernelSize = 120;
    const outputSize = 180;

    // Draw image
    const imageCellSize = imageSize / image.length;
    for (let i = 0; i < image.length; i++) {
      for (let j = 0; j < image[0].length; j++) {
        const x = padding + j * imageCellSize;
        const y = padding + i * imageCellSize;
        const value = image[i][j];
        
        // Highlight current window
        let highlighted = false;
        if (currentStep < steps.length) {
          const step = steps[currentStep];
          const kernelH = kernel.length;
          const kernelW = kernel[0].length;
          
          if (i >= step.row && i < step.row + kernelH &&
              j >= step.col && j < step.col + kernelW) {
            highlighted = true;
          }
        }
        
        const grayscale = Math.floor(255 * (1 - value));
        ctx.fillStyle = highlighted 
          ? `rgba(59, 130, 246, ${0.3 + value * 0.7})`
          : `rgb(${grayscale}, ${grayscale}, ${grayscale})`;
        ctx.fillRect(x, y, imageCellSize, imageCellSize);
        
        ctx.strokeStyle = highlighted ? '#2563eb' : '#d1d5db';
        ctx.lineWidth = highlighted ? 2 : 1;
        ctx.strokeRect(x, y, imageCellSize, imageCellSize);
      }
    }

    // Draw image label
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Input Image', padding + imageSize / 2, padding - 5);

    // Draw kernel
    const kernelX = padding + imageSize + 50;
    const kernelCellSize = kernelSize / kernel.length;
    for (let i = 0; i < kernel.length; i++) {
      for (let j = 0; j < kernel[0].length; j++) {
        const x = kernelX + j * kernelCellSize;
        const y = padding + 50 + i * kernelCellSize;
        const value = kernel[i][j];
        
        // Color based on value
        const hue = value > 0 ? 210 : 0;
        const saturation = Math.min(100, Math.abs(value) * 100);
        ctx.fillStyle = `hsl(${hue}, ${saturation}%, 70%)`;
        ctx.fillRect(x, y, kernelCellSize, kernelCellSize);
        
        ctx.strokeStyle = '#374151';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, kernelCellSize, kernelCellSize);
        
        // Draw value
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 12px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          value.toFixed(1),
          x + kernelCellSize / 2,
          y + kernelCellSize / 2
        );
      }
    }

    // Draw kernel label
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText('Kernel/Filter', kernelX + kernelSize / 2, padding + 35);

    // Draw arrow
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(kernelX + kernelSize + 10, height / 2);
    ctx.lineTo(kernelX + kernelSize + 30, height / 2);
    ctx.stroke();
    
    // Arrow head
    ctx.beginPath();
    ctx.moveTo(kernelX + kernelSize + 30, height / 2);
    ctx.lineTo(kernelX + kernelSize + 25, height / 2 - 5);
    ctx.lineTo(kernelX + kernelSize + 25, height / 2 + 5);
    ctx.closePath();
    ctx.fillStyle = '#6b7280';
    ctx.fill();

    // Draw feature map
    const outputX = kernelX + kernelSize + 50;
    const outputCellSize = outputSize / featureMap.length;
    
    // Normalize feature map for display
    const flatValues = featureMap.flat();
    const minVal = Math.min(...flatValues);
    const maxVal = Math.max(...flatValues);
    const range = maxVal - minVal || 1;

    for (let i = 0; i < featureMap.length; i++) {
      for (let j = 0; j < featureMap[0].length; j++) {
        const x = outputX + j * outputCellSize;
        const y = padding + i * outputCellSize;
        const value = featureMap[i][j];
        const normalized = (value - minVal) / range;
        
        // Highlight current output cell
        const isCurrentCell = currentStep < steps.length && 
                             steps[currentStep].row === i && 
                             steps[currentStep].col === j;
        
        const grayscale = Math.floor(255 * (1 - normalized));
        ctx.fillStyle = isCurrentCell
          ? '#fbbf24'
          : `rgb(${grayscale}, ${grayscale}, ${grayscale})`;
        ctx.fillRect(x, y, outputCellSize, outputCellSize);
        
        ctx.strokeStyle = isCurrentCell ? '#f59e0b' : '#d1d5db';
        ctx.lineWidth = isCurrentCell ? 2 : 1;
        ctx.strokeRect(x, y, outputCellSize, outputCellSize);
        
        // Draw value if cell is large enough
        if (outputCellSize > 30) {
          ctx.fillStyle = normalized > 0.5 ? '#ffffff' : '#000000';
          ctx.font = '10px monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(
            value.toFixed(1),
            x + outputCellSize / 2,
            y + outputCellSize / 2
          );
        }
      }
    }

    // Draw output label
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText('Feature Map', outputX + outputSize / 2, padding - 5);

    // Draw calculation if step is active
    if (currentStep < steps.length) {
      const step = steps[currentStep];
      const calcY = height - 60;
      
      ctx.fillStyle = '#1f2937';
      ctx.font = '12px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`Position: (${step.row}, ${step.col})`, padding, calcY);
      ctx.fillText(`Result: ${step.dotProduct.toFixed(3)}`, padding, calcY + 20);
    }

  }, [image, kernel, currentStep, steps, featureMap, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="border border-gray-200 rounded-lg bg-white"
    />
  );
}
