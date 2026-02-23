"use client";

import React, { useEffect, useRef } from 'react';
import { setupHDCanvas, clearCanvas } from '@/lib/canvas-utils';

interface LossChartProps {
  lossHistory: number[];
  width?: number;
  height?: number;
}

export function LossChart({ lossHistory, width = 400, height = 200 }: LossChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Setup HD canvas
    setupHDCanvas(canvas, width, height);
    clearCanvas(ctx);

    if (lossHistory.length === 0) {
      ctx.fillStyle = '#9ca3af';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No data yet - start training!', width / 2, height / 2);
      return;
    }

    const padding = 40;
    const plotWidth = width - 2 * padding;
    const plotHeight = height - 2 * padding;

    // Draw axes
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.stroke();
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Find min/max for scaling
    const maxLoss = Math.max(...lossHistory);
    const minLoss = Math.min(...lossHistory);
    const lossRange = maxLoss - minLoss || 1;

    // Draw grid
    ctx.strokeStyle = '#f3f4f6';
    ctx.lineWidth = 0.5;
    
    for (let i = 0; i <= 5; i++) {
      const y = padding + (i / 5) * plotHeight;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
      
      // Y-axis labels
      const lossValue = maxLoss - (i / 5) * lossRange;
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(lossValue.toFixed(2), padding - 5, y + 3);
    }

    // Draw loss curve
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();

    lossHistory.forEach((loss, index) => {
      const x = padding + (index / (lossHistory.length - 1 || 1)) * plotWidth;
      const y = height - padding - ((loss - minLoss) / lossRange) * plotHeight;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Draw points
    lossHistory.forEach((loss, index) => {
      const x = padding + (index / (lossHistory.length - 1 || 1)) * plotWidth;
      const y = height - padding - ((loss - minLoss) / lossRange) * plotHeight;
      
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Draw labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Iteration', width / 2, height - 5);
    
    ctx.save();
    ctx.translate(10, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Loss', 0, 0);
    ctx.restore();

  }, [lossHistory, width, height]);

  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      <h3 className="text-sm font-medium mb-2">Loss Over Time</h3>
      <canvas ref={canvasRef} width={width} height={height} />
      {lossHistory.length > 0 && (
        <p className="text-xs text-gray-600 mt-2">
          Current Loss: <span className="font-mono font-medium">{lossHistory[lossHistory.length - 1].toFixed(4)}</span>
        </p>
      )}
    </div>
  );
}
