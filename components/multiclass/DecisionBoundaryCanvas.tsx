"use client";

import React, { useEffect, useRef } from 'react';
import type { MultiClassDataPoint, NetworkWeights } from '@/lib/math/multiclass';
import { forwardPass } from '@/lib/math/multiclass';

interface DecisionBoundaryCanvasProps {
  dataset: MultiClassDataPoint[];
  weights: NetworkWeights;
  onPointClick: (point: MultiClassDataPoint) => void;
  selectedPoint: MultiClassDataPoint | null;
}

const COLORS = [
  '#ef4444', // red
  '#3b82f6', // blue
  '#22c55e', // green
  '#f59e0b', // orange
  '#a855f7', // purple
];

export function DecisionBoundaryCanvas({
  dataset,
  weights,
  onPointClick,
  selectedPoint,
}: DecisionBoundaryCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set up HD canvas
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const scale = Math.min(width, height) / 12;

    // Clear canvas
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, height);

    // Transform coordinates from data space to canvas space
    const toCanvasX = (x: number) => centerX + x * scale;
    const toCanvasY = (y: number) => centerY - y * scale;
    const toDataX = (px: number) => (px - centerX) / scale;
    const toDataY = (py: number) => -(py - centerY) / scale;

    // Draw decision boundary background
    const resolution = 3;
    for (let px = 0; px < width; px += resolution) {
      for (let py = 0; py < height; py += resolution) {
        const x = toDataX(px);
        const y = toDataY(py);
        const { a2 } = forwardPass([x, y], weights);
        const predictedClass = a2.indexOf(Math.max(...a2));
        const confidence = Math.max(...a2);

        ctx.fillStyle = COLORS[predictedClass] + Math.floor(confidence * 30).toString(16).padStart(2, '0');
        ctx.fillRect(px, py, resolution, resolution);
      }
    }

    // Draw axes
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.stroke();

    // Draw axis labels
    ctx.fillStyle = '#666';
    ctx.font = '10px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('x', width - 10, centerY - 5);
    ctx.fillText('y', centerX + 5, 15);

    // Draw data points
    dataset.forEach((point) => {
      const px = toCanvasX(point.x);
      const py = toCanvasY(point.y);

      // Determine if correctly classified
      const { a2 } = forwardPass([point.x, point.y], weights);
      const predictedClass = a2.indexOf(Math.max(...a2));
      const isCorrect = predictedClass === point.class;

      // Draw point
      ctx.beginPath();
      ctx.arc(px, py, isCorrect ? 5 : 6, 0, 2 * Math.PI);
      ctx.fillStyle = COLORS[point.class];
      ctx.fill();

      // Draw border (white for correct, red for incorrect)
      ctx.strokeStyle = isCorrect ? '#fff' : '#dc2626';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Highlight selected point
      if (selectedPoint && point === selectedPoint) {
        ctx.beginPath();
        ctx.arc(px, py, 10, 0, 2 * Math.PI);
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 3;
        ctx.stroke();
      }
    });

    // Handle clicks
    const handleClick = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;

      // Find closest point
      let closestPoint: MultiClassDataPoint | null = null;
      let closestDist = Infinity;

      dataset.forEach((point) => {
        const pointPx = toCanvasX(point.x);
        const pointPy = toCanvasY(point.y);
        const dist = Math.sqrt((px - pointPx) ** 2 + (py - pointPy) ** 2);
        if (dist < 15 && dist < closestDist) {
          closestDist = dist;
          closestPoint = point;
        }
      });

      if (closestPoint) {
        onPointClick(closestPoint);
      }
    };

    canvas.addEventListener('click', handleClick);
    return () => canvas.removeEventListener('click', handleClick);
  }, [dataset, weights, onPointClick, selectedPoint]);

  return (
    <div style={{ maxWidth: '600px', width: '100%', aspectRatio: '1', position: 'relative' }}>
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-pointer border rounded"
        style={{ position: 'absolute', inset: 0 }}
      />
    </div>
  );
}
