"use client";

import React, { useRef, useEffect } from 'react';
import { DataPoint } from '@/lib/math/decision-tree';

interface DecisionBoundaryCanvasProps {
  dataPoints: DataPoint[];
  decisionBoundary: number[][];
  width?: number;
  height?: number;
}

export function DecisionBoundaryCanvas({
  dataPoints,
  decisionBoundary,
  width = 600,
  height = 500
}: DecisionBoundaryCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const xMin = -5;
  const xMax = 5;
  const yMin = -5;
  const yMax = 5;

  const colors = [
    'rgba(239, 68, 68, 0.3)',   // red
    'rgba(59, 130, 246, 0.3)',  // blue
    'rgba(34, 197, 94, 0.3)',   // green
    'rgba(251, 191, 36, 0.3)',  // yellow
    'rgba(168, 85, 247, 0.3)',  // purple
    'rgba(236, 72, 153, 0.3)',  // pink
  ];

  const pointColors = [
    'rgb(220, 38, 38)',   // darker red
    'rgb(37, 99, 235)',   // darker blue
    'rgb(22, 163, 74)',   // darker green
    'rgb(217, 119, 6)',   // darker yellow
    'rgb(124, 58, 237)',  // darker purple
    'rgb(219, 39, 119)',  // darker pink
  ];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);

    // Draw decision boundary
    const cellWidth = width / decisionBoundary[0].length;
    const cellHeight = height / decisionBoundary.length;

    for (let i = 0; i < decisionBoundary.length; i++) {
      for (let j = 0; j < decisionBoundary[i].length; j++) {
        const label = decisionBoundary[i][j];
        ctx.fillStyle = colors[label % colors.length];
        ctx.fillRect(
          j * cellWidth,
          (decisionBoundary.length - 1 - i) * cellHeight,
          cellWidth,
          cellHeight
        );
      }
    }

    // Draw grid
    ctx.strokeStyle = 'rgba(200, 200, 200, 0.3)';
    ctx.lineWidth = 0.5;

    // Vertical lines
    for (let x = 0; x <= width; x += width / 10) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Horizontal lines
    for (let y = 0; y <= height; y += height / 10) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.lineWidth = 2;

    // X-axis
    const yAxisPos = height * (yMax / (yMax - yMin));
    ctx.beginPath();
    ctx.moveTo(0, yAxisPos);
    ctx.lineTo(width, yAxisPos);
    ctx.stroke();

    // Y-axis
    const xAxisPos = width * (-xMin / (xMax - xMin));
    ctx.beginPath();
    ctx.moveTo(xAxisPos, 0);
    ctx.lineTo(xAxisPos, height);
    ctx.stroke();

    // Draw data points
    dataPoints.forEach(point => {
      const x = ((point.features[0] - xMin) / (xMax - xMin)) * width;
      const y = height - ((point.features[1] - yMin) / (yMax - yMin)) * height;

      // Draw point with border
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = pointColors[point.label % pointColors.length];
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // Draw axis labels
    ctx.fillStyle = 'black';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    
    // X-axis labels
    for (let i = 0; i <= 5; i++) {
      const x = (i / 5) * width;
      const value = xMin + (i / 5) * (xMax - xMin);
      ctx.fillText(value.toFixed(1), x, yAxisPos + 20);
    }

    // Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const y = height - (i / 5) * height;
      const value = yMin + (i / 5) * (yMax - yMin);
      ctx.fillText(value.toFixed(1), xAxisPos - 10, y + 5);
    }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataPoints, decisionBoundary, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="border rounded-lg"
    />
  );
}
