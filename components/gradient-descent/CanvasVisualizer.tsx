"use client";

import React, { useRef, useEffect } from 'react';
import { setupHDCanvas, clearCanvas } from '@/lib/canvas-utils';

interface Point {
  x: number;
  y: number;
}

interface CanvasVisualizerProps {
  points: Point[];
  w: number;
  b: number;
  onPointDrag?: (index: number, x: number, y: number) => void;
  width?: number;
  height?: number;
}

export function CanvasVisualizer({
  points,
  w,
  b,
  onPointDrag,
  width = 600,
  height = 400
}: CanvasVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [draggingPoint, setDraggingPoint] = React.useState<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Setup HD canvas for crisp rendering
    setupHDCanvas(canvas, width, height);
    clearCanvas(ctx);

    // Set up coordinate system
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

    // Draw grid
    ctx.strokeStyle = '#f3f4f6';
    ctx.lineWidth = 0.5;
    
    for (let i = 0; i <= 10; i++) {
      const x = padding + (i / 10) * plotWidth;
      const y = padding + (i / 10) * plotHeight;
      
      // Vertical lines
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
      
      // Horizontal lines
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Find data range
    const xValues = points.map(p => p.x);
    const yValues = points.map(p => p.y);
    const xMin = Math.min(...xValues, 0);
    const xMax = Math.max(...xValues, 10);
    const yMin = Math.min(...yValues, 0);
    const yMax = Math.max(...yValues, 10);

    // Convert data coordinates to canvas coordinates
    const toCanvasX = (x: number) => {
      return padding + ((x - xMin) / (xMax - xMin)) * plotWidth;
    };
    
    const toCanvasY = (y: number) => {
      return height - padding - ((y - yMin) / (yMax - yMin)) * plotHeight;
    };

    // Draw regression line
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const lineY1 = w * xMin + b;
    const lineY2 = w * xMax + b;
    ctx.moveTo(toCanvasX(xMin), toCanvasY(lineY1));
    ctx.lineTo(toCanvasX(xMax), toCanvasY(lineY2));
    ctx.stroke();

    // Draw residuals (error lines)
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    
    points.forEach(point => {
      const predictedY = w * point.x + b;
      ctx.beginPath();
      ctx.moveTo(toCanvasX(point.x), toCanvasY(point.y));
      ctx.lineTo(toCanvasX(point.x), toCanvasY(predictedY));
      ctx.stroke();
    });
    
    ctx.setLineDash([]);

    // Draw points
    points.forEach((point, index) => {
      const cx = toCanvasX(point.x);
      const cy = toCanvasY(point.y);
      
      ctx.fillStyle = draggingPoint === index ? '#2563eb' : '#3b82f6';
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
      ctx.fill();
      
      // White border
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // Draw axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    
    // X-axis label
    ctx.fillText('X', width / 2, height - 10);
    
    // Y-axis label
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Y', 0, 0);
    ctx.restore();

  }, [points, w, b, draggingPoint, width, height]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onPointDrag) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const padding = 40;
    const plotWidth = width - 2 * padding;
    const plotHeight = height - 2 * padding;

    const xValues = points.map(p => p.x);
    const yValues = points.map(p => p.y);
    const xMin = Math.min(...xValues, 0);
    const xMax = Math.max(...xValues, 10);
    const yMin = Math.min(...yValues, 0);
    const yMax = Math.max(...yValues, 10);

    const toCanvasX = (x: number) => padding + ((x - xMin) / (xMax - xMin)) * plotWidth;
    const toCanvasY = (y: number) => height - padding - ((y - yMin) / (yMax - yMin)) * plotHeight;

    // Check if clicking on a point
    for (let i = 0; i < points.length; i++) {
      const cx = toCanvasX(points[i].x);
      const cy = toCanvasY(points[i].y);
      const distance = Math.sqrt((mouseX - cx) ** 2 + (mouseY - cy) ** 2);
      
      if (distance < 10) {
        setDraggingPoint(i);
        break;
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (draggingPoint === null || !onPointDrag) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const padding = 40;
    const plotWidth = width - 2 * padding;
    const plotHeight = height - 2 * padding;

    const xValues = points.map(p => p.x);
    const yValues = points.map(p => p.y);
    const xMin = Math.min(...xValues, 0);
    const xMax = Math.max(...xValues, 10);
    const yMin = Math.min(...yValues, 0);
    const yMax = Math.max(...yValues, 10);

    // Convert canvas coordinates back to data coordinates
    const dataX = xMin + ((mouseX - padding) / plotWidth) * (xMax - xMin);
    const dataY = yMin + ((height - padding - mouseY) / plotHeight) * (yMax - yMin);

    onPointDrag(draggingPoint, dataX, dataY);
  };

  const handleMouseUp = () => {
    setDraggingPoint(null);
  };

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="border border-gray-200 rounded-lg cursor-pointer bg-white"
      style={{ maxWidth: '100%', height: 'auto' }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    />
  );
}
