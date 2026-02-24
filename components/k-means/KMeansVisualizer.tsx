"use client";

import React, { useRef, useEffect } from 'react';
import { Point, Centroid } from '@/lib/math/k-means';

interface KMeansVisualizerProps {
  points: Point[];
  centroids: Centroid[];
  width?: number;
  height?: number;
}

export function KMeansVisualizer({
  points,
  centroids,
  width = 600,
  height = 500
}: KMeansVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const xMin = -6;
  const xMax = 6;
  const yMin = -6;
  const yMax = 6;

  const clusterColors = [
    'rgb(239, 68, 68)',    // red
    'rgb(59, 130, 246)',   // blue
    'rgb(34, 197, 94)',    // green
    'rgb(251, 191, 36)',   // yellow
    'rgb(168, 85, 247)',   // purple
    'rgb(236, 72, 153)',   // pink
    'rgb(20, 184, 166)',   // teal
    'rgb(249, 115, 22)',   // orange
  ];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = 'rgba(200, 200, 200, 0.3)';
    ctx.lineWidth = 0.5;

    // Vertical lines
    for (let x = 0; x <= width; x += width / 12) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Horizontal lines
    for (let y = 0; y <= height; y += height / 12) {
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

    // Helper function to convert coordinates
    const toCanvasX = (x: number) => ((x - xMin) / (xMax - xMin)) * width;
    const toCanvasY = (y: number) => height - ((y - yMin) / (yMax - yMin)) * height;

    // Draw lines from points to their centroids
    ctx.strokeStyle = 'rgba(150, 150, 150, 0.2)';
    ctx.lineWidth = 1;
    points.forEach(point => {
      if (point.cluster !== undefined) {
        const centroid = centroids[point.cluster];
        if (centroid) {
          ctx.beginPath();
          ctx.moveTo(toCanvasX(point.x), toCanvasY(point.y));
          ctx.lineTo(toCanvasX(centroid.x), toCanvasY(centroid.y));
          ctx.stroke();
        }
      }
    });

    // Draw data points
    points.forEach(point => {
      const x = toCanvasX(point.x);
      const y = toCanvasY(point.y);

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      
      if (point.cluster !== undefined) {
        ctx.fillStyle = clusterColors[point.cluster % clusterColors.length];
      } else {
        ctx.fillStyle = 'gray';
      }
      
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // Draw centroids
    centroids.forEach(centroid => {
      const x = toCanvasX(centroid.x);
      const y = toCanvasY(centroid.y);

      // Draw outer glow
      ctx.beginPath();
      ctx.arc(x, y, 12, 0, 2 * Math.PI);
      ctx.fillStyle = clusterColors[centroid.cluster % clusterColors.length] + '40';
      ctx.fill();

      // Draw centroid marker (X shape)
      ctx.strokeStyle = clusterColors[centroid.cluster % clusterColors.length];
      ctx.lineWidth = 3;
      
      // Draw X
      ctx.beginPath();
      ctx.moveTo(x - 6, y - 6);
      ctx.lineTo(x + 6, y + 6);
      ctx.moveTo(x + 6, y - 6);
      ctx.lineTo(x - 6, y + 6);
      ctx.stroke();

      // Draw white border
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 5;
      ctx.globalCompositeOperation = 'destination-over';
      ctx.beginPath();
      ctx.moveTo(x - 6, y - 6);
      ctx.lineTo(x + 6, y + 6);
      ctx.moveTo(x + 6, y - 6);
      ctx.lineTo(x - 6, y + 6);
      ctx.stroke();
      ctx.globalCompositeOperation = 'source-over';
    });

    // Draw axis labels
    ctx.fillStyle = 'black';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    
    // X-axis labels
    for (let i = 0; i <= 6; i++) {
      const x = (i / 6) * width;
      const value = xMin + (i / 6) * (xMax - xMin);
      ctx.fillText(value.toFixed(1), x, yAxisPos + 20);
    }

    // Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 6; i++) {
      const y = height - (i / 6) * height;
      const value = yMin + (i / 6) * (yMax - yMin);
      ctx.fillText(value.toFixed(1), xAxisPos - 10, y + 5);
    }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, centroids, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="border rounded-lg"
    />
  );
}
