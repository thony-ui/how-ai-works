"use client";

import React, { useRef, useEffect } from 'react';

interface ElbowChartProps {
  data: { k: number; inertia: number }[];
  currentK: number;
  width?: number;
  height?: number;
}

export function ElbowChart({
  data,
  currentK,
  width = 600,
  height = 300
}: ElbowChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const padding = 50;
  const chartWidth = width - 2 * padding;
  const chartHeight = height - 2 * padding;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);

    // Find min and max values
    const maxK = Math.max(...data.map(d => d.k));
    const minK = Math.min(...data.map(d => d.k));
    const maxInertia = Math.max(...data.map(d => d.inertia));
    const minInertia = Math.min(...data.map(d => d.inertia));

    // Helper functions
    const getX = (k: number) => padding + ((k - minK) / (maxK - minK)) * chartWidth;
    const getY = (inertia: number) => 
      height - padding - ((inertia - minInertia) / (maxInertia - minInertia)) * chartHeight;

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;

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

    // Draw grid lines
    ctx.strokeStyle = 'rgba(200, 200, 200, 0.5)';
    ctx.lineWidth = 1;

    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding + (i / 5) * chartHeight;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Vertical grid lines
    for (let i = 0; i <= maxK - minK; i++) {
      const x = padding + (i / (maxK - minK)) * chartWidth;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
    }

    // Draw line
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();

    data.forEach((point, index) => {
      const x = getX(point.k);
      const y = getY(point.inertia);

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Draw data points
    data.forEach(point => {
      const x = getX(point.k);
      const y = getY(point.inertia);

      ctx.beginPath();
      ctx.arc(x, y, point.k === currentK ? 8 : 5, 0, 2 * Math.PI);
      ctx.fillStyle = point.k === currentK ? '#ef4444' : '#3b82f6';
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // Draw axis labels
    ctx.fillStyle = '#333';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';

    // X-axis labels
    for (let k = minK; k <= maxK; k++) {
      const x = getX(k);
      ctx.fillText(k.toString(), x, height - padding + 25);
    }

    // Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const inertia = minInertia + (i / 5) * (maxInertia - minInertia);
      const y = height - padding - (i / 5) * chartHeight;
      ctx.fillText(inertia.toFixed(0), padding - 10, y + 5);
    }

    // Draw axis titles
    ctx.fillStyle = '#333';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';

    // X-axis title
    ctx.fillText('Number of Clusters (K)', width / 2, height - 5);

    // Y-axis title (rotated)
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Inertia (Within-Cluster Sum of Squares)', 0, 0);
    ctx.restore();

  }, [data, currentK, width, height, chartWidth, chartHeight]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="border rounded-lg"
    />
  );
}
