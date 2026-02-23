"use client";

import React, { useEffect, useRef } from 'react';

interface ProbabilityChartProps {
  probabilities: number[];
  targetClass: number;
  logits: number[];
}

export function ProbabilityChart({ probabilities, targetClass, logits }: ProbabilityChartProps) {
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

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Settings
    const padding = 60;
    const chartHeight = height - padding * 2;
    const chartWidth = width - padding * 2;
    const barWidth = Math.min(80, chartWidth / probabilities.length - 20);
    const spacing = chartWidth / probabilities.length;

    // Draw title
    ctx.font = 'bold 14px system-ui';
    ctx.fillStyle = '#000';
    ctx.textAlign = 'center';
    ctx.fillText('Softmax Probabilities', width / 2, 20);

    // Draw axes
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Draw y-axis labels and grid lines
    ctx.font = '12px system-ui';
    ctx.textAlign = 'right';
    ctx.fillStyle = '#666';
    for (let i = 0; i <= 5; i++) {
      const y = padding + chartHeight - (i / 5) * chartHeight;
      const value = (i / 5);
      
      // Label
      ctx.fillText(value.toFixed(1), padding - 10, y + 4);
      
      // Grid line
      ctx.strokeStyle = '#eee';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Draw bars
    probabilities.forEach((prob, i) => {
      const x = padding + spacing * i + spacing / 2 - barWidth / 2;
      const barHeight = prob * chartHeight;
      const y = height - padding - barHeight;

      // Determine color
      const isTarget = i === targetClass;
      const isPredicted = prob === Math.max(...probabilities);
      
      let color = '#94a3b8'; // default gray
      if (isTarget && isPredicted) {
        color = '#22c55e'; // green - correct prediction
      } else if (isTarget) {
        color = '#3b82f6'; // blue - target class
      } else if (isPredicted) {
        color = '#f59e0b'; // orange - predicted but wrong
      }

      // Bar
      ctx.fillStyle = color;
      ctx.fillRect(x, y, barWidth, barHeight);

      // Probability value on top of bar
      ctx.font = 'bold 12px monospace';
      ctx.fillStyle = '#000';
      ctx.textAlign = 'center';
      ctx.fillText((prob * 100).toFixed(1) + '%', x + barWidth / 2, y - 8);

      // Logit value below bar
      ctx.font = '11px monospace';
      ctx.fillStyle = '#666';
      ctx.fillText(`z=${logits[i].toFixed(2)}`, x + barWidth / 2, height - padding + 20);

      // Class label
      ctx.font = 'bold 12px system-ui';
      ctx.fillStyle = '#000';
      ctx.fillText(`Class ${i}`, x + barWidth / 2, height - padding + 35);

      // Badge/indicator
      if (isTarget) {
        ctx.font = 'bold 10px system-ui';
        ctx.fillStyle = '#3b82f6';
        ctx.fillText('TARGET', x + barWidth / 2, y - 24);
      }
      if (isPredicted) {
        ctx.font = 'bold 10px system-ui';
        ctx.fillStyle = isPredicted && isTarget ? '#22c55e' : '#f59e0b';
        ctx.fillText('PREDICTED', x + barWidth / 2, y - 40);
      }
    });

    // Legend
    const legendX = padding;
    const legendY = 40;
    const legendItemHeight = 20;

    ctx.font = '11px system-ui';
    ctx.textAlign = 'left';

    const legendItems = [
      { color: '#22c55e', label: 'Correct Prediction' },
      { color: '#3b82f6', label: 'Target Class' },
      { color: '#f59e0b', label: 'Wrong Prediction' },
      { color: '#94a3b8', label: 'Other Classes' },
    ];

    legendItems.forEach((item, i) => {
      const y = legendY + i * legendItemHeight;
      
      // Color box
      ctx.fillStyle = item.color;
      ctx.fillRect(legendX, y, 12, 12);
      
      // Label
      ctx.fillStyle = '#666';
      ctx.fillText(item.label, legendX + 18, y + 10);
    });

  }, [probabilities, targetClass, logits]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full"
      style={{ width: '100%', height: '450px' }}
    />
  );
}
