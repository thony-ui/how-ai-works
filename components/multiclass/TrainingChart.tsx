"use client";

import React, { useEffect, useRef } from 'react';

interface TrainingChartProps {
  lossHistory: number[];
  accuracyHistory: number[];
}

export function TrainingChart({ lossHistory, accuracyHistory }: TrainingChartProps) {
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
    const padding = 50;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Clear canvas
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, height);

    if (lossHistory.length === 0) {
      ctx.fillStyle = '#999';
      ctx.font = '14px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('Start training to see progress', width / 2, height / 2);
      return;
    }

    // Find max values for scaling
    const maxLoss = Math.max(...lossHistory, 1);
    const maxEpoch = lossHistory.length - 1;

    // Draw grid
    ctx.strokeStyle = '#eee';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Y-axis labels (Loss on left)
    ctx.fillStyle = '#dc2626';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight * (5 - i)) / 5;
      const value = (maxLoss * i) / 5;
      ctx.fillText(value.toFixed(2), padding - 5, y + 4);
    }

    // Y-axis labels (Accuracy on right)
    ctx.fillStyle = '#22c55e';
    ctx.textAlign = 'left';
    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight * (5 - i)) / 5;
      const value = (i / 5) * 100;
      ctx.fillText(value.toFixed(0) + '%', width - padding + 5, y + 4);
    }

    // X-axis labels
    ctx.fillStyle = '#666';
    ctx.textAlign = 'center';
    const epochStep = Math.ceil(maxEpoch / 5);
    for (let i = 0; i <= 5; i++) {
      const epoch = i * epochStep;
      if (epoch <= maxEpoch) {
        const x = padding + (chartWidth * epoch) / maxEpoch;
        ctx.fillText(epoch.toString(), x, height - padding + 20);
      }
    }

    // Axis titles
    ctx.fillStyle = '#000';
    ctx.font = 'bold 12px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', width / 2, height - 10);
    
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#dc2626';
    ctx.fillText('Loss', 0, 0);
    ctx.restore();

    ctx.save();
    ctx.translate(width - 15, height / 2);
    ctx.rotate(Math.PI / 2);
    ctx.fillStyle = '#22c55e';
    ctx.fillText('Accuracy', 0, 0);
    ctx.restore();

    // Draw loss line
    ctx.strokeStyle = '#dc2626';
    ctx.lineWidth = 2;
    ctx.beginPath();
    lossHistory.forEach((loss, i) => {
      const x = padding + (chartWidth * i) / maxEpoch;
      const y = padding + chartHeight - (loss / maxLoss) * chartHeight;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw accuracy line
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    accuracyHistory.forEach((acc, i) => {
      const x = padding + (chartWidth * i) / maxEpoch;
      const y = padding + chartHeight - acc * chartHeight;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Legend
    const legendX = width / 2 - 80;
    const legendY = padding - 35;

    // Loss legend
    ctx.fillStyle = '#dc2626';
    ctx.fillRect(legendX, legendY, 20, 3);
    ctx.fillStyle = '#666';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'left';
    ctx.fillText('Loss', legendX + 25, legendY + 3);

    // Accuracy legend
    ctx.fillStyle = '#22c55e';
    ctx.fillRect(legendX + 80, legendY, 20, 3);
    ctx.fillStyle = '#666';
    ctx.fillText('Accuracy', legendX + 105, legendY + 3);
  }, [lossHistory, accuracyHistory]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full"
      style={{ width: '100%', height: '300px' }}
    />
  );
}
