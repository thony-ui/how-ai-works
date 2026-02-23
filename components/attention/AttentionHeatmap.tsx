"use client";

import React, { useEffect, useRef } from 'react';

interface AttentionHeatmapProps {
  weights: number[][];
  queryLabels: string[];
  keyLabels: string[];
  selectedQueryIdx: number;
  onQuerySelect: (idx: number) => void;
}

export function AttentionHeatmap({
  weights,
  queryLabels,
  keyLabels,
  selectedQueryIdx,
  onQuerySelect,
}: AttentionHeatmapProps) {
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
    const padding = 80;
    const cellWidth = (width - padding * 2) / keyLabels.length;
    const cellHeight = (height - padding * 2) / queryLabels.length;

    // Clear canvas
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, height);

    // Draw title
    ctx.font = 'bold 14px system-ui';
    ctx.fillStyle = '#000';
    ctx.textAlign = 'center';
    ctx.fillText('Attention Weights (Query → Key)', width / 2, 20);

    // Color scale function (white to blue)
    const getColor = (value: number): string => {
      const intensity = Math.floor(value * 255);
      return `rgb(${255 - intensity}, ${255 - intensity / 2}, 255)`;
    };

    // Draw cells
    weights.forEach((row, i) => {
      row.forEach((weight, j) => {
        const x = padding + j * cellWidth;
        const y = padding + i * cellHeight;

        // Cell background
        ctx.fillStyle = getColor(weight);
        ctx.fillRect(x, y, cellWidth, cellHeight);

        // Cell border
        ctx.strokeStyle = i === selectedQueryIdx ? '#000' : '#ccc';
        ctx.lineWidth = i === selectedQueryIdx ? 2 : 0.5;
        ctx.strokeRect(x, y, cellWidth, cellHeight);

        // Weight value
        ctx.fillStyle = weight > 0.5 ? '#fff' : '#000';
        ctx.font = 'bold 11px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          (weight * 100).toFixed(0) + '%',
          x + cellWidth / 2,
          y + cellHeight / 2
        );
      });
    });

    // Draw row labels (queries)
    ctx.fillStyle = '#000';
    ctx.font = '12px system-ui';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    queryLabels.forEach((label, i) => {
      const y = padding + i * cellHeight + cellHeight / 2;
      ctx.fillStyle = i === selectedQueryIdx ? '#3b82f6' : '#000';
      ctx.font = i === selectedQueryIdx ? 'bold 12px system-ui' : '12px system-ui';
      ctx.fillText(`Q: "${label}"`, padding - 10, y);
    });

    // Draw column labels (keys)
    ctx.save();
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    keyLabels.forEach((label, i) => {
      const x = padding + i * cellWidth + cellWidth / 2;
      ctx.translate(x, padding - 10);
      ctx.rotate(-Math.PI / 4);
      ctx.fillStyle = '#000';
      ctx.font = '12px system-ui';
      ctx.fillText(`K: "${label}"`, 0, 0);
      ctx.rotate(Math.PI / 4);
      ctx.translate(-x, -(padding - 10));
    });
    ctx.restore();

    // Color scale legend
    const legendX = width - padding + 20;
    const legendY = padding;
    const legendHeight = height - padding * 2;
    const legendWidth = 20;

    ctx.font = 'bold 10px system-ui';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#666';
    ctx.fillText('Weight', legendX + legendWidth / 2, legendY - 10);

    // Draw gradient
    for (let i = 0; i < legendHeight; i++) {
      const value = 1 - i / legendHeight;
      ctx.fillStyle = getColor(value);
      ctx.fillRect(legendX, legendY + i, legendWidth, 1);
    }

    // Legend border
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);

    // Legend labels
    ctx.font = '10px system-ui';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#666';
    ctx.fillText('1.0', legendX + legendWidth + 5, legendY + 5);
    ctx.fillText('0.5', legendX + legendWidth + 5, legendY + legendHeight / 2);
    ctx.fillText('0.0', legendX + legendWidth + 5, legendY + legendHeight);

    // Handle clicks
    const handleClick = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      // Check if click is within the heatmap
      if (x >= padding && x < width - padding && y >= padding && y < height - padding) {
        const queryIdx = Math.floor((y - padding) / cellHeight);
        if (queryIdx >= 0 && queryIdx < queryLabels.length) {
          onQuerySelect(queryIdx);
        }
      }
    };

    canvas.addEventListener('click', handleClick);
    return () => canvas.removeEventListener('click', handleClick);
  }, [weights, queryLabels, keyLabels, selectedQueryIdx, onQuerySelect]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full cursor-pointer border rounded"
      style={{ width: '700px', height: '500px' }}
    />
  );
}
