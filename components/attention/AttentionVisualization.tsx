"use client";

import React, { useEffect, useRef } from 'react';

interface AttentionVisualizationProps {
  tokens: string[];
  weights: number[];
  queryIdx: number;
}

export function AttentionVisualization({
  tokens,
  weights,
  queryIdx,
}: AttentionVisualizationProps) {
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
    const centerY = height / 2;
    const padding = 100;
    const spacing = (width - padding * 2) / (tokens.length - 1);

    // Clear canvas
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, height);

    // Draw title
    ctx.font = 'bold 14px system-ui';
    ctx.fillStyle = '#000';
    ctx.textAlign = 'center';
    ctx.fillText(`Attention Flow from "${tokens[queryIdx]}"`, width / 2, 25);

    // Token positions
    const tokenPositions = tokens.map((_, i) => ({
      x: padding + i * spacing,
      y: centerY,
    }));

    // Draw attention connections
    weights.forEach((weight, i) => {
      if (i === queryIdx) return; // Skip self

      const startPos = tokenPositions[queryIdx];
      const endPos = tokenPositions[i];

      // Draw curved line
      const controlY = weight > 0.2 ? centerY - 80 : centerY - 40;
      const alpha = Math.max(0.1, weight);
      const lineWidth = Math.max(1, weight * 8);

      ctx.strokeStyle = `rgba(59, 130, 246, ${alpha})`;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(startPos.x, startPos.y);
      ctx.quadraticCurveTo(
        (startPos.x + endPos.x) / 2,
        controlY,
        endPos.x,
        endPos.y
      );
      ctx.stroke();

      // Draw arrow head
      const angle = Math.atan2(endPos.y - controlY, endPos.x - (startPos.x + endPos.x) / 2);
      const arrowSize = Math.max(5, weight * 15);
      ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`;
      ctx.beginPath();
      ctx.moveTo(endPos.x, endPos.y);
      ctx.lineTo(
        endPos.x - arrowSize * Math.cos(angle - Math.PI / 6),
        endPos.y - arrowSize * Math.sin(angle - Math.PI / 6)
      );
      ctx.lineTo(
        endPos.x - arrowSize * Math.cos(angle + Math.PI / 6),
        endPos.y - arrowSize * Math.sin(angle + Math.PI / 6)
      );
      ctx.closePath();
      ctx.fill();
    });

    // Draw self-attention loop
    const selfWeight = weights[queryIdx];
    const selfPos = tokenPositions[queryIdx];
    const loopRadius = 30;
    
    ctx.strokeStyle = `rgba(147, 51, 234, ${Math.max(0.2, selfWeight)})`;
    ctx.lineWidth = Math.max(1, selfWeight * 8);
    ctx.beginPath();
    ctx.arc(selfPos.x, selfPos.y - loopRadius, loopRadius, 0, Math.PI, true);
    ctx.stroke();

    // Self-attention arrow
    ctx.fillStyle = `rgba(147, 51, 234, ${Math.max(0.2, selfWeight)})`;
    ctx.beginPath();
    ctx.moveTo(selfPos.x, selfPos.y);
    ctx.lineTo(selfPos.x - 6, selfPos.y - 8);
    ctx.lineTo(selfPos.x + 6, selfPos.y - 8);
    ctx.closePath();
    ctx.fill();

    // Draw tokens
    tokens.forEach((token, i) => {
      const pos = tokenPositions[i];
      const isQuery = i === queryIdx;
      const weight = weights[i];

      // Circle
      const radius = isQuery ? 35 : 30;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);
      
      if (isQuery) {
        ctx.fillStyle = '#3b82f6';
      } else {
        const intensity = Math.floor(weight * 200);
        ctx.fillStyle = `rgb(${255 - intensity}, ${255 - intensity / 2}, 255)`;
      }
      ctx.fill();
      
      ctx.strokeStyle = isQuery ? '#1e40af' : '#94a3b8';
      ctx.lineWidth = isQuery ? 3 : 2;
      ctx.stroke();

      // Token text
      ctx.fillStyle = isQuery ? '#fff' : '#000';
      ctx.font = isQuery ? 'bold 14px system-ui' : '12px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(token, pos.x, pos.y);

      // Weight percentage below
      ctx.fillStyle = '#666';
      ctx.font = 'bold 11px monospace';
      ctx.fillText((weight * 100).toFixed(0) + '%', pos.x, pos.y + radius + 15);

      // Position label above
      ctx.fillStyle = '#999';
      ctx.font = '10px system-ui';
      ctx.fillText(`Pos ${i}`, pos.x, pos.y - radius - 8);
    });

    // Legend
    const legendY = height - 40;
    const legendX = padding;

    ctx.font = '11px system-ui';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#666';

    // Query indicator
    ctx.fillStyle = '#3b82f6';
    ctx.fillRect(legendX, legendY, 15, 15);
    ctx.fillStyle = '#666';
    ctx.fillText('Query Token', legendX + 20, legendY + 11);

    // Self-attention indicator
    ctx.strokeStyle = '#9333ea';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(legendX + 150, legendY + 7, 10, 0, Math.PI, true);
    ctx.stroke();
    ctx.fillStyle = '#666';
    ctx.fillText('Self-Attention', legendX + 170, legendY + 11);

    // Cross-attention indicator
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(legendX + 320, legendY);
    ctx.lineTo(legendX + 340, legendY + 15);
    ctx.stroke();
    ctx.fillStyle = '#666';
    ctx.fillText('Attention Flow', legendX + 350, legendY + 11);
  }, [tokens, weights, queryIdx]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full border rounded"
      style={{ width: '900px', height: '300px' }}
    />
  );
}
