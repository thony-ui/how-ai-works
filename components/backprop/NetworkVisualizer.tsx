"use client";

import React, { useRef, useEffect } from 'react';
import type { NetworkState } from '@/lib/math/network';
import { setupHDCanvas, clearCanvas } from '@/lib/canvas-utils';

interface NetworkVisualizerProps {
  state: NetworkState;
  showGradients?: boolean;
  width?: number;
  height?: number;
}

export function NetworkVisualizer({ 
  state, 
  showGradients = false,
  width = 700,
  height = 400
}: NetworkVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Setup HD canvas
    setupHDCanvas(canvas, width, height);
    clearCanvas(ctx);

    // Define layer positions
    const inputX = 100;
    const hiddenX = 350;
    const outputX = 600;
    const centerY = height / 2;

    // Node radius
    const nodeRadius = 20;

    // Input layer positions (2 nodes)
    const inputNodes = [
      { x: inputX, y: centerY - 40 },
      { x: inputX, y: centerY + 40 }
    ];

    // Hidden layer positions (4 nodes)
    const hiddenNodes = [
      { x: hiddenX, y: centerY - 90 },
      { x: hiddenX, y: centerY - 30 },
      { x: hiddenX, y: centerY + 30 },
      { x: hiddenX, y: centerY + 90 }
    ];

    // Output layer position (1 node)
    const outputNodes = [
      { x: outputX, y: centerY }
    ];

    // Helper to get color based on value
    const getValueColor = (value: number): string => {
      // Normalize to 0-1 for color (clip to reasonable range)
      const normalized = Math.max(-1, Math.min(1, value));
      if (normalized > 0) {
        const intensity = Math.floor(normalized * 200 + 55);
        return `rgb(${intensity}, ${intensity}, 255)`;
      } else {
        const intensity = Math.floor(Math.abs(normalized) * 200 + 55);
        return `rgb(255, ${intensity}, ${intensity})`;
      }
    };

    // Helper to get line width based on weight magnitude
    const getLineWidth = (weight: number): number => {
      return Math.min(5, Math.max(0.5, Math.abs(weight) * 2));
    };

    // Draw connections with weights
    const drawConnections = () => {
      // Input to hidden connections
      state.w1.forEach((inputWeights, i) => {
        inputWeights.forEach((weight, j) => {
          const gradient = showGradients && state.dw1 ? state.dw1[i][j] : 0;
          
          ctx.strokeStyle = showGradients && gradient !== 0
            ? (gradient > 0 ? 'rgba(239, 68, 68, 0.6)' : 'rgba(34, 197, 94, 0.6)')
            : 'rgba(156, 163, 175, 0.3)';
          
          ctx.lineWidth = showGradients && gradient !== 0
            ? Math.min(4, Math.abs(gradient) * 10 + 1)
            : getLineWidth(weight);

          ctx.beginPath();
          ctx.moveTo(inputNodes[i].x + nodeRadius, inputNodes[i].y);
          ctx.lineTo(hiddenNodes[j].x - nodeRadius, hiddenNodes[j].y);
          ctx.stroke();

          // Draw weight value
          if (!showGradients) {
            const midX = (inputNodes[i].x + hiddenNodes[j].x) / 2;
            const midY = (inputNodes[i].y + hiddenNodes[j].y) / 2;
            ctx.fillStyle = '#6b7280';
            ctx.font = '10px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(weight.toFixed(2), midX, midY - 5);
          }
        });
      });

      // Hidden to output connections
      state.w2.forEach((hiddenWeights, i) => {
        hiddenWeights.forEach((weight, j) => {
          const gradient = showGradients && state.dw2 ? state.dw2[i][j] : 0;
          
          ctx.strokeStyle = showGradients && gradient !== 0
            ? (gradient > 0 ? 'rgba(239, 68, 68, 0.6)' : 'rgba(34, 197, 94, 0.6)')
            : 'rgba(156, 163, 175, 0.3)';
          
          ctx.lineWidth = showGradients && gradient !== 0
            ? Math.min(4, Math.abs(gradient) * 10 + 1)
            : getLineWidth(weight);

          ctx.beginPath();
          ctx.moveTo(hiddenNodes[i].x + nodeRadius, hiddenNodes[i].y);
          ctx.lineTo(outputNodes[j].x - nodeRadius, outputNodes[j].y);
          ctx.stroke();

          // Draw weight value
          if (!showGradients) {
            const midX = (hiddenNodes[i].x + outputNodes[j].x) / 2;
            const midY = (hiddenNodes[i].y + outputNodes[j].y) / 2;
            ctx.fillStyle = '#6b7280';
            ctx.font = '10px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(weight.toFixed(2), midX, midY - 5);
          }
        });
      });
    };

    // Draw nodes
    const drawNode = (x: number, y: number, value: number | undefined, label: string) => {
      // Node circle
      ctx.fillStyle = value !== undefined ? getValueColor(value) : '#e5e7eb';
      ctx.beginPath();
      ctx.arc(x, y, nodeRadius, 0, 2 * Math.PI);
      ctx.fill();
      
      ctx.strokeStyle = '#374151';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Value text
      if (value !== undefined) {
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 12px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(value.toFixed(2), x, y);
      }

      // Label
      ctx.fillStyle = '#374151';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, x, y + nodeRadius + 15);
    };

    // Draw all connections first (so they appear behind nodes)
    drawConnections();

    // Draw input layer
    inputNodes.forEach((node, i) => {
      const value = state.input ? state.input[0][i] : undefined;
      const gradient = showGradients && state.dz1 ? state.dz1[0][i] : undefined;
      const displayValue = showGradients && gradient !== undefined ? gradient : value;
      drawNode(node.x, node.y, displayValue, `x${i + 1}`);
    });

    // Draw hidden layer
    hiddenNodes.forEach((node, i) => {
      const value = state.a1 ? state.a1[0][i] : undefined;
      const gradient = showGradients && state.da1 ? state.da1[0][i] : undefined;
      const displayValue = showGradients && gradient !== undefined ? gradient : value;
      drawNode(node.x, node.y, displayValue, `h${i + 1}`);
    });

    // Draw output layer
    outputNodes.forEach((node, i) => {
      const value = state.a2 ? state.a2[0][i] : undefined;
      const gradient = showGradients && state.dz2 ? state.dz2[0][i] : undefined;
      const displayValue = showGradients && gradient !== undefined ? gradient : value;
      drawNode(node.x, node.y, displayValue, 'output');
    });

    // Draw layer labels
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    
    ctx.fillText('Input Layer', inputX, 40);
    ctx.fillText('Hidden Layer (ReLU)', hiddenX, 40);
    ctx.fillText('Output Layer', outputX, 40);

    // Draw legend
    if (showGradients) {
      ctx.fillStyle = '#374151';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Red connections: positive gradient', 20, height - 40);
      ctx.fillText('Green connections: negative gradient', 20, height - 20);
    }

  }, [state, showGradients, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="border border-gray-200 rounded-lg bg-white"
      style={{ maxWidth: '100%', height: 'auto' }}
    />
  );
}
