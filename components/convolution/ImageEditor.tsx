"use client";

import React from 'react';
import { setupHDCanvas, clearCanvas } from '@/lib/canvas-utils';

interface ImageEditorProps {
  image: number[][];
  onImageChange: (image: number[][]) => void;
  width?: number;
  height?: number;
}

export function ImageEditor({ 
  image, 
  onImageChange,
  width = 300,
  height = 300
}: ImageEditorProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = React.useState(false);

  const rows = image.length;
  const cols = image[0].length;
  const cellWidth = width / cols;
  const cellHeight = height / rows;

  const drawGrid = React.useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Setup HD canvas
    setupHDCanvas(canvas, width, height);
    clearCanvas(ctx);

    // Draw cells
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const value = image[i][j];
        const x = j * cellWidth;
        const y = i * cellHeight;

        // Fill cell based on value (0 = white, 1 = black)
        const grayscale = Math.floor(255 * (1 - value));
        ctx.fillStyle = `rgb(${grayscale}, ${grayscale}, ${grayscale})`;
        ctx.fillRect(x, y, cellWidth, cellHeight);

        // Draw grid lines
        ctx.strokeStyle = '#d1d5db';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellWidth, cellHeight);
      }
    }
  }, [image, rows, cols, cellWidth, cellHeight, width, height]);

  React.useEffect(() => {
    drawGrid();
  }, [drawGrid]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    handleDraw(e);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    handleDraw(e);
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  const handleDraw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const col = Math.floor(x / cellWidth);
    const row = Math.floor(y / cellHeight);

    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      const newImage = image.map(r => [...r]);
      // Toggle or set to 1
      newImage[row][col] = e.shiftKey ? 0 : 1;
      onImageChange(newImage);
    }
  };

  return (
    <div className="space-y-2 flex flex-col items-center">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="border-2 border-gray-300 rounded cursor-crosshair"
        style={{ maxWidth: '100%', height: 'auto' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
      <p className="text-xs text-muted-foreground">
        Click to draw (Shift+Click to erase)
      </p>
    </div>
  );
}
