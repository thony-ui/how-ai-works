"use client";

import React from 'react';
import { Input } from '@/components/ui/input';

interface KernelEditorProps {
  kernel: number[][];
  onKernelChange: (kernel: number[][]) => void;
}

export function KernelEditor({ kernel, onKernelChange }: KernelEditorProps) {
  const handleCellChange = (row: number, col: number, value: string) => {
    const newKernel = kernel.map(r => [...r]);
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      newKernel[row][col] = numValue;
      onKernelChange(newKernel);
    }
  };

  return (
    <div className="space-y-2">
      <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${kernel[0].length}, 1fr)` }}>
        {kernel.map((row, i) =>
          row.map((val, j) => (
            <Input
              key={`${i}-${j}`}
              type="number"
              step="0.1"
              value={val}
              onChange={(e) => handleCellChange(i, j, e.target.value)}
              className="w-16 h-16 text-center font-mono text-sm p-1"
            />
          ))
        )}
      </div>
    </div>
  );
}
