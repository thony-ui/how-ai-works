"use client";

import React, { useRef, useEffect } from 'react';
import { TreeNode } from '@/lib/math/decision-tree';

interface DecisionTreeVisualizerProps {
  tree: TreeNode;
  width?: number;
  height?: number;
}

interface TreeLayout {
  x: number;
  y: number;
  node: TreeNode;
}

export function DecisionTreeVisualizer({ 
  tree, 
  width = 800, 
  height = 600 
}: DecisionTreeVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodeRadius = 30;
  const levelHeight = 100;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Calculate tree layout
    const calculateLayout = (
      node: TreeNode,
      x: number,
      y: number,
      width: number,
      depth: number = 0
    ): TreeLayout[] => {
      const layout: TreeLayout[] = [];
      
      layout.push({ x, y, node });
      
      if (node.left && node.right) {
        const childWidth = width / 2;
        const childY = y + levelHeight;
        
        layout.push(...calculateLayout(node.left, x - childWidth / 2, childY, childWidth, depth + 1));
        layout.push(...calculateLayout(node.right, x + childWidth / 2, childY, childWidth, depth + 1));
      }
      
      return layout;
    };

    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);

    const layout = calculateLayout(tree, width / 2, 50, width * 0.8);
    
    // Find connections
    const connections: { from: TreeLayout; to: TreeLayout; isLeft: boolean }[] = [];
    layout.forEach(item => {
      if (item.node.left) {
        const leftChild = layout.find(l => l.node === item.node.left);
        if (leftChild) {
          connections.push({ from: item, to: leftChild, isLeft: true });
        }
      }
      if (item.node.right) {
        const rightChild = layout.find(l => l.node === item.node.right);
        if (rightChild) {
          connections.push({ from: item, to: rightChild, isLeft: false });
        }
      }
    });

    // Draw connections
    connections.forEach(conn => {
      ctx.beginPath();
      ctx.moveTo(conn.from.x, conn.from.y + nodeRadius);
      ctx.lineTo(conn.to.x, conn.to.y - nodeRadius);
      ctx.strokeStyle = conn.isLeft ? '#3b82f6' : '#ef4444';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw edge label
      ctx.fillStyle = conn.isLeft ? '#3b82f6' : '#ef4444';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        conn.isLeft ? '≤' : '>',
        (conn.from.x + conn.to.x) / 2,
        (conn.from.y + conn.to.y) / 2
      );
    });

    // Draw nodes
    layout.forEach(item => {
      const isLeaf = item.node.value !== undefined;

      // Draw circle
      ctx.beginPath();
      ctx.arc(item.x, item.y, nodeRadius, 0, 2 * Math.PI);
      ctx.fillStyle = isLeaf ? '#10b981' : '#8b5cf6';
      ctx.fill();
      ctx.strokeStyle = '#1f2937';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw text
      ctx.fillStyle = 'white';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      if (isLeaf) {
        ctx.fillText('Class', item.x, item.y - 7);
        ctx.fillText(String(item.node.value), item.x, item.y + 7);
      } else {
        ctx.fillText(`X${item.node.featureIndex}`, item.x, item.y - 7);
        ctx.fillText(item.node.threshold?.toFixed(2) || '', item.x, item.y + 7);
      }

      // Draw gini and samples info
      ctx.fillStyle = '#6b7280';
      ctx.font = '9px sans-serif';
      ctx.fillText(
        `entropy: ${item.node.entropy?.toFixed(3)}`,
        item.x,
        item.y + nodeRadius + 15
      );
      ctx.fillText(
        `n: ${item.node.samples}`,
        item.x,
        item.y + nodeRadius + 27
      );
    });
  }, [tree, width, height]);
  
  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="border rounded-lg bg-white"
    />
  );
}
