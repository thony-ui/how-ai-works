/**
 * Canvas utilities for high-DPI rendering
 */

/**
 * Set up canvas for high-DPI (Retina) displays
 * Returns the scaling factor used
 */
export function setupHDCanvas(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
): number {
  const dpr = window.devicePixelRatio || 1;

  // Set display size (CSS pixels)
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  // Set actual size in memory (scaled to account for DPI)
  canvas.width = width * dpr;
  canvas.height = height * dpr;

  // Scale all drawing operations by the DPR
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.scale(dpr, dpr);
  }

  return dpr;
}

/**
 * Clear canvas with proper scaling
 */
export function clearCanvas(ctx: CanvasRenderingContext2D): void {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.restore();
}
