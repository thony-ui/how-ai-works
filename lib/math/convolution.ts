/**
 * Convolution operations for CNN visualization
 * Implements 2D convolution with step-by-step tracking
 */

export type Image2D = number[][];
export type Kernel = number[][];

export interface ConvolutionStep {
  row: number;
  col: number;
  window: number[][];
  dotProduct: number;
  calculation: string;
}

/**
 * Perform 2D convolution
 */
export function convolve2D(
  image: Image2D,
  kernel: Kernel,
  stride: number = 1,
): Image2D {
  const imageH = image.length;
  const imageW = image[0].length;
  const kernelH = kernel.length;
  const kernelW = kernel[0].length;

  const outputH = Math.floor((imageH - kernelH) / stride) + 1;
  const outputW = Math.floor((imageW - kernelW) / stride) + 1;

  const output: Image2D = Array(outputH)
    .fill(0)
    .map(() => Array(outputW).fill(0));

  for (let i = 0; i < outputH; i++) {
    for (let j = 0; j < outputW; j++) {
      const rowStart = i * stride;
      const colStart = j * stride;

      let sum = 0;
      for (let ki = 0; ki < kernelH; ki++) {
        for (let kj = 0; kj < kernelW; kj++) {
          sum += image[rowStart + ki][colStart + kj] * kernel[ki][kj];
        }
      }

      output[i][j] = sum;
    }
  }

  return output;
}

/**
 * Get convolution steps for visualization
 */
export function getConvolutionSteps(
  image: Image2D,
  kernel: Kernel,
  stride: number = 1,
): ConvolutionStep[] {
  const steps: ConvolutionStep[] = [];
  const imageH = image.length;
  const imageW = image[0].length;
  const kernelH = kernel.length;
  const kernelW = kernel[0].length;

  const outputH = Math.floor((imageH - kernelH) / stride) + 1;
  const outputW = Math.floor((imageW - kernelW) / stride) + 1;

  for (let i = 0; i < outputH; i++) {
    for (let j = 0; j < outputW; j++) {
      const rowStart = i * stride;
      const colStart = j * stride;

      // Extract the window
      const window: number[][] = [];
      const calculations: string[] = [];
      let sum = 0;

      for (let ki = 0; ki < kernelH; ki++) {
        const windowRow: number[] = [];
        for (let kj = 0; kj < kernelW; kj++) {
          const imageVal = image[rowStart + ki][colStart + kj];
          const kernelVal = kernel[ki][kj];
          windowRow.push(imageVal);

          const product = imageVal * kernelVal;
          sum += product;
          calculations.push(
            `(${imageVal.toFixed(1)} × ${kernelVal.toFixed(1)})`,
          );
        }
        window.push(windowRow);
      }

      steps.push({
        row: i,
        col: j,
        window,
        dotProduct: sum,
        calculation: calculations.join(" + ") + ` = ${sum.toFixed(3)}`,
      });
    }
  }

  return steps;
}

/**
 * Common convolution kernels
 */
export const KERNELS = {
  edgeDetection: [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1],
  ],
  sharpen: [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ],
  blur: [
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
  ],
  emboss: [
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2],
  ],
  identity: [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
  ],
  horizontalEdge: [
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1],
  ],
  verticalEdge: [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
  ],
};

/**
 * Normalize values to 0-1 range
 */
export function normalize(image: Image2D): Image2D {
  const flat = image.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const range = max - min;

  if (range === 0) return image;

  return image.map((row) => row.map((val) => (val - min) / range));
}
