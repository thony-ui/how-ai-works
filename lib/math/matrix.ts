/**
 * Matrix operations utility
 * Provides core matrix math operations for neural network computations
 */

export type Matrix = number[][];
export type Vector = number[];

/**
 * Create a matrix filled with zeros
 */
export function zeros(rows: number, cols: number): Matrix {
  return Array(rows)
    .fill(0)
    .map(() => Array(cols).fill(0));
}

/**
 * Create a matrix filled with random values between -1 and 1
 */
export function randomMatrix(rows: number, cols: number): Matrix {
  return Array(rows)
    .fill(0)
    .map(() =>
      Array(cols)
        .fill(0)
        .map(() => Math.random() * 2 - 1),
    );
}

/**
 * Matrix multiplication
 */
export function matrixMultiply(a: Matrix, b: Matrix): Matrix {
  const rowsA = a.length;
  const colsA = a[0].length;
  const colsB = b[0].length;

  const result = zeros(rowsA, colsB);

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      for (let k = 0; k < colsA; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return result;
}

/**
 * Element-wise matrix addition
 */
export function matrixAdd(a: Matrix, b: Matrix): Matrix {
  return a.map((row, i) => row.map((val, j) => val + b[i][j]));
}

/**
 * Element-wise matrix subtraction
 */
export function matrixSubtract(a: Matrix, b: Matrix): Matrix {
  return a.map((row, i) => row.map((val, j) => val - b[i][j]));
}

/**
 * Scalar multiplication
 */
export function scalarMultiply(matrix: Matrix, scalar: number): Matrix {
  return matrix.map((row) => row.map((val) => val * scalar));
}

/**
 * Matrix transpose
 */
export function transpose(matrix: Matrix): Matrix {
  return matrix[0].map((_, colIdx) => matrix.map((row) => row[colIdx]));
}

/**
 * Dot product of two vectors
 */
export function dotProduct(a: Vector, b: Vector): number {
  return a.reduce((sum, val, idx) => sum + val * b[idx], 0);
}

/**
 * Convert vector to column matrix
 */
export function vectorToMatrix(vector: Vector): Matrix {
  return vector.map((val) => [val]);
}

/**
 * Convert column matrix to vector
 */
export function matrixToVector(matrix: Matrix): Vector {
  return matrix.map((row) => row[0]);
}
