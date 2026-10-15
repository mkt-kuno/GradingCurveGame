// ================================================================
// Uniform Grid — allocation-free O(N) broad-phase collision detection
// Fixed Int32Array backing stores; zero GC pressure during physics.
// Because build() inserts particle indices in ascending order, scanning
// cells in row-major order yields neighbor candidates grouped
// ascending, preserving a deterministic Gauss-Seidel iteration order.
// ================================================================

import { GAME_W, GAME_H } from './constants';

// Largest radius (231px) / cellSize (500px) -> a particle never spans
// more than 2x2 cells, but keep margin.
const CELL_SPAN = 2;
const SLOTS_PER_CELL = 64;

export class UniformGrid {
  private invCell: number;
  private cols: number;
  private rows: number;
  private counts: Int32Array;
  private entries: Int32Array;

  // Reusable query buffers (no allocation during physics)
  private stamp = 0;
  private seenStamp: Int32Array;
  private pairI: Int32Array;
  private pairJ: Int32Array;
  pairCount = 0;
  private nearBuf: Int32Array;
  nearCount = 0;

  constructor(cellSize: number, maxParticles = 2048) {
    this.invCell = 1 / cellSize;
    this.cols = Math.ceil((GAME_W + cellSize * 2) / cellSize) + 2;
    this.rows = Math.ceil((GAME_H + cellSize * 2) / cellSize) + 2;
    this.counts = new Int32Array(this.cols * this.rows);
    this.entries = new Int32Array(this.cols * this.rows * SLOTS_PER_CELL);
    // Exact pair dedup needs n^2 keys; hard cap keeps memory bounded.
    this.maxParticles = Math.min(maxParticles, 1024);
    this.seenStamp = new Int32Array(this.maxParticles * this.maxParticles);
    this.pairI = new Int32Array(maxParticles * 12);
    this.pairJ = new Int32Array(maxParticles * 12);
    this.nearBuf = new Int32Array(maxParticles);
  }

  private maxParticles: number;

  private cellIndex(cx: number, cy: number): number {
    const gx = cx + 1 < 0 ? 0 : cx + 1 >= this.cols ? this.cols - 1 : cx + 1;
    const gy = cy + 1 < 0 ? 0 : cy + 1 >= this.rows ? this.rows - 1 : cy + 1;
    return gy * this.cols + gx;
  }

  build(px: Float32Array, py: Float32Array, pr: Float32Array, count: number) {
    this.counts.fill(0);
    for (let i = 0; i < count; i++) {
      const x = px[i];
      const y = py[i];
      const r = pr[i];
      const minCX = Math.floor((x - r) * this.invCell);
      const maxCX = Math.floor((x + r) * this.invCell);
      const minCY = Math.floor((y - r) * this.invCell);
      const maxCY = Math.floor((y + r) * this.invCell);
      for (let cy = minCY; cy <= maxCY; cy++) {
        for (let cx = minCX; cx <= maxCX; cx++) {
          const idx = this.cellIndex(cx, cy);
          const slot = this.counts[idx];
          if (slot < SLOTS_PER_CELL) {
            this.entries[idx * SLOTS_PER_CELL + slot] = i;
            this.counts[idx] = slot + 1;
          }
        }
      }
    }
  }

  /**
   * Collect overlapping-or-nearby candidate pairs into pairI/pairJ.
   * Deterministic order: ascending i, and ascending j within each i,
   * because entries are inserted in ascending index order and the
   * dedup stamp keeps only the first (smallest-cell) occurrence.
   */
  collectPairs(px: Float32Array, py: Float32Array, pr: Float32Array, count: number): number {
    this.pairCount = 0;
    this.stamp++;
    const seen = this.seenStamp;
    const stamp = this.stamp;
    for (let i = 0; i < count; i++) {
      const x = px[i];
      const y = py[i];
      const r = pr[i];
      const minCX = Math.floor((x - r) * this.invCell);
      const maxCX = Math.floor((x + r) * this.invCell);
      const minCY = Math.floor((y - r) * this.invCell);
      const maxCY = Math.floor((y + r) * this.invCell);
      for (let cy = minCY; cy <= maxCY; cy++) {
        for (let cx = minCX; cx <= maxCX; cx++) {
          const idx = this.cellIndex(cx, cy);
          const base = idx * SLOTS_PER_CELL;
          const n = this.counts[idx];
          for (let e = 0; e < n; e++) {
            const j = this.entries[base + e];
            if (j <= i) continue;
            const key = i * count + j;
            if (seen[key] === stamp) continue;
            seen[key] = stamp;
            const dx = px[j] - x;
            const dy = py[j] - y;
            const minDist = r + pr[j];
            if (dx * dx + dy * dy < minDist * minDist) {
              this.pairI[this.pairCount] = i;
              this.pairJ[this.pairCount] = j;
              this.pairCount++;
            }
          }
        }
      }
    }
    return this.pairCount;
  }

  getPairI(k: number): number { return this.pairI[k]; }
  getPairJ(k: number): number { return this.pairJ[k]; }

  /** Indices of particles whose cell range overlaps particle `index`. */
  collectNearby(
    px: Float32Array, py: Float32Array, pr: Float32Array,
    count: number, index: number, extraRadius = 0,
  ): number {
    this.nearCount = 0;
    this.stamp++;
    const stamp = this.stamp;
    const x = px[index];
    const y = py[index];
    const reach = pr[index] + extraRadius;
    const minCX = Math.floor((x - reach) * this.invCell);
    const maxCX = Math.floor((x + reach) * this.invCell);
    const minCY = Math.floor((y - reach) * this.invCell);
    const maxCY = Math.floor((y + reach) * this.invCell);
    for (let cy = minCY; cy <= maxCY; cy++) {
      for (let cx = minCX; cx <= maxCX; cx++) {
        const idx = this.cellIndex(cx, cy);
        const base = idx * SLOTS_PER_CELL;
        const n = this.counts[idx];
        for (let e = 0; e < n; e++) {
          const j = this.entries[base + e];
          if (j === index || this.seenStamp[j] === stamp) continue;
          this.seenStamp[j] = stamp;
          this.nearBuf[this.nearCount++] = j;
        }
      }
    }
    return this.nearCount;
  }

  getNearby(k: number): number { return this.nearBuf[k]; }
}
