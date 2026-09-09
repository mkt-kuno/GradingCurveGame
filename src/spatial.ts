// ================================================================
// Spatial Hash Grid — O(N) broad-phase collision detection
// ================================================================

export class SpatialHash {
  private cellSize: number;
  private invCell: number;
  private cells: Map<number, number[]>;

  constructor(cellSize: number) {
    this.cellSize = cellSize;
    this.invCell = 1 / cellSize;
    this.cells = new Map();
  }

  private hashKey(cx: number, cy: number): number {
    return ((cx * 92837111) ^ (cy * 689287499)) | 0;
  }

  build(particles: Array<{ x: number; y: number; radius: number }>) {
    this.cells.clear();
    for (let i = 0; i < particles.length; i++) {
      const { x, y, radius } = particles[i];
      const minCX = Math.floor((x - radius) * this.invCell);
      const maxCX = Math.floor((x + radius) * this.invCell);
      const minCY = Math.floor((y - radius) * this.invCell);
      const maxCY = Math.floor((y + radius) * this.invCell);
      for (let cx = minCX; cx <= maxCX; cx++) {
        for (let cy = minCY; cy <= maxCY; cy++) {
          const k = this.hashKey(cx, cy);
          let cell = this.cells.get(k);
          if (!cell) { cell = []; this.cells.set(k, cell); }
          cell.push(i);
        }
      }
    }
  }

  findPairs(particles: Array<{ x: number; y: number; radius: number }>): [number, number][] {
    const n = particles.length;
    const pairs: [number, number][] = [];
    const checked = new Set<number>();
    for (let i = 0; i < n; i++) {
      const { x, y, radius } = particles[i];
      const minCX = Math.floor((x - radius) * this.invCell);
      const maxCX = Math.floor((x + radius) * this.invCell);
      const minCY = Math.floor((y - radius) * this.invCell);
      const maxCY = Math.floor((y + radius) * this.invCell);
      for (let cx = minCX; cx <= maxCX; cx++) {
        for (let cy = minCY; cy <= maxCY; cy++) {
          const k = this.hashKey(cx, cy);
          const cell = this.cells.get(k);
          if (!cell) continue;
          for (const j of cell) {
            if (j <= i) continue;
            const pk = i * n + j;
            if (checked.has(pk)) continue;
            checked.add(pk);
            const dx = particles[j].x - x;
            const dy = particles[j].y - y;
            const minDist = radius + particles[j].radius;
            if (dx * dx + dy * dy < minDist * minDist) {
              pairs.push([i, j]);
            }
          }
        }
      }
    }
    // Sort to preserve deterministic Gauss-Seidel-like iteration order
    pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    return pairs;
  }

  findNearbyIndices(
    particles: Array<{ x: number; y: number; radius: number }>,
    index: number,
    extraRadius = 0,
  ): number[] {
    const p = particles[index];
    const reach = p.radius + extraRadius;
    const minCX = Math.floor((p.x - reach) * this.invCell);
    const maxCX = Math.floor((p.x + reach) * this.invCell);
    const minCY = Math.floor((p.y - reach) * this.invCell);
    const maxCY = Math.floor((p.y + reach) * this.invCell);
    const nearby: number[] = [];
    const seen = new Set<number>();
    for (let cx = minCX; cx <= maxCX; cx++) {
      for (let cy = minCY; cy <= maxCY; cy++) {
        const cell = this.cells.get(this.hashKey(cx, cy));
        if (!cell) continue;
        for (const j of cell) {
          if (j === index || seen.has(j)) continue;
          seen.add(j);
          nearby.push(j);
        }
      }
    }
    nearby.sort((a, b) => a - b);
    return nearby;
  }
}
