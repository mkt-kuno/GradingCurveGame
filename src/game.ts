// ================================================================
// 粒度分布ゲーム - 土質工学メロンゲーム
// DEM Physics + GPU Rendering (WebGPU / Canvas2D fallback)
// Phase 1: CPU optimizations (spatial hash, no shadowBlur, cache, throttled chart)
// Phase 3: WebGPU rendering
// ================================================================

import {
  LEVELS, SIEVE_SIZES, GAME_W, GAME_H, WALL_T, CL, CR, CB, CONTAINER_W, CONTAINER_H,
  DROP_Y, DANGER_Y, GRAVITY, ESTAR_PP, ESTAR_PW, KN_PP, KN_PW,
  MU_PP, MU_PW, REST_PP, REST_PW, MU_ROLL_PP, MU_ROLL_PW,
  SUB_STEPS, MAX_DELTA_RATIO, MAX_VEL, MAX_OMEGA, VEL_DAMP, ANG_DAMP,
  BETA_PP, BETA_PW,
} from './constants';

import { SpatialHash } from './spatial';
import { WebGPURenderer } from './renderer-webgpu';

// ================================================================
// Types
// ================================================================

interface Particle {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  mass: number;
  inertia: number;
  level: number;
  angle: number;
  omega: number;
  active: boolean;
  graceFrames: number;
}

interface ContactVis { x: number; y: number; force: number; }
interface Effect { x: number; y: number; r: number; alpha: number; color: string; }
interface ScorePopup { x: number; y: number; text: string; timer: number; }

type ContactModel = 'hooke' | 'hertz';

// ================================================================
// State
// ================================================================

let particles: Particle[] = [];
let nextId = 0;
let contactVis: ContactVis[] = [];
let effects: Effect[] = [];
let scorePopups: ScorePopup[] = [];
let mergeQueue: [number, number][] = [];
let contactedPairs = new Set<string>();

let score = 0;
let currentLevel = 0;
let nextLevel = 0;
let dropX = GAME_W / 2;
let canDrop = true;
let gameOver = false;
let mergeCount = 0;
let contactModel: ContactModel = 'hooke';
let showForceChains = true;
let comboCount = 0;
let comboTimer = 0;
let lastDroppedId: number | null = null;

// Spatial hash
const spatialHash = new SpatialHash(500);

// Chart dirty flag for throttling
let chartDirty = true;
let lastChartFrame = 0;

// Renderer
let gpuRenderer: WebGPURenderer | null = null;
let rendererName = 'Canvas2D';

// ================================================================
// DOM
// ================================================================

const gameCanvas = document.getElementById('game-canvas') as HTMLCanvasElement;
let gCtx: CanvasRenderingContext2D | null = null;
const chartCanvas = document.getElementById('chart-canvas') as HTMLCanvasElement;
const cCtx = chartCanvas.getContext('2d')!;
const nextCanvas = document.getElementById('next-canvas') as HTMLCanvasElement;
const nCtx = nextCanvas.getContext('2d')!;

const scoreEl = document.getElementById('score-value')!;
const highScoreEl = document.getElementById('high-score')!;
const nextNameEl = document.getElementById('next-name')!;
const gameOverEl = document.getElementById('game-over')!;
const finalScoreEl = document.getElementById('final-score')!;
const restartBtn = document.getElementById('restart-btn')!;
const paramsEl = document.getElementById('grading-params')!;
const btnHooke = document.getElementById('btn-hooke') as HTMLButtonElement;
const btnHertz = document.getElementById('btn-hertz') as HTMLButtonElement;
const chkForces = document.getElementById('chk-forces') as HTMLInputElement;

function getTodayKey(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

function loadHighScore(): number {
  try {
    const data = localStorage.getItem('granularity_highscore');
    if (!data) return 0;
    const parsed = JSON.parse(data) as { date: string; score: number };
    if (parsed.date === getTodayKey()) return parsed.score;
    return 0;
  } catch { return 0; }
}

function saveHighScore(s: number) {
  localStorage.setItem('granularity_highscore', JSON.stringify({ date: getTodayKey(), score: s }));
}

let highScore = loadHighScore();
highScoreEl.textContent = `今日のハイスコア: ${highScore}`;

// ================================================================
// Renderer Detection
// ================================================================

async function initRenderer() {
  // Try WebGPU first
  try {
    if ('gpu' in navigator) {
      const adapter = await (navigator as any).gpu.requestAdapter();
      if (adapter) {
        const device = await adapter.requestDevice();
        gpuRenderer = new WebGPURenderer(gameCanvas, device);
        rendererName = 'WebGPU Enable';
        return;
      }
    }
  } catch (e) {
    console.log('WebGPU not available:', e);
  }

  // Canvas2D fallback
  gCtx = gameCanvas.getContext('2d')!;
  rendererName = 'Canvas2D';
}

// ================================================================
// Particle Cache (Canvas2D fallback)
// ================================================================

const PARTICLE_CACHE: HTMLCanvasElement[] = [];

function hexToRGBA(hex: string, alpha: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 0xFF},${(n >> 8) & 0xFF},${n & 0xFF},${alpha})`;
}

function lighten(hex: string, amt: number): string {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.min(255, ((n >> 16) & 0xFF) + amt);
  const g = Math.min(255, ((n >> 8) & 0xFF) + amt);
  const b = Math.min(255, (n & 0xFF) + amt);
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')}`;
}

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function initParticleCache() {
  if (PARTICLE_CACHE.length > 0) return;
  for (let level = 0; level < LEVELS.length; level++) {
    const info = LEVELS[level];
    const r = info.radius;
    const pad = 10;
    const size = r * 2 + pad * 2;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d')!;
    const cx = size / 2;
    const cy = size / 2;

    // Shadow
    ctx.beginPath();
    ctx.arc(cx + 2, cy + 3, r, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,0,0,0.15)';
    ctx.fill();

    // Gradient
    const grad = ctx.createRadialGradient(cx - r * 0.25, cy - r * 0.25, r * 0.05, cx, cy, r);
    grad.addColorStop(0, lighten(info.color, 35));
    grad.addColorStop(1, info.color);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.strokeStyle = info.strokeColor;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Dots
    const rng = mulberry32(level * 54321 + 7);
    const dotN = Math.min(level * 6 + 4, 27);
    for (let i = 0; i < dotN; i++) {
      const ddx = (rng() - 0.5) * r * 1.4;
      const ddy = (rng() - 0.5) * r * 1.4;
      if (ddx * ddx + ddy * ddy < (r * 0.7) ** 2) {
        ctx.beginPath();
        ctx.arc(cx + ddx, cy + ddy, Math.max(1, r * 0.05), 0, Math.PI * 2);
        ctx.fillStyle = hexToRGBA(info.strokeColor, 0.3);
        ctx.fill();
      }
    }

    PARTICLE_CACHE[level] = canvas;
  }
}

function drawParticleCached(ctx: CanvasRenderingContext2D, x: number, y: number, angle: number, level: number) {
  const cache = PARTICLE_CACHE[level];
  const r = LEVELS[level].radius;
  const pad = 10;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.drawImage(cache, -r - pad, -r - pad);

  // Sieve text
  const fontSize = Math.max(8, Math.floor(r * 0.3));
  ctx.font = `bold ${fontSize}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#3E2723';
  ctx.fillText(LEVELS[level].sieve, 0, 0);
  ctx.restore();
}

// ================================================================
// DEM Physics (CPU with spatial hash)
// ================================================================

function createParticle(x: number, y: number, level: number): Particle {
  const r = LEVELS[level].radius;
  const m = r * r * 0.008;
  return {
    id: nextId++, x, y, vx: 0, vy: 0, radius: r, mass: m,
    inertia: 0.5 * m * r * r, level, angle: 0, omega: 0,
    active: false, graceFrames: 0,
  };
}

function pairKey(a: number, b: number): string {
  return a < b ? `${a}_${b}` : `${b}_${a}`;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function hertzNormalForce(delta: number, rEff: number, eStar: number): number {
  return (4 / 3) * eStar * Math.sqrt(Math.max(rEff, 0.1)) * Math.pow(Math.max(delta, 0), 1.5);
}

function hookeNormalForce(delta: number, kn: number): number {
  return kn * delta;
}

function computeNormalForce(
  delta: number, rEff: number, mEff: number, vn: number,
  eStar: number, kn: number, b: number,
): number {
  if (delta <= 0) return 0;
  let fElastic: number;
  let kEff: number;
  if (contactModel === 'hertz') {
    fElastic = hertzNormalForce(delta, rEff, eStar);
    kEff = 2 * eStar * Math.sqrt(Math.max(rEff * delta, 0.01));
  } else {
    fElastic = hookeNormalForce(delta, kn);
    kEff = kn;
  }
  const eta = 2 * b * Math.sqrt(Math.max(mEff * kEff, 0.001));
  return Math.max(0, fElastic - eta * vn);
}

function physicsStep(dt: number) {
  const newContactVis: ContactVis[] = [];
  const newPairs = new Set<string>();

  // Gravity
  for (const p of particles) {
    p.vy += GRAVITY * dt;
  }

  // Build spatial hash
  spatialHash.build(particles);

  // Particle-Particle contacts via spatial hash
  for (const [i, j] of spatialHash.findPairs(particles)) {
    const a = particles[i];
    const b = particles[j];

    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const distSq = dx * dx + dy * dy;
    const minDist = a.radius + b.radius;
    if (distSq >= minDist * minDist) continue;

    const dist = Math.sqrt(Math.max(distSq, 1e-8));
    let delta = minDist - dist;
    if (delta <= 0) continue;

    const maxDelta = MAX_DELTA_RATIO * Math.min(a.radius, b.radius);
    delta = Math.min(delta, maxDelta);

    const nx = dx / dist;
    const ny = dy / dist;
    const tx = -ny;
    const ty = nx;

    const dvx = b.vx - a.vx;
    const dvy = b.vy - a.vy;
    const vn = dvx * nx + dvy * ny;
    const vSlide = (dvx * tx + dvy * ty) - a.omega * a.radius - b.omega * b.radius;

    const mEff = (a.mass * b.mass) / (a.mass + b.mass);
    const rEff = (a.radius * b.radius) / (a.radius + b.radius);

    const fn = computeNormalForce(delta, rEff, mEff, vn, ESTAR_PP, KN_PP, BETA_PP);

    const kTEffPP = contactModel === 'hertz'
      ? 2 * ESTAR_PP * Math.sqrt(Math.max(rEff * delta, 0.01))
      : KN_PP;
    const ftDamp = 2 * BETA_PP * Math.sqrt(Math.max(mEff * kTEffPP, 0.001));
    const ftMag = Math.min(MU_PP * fn, ftDamp * Math.abs(vSlide));
    const ftSign = vSlide > 0.001 ? -1 : vSlide < -0.001 ? 1 : 0;

    const fx = fn * nx + ftSign * ftMag * tx;
    const fy = fn * ny + ftSign * ftMag * ty;

    b.vx += (fx / b.mass) * dt;
    b.vy += (fy / b.mass) * dt;
    a.vx -= (fx / a.mass) * dt;
    a.vy -= (fy / a.mass) * dt;

    const torqueSign = vSlide > 0.001 ? 1 : vSlide < -0.001 ? -1 : 0;
    const torqueMag = ftMag;

    a.omega += (torqueSign * torqueMag * a.radius / a.inertia) * dt;
    b.omega += (torqueSign * torqueMag * b.radius / b.inertia) * dt;

    const omegaRel = b.omega - a.omega;
    if (Math.abs(omegaRel) > 0.01) {
      const tauRoll = MU_ROLL_PP * rEff * fn;
      const rollSign = omegaRel > 0 ? 1 : -1;
      const rollImpulse = Math.min(tauRoll * dt, Math.abs(omegaRel) * 0.5 * (a.inertia * b.inertia) / (a.inertia + b.inertia));
      a.omega += rollSign * rollImpulse / a.inertia;
      b.omega -= rollSign * rollImpulse / b.inertia;
    }

    const totalM = a.mass + b.mass;
    const corr = delta * 0.2;
    a.x -= nx * corr * (b.mass / totalM);
    a.y -= ny * corr * (b.mass / totalM);
    b.x += nx * corr * (a.mass / totalM);
    b.y += ny * corr * (a.mass / totalM);

    const fMag = Math.sqrt(fx * fx + fy * fy);
    newContactVis.push({
      x: (a.x * b.radius + b.x * a.radius) / (a.radius + b.radius),
      y: (a.y * b.radius + b.y * a.radius) / (a.radius + b.radius),
      force: fMag,
    });

    const key = pairKey(a.id, b.id);
    newPairs.add(key);
    if (!contactedPairs.has(key) && a.level === b.level) {
      mergeQueue.push([a.id, b.id]);
    }
  }

  // Wall contacts
  for (const p of particles) {
    // Bottom
    const oBot = p.y + p.radius - CB;
    if (oBot > 0) {
      const d = Math.min(oBot, MAX_DELTA_RATIO * p.radius);
      const vn = -p.vy;
      const fn = computeNormalForce(d, p.radius, p.mass, vn, ESTAR_PW, KN_PW, BETA_PW);
      p.vy -= (fn / p.mass) * dt;
      const vSlideBot = p.vx - p.omega * p.radius;
      if (Math.abs(vSlideBot) > 0.01) {
        const kTEffW = contactModel === 'hertz'
          ? 2 * ESTAR_PW * Math.sqrt(Math.max(p.radius * d, 0.01))
          : KN_PW;
        const ftDampW = 2 * BETA_PW * Math.sqrt(Math.max(p.mass * kTEffW, 0.001));
        const ft = Math.min(MU_PW * fn, ftDampW * Math.abs(vSlideBot));
        p.vx -= Math.sign(vSlideBot) * (ft / p.mass) * dt;
        p.omega += Math.sign(vSlideBot) * (ft * p.radius / p.inertia) * dt;
      }
      if (Math.abs(p.omega) > 0.01) {
        const tauR = MU_ROLL_PW * p.radius * fn;
        const imp = Math.min(tauR * dt, Math.abs(p.omega) * p.inertia * 0.5);
        p.omega -= Math.sign(p.omega) * imp / p.inertia;
      }
      p.y = Math.min(p.y, CB - p.radius);
    }

    // Left
    const oL = CL - (p.x - p.radius);
    if (oL > 0) {
      const d = Math.min(oL, MAX_DELTA_RATIO * p.radius);
      const vn = p.vx;
      const fn = computeNormalForce(d, p.radius, p.mass, vn, ESTAR_PW, KN_PW, BETA_PW);
      p.vx += (fn / p.mass) * dt;
      const vSlideL = p.vy - p.omega * p.radius;
      if (Math.abs(vSlideL) > 0.01) {
        const kTEffW = contactModel === 'hertz'
          ? 2 * ESTAR_PW * Math.sqrt(Math.max(p.radius * d, 0.01))
          : KN_PW;
        const ftDampW = 2 * BETA_PW * Math.sqrt(Math.max(p.mass * kTEffW, 0.001));
        const ft = Math.min(MU_PW * fn, ftDampW * Math.abs(vSlideL));
        p.vy -= Math.sign(vSlideL) * (ft / p.mass) * dt;
        p.omega += Math.sign(vSlideL) * (ft * p.radius / p.inertia) * dt;
      }
      if (Math.abs(p.omega) > 0.01) {
        const tauR = MU_ROLL_PW * p.radius * fn;
        const imp = Math.min(tauR * dt, Math.abs(p.omega) * p.inertia * 0.5);
        p.omega -= Math.sign(p.omega) * imp / p.inertia;
      }
      p.x = Math.max(p.x, CL + p.radius);
    }

    // Right
    const oR = (p.x + p.radius) - CR;
    if (oR > 0) {
      const d = Math.min(oR, MAX_DELTA_RATIO * p.radius);
      const vn = -p.vx;
      const fn = computeNormalForce(d, p.radius, p.mass, vn, ESTAR_PW, KN_PW, BETA_PW);
      p.vx -= (fn / p.mass) * dt;
      const vSlideR = p.vy + p.omega * p.radius;
      if (Math.abs(vSlideR) > 0.01) {
        const kTEffW = contactModel === 'hertz'
          ? 2 * ESTAR_PW * Math.sqrt(Math.max(p.radius * d, 0.01))
          : KN_PW;
        const ftDampW = 2 * BETA_PW * Math.sqrt(Math.max(p.mass * kTEffW, 0.001));
        const ft = Math.min(MU_PW * fn, ftDampW * Math.abs(vSlideR));
        p.vy -= Math.sign(vSlideR) * (ft / p.mass) * dt;
        p.omega -= Math.sign(vSlideR) * (ft * p.radius / p.inertia) * dt;
      }
      if (Math.abs(p.omega) > 0.01) {
        const tauR = MU_ROLL_PW * p.radius * fn;
        const imp = Math.min(tauR * dt, Math.abs(p.omega) * p.inertia * 0.5);
        p.omega -= Math.sign(p.omega) * imp / p.inertia;
      }
      p.x = Math.min(p.x, CR - p.radius);
    }
  }

  // Integration + damping
  for (const p of particles) {
    p.vx *= VEL_DAMP;
    p.vy *= VEL_DAMP;
    p.omega *= ANG_DAMP;

    p.vx = clamp(p.vx, -MAX_VEL, MAX_VEL);
    p.vy = clamp(p.vy, -MAX_VEL, MAX_VEL);
    p.omega = clamp(p.omega, -MAX_OMEGA, MAX_OMEGA);

    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.angle += p.omega * dt;

    if (!p.active && p.y > DANGER_Y) {
      p.active = true;
    }
  }

  // Iterative position correction (resolve overlaps without adding energy)
  for (let iter = 0; iter < 3; iter++) {
    // Particle-particle separation
    for (const [i, j] of spatialHash.findPairs(particles)) {
      const a = particles[i];
      const b = particles[j];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distSq = dx * dx + dy * dy;
      const minDist = a.radius + b.radius;
      if (distSq >= minDist * minDist) continue;
      const dist = Math.sqrt(Math.max(distSq, 1e-8));
      const overlap = minDist - dist;
      if (overlap <= 0) continue;
      const nx = dx / dist;
      const ny = dy / dist;
      const totalM = a.mass + b.mass;
      const corr = overlap * 0.3;
      a.x -= nx * corr * (b.mass / totalM);
      a.y -= ny * corr * (b.mass / totalM);
      b.x += nx * corr * (a.mass / totalM);
      b.y += ny * corr * (a.mass / totalM);
    }

    // Wall clamping
    for (const p of particles) {
      if (p.y + p.radius > CB) p.y = CB - p.radius;
      if (p.x - p.radius < CL) p.x = CL + p.radius;
      if (p.x + p.radius > CR) p.x = CR - p.radius;
    }

    // Wall-squeeze detection: if a particle is pressed against a wall by another particle, push the other particle away
    for (const p of particles) {
      const atLeftWall = (p.x - p.radius) < CL + 1;
      const atRightWall = (p.x + p.radius) > CR - 1;
      const atBottom = (p.y + p.radius) > CB - 1;
      if (!atLeftWall && !atRightWall && !atBottom) continue;

      for (const q of particles) {
        if (p === q) continue;
        const dx = q.x - p.x;
        const dy = q.y - p.y;
        const distSq = dx * dx + dy * dy;
        const minDist = p.radius + q.radius;
        if (distSq >= minDist * minDist) continue;
        const dist = Math.sqrt(Math.max(distSq, 1e-8));
        const overlap = minDist - dist;
        if (overlap <= 0) continue;
        // Push q away from p (p is against wall, so only move q)
        const nx = dx / dist;
        const ny = dy / dist;
        q.x += nx * overlap * 0.5;
        q.y += ny * overlap * 0.5;
      }
    }
  }

  contactVis = newContactVis;
  contactedPairs = newPairs;
}

// ================================================================
// Game Over Logic
// ================================================================

function doGameOver() {
  gameOver = true;
  finalScoreEl.textContent = score.toString();
  gameOverEl.style.display = 'flex';
  if (score > highScore) {
    highScore = score;
    saveHighScore(highScore);
    highScoreEl.textContent = `今日のハイスコア: ${highScore}`;
  }
}

function checkGameOver() {
  for (const p of particles) {
    if (p.graceFrames > 0) continue;
    if (p.active && p.y < DANGER_Y) { doGameOver(); return; }
    if (!p.active && p.y < DANGER_Y) {
      for (const q of particles) {
        if (p === q) continue;
        const dx = q.x - p.x;
        const dy = q.y - p.y;
        const distSq = dx * dx + dy * dy;
        const minDist = p.radius + q.radius;
        if (distSq < minDist * minDist) { doGameOver(); return; }
      }
    }
  }
}

// ================================================================
// Merge Logic
// ================================================================

function processMerges() {
  const consumed = new Set<number>();
  let mergedAny = false;

  for (const [idA, idB] of mergeQueue) {
    if (consumed.has(idA) || consumed.has(idB)) continue;

    const a = particles.find(p => p.id === idA);
    const b = particles.find(p => p.id === idB);
    if (!a || !b || a.level !== b.level) continue;

    const mx = (a.x + b.x) / 2;
    const my = (a.y + b.y) / 2;
    const newLevel = a.level + 1;

    consumed.add(idA);
    consumed.add(idB);
    particles = particles.filter(p => p.id !== idA && p.id !== idB);

    mergedAny = true;

    if (newLevel >= LEVELS.length) {
      const pts = LEVELS[a.level].score * 5;
      score += pts;
      mergeCount++;
      comboCount++;
      comboTimer = 45;
      effects.push({ x: mx, y: my, r: 10, alpha: 1, color: '#FFFFFF' });
      effects.push({ x: mx, y: my, r: 30, alpha: 0.7, color: '#F5E6CC' });
      scorePopups.push({ x: mx, y: my, text: `+${pts} MAX!`, timer: 90 });
      scoreEl.textContent = score.toString();
      continue;
    }

    const tm = a.mass + b.mass;
    const np = createParticle(mx, my, newLevel);
    const avgVx = (a.vx * a.mass + b.vx * b.mass) / tm;
    const avgVy = (a.vy * a.mass + b.vy * b.mass) / tm;
    np.vx = avgVx * 0.15;
    np.vy = Math.max(avgVy * 0.15, 0);
    np.omega = 0;
    np.active = a.active || b.active;
    np.graceFrames = 30;
    particles.push(np);

    effects.push({ x: mx, y: my, r: LEVELS[newLevel].radius * 0.3, alpha: 1, color: LEVELS[newLevel].color });

    comboCount++;
    comboTimer = 45;
    const pts = Math.floor(LEVELS[newLevel].score * (1 + (comboCount - 1) * 0.5));
    score += pts;
    mergeCount++;
    scorePopups.push({ x: mx, y: my, text: `+${pts}`, timer: 60 });
    scoreEl.textContent = score.toString();
  }

  mergeQueue = [];
  if (comboTimer > 0) {
    comboTimer--;
    if (comboTimer === 0) comboCount = 0;
  }

  if (mergedAny || particles.length > 0) {
    chartDirty = true;
  }
}

// ================================================================
// Game Logic
// ================================================================

function getRandomLevel(): number {
  const r = Math.random();
  if (r < 0.35) return 0;
  if (r < 0.65) return 1;
  if (r < 0.85) return 2;
  return 3;
}

function drop() {
  if (!canDrop || gameOver) return;
  const r = LEVELS[currentLevel].radius;
  const cx = Math.max(CL + r + 2, Math.min(CR - r - 2, dropX));
  const p = createParticle(cx, DROP_Y, currentLevel);
  particles.push(p);
  canDrop = false;
  lastDroppedId = p.id;
  currentLevel = nextLevel;
  nextLevel = getRandomLevel();
  updateNextPreview();
  chartDirty = true;
}

function restart() {
  particles = [];
  contactVis = [];
  effects = [];
  scorePopups = [];
  mergeQueue = [];
  contactedPairs.clear();
  nextId = 0;

  score = 0;
  currentLevel = getRandomLevel();
  nextLevel = getRandomLevel();
  dropX = GAME_W / 2;
  canDrop = true;
  gameOver = false;
  comboCount = 0;
  comboTimer = 0;
  mergeCount = 0;
  lastDroppedId = null;

  scoreEl.textContent = '0';
  gameOverEl.style.display = 'none';
  highScore = loadHighScore();
  highScoreEl.textContent = `今日のハイスコア: ${highScore}`;
  updateNextPreview();
  chartDirty = true;
}

// ================================================================
// Canvas2D Rendering (optimized fallback)
// ================================================================

let dangerTimer = 0;

function drawGameCanvas2D() {
  if (!gCtx) return;
  const ctx = gCtx;
  ctx.clearRect(0, 0, GAME_W, GAME_H);

  // Background
  ctx.fillStyle = '#0f0f23';
  ctx.fillRect(0, 0, GAME_W, GAME_H);
  const bgGrad = ctx.createLinearGradient(0, 0, 0, GAME_H);
  bgGrad.addColorStop(0, '#151530');
  bgGrad.addColorStop(1, '#0d0d20');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(CL, 0, CONTAINER_W, GAME_H);
  ctx.fillStyle = '#1e1e45';
  ctx.fillRect(0, 0, WALL_T, GAME_H);
  ctx.fillRect(CR, 0, WALL_T, GAME_H);
  ctx.fillRect(0, CB, GAME_W, WALL_T);
  ctx.strokeStyle = '#3a3a7a';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(CL, 0); ctx.lineTo(CL, CB); ctx.lineTo(CR, CB); ctx.lineTo(CR, 0);
  ctx.stroke();

  // Danger line
  const dAlpha = dangerTimer > 0 ? 0.7 + 0.3 * Math.abs(Math.sin(Date.now() / 130)) : 0.6;
  ctx.strokeStyle = `rgba(255,160,0,${dAlpha})`;
  ctx.lineWidth = 5;
  ctx.setLineDash([12, 6]);
  ctx.beginPath();
  ctx.moveTo(CL, DANGER_Y); ctx.lineTo(CR, DANGER_Y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = `rgba(255,160,0,${dAlpha * 0.8})`;
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('DEAD LINE', CR - 4, DANGER_Y - 4);

  // Force chains (optimized, no shadowBlur)
  if (showForceChains && contactVis.length > 0) {
    const maxF = Math.max(...contactVis.map(c => c.force), 1);
    for (const c of contactVis) {
      const t = Math.min(c.force / maxF, 1);
      const baseR = 8 + t * 24;
      const grad = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, baseR);
      grad.addColorStop(0, `rgba(255,255,${Math.floor(220*(1-t))},1)`);
      grad.addColorStop(0.5, `rgba(255,${Math.floor(255*(1-t*0.8))},${Math.floor(50*(1-t))},${0.6+t*0.2})`);
      grad.addColorStop(1, `rgba(${Math.floor(255-40*t)},${Math.floor(80*(1-t))},0,0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(c.x, c.y, baseR, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = `rgba(255,255,200,${0.5+t*0.5})`;
      ctx.lineWidth = 2 + t * 3;
      ctx.beginPath();
      ctx.arc(c.x, c.y, baseR * 0.3, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // Effects
  for (const e of effects) {
    ctx.beginPath();
    ctx.arc(e.x, e.y, e.r, 0, Math.PI * 2);
    ctx.strokeStyle = hexToRGBA(e.color, e.alpha);
    ctx.lineWidth = 3;
    ctx.stroke();
  }

  // Particles (cached)
  initParticleCache();
  for (const p of particles) {
    drawParticleCached(ctx, p.x, p.y, p.angle, p.level);
  }

  // Score popups
  for (const sp of scorePopups) {
    const alpha = sp.timer / 60;
    const sz = 14 + (1 - alpha) * 8;
    ctx.font = `bold ${sz}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.strokeStyle = `rgba(78,52,32,${alpha})`;
    ctx.lineWidth = sz * 0.35;
    ctx.lineJoin = 'round';
    ctx.strokeText(sp.text, sp.x, sp.y - 30 + (1 - alpha) * 20);
    ctx.fillStyle = `rgba(247,220,111,${alpha})`;
    ctx.fillText(sp.text, sp.x, sp.y - 30 + (1 - alpha) * 20);
  }

  // Combo
  if (comboCount > 1 && comboTimer > 0) {
    const alpha = comboTimer / 45;
    const sz = 22 + comboCount * 2;
    ctx.font = `bold ${sz}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.strokeStyle = `rgba(78,52,32,${alpha})`;
    ctx.lineWidth = sz * 0.35;
    ctx.lineJoin = 'round';
    ctx.strokeText(`${comboCount} COMBO!`, GAME_W / 2, GAME_H / 2 - 60);
    ctx.fillStyle = `rgba(255,100,100,${alpha})`;
    ctx.fillText(`${comboCount} COMBO!`, GAME_W / 2, GAME_H / 2 - 60);
  }

  // Drop preview
  if (!gameOver && canDrop) {
    const r = LEVELS[currentLevel].radius;
    const cx = Math.max(CL + r + 2, Math.min(CR - r - 2, dropX));
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, DROP_Y + r);
    ctx.lineTo(cx, CB);
    ctx.stroke();
    ctx.setLineDash([]);
    drawParticleCached(ctx, cx, DROP_Y, 0, currentLevel);
  }

  // Status line
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  const modelText = contactModel === 'hertz' ? 'Hertz Contact' : 'Hooke Contact';
  ctx.fillText(`${modelText}  N=${particles.length}  ${rendererName}`, CL + 4, 14);
}

// ================================================================
// Grading Chart (throttled)
// ================================================================

function drawGradingChart() {
  const ctx = cCtx;
  const W = chartCanvas.width;
  const H = chartCanvas.height;
  const pad = { top: 28, right: 24, bottom: 56, left: 62 };
  const pW = W - pad.left - pad.right;
  const pH = H - pad.top - pad.bottom;

  ctx.fillStyle = '#f5f5f5';
  ctx.fillRect(0, 0, W, H);

  const counts = new Array(LEVELS.length).fill(0);
  let totalMass = 0;
  for (const p of particles) {
    counts[p.level] += p.mass;
    totalMass += p.mass;
  }

  ctx.strokeStyle = '#555';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + pH);
  ctx.lineTo(pad.left + pW, pad.top + pH);
  ctx.stroke();

  ctx.fillStyle = '#444';
  ctx.font = '13px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const pct of [0, 20, 40, 60, 80, 100]) {
    const y = pad.top + pH - (pct / 100) * pH;
    ctx.fillText(`${pct}`, pad.left - 4, y);
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + pW, y);
    ctx.stroke();
  }

  const logMin = Math.log10(0.5);
  const logMax = Math.log10(100);

  function toX(mm: number): number {
    return pad.left + ((Math.log10(mm) - logMin) / (logMax - logMin)) * pW;
  }

  const xTicks = [0.5, 0.75, 2, 4.75, 9.5, 19, 26.5, 37.5, 53, 75, 100];
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const s of xTicks) {
    const x = toX(s);
    const isMain = SIEVE_SIZES.includes(s);
    ctx.strokeStyle = isMain ? '#ccc' : '#e5e5e5';
    ctx.lineWidth = isMain ? 0.8 : 0.4;
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + pH);
    ctx.stroke();
    ctx.fillStyle = isMain ? '#333' : '#999';
    ctx.font = isMain ? '13px sans-serif' : '10px sans-serif';
    ctx.fillText(`${s}`, x, pad.top + pH + 2);
  }

  ctx.font = '15px sans-serif';
  ctx.fillStyle = '#555';
  ctx.textAlign = 'center';
  ctx.fillText('粒径 (mm)', pad.left + pW / 2, pad.top + pH + 32);

  ctx.save();
  ctx.translate(14, pad.top + pH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.font = '15px sans-serif';
  ctx.fillText('通過質量百分率 (%)', 0, 0);
  ctx.restore();

  if (totalMass < 0.01) {
    ctx.fillStyle = '#aaa';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('粒子を落としてください', W / 2, H / 2);
    paramsEl.innerHTML = 'D\u2081\u2080=--  D\u2083\u2080=--  D\u2085\u2080=--  D\u2086\u2080=--  Uc=--  Uc\'=--<br>土の種類: ----';
    return;
  }

  const cumMass: number[] = [];
  let cm = 0;
  for (let i = 0; i < LEVELS.length; i++) {
    cm += counts[i];
    cumMass.push(cm);
  }

  const points: { x: number; y: number }[] = [];
  points.push({ x: 100, y: 100 });
  for (let i = SIEVE_SIZES.length - 1; i >= 0; i--) {
    points.push({ x: SIEVE_SIZES[i], y: (cumMass[i] / totalMass) * 100 });
  }
  points.push({ x: 0.5, y: 0 });
  points.sort((a, b) => a.x - b.x);

  ctx.strokeStyle = '#8B6F47';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const px = toX(points[i].x);
    const py = pad.top + pH - (points[i].y / 100) * pH;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  ctx.fillStyle = 'rgba(139,111,71,0.1)';
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const px = toX(points[i].x);
    const py = pad.top + pH - (points[i].y / 100) * pH;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.lineTo(pad.left + pW, pad.top + pH);
  ctx.lineTo(pad.left, pad.top + pH);
  ctx.closePath();
  ctx.fill();

  for (let i = 1; i < points.length - 1; i++) {
    const px = toX(points[i].x);
    const py = pad.top + pH - (points[i].y / 100) * pH;
    ctx.beginPath();
    ctx.arc(px, py, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = '#6B5235';
    ctx.fill();
  }

  const d10 = interpD(10, points);
  const d30 = interpD(30, points);
  const d50 = interpD(50, points);
  const d60 = interpD(60, points);

  const dLines = [
    { d: d10, pct: 10, label: 'D\u2081\u2080', color: '#2196F3' },
    { d: d30, pct: 30, label: 'D\u2083\u2080', color: '#4CAF50' },
    { d: d50, pct: 50, label: 'D\u2085\u2080', color: '#9C27B0' },
    { d: d60, pct: 60, label: 'D\u2086\u2080', color: '#FF5722' },
  ];

  for (const dl of dLines) {
    if (dl.d === null) continue;
    const hx = toX(dl.d);
    const hy = pad.top + pH - (dl.pct / 100) * pH;
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = dl.color;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.65;
    ctx.beginPath();
    ctx.moveTo(hx, pad.top + pH); ctx.lineTo(hx, hy);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad.left, hy); ctx.lineTo(hx, hy);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(hx, hy, 4, 0, Math.PI * 2);
    ctx.fillStyle = dl.color;
    ctx.fill();
  }

  let txt = '';
  if (d10 !== null) txt += `D\u2081\u2080=${d10.toFixed(1)}mm  `;
  else txt += 'D\u2081\u2080=--  ';
  if (d30 !== null) txt += `D\u2083\u2080=${d30.toFixed(1)}mm  `;
  else txt += 'D\u2083\u2080=--  ';
  if (d50 !== null) txt += `D\u2085\u2080=${d50.toFixed(1)}mm  `;
  else txt += 'D\u2085\u2080=--  ';
  if (d60 !== null) txt += `D\u2086\u2080=${d60.toFixed(1)}mm`;
  else txt += 'D\u2086\u2080=--';
  if (d10 !== null && d60 !== null) {
    const Uc = d60 / d10;
    txt += `  Uc=${Uc.toFixed(2)}`;
    if (d30 !== null) {
      const Ucp = (d30 * d30) / (d10 * d60);
      txt += `  Uc'=${Ucp.toFixed(2)}`;
    } else txt += "  Uc'=--";
  } else txt += '  Uc=--  Uc\'=--';
  txt += '<br>土の種類: ';
  if (d10 !== null && d30 !== null && d60 !== null) {
    const Uc = d60 / d10;
    const Ucp = (d30 * d30) / (d10 * d60);
    if (Uc >= 4 && Ucp >= 1 && Ucp <= 3) txt += '良粒度礫 (GW)';
    else if (Uc >= 6 && Ucp >= 1 && Ucp <= 3) txt += '良粒度砂 (SW)';
    else txt += '不良粒度 (GP/SP)';
  } else {
    txt += '----';
  }
  for (const dl of dLines) {
    if (dl.d === null) continue;
    const hx = toX(dl.d);
    const hy = pad.top + pH - (dl.pct / 100) * pH;
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;
    ctx.lineJoin = 'round';
    ctx.miterLimit = 2;
    ctx.strokeText(dl.label, hx + 4, hy - 3);
    ctx.strokeText(dl.label, hx + 4, hy - 3);
    ctx.fillStyle = dl.color;
    ctx.fillText(dl.label, hx + 4, hy - 3);
    ctx.textBaseline = 'alphabetic';
  }

  paramsEl.innerHTML = txt;
}

function interpD(target: number, pts: { x: number; y: number }[]): number | null {
  for (let i = 1; i < pts.length; i++) {
    const y0 = pts[i - 1].y;
    const y1 = pts[i].y;
    if ((y0 <= target && y1 >= target) || (y0 >= target && y1 <= target)) {
      const dy = y1 - y0;
      if (Math.abs(dy) < 0.001) continue;
      const t = (target - y0) / dy;
      if (t < 0 || t > 1) continue;
      const logX = Math.log10(Math.max(pts[i - 1].x, 0.01)) + t * (Math.log10(pts[i].x) - Math.log10(Math.max(pts[i - 1].x, 0.01)));
      return Math.pow(10, logX);
    }
  }
  return null;
}

// ================================================================
// Next Preview
// ================================================================

function updateNextPreview() {
  const ctx = nCtx;
  const s = 96;
  ctx.clearRect(0, 0, s, s);
  const r = LEVELS[nextLevel].radius;
  const sc = Math.min(40 / r, 1);
  ctx.save();
  ctx.translate(s / 2, s / 2);
  ctx.scale(sc, sc);

  const info = LEVELS[nextLevel];
  const grad = ctx.createRadialGradient(-r * 0.25, -r * 0.25, r * 0.05, 0, 0, r);
  grad.addColorStop(0, lighten(info.color, 35));
  grad.addColorStop(1, info.color);
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.strokeStyle = info.strokeColor;
  ctx.lineWidth = 2;
  ctx.stroke();

  const fontSize = Math.max(8, Math.floor(r * 0.3));
  ctx.font = `bold ${fontSize}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#3E2723';
  ctx.fillText(info.sieve, 0, 0);

  ctx.restore();
  nextNameEl.textContent = `${info.name} (${info.sieve})`;
}

// ================================================================
// Input
// ================================================================

function setupInput() {
  gameCanvas.addEventListener('mousemove', (e) => {
    const rect = gameCanvas.getBoundingClientRect();
    dropX = (e.clientX - rect.left) * (GAME_W / rect.width);
  });

  gameCanvas.addEventListener('click', () => drop());

  gameCanvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const rect = gameCanvas.getBoundingClientRect();
    dropX = (e.touches[0].clientX - rect.left) * (GAME_W / rect.width);
  }, { passive: false });

  gameCanvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    drop();
  });

  const keys = new Set<string>();
  document.addEventListener('keydown', (e) => {
    keys.add(e.key);
    if (e.key === ' ') { e.preventDefault(); drop(); }
  });
  document.addEventListener('keyup', (e) => keys.delete(e.key));

  setInterval(() => {
    if (keys.has('ArrowLeft') || keys.has('a') || keys.has('A')) dropX = Math.max(CL + 20, dropX - 6);
    if (keys.has('ArrowRight') || keys.has('d') || keys.has('D')) dropX = Math.min(CR - 20, dropX + 6);
  }, 16);

  restartBtn.addEventListener('click', restart);
  document.getElementById('score-restart-btn')!.addEventListener('click', restart);

  btnHooke.addEventListener('click', () => {
    contactModel = 'hooke';
    btnHooke.classList.add('active');
    btnHertz.classList.remove('active');
  });
  btnHertz.addEventListener('click', () => {
    contactModel = 'hertz';
    btnHertz.classList.add('active');
    btnHooke.classList.remove('active');
  });
  chkForces.addEventListener('change', () => { showForceChains = chkForces.checked; });
}

// ================================================================
// Main Loop
// ================================================================

function update() {
  if (!gameOver) {
    const dt = (1 / 60) / SUB_STEPS;
    for (let i = 0; i < SUB_STEPS; i++) {
      physicsStep(dt);
    }
    processMerges();
    checkGameOver();
    for (const p of particles) {
      if (p.graceFrames > 0) p.graceFrames--;
    }

    // Check if the last dropped particle has cleared the danger line or touched another particle
    if (!canDrop && lastDroppedId !== null) {
      const dp = particles.find(p => p.id === lastDroppedId);
      if (!dp) {
        // Particle was consumed by merge — allow next drop
        canDrop = true;
        lastDroppedId = null;
      } else {
        const fullyBelowLine = (dp.y - dp.radius) > DANGER_Y;
        let hasContact = false;
        if (!fullyBelowLine) {
          for (const q of particles) {
            if (q === dp) continue;
            const dx = q.x - dp.x;
            const dy = q.y - dp.y;
            const distSq = dx * dx + dy * dy;
            const minDist = dp.radius + q.radius + 2; // small tolerance for near-contact
            if (distSq < minDist * minDist) { hasContact = true; break; }
          }
        }
        if (fullyBelowLine || hasContact) {
          canDrop = true;
          lastDroppedId = null;
        }
      }
    }
  }

  for (const e of effects) {
    e.r += 2.5;
    e.alpha -= 0.025;
  }
  effects = effects.filter(e => e.alpha > 0);

  for (const sp of scorePopups) sp.timer--;
  scorePopups = scorePopups.filter(sp => sp.timer > 0);

  // Render
  if (gpuRenderer) {
    gpuRenderer.drawFrame(
      particles, contactVis, effects, scorePopups,
      comboCount, comboTimer, currentLevel, dropX, canDrop, gameOver,
      contactModel, showForceChains,
      dangerTimer > 0 ? 0.7 + 0.3 * Math.abs(Math.sin(Date.now() / 130)) : 0.6,
      Date.now() / 1000,
      rendererName,
    );
  } else {
    drawGameCanvas2D();
  }

  // Chart (throttled)
  if (chartDirty || (Date.now() - lastChartFrame > 500)) {
    drawGradingChart();
    chartDirty = false;
    lastChartFrame = Date.now();
  }

  if (gameOver && dangerTimer > 0) dangerTimer--;

  requestAnimationFrame(update);
}

// ================================================================
// Init
// ================================================================

async function main() {
  await initRenderer();
  currentLevel = getRandomLevel();
  nextLevel = getRandomLevel();
  updateNextPreview();
  setupInput();
  update();
}

main().catch(console.error);

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
