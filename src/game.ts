// ================================================================
// 粒度分布ゲーム - 土質工学メロンゲーム
// DEM Physics (SoA Float32Array, allocation-free hot loop)
// Rendering: WebGPU -> WebGL -> Canvas2D
// ================================================================

import {
  LEVELS, SIEVE_SIZES, GAME_W, GAME_H, WALL_T, CL, CR, CB, CONTAINER_W,
  DROP_Y, DANGER_Y, GRAVITY, ESTAR_PP, ESTAR_PW, KN_PP, KN_PW,
  MU_PP, MU_PW, MU_ROLL_PP, MU_ROLL_PW,
  SUB_STEPS, COMBO_TIMEOUT_STEPS, GRACE_STEPS, MAX_DELTA_RATIO, MAX_VEL, MAX_OMEGA, VEL_DAMP, ANG_DAMP,
  BETA_PP, BETA_PW,
} from './constants';

import { UniformGrid } from './spatial';
import { WebGPURenderer } from './renderer-webgpu';
import { WebGLRenderer } from './renderer-webgl';

// ================================================================
// Types
// ================================================================

interface ContactVis { x: number; y: number; force: number; }
interface Effect { x: number; y: number; r: number; alpha: number; color: string; }
interface ScorePopup { x: number; y: number; text: string; timer: number; }
interface PerformanceProfile {
  compact: boolean;
  subSteps: number;
  correctionIterations: number;
  forceChainStride: number;
  effectiveForceChains: boolean;
  drawParticleLabels: boolean;
  minLabelRadius: number;
  maxScorePopups: number;
  chartIntervalMs: number;
  showDropGuide: boolean;
}

type ContactModel = 'hooke' | 'hertz';
interface AnyRenderer { drawFrame: (...args: any[]) => void; }

// ================================================================
// Particle Store — Structure of Arrays (f32/i32, zero GC in hot loop)
// ================================================================

const MAX_PARTICLES = 1024;

const px = new Float32Array(MAX_PARTICLES);
const py = new Float32Array(MAX_PARTICLES);
const pvx = new Float32Array(MAX_PARTICLES);
const pvy = new Float32Array(MAX_PARTICLES);
const pRadius = new Float32Array(MAX_PARTICLES);
const pMass = new Float32Array(MAX_PARTICLES);
const pInertia = new Float32Array(MAX_PARTICLES);
const pAngle = new Float32Array(MAX_PARTICLES);
const pOmega = new Float32Array(MAX_PARTICLES);
const pLevel = new Int32Array(MAX_PARTICLES);
const pActive = new Int32Array(MAX_PARTICLES);
const pGrace = new Int32Array(MAX_PARTICLES);
const pId = new Int32Array(MAX_PARTICLES);
let pCount = 0;
let nextId = 0;

function spawnParticle(x: number, y: number, level: number): number {
  if (pCount >= MAX_PARTICLES) return -1;
  const i = pCount++;
  const r = LEVELS[level].radius;
  const m = r * r * 0.008;
  px[i] = x; py[i] = y;
  pvx[i] = 0; pvy[i] = 0;
  pRadius[i] = r; pMass[i] = m;
  pInertia[i] = 0.5 * m * r * r;
  pAngle[i] = 0; pOmega[i] = 0;
  pLevel[i] = level;
  pActive[i] = 0; pGrace[i] = 0;
  pId[i] = nextId++;
  return i;
}

/** Swap-remove; returns the index now occupying slot i (re-check it). */
function removeParticle(i: number) {
  const last = --pCount;
  if (i !== last) {
    px[i] = px[last]; py[i] = py[last];
    pvx[i] = pvx[last]; pvy[i] = pvy[last];
    pRadius[i] = pRadius[last]; pMass[i] = pMass[last];
    pInertia[i] = pInertia[last]; pAngle[i] = pAngle[last];
    pOmega[i] = pOmega[last]; pLevel[i] = pLevel[last];
    pActive[i] = pActive[last]; pGrace[i] = pGrace[last];
    pId[i] = pId[last];
  }
}

function findIndexById(id: number): number {
  for (let i = 0; i < pCount; i++) if (pId[i] === id) return i;
  return -1;
}

// ================================================================
// State
// ================================================================

let contactVis: ContactVis[] = [];
let contactVisLen = 0;
let effects: Effect[] = [];
let scorePopups: ScorePopup[] = [];
const mergeQueue = new Int32Array(MAX_PARTICLES * 8); // flattened [idA, idB, ...]
let mergeQueueLen = 0;
let contactedPairs = new Set<number>();

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
let lastDroppedLevel = 0;
let lastDropAnchorX = GAME_W / 2;
let lastDropAnchorY = DROP_Y;
let totalSimSteps = 0;
let simStepSerial = 0;
let stepsSinceDrop = 0;

const grid = new UniformGrid(500, MAX_PARTICLES);

let chartDirty = true;
let lastChartFrame = 0;

// Renderers
let gpuRenderer: AnyRenderer | null = null; // WebGPU or WebGL
let rendererKind: 'webgpu' | 'webgl' | 'canvas2d' = 'canvas2d';

// fps-adaptive quality
let fpsEma = 60;
let lastFrameTime = 0;
let qualityTier = 0; // 0 = full, 1 = reduced, 2 = minimal

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
const topScoresEl = document.getElementById('top-scores')!;
const restartBtn = document.getElementById('restart-btn')!;
const paramsEl = document.getElementById('grading-params')!;
const btnHooke = document.getElementById('btn-hooke') as HTMLButtonElement;
const btnHertz = document.getElementById('btn-hertz') as HTMLButtonElement;
const chkForces = document.getElementById('chk-forces') as HTMLInputElement;
const compactHintEl = document.getElementById('compact-hint')!;
const compactScoreEl = document.getElementById('compact-score')!;
const compactNextEl = document.getElementById('compact-next')!;
const compactLeftBtn = document.getElementById('compact-left-btn') as HTMLButtonElement;
const compactDropBtn = document.getElementById('compact-drop-btn') as HTMLButtonElement;
const compactRightBtn = document.getElementById('compact-right-btn') as HTMLButtonElement;
const compactModelBtn = document.getElementById('compact-model-btn') as HTMLButtonElement;
const compactForceBtn = document.getElementById('compact-force-btn') as HTMLButtonElement;
const bgmEl = document.getElementById('bgm') as HTMLAudioElement;
const muteBtn = document.getElementById('mute-btn') as HTMLButtonElement;

const pressedKeys = new Set<string>();
let cachedCanvasRect = new DOMRect(0, 0, GAME_W, GAME_H);
let activePointerId: number | null = null;
let pointerStartX = 0;
let pointerStartY = 0;
let pointerDragDistance = 0;
let pointerVerticalDistance = 0;
let pointerSwipeDropTriggered = false;
let compactMoveDir = 0;

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

const APP_VERSION = '3.6.0';

interface TopScoreEntry { date: string; version: string; model: string; score: number; }

function loadTopScores(): TopScoreEntry[] {
  try {
    const data = localStorage.getItem('granularity_top3');
    if (!data) return [];
    return (JSON.parse(data) as TopScoreEntry[]).slice(0, 3);
  } catch { return []; }
}

function saveTopScores(scores: TopScoreEntry[]) {
  localStorage.setItem('granularity_top3', JSON.stringify(scores.slice(0, 3)));
}

function addScoreToTop(s: number): TopScoreEntry[] {
  const scores = loadTopScores();
  scores.push({ date: getTodayKey(), version: APP_VERSION, model: contactModel, score: s });
  scores.sort((a, b) => b.score - a.score);
  const top3 = scores.slice(0, 3);
  saveTopScores(top3);
  return top3;
}

// ================================================================
// BGM mute (persisted in cookie)
// ================================================================

function readMuteCookie(): boolean {
  return /(?:^|;\s*)bgm_muted=1/.test(document.cookie);
}

function writeMuteCookie(muted: boolean) {
  const expires = new Date(Date.now() + 365 * 864e5).toUTCString();
  document.cookie = `bgm_muted=${muted ? 1 : 0}; expires=${expires}; path=/; SameSite=Lax`;
}

function applyMuteState(muted: boolean) {
  bgmEl.muted = muted;
  muteBtn.textContent = muted ? '\u{1F507}' : '\u{1F50A}';
  muteBtn.classList.toggle('muted', muted);
  muteBtn.setAttribute('aria-label', muted ? 'Unmute BGM' : 'Mute BGM');
}

function setupMute() {
  applyMuteState(readMuteCookie());
  muteBtn.addEventListener('click', () => {
    const muted = !bgmEl.muted;
    writeMuteCookie(muted);
    applyMuteState(muted);
  });
}

function updateHudStats() {
  compactScoreEl.textContent = score.toLocaleString();
}

function updateLayoutHint() {
  const compact = window.matchMedia('(max-width: 900px), (max-height: 760px)').matches;
  compactHintEl.textContent = compact
    ? `Compact mobile layout active (${rendererKind})`
    : `Desktop layout active (${rendererKind})`;
}

function getPerformanceProfile(): PerformanceProfile {
  const compact = document.body.classList.contains('compact');
  const n = pCount;
  const extreme = compact || n >= 220 || qualityTier >= 2;
  const heavy = n >= 140 || qualityTier >= 1;
  return {
    compact,
    subSteps: qualityTier >= 2 ? 8 : qualityTier === 1 ? 12 : SUB_STEPS,
    correctionIterations: extreme ? 1 : heavy ? 2 : 3,
    forceChainStride: n >= 260 ? 6 : n >= 180 ? 4 : n >= 120 ? 2 : 1,
    effectiveForceChains: showForceChains && !(compact && n >= 80),
    drawParticleLabels: !compact && n <= 180,
    minLabelRadius: compact || n > 180 ? 80 : 0,
    maxScorePopups: compact ? 6 : n >= 180 ? 8 : 12,
    chartIntervalMs: compact ? (n >= 120 ? 1400 : 900) : n >= 200 ? 1200 : n >= 120 ? 800 : 450,
    showDropGuide: !compact && n < 220,
  };
}

function updateFpsAdaptation(now: number) {
  if (lastFrameTime > 0) {
    const dt = now - lastFrameTime;
    if (dt > 0 && dt < 500) {
      const fps = 1000 / dt;
      fpsEma += (fps - fpsEma) * 0.05;
    }
  }
  lastFrameTime = now;
  // Adapt once per ~2 seconds
  if ((totalSimSteps & 127) === 0) {
    if (fpsEma < 50 && qualityTier < 2) qualityTier++;
    else if (fpsEma > 58 && qualityTier > 0) qualityTier--;
  }
}

function updateScoreDisplays() {
  scoreEl.textContent = score.toString();
  compactScoreEl.textContent = score.toLocaleString();
}

function syncControlState() {
  btnHooke.classList.toggle('active', contactModel === 'hooke');
  btnHertz.classList.toggle('active', contactModel === 'hertz');
  compactModelBtn.textContent = contactModel === 'hertz' ? 'HERTZ' : 'HOOKE';
  compactModelBtn.classList.toggle('is-active', contactModel === 'hertz');
  chkForces.checked = showForceChains;
  compactForceBtn.textContent = showForceChains ? 'CHAIN ON' : 'CHAIN OFF';
  compactForceBtn.classList.toggle('is-active', showForceChains);
}

function setContactModel(model: ContactModel) {
  contactModel = model;
  syncControlState();
}

function pushScorePopup(popup: ScorePopup) {
  const maxPopups = getPerformanceProfile().maxScorePopups;
  if (scorePopups.length >= maxPopups) {
    scorePopups.splice(0, scorePopups.length - maxPopups + 1);
  }
  scorePopups.push(popup);
}

let highScore = loadHighScore();
highScoreEl.textContent = `今日のハイスコア / Todays HighScore: ${highScore}`;

// ================================================================
// Renderer Detection: WebGPU -> WebGL -> Canvas2D
// ================================================================

async function initRenderer() {
  try {
    if ('gpu' in navigator) {
      const adapter = await (navigator as any).gpu.requestAdapter();
      if (adapter) {
        const device = await adapter.requestDevice();
        gpuRenderer = new WebGPURenderer(gameCanvas, device);
        rendererKind = 'webgpu';
        return;
      }
    }
  } catch (e) {
    console.log('WebGPU not available:', e);
  }

  try {
    initParticleCache();
    gpuRenderer = new WebGLRenderer(gameCanvas, PARTICLE_CACHE);
    rendererKind = 'webgl';
    return;
  } catch (e) {
    console.log('WebGL not available:', e);
  }

  gCtx = gameCanvas.getContext('2d')!;
  rendererKind = 'canvas2d';
}

// ================================================================
// Particle Sprite Cache (shared by WebGL textures + Canvas2D fallback)
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

    ctx.beginPath();
    ctx.arc(cx + 2, cy + 3, r, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,0,0,0.15)';
    ctx.fill();

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

  const fontSize = Math.max(8, Math.floor(r * 0.3));
  ctx.font = `bold ${fontSize}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#3E2723';
  ctx.fillText(LEVELS[level].sieve, 0, 0);
  ctx.restore();
}

// ================================================================
// DEM Physics (SoA, allocation-free)
// ================================================================

function pairKey(a: number, b: number): number {
  return a < b ? a * 65536 + b : b * 65536 + a;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function updateCanvasRect() {
  cachedCanvasRect = gameCanvas.getBoundingClientRect();
}

function setDropFromClientX(clientX: number) {
  const width = Math.max(cachedCanvasRect.width, 1);
  dropX = clamp((clientX - cachedCanvasRect.left) * (GAME_W / width), CL + 20, CR - 20);
}

function computeStepBonus(steps: number, level: number): number {
  const base = LEVELS[level].score;
  if (steps <= SUB_STEPS * 6) return base * 3;
  if (steps <= SUB_STEPS * 12) return base * 2;
  if (steps <= SUB_STEPS * 20) return base;
  return 0;
}

function vibrate(pattern: number | number[]) {
  if (typeof navigator.vibrate === 'function') navigator.vibrate(pattern);
}

function releaseDropGate(anchorX?: number, anchorY?: number) {
  const bonus = computeStepBonus(stepsSinceDrop, lastDroppedLevel);
  if (bonus > 0) {
    score += bonus;
    updateScoreDisplays();
    pushScorePopup({
      x: anchorX ?? lastDropAnchorX,
      y: anchorY ?? lastDropAnchorY,
      text: `STEP +${bonus}`, timer: 55,
    });
  }
  canDrop = true;
  lastDroppedId = null;
  stepsSinceDrop = 0;
  updateHudStats();
}

function computeNormalForcePP(delta: number, rEff: number, mEff: number, vn: number): number {
  let fElastic: number;
  let kEff: number;
  if (contactModel === 'hertz') {
    fElastic = (4 / 3) * ESTAR_PP * Math.sqrt(rEff) * delta * Math.sqrt(delta);
    kEff = 2 * ESTAR_PP * Math.sqrt(Math.max(rEff * delta, 0.01));
  } else {
    fElastic = KN_PP * delta;
    kEff = KN_PP;
  }
  const eta = 2 * BETA_PP * Math.sqrt(mEff * kEff);
  const f = fElastic - eta * vn;
  return f > 0 ? f : 0;
}

function computeNormalForcePW(delta: number, radius: number, mass: number, vn: number): number {
  let fElastic: number;
  let kEff: number;
  if (contactModel === 'hertz') {
    fElastic = (4 / 3) * ESTAR_PW * Math.sqrt(radius) * delta * Math.sqrt(delta);
    kEff = 2 * ESTAR_PW * Math.sqrt(Math.max(radius * delta, 0.01));
  } else {
    fElastic = KN_PW * delta;
    kEff = KN_PW;
  }
  const eta = 2 * BETA_PW * Math.sqrt(mass * kEff);
  const f = fElastic - eta * vn;
  return f > 0 ? f : 0;
}

function tangentialDampPP(delta: number, rEff: number, mEff: number): number {
  const kT = contactModel === 'hertz'
    ? 2 * ESTAR_PP * Math.sqrt(Math.max(rEff * delta, 0.01))
    : KN_PP;
  return 2 * BETA_PP * Math.sqrt(mEff * kT);
}

function tangentialDampPW(delta: number, radius: number, mass: number): number {
  const kT = contactModel === 'hertz'
    ? 2 * ESTAR_PW * Math.sqrt(Math.max(radius * delta, 0.01))
    : KN_PW;
  return 2 * BETA_PW * Math.sqrt(mass * kT);
}

function physicsStep(dt: number, profile: PerformanceProfile) {
  const n = pCount;
  const collectContactVis = profile.effectiveForceChains;
  contactVisLen = 0;

  // Gravity
  for (let i = 0; i < n; i++) pvy[i] += GRAVITY * dt;

  grid.build(px, py, pRadius, n);
  const newPairs = new Set<number>();
  const pairTotal = grid.collectPairs(px, py, pRadius, n);

  for (let k = 0; k < pairTotal; k++) {
    const i = grid.getPairI(k);
    const j = grid.getPairJ(k);

    const dx = px[j] - px[i];
    const dy = py[j] - py[i];
    const distSq = dx * dx + dy * dy;
    const minDist = pRadius[i] + pRadius[j];
    if (distSq >= minDist * minDist) continue;

    const dist = Math.sqrt(distSq > 1e-8 ? distSq : 1e-8);
    let delta = minDist - dist;
    if (delta <= 0) continue;

    const ri = pRadius[i];
    const rj = pRadius[j];
    const maxDelta = MAX_DELTA_RATIO * (ri < rj ? ri : rj);
    if (delta > maxDelta) delta = maxDelta;

    const nx = dx / dist;
    const ny = dy / dist;
    const tx = -ny;
    const ty = nx;

    const dvx = pvx[j] - pvx[i];
    const dvy = pvy[j] - pvy[i];
    const vn = dvx * nx + dvy * ny;
    const vSlide = (dvx * tx + dvy * ty) - pOmega[i] * ri - pOmega[j] * rj;

    const mi = pMass[i];
    const mj = pMass[j];
    const mEff = (mi * mj) / (mi + mj);
    const rEff = (ri * rj) / (ri + rj);

    const fn = computeNormalForcePP(delta, rEff, mEff, vn);
    const ftDamp = tangentialDampPP(delta, rEff, mEff);
    const ftCap = MU_PP * fn;
    const ftRaw = ftDamp * Math.abs(vSlide);
    const ftMag = ftCap < ftRaw ? ftCap : ftRaw;
    const ftSign = vSlide > 0 ? -1 : vSlide < 0 ? 1 : 0;

    const fx = fn * nx + ftSign * ftMag * tx;
    const fy = fn * ny + ftSign * ftMag * ty;

    pvx[j] += (fx / mj) * dt;
    pvy[j] += (fy / mj) * dt;
    pvx[i] -= (fx / mi) * dt;
    pvy[i] -= (fy / mi) * dt;

    const torqueSign = -ftSign;
    pOmega[i] += (torqueSign * ftMag * ri / pInertia[i]) * dt;
    pOmega[j] += (torqueSign * ftMag * rj / pInertia[j]) * dt;

    // Rolling resistance: oppose relative spin
    const omegaRel = pOmega[j] - pOmega[i];
    if (omegaRel > 1e-6 || omegaRel < -1e-6) {
      const tauRoll = MU_ROLL_PP * rEff * fn;
      const rollSign = omegaRel > 0 ? 1 : -1;
      const ii = pInertia[i];
      const ij = pInertia[j];
      const maxImpulse = Math.abs(omegaRel) * 0.5 * (ii * ij) / (ii + ij);
      const rollImpulse = Math.min(tauRoll * dt, maxImpulse);
      pOmega[i] += rollSign * rollImpulse / ii;
      pOmega[j] -= rollSign * rollImpulse / ij;
    }

    const totalM = mi + mj;
    const corr = delta * 0.2;
    px[i] -= nx * corr * (mj / totalM);
    py[i] -= ny * corr * (mj / totalM);
    px[j] += nx * corr * (mi / totalM);
    py[j] += ny * corr * (mi / totalM);

    if (collectContactVis) {
      let cv: ContactVis;
      if (contactVisLen < contactVis.length) {
        cv = contactVis[contactVisLen];
      } else {
        cv = { x: 0, y: 0, force: 0 };
        contactVis.push(cv);
      }
      contactVisLen++;
      cv.x = (px[i] * rj + px[j] * ri) / (ri + rj);
      cv.y = (py[i] * rj + py[j] * ri) / (ri + rj);
      cv.force = Math.sqrt(fx * fx + fy * fy);
    }

    const key = pairKey(pId[i], pId[j]);
    newPairs.add(key);
    if (!contactedPairs.has(key) && pLevel[i] === pLevel[j] && mergeQueueLen + 2 <= mergeQueue.length) {
      mergeQueue[mergeQueueLen++] = pId[i];
      mergeQueue[mergeQueueLen++] = pId[j];
    }
  }
  if (contactVisLen < contactVis.length) contactVis.length = contactVisLen;

  // Wall contacts
  for (let i = 0; i < n; i++) {
    const r = pRadius[i];
    const m = pMass[i];

    // Bottom
    const oBot = py[i] + r - CB;
    if (oBot > 0) {
      const d = Math.min(oBot, MAX_DELTA_RATIO * r);
      const fn = computeNormalForcePW(d, r, m, -pvy[i]);
      pvy[i] -= (fn / m) * dt;
      const vSlide = pvx[i] - pOmega[i] * r;
      const ft = Math.min(MU_PW * fn, tangentialDampPW(d, r, m) * Math.abs(vSlide));
      const ftDir = vSlide > 0 ? 1 : vSlide < 0 ? -1 : 0;
      pvx[i] -= ftDir * (ft / m) * dt;
      pOmega[i] += ftDir * (ft * r / pInertia[i]) * dt;
      const tauR = MU_ROLL_PW * r * fn;
      const imp = Math.min(tauR * dt, Math.abs(pOmega[i]) * pInertia[i] * 0.5);
      if (Math.abs(pOmega[i]) > 1e-6) pOmega[i] -= Math.sign(pOmega[i]) * imp / pInertia[i];
      if (py[i] > CB - r) py[i] = CB - r;
    }

    // Left
    const oL = CL - (px[i] - r);
    if (oL > 0) {
      const d = Math.min(oL, MAX_DELTA_RATIO * r);
      const fn = computeNormalForcePW(d, r, m, pvx[i]);
      pvx[i] += (fn / m) * dt;
      const vSlide = pvy[i] - pOmega[i] * r;
      const ft = Math.min(MU_PW * fn, tangentialDampPW(d, r, m) * Math.abs(vSlide));
      const ftDir = vSlide > 0 ? 1 : vSlide < 0 ? -1 : 0;
      pvy[i] -= ftDir * (ft / m) * dt;
      pOmega[i] += ftDir * (ft * r / pInertia[i]) * dt;
      const tauR = MU_ROLL_PW * r * fn;
      const imp = Math.min(tauR * dt, Math.abs(pOmega[i]) * pInertia[i] * 0.5);
      if (Math.abs(pOmega[i]) > 1e-6) pOmega[i] -= Math.sign(pOmega[i]) * imp / pInertia[i];
      if (px[i] < CL + r) px[i] = CL + r;
    }

    // Right
    const oR = (px[i] + r) - CR;
    if (oR > 0) {
      const d = Math.min(oR, MAX_DELTA_RATIO * r);
      const fn = computeNormalForcePW(d, r, m, -pvx[i]);
      pvx[i] -= (fn / m) * dt;
      const vSlide = pvy[i] + pOmega[i] * r;
      const ft = Math.min(MU_PW * fn, tangentialDampPW(d, r, m) * Math.abs(vSlide));
      const ftDir = vSlide > 0 ? 1 : vSlide < 0 ? -1 : 0;
      pvy[i] -= ftDir * (ft / m) * dt;
      pOmega[i] -= ftDir * (ft * r / pInertia[i]) * dt;
      const tauR = MU_ROLL_PW * r * fn;
      const imp = Math.min(tauR * dt, Math.abs(pOmega[i]) * pInertia[i] * 0.5);
      if (Math.abs(pOmega[i]) > 1e-6) pOmega[i] -= Math.sign(pOmega[i]) * imp / pInertia[i];
      if (px[i] > CR - r) px[i] = CR - r;
    }
  }

  // Integration + damping
  for (let i = 0; i < n; i++) {
    pvx[i] *= VEL_DAMP;
    pvy[i] *= VEL_DAMP;
    pOmega[i] *= ANG_DAMP;

    pvx[i] = clamp(pvx[i], -MAX_VEL, MAX_VEL);
    pvy[i] = clamp(pvy[i], -MAX_VEL, MAX_VEL);
    pOmega[i] = clamp(pOmega[i], -MAX_OMEGA, MAX_OMEGA);

    px[i] += pvx[i] * dt;
    py[i] += pvy[i] * dt;
    pAngle[i] += pOmega[i] * dt;

    if (pActive[i] === 0 && py[i] > DANGER_Y) pActive[i] = 1;
  }

  // Iterative position correction
  for (let iter = 0; iter < profile.correctionIterations; iter++) {
    const pc = grid.collectPairs(px, py, pRadius, n);
    for (let k = 0; k < pc; k++) {
      const i = grid.getPairI(k);
      const j = grid.getPairJ(k);
      const dx = px[j] - px[i];
      const dy = py[j] - py[i];
      const distSq = dx * dx + dy * dy;
      const minDist = pRadius[i] + pRadius[j];
      if (distSq >= minDist * minDist) continue;
      const dist = Math.sqrt(distSq > 1e-8 ? distSq : 1e-8);
      const overlap = minDist - dist;
      if (overlap <= 0) continue;
      const nx = dx / dist;
      const ny = dy / dist;
      const totalM = pMass[i] + pMass[j];
      const corr = overlap * 0.3;
      px[i] -= nx * corr * (pMass[j] / totalM);
      py[i] -= ny * corr * (pMass[j] / totalM);
      px[j] += nx * corr * (pMass[i] / totalM);
      py[j] += ny * corr * (pMass[i] / totalM);
    }

    for (let i = 0; i < n; i++) {
      const r = pRadius[i];
      if (py[i] + r > CB) py[i] = CB - r;
      if (px[i] - r < CL) px[i] = CL + r;
      if (px[i] + r > CR) px[i] = CR - r;
    }

    if ((simStepSerial & 7) === 0) {
      // Wall-squeeze relief: push the non-wall particle out
      for (let i = 0; i < n; i++) {
        const r = pRadius[i];
        const atLeftWall = (px[i] - r) < CL + 1;
        const atRightWall = (px[i] + r) > CR - 1;
        const atBottom = (py[i] + r) > CB - 1;
        if (!atLeftWall && !atRightWall && !atBottom) continue;

        const nc = grid.collectNearby(px, py, pRadius, n, i, r);
        for (let k = 0; k < nc; k++) {
          const j = grid.getNearby(k);
          const dx = px[j] - px[i];
          const dy = py[j] - py[i];
          const distSq = dx * dx + dy * dy;
          const minDist = r + pRadius[j];
          if (distSq >= minDist * minDist) continue;
          const dist = Math.sqrt(distSq > 1e-8 ? distSq : 1e-8);
          const overlap = minDist - dist;
          if (overlap <= 0) continue;
          const nx = dx / dist;
          const ny = dy / dist;
          px[j] += nx * overlap * 0.5;
          py[j] += ny * overlap * 0.5;
        }
      }
    }
  }

  contactedPairs = newPairs;
  simStepSerial++;
}

function overlapsAnyParticle(index: number, extra = 0): boolean {
  const nc = grid.collectNearby(px, py, pRadius, pCount, index, extra);
  const r = pRadius[index];
  for (let k = 0; k < nc; k++) {
    const j = grid.getNearby(k);
    const dx = px[j] - px[index];
    const dy = py[j] - py[index];
    const minDist = r + pRadius[j] + extra;
    if (dx * dx + dy * dy < minDist * minDist) return true;
  }
  return false;
}

// ================================================================
// Game Over Logic
// ================================================================

function doGameOver() {
  gameOver = true;
  finalScoreEl.textContent = score.toString();
  const top3 = addScoreToTop(score);
  topScoresEl.innerHTML = top3.map((e, i) =>
    `<div class="top-score-row">${i + 1}. ${e.date} &nbsp; v${e.version} &nbsp; ${e.model === 'hertz' ? 'Hertz' : 'Hooke'} &nbsp; <b>${e.score}</b></div>`
  ).join('');
  gameOverEl.style.display = 'flex';
  if (score > highScore) {
    highScore = score;
    saveHighScore(highScore);
    highScoreEl.textContent = `今日のハイスコア / Todays HighScore: ${highScore}`;
  }
}

function checkGameOver() {
  for (let i = 0; i < pCount; i++) {
    if (pGrace[i] > 0) continue;
    if (pActive[i] === 1 && py[i] < DANGER_Y) { doGameOver(); return; }
    if (pActive[i] === 0 && py[i] < DANGER_Y && overlapsAnyParticle(i)) {
      doGameOver();
      return;
    }
  }
}

// ================================================================
// Merge Logic
// ================================================================

function processMerges() {
  const consumed = new Set<number>();
  let mergedAny = false;

  for (let k = 0; k < mergeQueueLen; k += 2) {
    const idA = mergeQueue[k];
    const idB = mergeQueue[k + 1];
    if (consumed.has(idA) || consumed.has(idB)) continue;

    const ia = findIndexById(idA);
    const ib = findIndexById(idB);
    if (ia < 0 || ib < 0 || pLevel[ia] !== pLevel[ib]) continue;

    const mx = (px[ia] + px[ib]) / 2;
    const my = (py[ia] + py[ib]) / 2;
    const level = pLevel[ia];
    const newLevel = level + 1;

    consumed.add(idA);
    consumed.add(idB);

    const ma = pMass[ia];
    const mb = pMass[ib];
    const tm = ma + mb;
    const avgVx = (pvx[ia] * ma + pvx[ib] * mb) / tm;
    const avgVy = (pvy[ia] * ma + pvy[ib] * mb) / tm;
    const wasActive = pActive[ia] === 1 || pActive[ib] === 1;

    // Remove higher index first so the lower index stays valid
    if (ia > ib) { removeParticle(ia); removeParticle(ib); }
    else { removeParticle(ib); removeParticle(ia); }

    mergedAny = true;

    if (newLevel >= LEVELS.length) {
      const pts = LEVELS[level].score * 5;
      score += pts;
      mergeCount++;
      comboCount++;
      comboTimer = COMBO_TIMEOUT_STEPS;
      effects.push({ x: mx, y: my, r: 10, alpha: 1, color: '#FFFFFF' });
      effects.push({ x: mx, y: my, r: 30, alpha: 0.7, color: '#F5E6CC' });
      pushScorePopup({ x: mx, y: my, text: `+${pts} MAX!`, timer: 90 });
      updateScoreDisplays();
      continue;
    }

    const ni = spawnParticle(mx, my, newLevel);
    pvx[ni] = avgVx * 0.15;
    pvy[ni] = Math.max(avgVy * 0.15, 0);
    pOmega[ni] = 0;
    pActive[ni] = wasActive ? 1 : 0;
    pGrace[ni] = GRACE_STEPS;

    effects.push({ x: mx, y: my, r: LEVELS[newLevel].radius * 0.3, alpha: 1, color: LEVELS[newLevel].color });

    comboCount++;
    comboTimer = COMBO_TIMEOUT_STEPS;
    const pts = Math.floor(LEVELS[newLevel].score * (1 + (comboCount - 1) * 0.5));
    score += pts;
    mergeCount++;
    pushScorePopup({ x: mx, y: my, text: `+${pts}`, timer: 60 });
    updateScoreDisplays();
  }

  mergeQueueLen = 0;
  if (comboTimer > 0) {
    comboTimer -= SUB_STEPS;
    if (comboTimer <= 0) { comboTimer = 0; comboCount = 0; }
  }

  if (mergedAny || pCount > 0) chartDirty = true;
  if (mergedAny) grid.build(px, py, pRadius, pCount);
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
  const jitter = (Math.random() - 0.5) * 2;
  const cx = Math.max(CL + r + 2, Math.min(CR - r - 2, dropX + jitter));
  lastDroppedLevel = currentLevel;
  const i = spawnParticle(cx, DROP_Y, currentLevel);
  if (i < 0) return;
  canDrop = false;
  lastDroppedId = pId[i];
  lastDropAnchorX = cx;
  lastDropAnchorY = DROP_Y;
  stepsSinceDrop = 0;
  currentLevel = nextLevel;
  nextLevel = getRandomLevel();
  updateNextPreview();
  chartDirty = true;
  updateHudStats();
  vibrate(12);
}

function restart() {
  pCount = 0;
  contactVis.length = 0;
  contactVisLen = 0;
  effects = [];
  scorePopups = [];
  mergeQueueLen = 0;
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
  lastDroppedLevel = 0;
  lastDropAnchorX = GAME_W / 2;
  lastDropAnchorY = DROP_Y;
  totalSimSteps = 0;
  simStepSerial = 0;
  stepsSinceDrop = 0;

  updateScoreDisplays();
  gameOverEl.style.display = 'none';
  topScoresEl.innerHTML = '';
  highScore = loadHighScore();
  highScoreEl.textContent = `今日のハイスコア / Todays HighScore: ${highScore}`;
  updateNextPreview();
  chartDirty = true;
  syncControlState();
  updateHudStats();
}

// ================================================================
// Canvas2D Rendering (fallback)
// ================================================================

let dangerTimer = 0;

function drawGameCanvas2D(profile: PerformanceProfile) {
  if (!gCtx) return;
  const ctx = gCtx;
  ctx.clearRect(0, 0, GAME_W, GAME_H);

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

  if (profile.effectiveForceChains && contactVisLen > 0) {
    let maxF = 1;
    for (let i = 0; i < contactVisLen; i += profile.forceChainStride) {
      if (contactVis[i].force > maxF) maxF = contactVis[i].force;
    }
    for (let i = 0; i < contactVisLen; i += profile.forceChainStride) {
      const c = contactVis[i];
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
    }
  }

  for (const e of effects) {
    ctx.beginPath();
    ctx.arc(e.x, e.y, e.r, 0, Math.PI * 2);
    ctx.strokeStyle = hexToRGBA(e.color, e.alpha);
    ctx.lineWidth = 3;
    ctx.stroke();
  }

  initParticleCache();
  for (let i = 0; i < pCount; i++) {
    drawParticleCached(ctx, px[i], py[i], pAngle[i], pLevel[i]);
  }

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

  if (comboCount > 1 && comboTimer > 0) {
    const alpha = comboTimer / COMBO_TIMEOUT_STEPS;
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

  if (!gameOver && canDrop) {
    const r = LEVELS[currentLevel].radius;
    const cx = Math.max(CL + r + 2, Math.min(CR - r - 2, dropX));
    if (profile.showDropGuide) {
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(cx, DROP_Y + r);
      ctx.lineTo(cx, CB);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    drawParticleCached(ctx, cx, DROP_Y, 0, currentLevel);
  }
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
  for (let i = 0; i < pCount; i++) {
    counts[pLevel[i]] += pMass[i];
    totalMass += pMass[i];
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
    const pxx = toX(points[i].x);
    const pyy = pad.top + pH - (points[i].y / 100) * pH;
    if (i === 0) ctx.moveTo(pxx, pyy); else ctx.lineTo(pxx, pyy);
  }
  ctx.stroke();

  ctx.fillStyle = 'rgba(139,111,71,0.1)';
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const pxx = toX(points[i].x);
    const pyy = pad.top + pH - (points[i].y / 100) * pH;
    if (i === 0) ctx.moveTo(pxx, pyy); else ctx.lineTo(pxx, pyy);
  }
  ctx.lineTo(pad.left + pW, pad.top + pH);
  ctx.lineTo(pad.left, pad.top + pH);
  ctx.closePath();
  ctx.fill();

  for (let i = 1; i < points.length - 1; i++) {
    const pxx = toX(points[i].x);
    const pyy = pad.top + pH - (points[i].y / 100) * pH;
    ctx.beginPath();
    ctx.arc(pxx, pyy, 2.5, 0, Math.PI * 2);
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
  compactNextEl.textContent = info.sieve;
}

// ================================================================
// Input
// ================================================================

function setupInput() {
  const clearCompactMove = () => {
    compactMoveDir = 0;
    compactLeftBtn.classList.remove('is-active');
    compactRightBtn.classList.remove('is-active');
  };
  const bindCompactMoveButton = (button: HTMLButtonElement, dir: number) => {
    button.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      compactMoveDir = dir;
      compactLeftBtn.classList.toggle('is-active', dir < 0);
      compactRightBtn.classList.toggle('is-active', dir > 0);
    });
    button.addEventListener('pointerup', clearCompactMove);
    button.addEventListener('pointercancel', clearCompactMove);
    button.addEventListener('pointerleave', clearCompactMove);
  };
  updateCanvasRect();
  if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(() => updateCanvasRect()).observe(gameCanvas);
  }
  window.addEventListener('resize', () => {
    updateCanvasRect();
    updateLayoutHint();
  });
  window.addEventListener('orientationchange', () => {
    updateCanvasRect();
    updateLayoutHint();
  });

  gameCanvas.addEventListener('pointerdown', (e) => {
    if (!e.isPrimary) return;
    updateCanvasRect();
    activePointerId = e.pointerId;
    pointerStartX = e.clientX;
    pointerStartY = e.clientY;
    pointerDragDistance = 0;
    pointerVerticalDistance = 0;
    pointerSwipeDropTriggered = false;
    setDropFromClientX(e.clientX);
    gameCanvas.setPointerCapture(e.pointerId);
  });

  gameCanvas.addEventListener('pointermove', (e) => {
    if (activePointerId !== null && e.pointerId === activePointerId) {
      pointerDragDistance = Math.max(pointerDragDistance, Math.abs(e.clientX - pointerStartX));
      pointerVerticalDistance = Math.max(pointerVerticalDistance, e.clientY - pointerStartY);
      if (!pointerSwipeDropTriggered && e.pointerType !== 'mouse' && pointerVerticalDistance > 36 && pointerVerticalDistance > pointerDragDistance) {
        pointerSwipeDropTriggered = true;
        activePointerId = null;
        pointerDragDistance = 0;
        drop();
      }
    }
    setDropFromClientX(e.clientX);
  });

  gameCanvas.addEventListener('pointerup', (e) => {
    if (activePointerId !== e.pointerId) return;
    setDropFromClientX(e.clientX);
    const shouldDrop = !pointerSwipeDropTriggered && (e.pointerType !== 'mouse' || pointerDragDistance < 8);
    activePointerId = null;
    pointerDragDistance = 0;
    pointerVerticalDistance = 0;
    pointerSwipeDropTriggered = false;
    if (shouldDrop) drop();
  });

  gameCanvas.addEventListener('pointercancel', () => {
    activePointerId = null;
    pointerDragDistance = 0;
    pointerVerticalDistance = 0;
    pointerSwipeDropTriggered = false;
  });

  document.addEventListener('keydown', (e) => {
    pressedKeys.add(e.key);
    if (e.key === ' ') { e.preventDefault(); drop(); }
  });
  document.addEventListener('keyup', (e) => pressedKeys.delete(e.key));

  restartBtn.addEventListener('click', restart);
  document.getElementById('score-restart-btn')!.addEventListener('click', restart);

  btnHooke.addEventListener('click', () => setContactModel('hooke'));
  btnHertz.addEventListener('click', () => setContactModel('hertz'));
  chkForces.addEventListener('change', () => {
    showForceChains = chkForces.checked;
    syncControlState();
  });
  bindCompactMoveButton(compactLeftBtn, -1);
  bindCompactMoveButton(compactRightBtn, 1);
  compactDropBtn.addEventListener('click', () => drop());
  compactModelBtn.addEventListener('click', () => setContactModel(contactModel === 'hooke' ? 'hertz' : 'hooke'));
  compactForceBtn.addEventListener('click', () => {
    showForceChains = !showForceChains;
    syncControlState();
  });
  syncControlState();
  updateLayoutHint();
}

// ================================================================
// Main Loop
// ================================================================

function update(now: number) {
  updateFpsAdaptation(now);
  const profile = getPerformanceProfile();
  if (!gameOver) {
    if (pressedKeys.has('ArrowLeft') || pressedKeys.has('a') || pressedKeys.has('A')) dropX = Math.max(CL + 20, dropX - 6);
    if (pressedKeys.has('ArrowRight') || pressedKeys.has('d') || pressedKeys.has('D')) dropX = Math.min(CR - 20, dropX + 6);
    if (compactMoveDir < 0) dropX = Math.max(CL + 20, dropX - 10);
    if (compactMoveDir > 0) dropX = Math.min(CR - 20, dropX + 10);
    const sub = profile.subSteps;
    const dt = (1 / 60) / sub;
    for (let i = 0; i < sub; i++) {
      physicsStep(dt, profile);
    }
    totalSimSteps += sub;
    if (!canDrop) stepsSinceDrop += sub;
    processMerges();
    checkGameOver();
    for (let i = 0; i < pCount; i++) {
      if (pGrace[i] > 0) pGrace[i] -= sub;
    }

    // Release the drop gate once the last particle settles below the line or contacts
    if (!canDrop && lastDroppedId !== null) {
      const droppedIndex = findIndexById(lastDroppedId);
      if (droppedIndex === -1) {
        releaseDropGate();
      } else {
        const fullyBelowLine = (py[droppedIndex] - pRadius[droppedIndex]) > DANGER_Y;
        const hasContact = !fullyBelowLine && overlapsAnyParticle(droppedIndex, 2);
        if (fullyBelowLine || hasContact) {
          releaseDropGate(px[droppedIndex], py[droppedIndex]);
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
      pCount, px, py, pRadius, pAngle, pLevel,
      contactVis, contactVisLen, effects, scorePopups,
      comboCount, comboTimer, currentLevel, dropX, canDrop, gameOver,
      contactModel, showForceChains,
      dangerTimer > 0 ? 0.7 + 0.3 * Math.abs(Math.sin(Date.now() / 130)) : 0.6,
      Date.now() / 1000,
      profile,
    );
  } else {
    drawGameCanvas2D(profile);
  }

  if (chartDirty || (Date.now() - lastChartFrame > profile.chartIntervalMs)) {
    drawGradingChart();
    chartDirty = false;
    lastChartFrame = Date.now();
  }

  if ((totalSimSteps & 31) === 0) updateHudStats();

  if (gameOver && dangerTimer > 0) dangerTimer--;

  requestAnimationFrame(update);
}

// ================================================================
// Init
// ================================================================

async function main() {
  await initRenderer();
  setupMute();
  currentLevel = getRandomLevel();
  nextLevel = getRandomLevel();
  updateNextPreview();
  updateHudStats();
  setupInput();
  update(performance.now());
}

main().catch(console.error);

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
