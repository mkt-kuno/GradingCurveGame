// ================================================================
// 粒度分布ゲーム - 土質工学スイカゲーム
// DEM (Discrete Element Method) Physics Engine
// Contact Models: Hooke (linear) / Hertz (non-linear)
// References: Tsuji et al. (1992), LAMMPS pair_gran
// ================================================================

// ================================================================
// Particle Level Definitions (JIS A 1204 ふるい寸法)
// ================================================================

interface ParticleLevel {
  name: string;
  sieve: string;
  upperSieveMM: number;
  radius: number;
  color: string;
  strokeColor: string;
  score: number;
}

const LEVELS: ParticleLevel[] = [
  { name: '細粒分', sieve: '<4.75mm', upperSieveMM: 4.75,  radius: 14, color: '#F5CBA7', strokeColor: '#C9956A', score: 1 },
  { name: '細礫',   sieve: '4.75mm',  upperSieveMM: 9.5,   radius: 20, color: '#F0B27A', strokeColor: '#C97833', score: 3 },
  { name: '小礫',   sieve: '9.5mm',   upperSieveMM: 19,    radius: 27, color: '#E67E22', strokeColor: '#BA6518', score: 6 },
  { name: '中礫',   sieve: '19mm',    upperSieveMM: 26.5,  radius: 35, color: '#E74C3C', strokeColor: '#B83227', score: 10 },
  { name: '粗礫',   sieve: '26.5mm',  upperSieveMM: 37.5,  radius: 44, color: '#AF7AC5', strokeColor: '#7D3C98', score: 15 },
  { name: '大礫',   sieve: '37.5mm',  upperSieveMM: 52,    radius: 54, color: '#5DADE2', strokeColor: '#2874A6', score: 21 },
  { name: '巨礫',   sieve: '52mm',    upperSieveMM: 75,    radius: 65, color: '#58D68D', strokeColor: '#1E8449', score: 28 },
  { name: '転石',   sieve: '75mm+',   upperSieveMM: 100,   radius: 77, color: '#F7DC6F', strokeColor: '#B7950B', score: 36 },
];

const SIEVE_SIZES = [4.75, 9.5, 19, 26.5, 37.5, 52, 75];

// ================================================================
// Game Dimensions
// ================================================================

const GAME_W = 450;
const GAME_H = 700;
const WALL_T = 25;
const CL = WALL_T;
const CR = GAME_W - WALL_T;
const CB = GAME_H - WALL_T;
const DROP_Y = 55;
const DANGER_Y = 95;

// ================================================================
// Material Properties
// ================================================================
//
// Silica sand (珪砂, e.g. Toyoura sand):
//   E = 70 GPa,  ν = 0.17,  ρ = 2650 kg/m³
//   Internal friction angle φ ≈ 30-40° → μ ≈ tan(24°) ≈ 0.45
//   Restitution e ≈ 0.50
//
// NBR rubber (wall liner):
//   E = 10 MPa,  ν = 0.49 (nearly incompressible)
//   μ ≈ 0.70 (rubber-sand friction)
//   Restitution e ≈ 0.55
//
// Effective contact modulus E* (Hertz theory):
//   1/E* = (1-ν₁²)/E₁ + (1-ν₂²)/E₂
//
//   Particle-Particle (sand-sand):
//     1/E* = 2×(1-0.17²)/70e9 = 0.02775/GPa → E* = 36.0 GPa
//
//   Particle-Wall (sand-NBR):
//     1/E* = (1-0.17²)/70e9 + (1-0.49²)/10e6
//          ≈ 1.39e-11 + 7.60e-8 ≈ 7.60e-8 → E* = 13.2 MPa
//
//   Real ratio E*_PP/E*_PW ≈ 2740 (sand-sand >> sand-rubber)
//
// Effective radius R*:
//   Particle-Particle: R* = (R₁·R₂)/(R₁+R₂)
//   Particle-Wall:     R* = R_particle  (flat wall → R_wall = ∞)
//
// ================================================================

const GRAVITY = 700;

// Pixel-space effective modulus for Hertz model
// F_n = (4/3) · E*_px · √R* · δ^(3/2)
const ESTAR_PP = 2000;
const ESTAR_PW = 200;

// Pixel-space stiffness for Hooke model
// F_n = k_n · δ
const KN_PP = 25000;
const KN_PW = 2500;

// Friction coefficients
const MU_PP = 0.45;
const MU_PW = 0.70;

// Coefficients of restitution
const RESTITUTION_PP = 0.50;
const RESTITUTION_PW = 0.55;

// Damping ratio β from restitution (Tsuji et al. 1992):
//   β = -ln(e) / √(π² + ln²(e))
function beta(e: number): number {
  if (e <= 0) return 1;
  if (e >= 1) return 0;
  const lnE = Math.log(e);
  return -lnE / Math.sqrt(Math.PI * Math.PI + lnE * lnE);
}

const BETA_PP = beta(RESTITUTION_PP);
const BETA_PW = beta(RESTITUTION_PW);

// Tangential stiffness ratio (Mindlin-Deresiewicz):
//   k_t / k_n = 4·(1-ν) / (2-ν)
//   For ν=0.17: k_t/k_n = 4·0.83/1.83 ≈ 1.81
const KT_RATIO = 4 * (1 - 0.17) / (2 - 0.17);

const SUB_STEPS = 8;
const AIR_DRAG = 0.9992;

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
  level: number;
  angle: number;
}

interface Contact {
  idA: number;
  idB: number;
  x: number;
  y: number;
  force: number;
}

interface Effect {
  x: number;
  y: number;
  r: number;
  alpha: number;
  color: string;
}

interface ScorePopup {
  x: number;
  y: number;
  text: string;
  timer: number;
}

type ContactModel = 'hooke' | 'hertz';

// ================================================================
// Game State
// ================================================================

let particles: Particle[] = [];
let nextId = 0;
let activeContacts: Contact[] = [];
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
let dangerTimer = 0;
let mergeCount = 0;
let contactModel: ContactModel = 'hertz';
let showForceChains = false;
let comboCount = 0;
let comboTimer = 0;

// ================================================================
// DOM
// ================================================================

const gameCanvas = document.getElementById('game-canvas') as HTMLCanvasElement;
const gCtx = gameCanvas.getContext('2d')!;
const chartCanvas = document.getElementById('chart-canvas') as HTMLCanvasElement;
const cCtx = chartCanvas.getContext('2d')!;
const nextCanvas = document.getElementById('next-canvas') as HTMLCanvasElement;
const nCtx = nextCanvas.getContext('2d')!;

const scoreEl = document.getElementById('score-value')!;
const mergeEl = document.getElementById('merge-count')!;
const nextNameEl = document.getElementById('next-name')!;
const gameOverEl = document.getElementById('game-over')!;
const finalScoreEl = document.getElementById('final-score')!;
const restartBtn = document.getElementById('restart-btn')!;
const paramsEl = document.getElementById('grading-params')!;
const btnHooke = document.getElementById('btn-hooke') as HTMLButtonElement;
const btnHertz = document.getElementById('btn-hertz') as HTMLButtonElement;
const chkForces = document.getElementById('chk-forces') as HTMLInputElement;

// ================================================================
// DEM Physics Engine
// ================================================================
//
// Normal contact force (two models):
//
//   Hooke (linear spring-dashpot):
//     F_n = k_n · δ                                    (elastic)
//         - c_n · v_n                                   (damping)
//     where c_n = 2·β·√(m*·k_n)                        (Tsuji damping)
//
//   Hertz (non-linear, Johnson 1985):
//     F_n = (4/3)·E*·√R* · δ^(3/2)                     (elastic)
//         - c_n(δ) · v_n                                 (damping)
//     where c_n = 2·β·√(m*·k_n_eff)                    (Tsuji damping)
//           k_n_eff = dF/dδ = 2·E*·√(R*·δ)             (incremental stiffness)
//
//   In both cases: F_n = max(0, F_elastic + F_damp)
//     (no tensile force)
//
// Tangential contact force:
//   Simplified Coulomb friction (no tangential spring tracking):
//     |F_t| = min(μ · F_n, m* · |v_t| / dt)
//
//   Full Mindlin-Deresiewicz would track δ_t incrementally:
//     k_t = 8·G*·√(R*·δ)  for Hertz-Mindlin
//     k_t = KT_RATIO · k_n for simplified
//
//   Sign convention:
//     v_n > 0 : separating
//     v_n < 0 : approaching
//     F_damp = -c_n · v_n  (opposes relative velocity)
//
// ================================================================

function createParticle(x: number, y: number, level: number): Particle {
  const r = LEVELS[level].radius;
  return {
    id: nextId++,
    x, y, vx: 0, vy: 0,
    radius: r,
    mass: r * r * 0.012,
    level,
    angle: 0,
  };
}

function pairKey(a: number, b: number): string {
  return a < b ? `${a}_${b}` : `${b}_${a}`;
}

function computeNormalForce(
  delta: number,
  rEff: number,
  mEff: number,
  vn: number,
  eStar: number,
  kn: number,
  b: number,
): number {
  if (delta <= 0) return 0;

  let fn_elastic: number;
  let kn_eff: number;

  if (contactModel === 'hertz') {
    fn_elastic = (4 / 3) * eStar * Math.sqrt(rEff) * Math.pow(delta, 1.5);
    kn_eff = 2 * eStar * Math.sqrt(rEff * delta);
  } else {
    fn_elastic = kn * delta;
    kn_eff = kn;
  }

  const cn = 2 * b * Math.sqrt(mEff * kn_eff);
  return Math.max(0, fn_elastic - cn * vn);
}

function physicsStep(dt: number) {
  const newContacts: Contact[] = [];
  const newPairs = new Set<string>();

  for (const p of particles) {
    p.vy += GRAVITY * dt;
  }

  // --- Particle-Particle contacts ---
  for (let i = 0; i < particles.length; i++) {
    for (let j = i + 1; j < particles.length; j++) {
      const a = particles[i];
      const b = particles[j];

      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distSq = dx * dx + dy * dy;
      const minDist = a.radius + b.radius;

      if (distSq >= minDist * minDist) continue;

      const dist = Math.sqrt(Math.max(distSq, 1e-8));
      const delta = minDist - dist;
      if (delta <= 0) continue;

      const nx = dx / dist;
      const ny = dy / dist;

      // Relative velocity of B w.r.t. A
      const dvx = b.vx - a.vx;
      const dvy = b.vy - a.vy;

      // Normal relative velocity (positive = separating)
      const vn = dvx * nx + dvy * ny;

      // Tangential relative velocity
      const vtx = dvx - vn * nx;
      const vty = dvy - vn * ny;
      const vtMag = Math.sqrt(vtx * vtx + vty * vty);

      // Effective mass: m* = (m₁·m₂)/(m₁+m₂)
      const mEff = (a.mass * b.mass) / (a.mass + b.mass);

      // Effective radius: R* = (R₁·R₂)/(R₁+R₂)
      const rEff = (a.radius * b.radius) / (a.radius + b.radius);

      // Normal force (Hooke or Hertz with Tsuji damping)
      const fn = computeNormalForce(delta, rEff, mEff, vn, ESTAR_PP, KN_PP, BETA_PP);

      // Tangential force (Coulomb friction)
      let ft = 0;
      if (vtMag > 0.001) {
        ft = Math.min(MU_PP * fn, mEff * vtMag / dt);
      }

      // Force on B: F_n·n - F_t·t_hat (friction opposes sliding)
      const fx = fn * nx - ft * (vtMag > 0.001 ? vtx / vtMag : 0);
      const fy = fn * ny - ft * (vtMag > 0.001 ? vty / vtMag : 0);

      // Apply: a = F/m (Newton's 2nd law)
      b.vx += (fx / b.mass) * dt;
      b.vy += (fy / b.mass) * dt;
      a.vx -= (fx / a.mass) * dt;
      a.vy -= (fy / a.mass) * dt;

      // Position correction (prevent excessive overlap)
      const totalM = a.mass + b.mass;
      const corr = delta * 0.4;
      a.x -= nx * corr * (b.mass / totalM);
      a.y -= ny * corr * (b.mass / totalM);
      b.x += nx * corr * (a.mass / totalM);
      b.y += ny * corr * (a.mass / totalM);

      // Store contact for force chain visualization
      const fMag = Math.sqrt(fx * fx + fy * fy);
      newContacts.push({
        idA: a.id, idB: b.id,
        x: (a.x * b.radius + b.x * a.radius) / (a.radius + b.radius),
        y: (a.y * b.radius + b.y * a.radius) / (a.radius + b.radius),
        force: fMag,
      });

      // Merge detection: same-level particles on first contact
      const key = pairKey(a.id, b.id);
      newPairs.add(key);
      if (!contactedPairs.has(key) && a.level === b.level) {
        mergeQueue.push([a.id, b.id]);
      }
    }
  }

  // --- Particle-Wall contacts (NBR rubber) ---
  // Wall: R* = R_particle (flat surface), wall mass = ∞ → m* = m_particle
  for (const p of particles) {
    // Bottom wall: n = (0, -1), v_n = -vy
    const oBot = p.y + p.radius - CB;
    if (oBot > 0) {
      const vn = -p.vy;
      const fn = computeNormalForce(oBot, p.radius, p.mass, vn, ESTAR_PW, KN_PW, BETA_PW);
      p.vy -= (fn / p.mass) * dt;
      if (Math.abs(p.vx) > 0.01) {
        const ft = Math.min(MU_PW * fn, p.mass * Math.abs(p.vx) / dt);
        p.vx -= Math.sign(p.vx) * (ft / p.mass) * dt;
      }
      p.y = Math.min(p.y, CB - p.radius);
    }

    // Left wall: n = (1, 0), v_n = vx
    const oLeft = CL - (p.x - p.radius);
    if (oLeft > 0) {
      const vn = p.vx;
      const fn = computeNormalForce(oLeft, p.radius, p.mass, vn, ESTAR_PW, KN_PW, BETA_PW);
      p.vx += (fn / p.mass) * dt;
      p.x = Math.max(p.x, CL + p.radius);
    }

    // Right wall: n = (-1, 0), v_n = -vx
    const oRight = (p.x + p.radius) - CR;
    if (oRight > 0) {
      const vn = -p.vx;
      const fn = computeNormalForce(oRight, p.radius, p.mass, vn, ESTAR_PW, KN_PW, BETA_PW);
      p.vx -= (fn / p.mass) * dt;
      p.x = Math.min(p.x, CR - p.radius);
    }
  }

  // --- Integration (symplectic Euler) ---
  for (const p of particles) {
    p.vx *= AIR_DRAG;
    p.vy *= AIR_DRAG;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.angle += (p.vx / Math.max(p.radius, 1)) * dt;
  }

  activeContacts = newContacts;
  contactedPairs = newPairs;
}

// ================================================================
// Merge Logic
// ================================================================
// Same-level particles merge on first contact:
//   level < max → next level (standard Suika mechanic)
//   level == max → annihilation (both disappear, big bonus)

function processMerges() {
  const consumed = new Set<number>();

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

    if (newLevel >= LEVELS.length) {
      // Max-level annihilation (転石 + 転石 → disappear)
      const pts = LEVELS[a.level].score * 5;
      score += pts;
      mergeCount++;
      comboCount++;
      comboTimer = 45;
      effects.push({ x: mx, y: my, r: 10, alpha: 1, color: '#FFFFFF' });
      effects.push({ x: mx, y: my, r: 30, alpha: 0.7, color: '#F7DC6F' });
      scorePopups.push({ x: mx, y: my, text: `+${pts} MAX!`, timer: 90 });
      scoreEl.textContent = score.toString();
      mergeEl.textContent = `合体回数: ${mergeCount}`;
      continue;
    }

    // Normal merge: conservation of momentum
    const tm = a.mass + b.mass;
    const nvx = (a.vx * a.mass + b.vx * b.mass) / tm;
    const nvy = (a.vy * a.mass + b.vy * b.mass) / tm;

    const np = createParticle(mx, my, newLevel);
    np.vx = nvx;
    np.vy = nvy - 40;
    particles.push(np);

    effects.push({ x: mx, y: my, r: LEVELS[newLevel].radius * 0.3, alpha: 1, color: LEVELS[newLevel].color });

    comboCount++;
    comboTimer = 45;
    const pts = Math.floor(LEVELS[newLevel].score * (1 + (comboCount - 1) * 0.5));
    score += pts;
    mergeCount++;

    scorePopups.push({ x: mx, y: my, text: `+${pts}`, timer: 60 });

    scoreEl.textContent = score.toString();
    mergeEl.textContent = `合体回数: ${mergeCount}`;
  }

  mergeQueue = [];
  if (comboTimer > 0) {
    comboTimer--;
    if (comboTimer === 0) comboCount = 0;
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
  currentLevel = nextLevel;
  nextLevel = getRandomLevel();
  updateNextPreview();

  setTimeout(() => { if (!gameOver) canDrop = true; }, 450);
}

function checkGameOver() {
  let above = false;
  for (const p of particles) {
    const spd = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
    if (spd < 80 && p.y - p.radius < DANGER_Y) {
      above = true;
      break;
    }
  }

  if (above) {
    dangerTimer++;
    if (dangerTimer > 90) {
      gameOver = true;
      finalScoreEl.textContent = score.toString();
      gameOverEl.style.display = 'flex';
    }
  } else {
    dangerTimer = Math.max(0, dangerTimer - 2);
  }
}

function restart() {
  particles = [];
  activeContacts = [];
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
  dangerTimer = 0;
  comboCount = 0;
  comboTimer = 0;
  mergeCount = 0;

  scoreEl.textContent = '0';
  mergeEl.textContent = '合体回数: 0';
  gameOverEl.style.display = 'none';
  updateNextPreview();
}

// ================================================================
// Rendering
// ================================================================

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

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

function drawParticle(ctx: CanvasRenderingContext2D, x: number, y: number, angle: number, level: number) {
  const info = LEVELS[level];
  const r = info.radius;

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);

  ctx.beginPath();
  ctx.arc(2, 3, r, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(0,0,0,0.18)';
  ctx.fill();

  const grad = ctx.createRadialGradient(-r * 0.25, -r * 0.25, r * 0.05, 0, 0, r);
  grad.addColorStop(0, lighten(info.color, 40));
  grad.addColorStop(1, info.color);
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.strokeStyle = info.strokeColor;
  ctx.lineWidth = 2;
  ctx.stroke();

  const rng = mulberry32(level * 54321 + 7);
  const dotN = Math.min(level * 4 + 3, 18);
  for (let i = 0; i < dotN; i++) {
    const ddx = (rng() - 0.5) * r * 1.4;
    const ddy = (rng() - 0.5) * r * 1.4;
    if (ddx * ddx + ddy * ddy < (r * 0.7) ** 2) {
      ctx.beginPath();
      ctx.arc(ddx, ddy, Math.max(1, r * 0.055), 0, Math.PI * 2);
      ctx.fillStyle = hexToRGBA(info.strokeColor, 0.35);
      ctx.fill();
    }
  }

  const fontSize = Math.max(7, Math.floor(r * 0.34));
  ctx.font = `bold ${fontSize}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#222';
  ctx.fillText(info.sieve, 0, 0);

  ctx.restore();
}

function drawGame() {
  const ctx = gCtx;

  ctx.fillStyle = '#0f0f23';
  ctx.fillRect(0, 0, GAME_W, GAME_H);

  const bgGrad = ctx.createLinearGradient(0, 0, 0, GAME_H);
  bgGrad.addColorStop(0, '#151530');
  bgGrad.addColorStop(1, '#0d0d20');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(CL, 0, CR - CL, GAME_H);

  ctx.fillStyle = '#1e1e45';
  ctx.fillRect(0, 0, WALL_T, GAME_H);
  ctx.fillRect(CR, 0, WALL_T, GAME_H);
  ctx.fillRect(0, CB, GAME_W, WALL_T);

  ctx.strokeStyle = '#3a3a7a';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(CL, 0);
  ctx.lineTo(CL, CB);
  ctx.lineTo(CR, CB);
  ctx.lineTo(CR, 0);
  ctx.stroke();

  const dAlpha = dangerTimer > 0 ? 0.25 + 0.5 * Math.abs(Math.sin(Date.now() / 130)) : 0.15;
  ctx.strokeStyle = `rgba(255,60,60,${dAlpha})`;
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 5]);
  ctx.beginPath();
  ctx.moveTo(CL, DANGER_Y);
  ctx.lineTo(CR, DANGER_Y);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = `rgba(255,60,60,${dAlpha * 0.7})`;
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('DANGER', CR - 4, DANGER_Y - 4);

  if (showForceChains && activeContacts.length > 0) {
    const maxF = Math.max(...activeContacts.map(c => c.force), 1);
    for (const c of activeContacts) {
      const t = Math.min(c.force / maxF, 1);
      const w = 1 + t * 5;
      ctx.strokeStyle = `rgba(${Math.floor(50 + 205 * t)},${Math.floor(200 * (1 - t))},${Math.floor(255 * (1 - t))},${0.3 + t * 0.5})`;
      ctx.lineWidth = w;
      ctx.beginPath();
      ctx.arc(c.x, c.y, w * 2, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  for (const e of effects) {
    ctx.beginPath();
    ctx.arc(e.x, e.y, e.r, 0, Math.PI * 2);
    ctx.strokeStyle = hexToRGBA(e.color, e.alpha);
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(e.x, e.y, e.r * 0.5, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(255,255,255,${e.alpha * 0.5})`;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  for (const p of particles) {
    drawParticle(ctx, p.x, p.y, p.angle, p.level);
  }

  for (const sp of scorePopups) {
    const alpha = sp.timer / 60;
    ctx.fillStyle = `rgba(247,220,111,${alpha})`;
    ctx.font = `bold ${14 + (1 - alpha) * 8}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText(sp.text, sp.x, sp.y - 30 + (1 - alpha) * 20);
  }

  if (comboCount > 1 && comboTimer > 0) {
    const alpha = comboTimer / 45;
    ctx.fillStyle = `rgba(255,100,100,${alpha})`;
    ctx.font = `bold ${22 + comboCount * 2}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText(`${comboCount} COMBO!`, GAME_W / 2, GAME_H / 2 - 60);
  }

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

    drawParticle(ctx, cx, DROP_Y, 0, currentLevel);
  }

  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(contactModel === 'hertz' ? 'Hertz Contact' : 'Hooke Contact', CL + 4, 14);
  ctx.fillText(`N=${particles.length}`, CL + 4, 26);
}

// ================================================================
// Grading Chart
// ================================================================

function drawGradingChart() {
  const ctx = cCtx;
  const W = chartCanvas.width;
  const H = chartCanvas.height;
  const pad = { top: 15, right: 12, bottom: 32, left: 38 };
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
  ctx.font = '8px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const p of [0, 20, 40, 60, 80, 100]) {
    const y = pad.top + pH - (p / 100) * pH;
    ctx.fillText(`${p}`, pad.left - 3, y);
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + pW, y);
    ctx.stroke();
  }

  const logMin = Math.log10(1);
  const logMax = Math.log10(150);

  function toX(mm: number): number {
    return pad.left + ((Math.log10(mm) - logMin) / (logMax - logMin)) * pW;
  }

  const xTicks = [1, 2, 4.75, 9.5, 19, 26.5, 37.5, 52, 75, 100];
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
    ctx.font = isMain ? '8px sans-serif' : '7px sans-serif';
    ctx.fillText(`${s}`, x, pad.top + pH + 2);
  }

  ctx.font = '8px sans-serif';
  ctx.fillStyle = '#555';
  ctx.textAlign = 'center';
  ctx.fillText('粒径 (mm)', pad.left + pW / 2, pad.top + pH + 20);

  ctx.save();
  ctx.translate(9, pad.top + pH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('通過質量百分率 (%)', 0, 0);
  ctx.restore();

  if (totalMass < 0.01) {
    ctx.fillStyle = '#aaa';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('粒子を落としてください', W / 2, H / 2);
    paramsEl.textContent = '';
    return;
  }

  const cumMass: number[] = [];
  let cm = 0;
  for (let i = 0; i < LEVELS.length; i++) {
    cm += counts[i];
    cumMass.push(cm);
  }

  const points: { x: number; y: number }[] = [];
  points.push({ x: 150, y: 100 });
  for (let i = SIEVE_SIZES.length - 1; i >= 0; i--) {
    points.push({ x: SIEVE_SIZES[i], y: (cumMass[i] / totalMass) * 100 });
  }
  points.push({ x: 0.8, y: 0 });
  points.sort((a, b) => a.x - b.x);

  ctx.strokeStyle = '#e74c3c';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const px = toX(points[i].x);
    const py = pad.top + pH - (points[i].y / 100) * pH;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  ctx.fillStyle = 'rgba(231,76,60,0.1)';
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    const px = toX(points[i].x);
    const py = pad.top + pH - (points[i].y / 100) * pH;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
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
    ctx.fillStyle = '#c0392b';
    ctx.fill();
  }

  const d10 = interpD(10, points);
  const d30 = interpD(30, points);
  const d60 = interpD(60, points);

  let txt = '';
  if (d10 !== null) txt += `D\u2081\u2080=${d10.toFixed(1)}mm  `;
  if (d30 !== null) txt += `D\u2083\u2080=${d30.toFixed(1)}mm  `;
  if (d60 !== null) txt += `D\u2086\u2080=${d60.toFixed(1)}mm`;
  if (d10 !== null && d60 !== null) {
    const Cu = d60 / d10;
    txt += `\nCu=${Cu.toFixed(2)}`;
    if (d30 !== null) {
      const Cc = (d30 * d30) / (d10 * d60);
      txt += `  Cc=${Cc.toFixed(2)}`;
      if (Cu >= 4 && Cc >= 1 && Cc <= 3) {
        txt += '\n\u2192 良粒度礫 (GW)';
      } else if (Cu >= 6 && Cc >= 1 && Cc <= 3) {
        txt += '\n\u2192 良粒度砂 (SW)';
      } else {
        txt += '\n\u2192 不良粒度 (GP/SP)';
      }
    }
  }
  paramsEl.textContent = txt;
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
  const s = 60;
  ctx.clearRect(0, 0, s, s);
  const r = LEVELS[nextLevel].radius;
  const sc = Math.min(24 / r, 1);
  ctx.save();
  ctx.translate(s / 2, s / 2);
  ctx.scale(sc, sc);
  drawParticle(ctx, 0, 0, 0, nextLevel);
  ctx.restore();
  nextNameEl.textContent = `${LEVELS[nextLevel].name} (${LEVELS[nextLevel].sieve})`;
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
  }

  for (const e of effects) {
    e.r += 2.5;
    e.alpha -= 0.025;
  }
  effects = effects.filter(e => e.alpha > 0);

  for (const sp of scorePopups) sp.timer--;
  scorePopups = scorePopups.filter(sp => sp.timer > 0);

  drawGame();
  drawGradingChart();

  requestAnimationFrame(update);
}

// ================================================================
// Init
// ================================================================

currentLevel = getRandomLevel();
nextLevel = getRandomLevel();
updateNextPreview();
setupInput();
update();
