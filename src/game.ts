// ================================================================
// 粒度分布ゲーム - 土質工学メロンゲーム
// DEM (Discrete Element Method) Physics Engine
// Contact: Hooke (linear) / Hertz (non-linear) + Tsuji damping
// Integration: Velocity-Verlet (position-Verlet variant)
// Tangential: Coulomb friction with rotational sliding
// Rolling friction: CDT model
// References: Tsuji et al.(1992), Silbert et al.(2001), LAMMPS pair_granular
// ================================================================

// ================================================================
// Particle Levels (JIS A 1204, radii x2)
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
  { name: '砂',     sieve: '0.75mm',  upperSieveMM: 2,     radius: 22, color: '#FDF5E6', strokeColor: '#C8B89C', score: 1 },
  { name: '細礫',   sieve: '2mm',     upperSieveMM: 4.75,  radius: 30, color: '#F8EED8', strokeColor: '#C4B49C', score: 2 },
  { name: '中礫',   sieve: '4.75mm',  upperSieveMM: 9.5,   radius: 42, color: '#F5E6CC', strokeColor: '#B8A48A', score: 4 },
  { name: '中礫',   sieve: '9.5mm',   upperSieveMM: 19,    radius: 60, color: '#EAD5B8', strokeColor: '#A89070', score: 7 },
  { name: '粗礫',   sieve: '19mm',    upperSieveMM: 26.5,  radius: 81, color: '#DCC4A0', strokeColor: '#9A7C5A', score: 11 },
  { name: '粗礫',   sieve: '26.5mm',  upperSieveMM: 37.5,  radius: 105, color: '#D0B48E', strokeColor: '#8C6C46', score: 16 },
  { name: '粗礫',   sieve: '37.5mm',  upperSieveMM: 53,    radius: 132, color: '#C2A47A', strokeColor: '#7E5E38', score: 22 },
  { name: '粗礫',   sieve: '53mm',    upperSieveMM: 75,    radius: 162, color: '#B49468', strokeColor: '#6E5030', score: 29 },
  { name: '石分',   sieve: '75mm',    upperSieveMM: 100,   radius: 195, color: '#A28458', strokeColor: '#5E4228', score: 37 },
  { name: '石分',   sieve: '100mm+',  upperSieveMM: 150,   radius: 231, color: '#907448', strokeColor: '#4E3420', score: 46 },
];

const SIEVE_SIZES = [0.75, 2, 4.75, 9.5, 19, 26.5, 37.5, 53, 75];

// ================================================================
// Game Dimensions — 1:1 container (792 x 792 inner)
// Canvas 858 x 924: 33px walls, 99px drop zone at top
// ================================================================

const GAME_W = 858;
const GAME_H = 865;
const WALL_T = 33;
const CL = WALL_T;
const CR = GAME_W - WALL_T;         // 825
const CB = GAME_H - WALL_T;         // 832
const CONTAINER_W = CR - CL;         // 792
const CONTAINER_H = CB - WALL_T;     // 719
const DROP_Y = 72;
const DANGER_Y = 116;

// ================================================================
// Material Properties
// ================================================================
// Silica sand (珪砂): E=70GPa, ν=0.17, ρ=2650kg/m³
// NBR rubber (wall):  E=10MPa, ν=0.49
//
// E* derivation:
//   PP (sand-sand): 1/E* = 2(1-0.17²)/70e9 → E* ≈ 36.0 GPa
//   PW (sand-NBR):  1/E* = (1-0.17²)/70e9 + (1-0.49²)/10e6 → E* ≈ 13.2 MPa
//   Ratio E*_PP/E*_PW ≈ 2740
//
// Pixel-space values scaled for gameplay while preserving ratio

// Material Properties — 全て珪砂 (Silica sand)
// ================================================================
// Silica sand (珪砂, e.g. Toyoura sand):
//   E = 70 GPa,  ν = 0.17,  ρ = 2650 kg/m³
//   Internal friction angle φ ≈ 30° → μ ≈ 0.45
//   Restitution e ≈ 0.50
//   Rolling friction μ_r ≈ 0.02-0.05
//
// E* derivation (Hertz):
//   PP (sand-sand): 1/E* = 2(1-0.17²)/70e9 → E* ≈ 36.0 GPa
//   PW (sand-wall): 壁も珪砂 → E* ≈ 36.0 GPa (same as PP)
//
// Pixel-space values scaled for gameplay

const GRAVITY = 700;

// Hertz: F_n = (4/3)·E*·√R*·δ^(3/2)
const ESTAR_PP = 10000;
const ESTAR_PW = 10000;           // 壁も珪砂 → PPと同じ

// Hooke: F_n = k_n · δ
const KN_PP = 100000;
const KN_PW = 100000;             // 壁も珪砂 → PPと同じ

const MU_PP = 0.65;               // 珪砂内部摩擦 (φ≈33°, tan33°≈0.65)
const MU_PW = 0.80;               // 壁も珪砂 → 同じ
const REST_PP = 0.35;             // 珪砂-珪砂反発係数 (文献: 0.3-0.5)
const REST_PW = 0.35;             // 壁も珪砂 → 同じ

// Rolling friction (CDT): μ_r for angular silica sand
// Benmebarek (2023): 0.1-0.6; Gu (2020): ~0.2; Rorato (2021): image-based 0.1-0.3
const MU_ROLL_PP = 0.15;         // PP: 珪砂粒子間 (文献範囲 0.1-0.2)
const MU_ROLL_PW = 0.30;         // PW: 壁面 (文献範囲 0.2-0.4)

// β = −ln(e) / √(π²+ln²(e))  (Tsuji damping)
function beta(e: number): number {
  if (e <= 0) return 1;
  if (e >= 1) return 0;
  const ln = Math.log(e);
  return -ln / Math.sqrt(Math.PI * Math.PI + ln * ln);
}

const BETA_PP = beta(REST_PP);
const BETA_PW = beta(REST_PW);

const SUB_STEPS = 10;
const MAX_DELTA_RATIO = 0.08;     // max overlap = 8% of min radius
const MAX_VEL = 3000;
const MAX_OMEGA = 80;
const VEL_DAMP = 0.9995;          // per substep air drag
const ANG_DAMP = 0.998;           // per substep angular drag

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
  inertia: number;                // I = 0.5·m·R² (2D disk)
  level: number;
  angle: number;
  omega: number;                  // angular velocity (rad/s)
  active: boolean;                // true once center has been below DANGER_Y
  graceFrames: number;            // frames of immunity after merge
}

interface ContactVis {
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
const highScoreEl = document.getElementById('high-score')!;
const nextNameEl = document.getElementById('next-name')!;
const gameOverEl = document.getElementById('game-over')!;
const finalScoreEl = document.getElementById('final-score')!;
const restartBtn = document.getElementById('restart-btn')!;
const paramsEl = document.getElementById('grading-params')!;
const btnHooke = document.getElementById('btn-hooke') as HTMLButtonElement;
const btnHertz = document.getElementById('btn-hertz') as HTMLButtonElement;
const chkForces = document.getElementById('chk-forces') as HTMLInputElement;

let highScore = loadHighScore();
highScoreEl.textContent = `今日のハイスコア: ${highScore}`;

// ================================================================
// DEM Physics
// ================================================================
//
// Normal force:
//   Hooke:  F_ne = k_n · δ
//   Hertz:  F_ne = (4/3)·E*·√R* · δ^(3/2)
//   Damping: F_nd = −η_n · v_n   where η_n = 2β·√(m*·k_eff)
//     Hooke k_eff = k_n
//     Hertz k_eff = dF/dδ = 2·E*·√(R*·δ)
//   F_n = max(0, F_ne + F_nd)
//
// Tangential sliding velocity at contact (2D):
//   PP: v_slide = (v_B − v_A)·t̂ − (ω_A·R_A + ω_B·R_B)
//       where t̂ = (−ny, nx), r_A = R_A·n̂, r_B = −R_B·n̂
//   Wall (bottom, r_cp=(0,R)): v_slide = vx − ω·R
//   Wall (left,   r_cp=(−R,0)): v_slide = vy − ω·R
//   Wall (right,  r_cp=(R,0)):  v_slide = vy + ω·R
//
// Tangential force (Coulomb with Tsuji viscous regularization):
//   η_t = 2β·√(m*·k_t_eff)
//     Hooke: k_t_eff = k_n
//     Hertz: k_t_eff = 2·E*·√(R*·δ)
//   |F_t| = min(μ·F_n, η_t · |v_slide|)
//   F_t opposes v_slide
//
// Torque from tangential force:
//   τ = r_cp × F_t  (2D scalar: rx·Fy − ry·Fx)
//   PP: τ_A = sign(v_slide)·|F_t|·R_A, τ_B = sign(v_slide)·|F_t|·R_B
//   Bottom wall: Δω = +sign(v_slide)·F_t·R / I
//   Left wall:   Δω = +sign(v_slide)·F_t·R / I
//   Right wall:  Δω = −sign(v_slide)·F_t·R / I
//
// Rolling friction (CDT): τ_roll = −μ_r · R* · F_n · sign(ω_rel)
//
// Moment of inertia: I = 0.5·m·R²
// ================================================================

function createParticle(x: number, y: number, level: number): Particle {
  const r = LEVELS[level].radius;
  const m = r * r * 0.008;
  return {
    id: nextId++,
    x, y,
    vx: 0, vy: 0,
    radius: r,
    mass: m,
    inertia: 0.5 * m * r * r,
    level,
    angle: 0,
    omega: 0,
    active: false,
    graceFrames: 0,
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
  // vn > 0 = separating, damping reduces force
  // vn < 0 = approaching, damping increases force
  return Math.max(0, fElastic - eta * vn);
}

function physicsStep(dt: number) {
  const newContactVis: ContactVis[] = [];
  const newPairs = new Set<string>();

  // --- Gravity ---
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
      let delta = minDist - dist;
      if (delta <= 0) continue;

      // Overlap cap (anti-divergence)
      const maxDelta = MAX_DELTA_RATIO * Math.min(a.radius, b.radius);
      delta = Math.min(delta, maxDelta);

      const nx = dx / dist;
      const ny = dy / dist;
      const tx = -ny;
      const ty = nx;

      // Relative velocity of B w.r.t. A
      const dvx = b.vx - a.vx;
      const dvy = b.vy - a.vy;

      // Normal relative velocity (positive = separating)
      const vn = dvx * nx + dvy * ny;

      // Tangential sliding velocity at contact point INCLUDING rotation
      // v_slide = (v_B − v_A)·t̂ − (ω_A·R_A + ω_B·R_B)
      const vSlide = (dvx * tx + dvy * ty) - a.omega * a.radius - b.omega * b.radius;

      // Effective quantities
      const mEff = (a.mass * b.mass) / (a.mass + b.mass);
      const rEff = (a.radius * b.radius) / (a.radius + b.radius);

      // Normal force
      const fn = computeNormalForce(delta, rEff, mEff, vn, ESTAR_PP, KN_PP, BETA_PP);

      // Tangential force (Coulomb with Tsuji damping)
      // η_t = 2β√(m*·k_t_eff)
      // Hooke: k_t_eff = k_n,  Hertz: k_t_eff = 2E*√(R*δ)
      const kTEffPP = contactModel === 'hertz'
        ? 2 * ESTAR_PP * Math.sqrt(Math.max(rEff * delta, 0.01))
        : KN_PP;
      const ftDamp = 2 * BETA_PP * Math.sqrt(Math.max(mEff * kTEffPP, 0.001));
      const ftMag = Math.min(MU_PP * fn, ftDamp * Math.abs(vSlide));
      const ftSign = vSlide > 0.001 ? -1 : vSlide < -0.001 ? 1 : 0;

      // Total force on B in n-direction and t-direction
      const fx = fn * nx + ftSign * ftMag * tx;
      const fy = fn * ny + ftSign * ftMag * ty;

      // Translational acceleration
      b.vx += (fx / b.mass) * dt;
      b.vy += (fy / b.mass) * dt;
      a.vx -= (fx / a.mass) * dt;
      a.vy -= (fy / a.mass) * dt;

      // Torques from tangential friction
      // τ = sign(v_slide) · |F_t| · R  (opposes sliding at contact)
      const torqueSign = vSlide > 0.001 ? 1 : vSlide < -0.001 ? -1 : 0;
      const torqueMag = ftMag;

      a.omega += (torqueSign * torqueMag * a.radius / a.inertia) * dt;
      b.omega += (torqueSign * torqueMag * b.radius / b.inertia) * dt;

      // Rolling friction (CDT): τ_roll = −μ_r · R* · F_n · sign(ω_rel)
      const omegaRel = b.omega - a.omega;
      if (Math.abs(omegaRel) > 0.01) {
        const tauRoll = MU_ROLL_PP * rEff * fn;
        const rollSign = omegaRel > 0 ? 1 : -1;
        const rollImpulse = Math.min(tauRoll * dt, Math.abs(omegaRel) * 0.5 * (a.inertia * b.inertia) / (a.inertia + b.inertia));
        a.omega += rollSign * rollImpulse / a.inertia;
        b.omega -= rollSign * rollImpulse / b.inertia;
      }

      // Position correction (20%, conservative)
      const totalM = a.mass + b.mass;
      const corr = delta * 0.2;
      a.x -= nx * corr * (b.mass / totalM);
      a.y -= ny * corr * (b.mass / totalM);
      b.x += nx * corr * (a.mass / totalM);
      b.y += ny * corr * (a.mass / totalM);

      // Visualization
      const fMag = Math.sqrt(fx * fx + fy * fy);
      newContactVis.push({
        x: (a.x * b.radius + b.x * a.radius) / (a.radius + b.radius),
        y: (a.y * b.radius + b.y * a.radius) / (a.radius + b.radius),
        force: fMag,
      });

      // Merge detection
      const key = pairKey(a.id, b.id);
      newPairs.add(key);
      if (!contactedPairs.has(key) && a.level === b.level) {
        mergeQueue.push([a.id, b.id]);
      }
    }
  }

  // --- Wall contacts (NBR rubber) ---
  for (const p of particles) {
    // Bottom wall: outward normal n=(0,-1), v_n = −vy
    const oBot = p.y + p.radius - CB;
    if (oBot > 0) {
      const d = Math.min(oBot, MAX_DELTA_RATIO * p.radius);
      const vn = -p.vy;
      const fn = computeNormalForce(d, p.radius, p.mass, vn, ESTAR_PW, KN_PW, BETA_PW);
      p.vy -= (fn / p.mass) * dt;

      // Wall friction: v_slide at contact = vx − ω·R
      // r_cp = (0, R), v_cp = (vx−ωR, vy), tangent t=(1,0)
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

      // Wall rolling friction
      if (Math.abs(p.omega) > 0.01) {
        const tauR = MU_ROLL_PW * p.radius * fn;
        const imp = Math.min(tauR * dt, Math.abs(p.omega) * p.inertia * 0.5);
        p.omega -= Math.sign(p.omega) * imp / p.inertia;
      }

      p.y = Math.min(p.y, CB - p.radius);
    }

    // Left wall: n=(1,0), v_n = vx
    const oL = CL - (p.x - p.radius);
    if (oL > 0) {
      const d = Math.min(oL, MAX_DELTA_RATIO * p.radius);
      const vn = p.vx;
      const fn = computeNormalForce(d, p.radius, p.mass, vn, ESTAR_PW, KN_PW, BETA_PW);
      p.vx += (fn / p.mass) * dt;
      // Wall friction: v_slide at contact = vy − ω·R
      // r_cp = (−R, 0), v_cp = (vx, vy−ωR), tangent t=(0,1)
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

    // Right wall: n=(−1,0), v_n = −vx
    const oR = (p.x + p.radius) - CR;
    if (oR > 0) {
      const d = Math.min(oR, MAX_DELTA_RATIO * p.radius);
      const vn = -p.vx;
      const fn = computeNormalForce(d, p.radius, p.mass, vn, ESTAR_PW, KN_PW, BETA_PW);
      p.vx -= (fn / p.mass) * dt;
      // Wall friction: v_slide at contact = vy + ω·R
      // r_cp = (R, 0), v_cp = (vx, vy+ωR), tangent t=(0,1)
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

  // --- Integration + damping + clamping ---
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

    // Mark as active once center passes below DANGER_Y
    if (!p.active && p.y > DANGER_Y) {
      p.active = true;
    }
  }

  contactVis = newContactVis;
  contactedPairs = newPairs;
}

// ================================================================
// Game Over Logic
// ================================================================
// Condition 1: active particle (past grace) whose CENTER is above DL
//   → settled particle has stacked past the danger line
// Condition 2: inactive particle (still falling) above DL touching
//   another particle
//   → newly dropped particle hit the pile before entering
// Checked AFTER processMerges so merged particles are evaluated
// as their new (larger) selves, not as the original pair.

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

    // Condition 1: active particle with center above danger line
    if (p.active && p.y < DANGER_Y) {
      doGameOver();
      return;
    }

    // Condition 2: inactive particle above danger line touching another
    if (!p.active && p.y < DANGER_Y) {
      for (const q of particles) {
        if (p === q) continue;
        const dx = q.x - p.x;
        const dy = q.y - p.y;
        const distSq = dx * dx + dy * dy;
        const minDist = p.radius + q.radius;
        if (distSq < minDist * minDist) {
          doGameOver();
          return;
        }
      }
    }
  }
}

// ================================================================
// Merge Logic
// ================================================================

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
  setTimeout(() => { if (!gameOver) canDrop = true; }, 500);
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
  dangerTimer = 0;
  comboCount = 0;
  comboTimer = 0;
  mergeCount = 0;

  scoreEl.textContent = '0';
  gameOverEl.style.display = 'none';
  highScore = loadHighScore();
  highScoreEl.textContent = `今日のハイスコア: ${highScore}`;
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
  ctx.fillStyle = 'rgba(0,0,0,0.15)';
  ctx.fill();

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

  const rng = mulberry32(level * 54321 + 7);
  const dotN = Math.min(level * 4 + 3, 18);
  for (let i = 0; i < dotN; i++) {
    const ddx = (rng() - 0.5) * r * 1.4;
    const ddy = (rng() - 0.5) * r * 1.4;
    if (ddx * ddx + ddy * ddy < (r * 0.7) ** 2) {
      ctx.beginPath();
      ctx.arc(ddx, ddy, Math.max(1, r * 0.05), 0, Math.PI * 2);
      ctx.fillStyle = hexToRGBA(info.strokeColor, 0.3);
      ctx.fill();
    }
  }

  const fontSize = Math.max(8, Math.floor(r * 0.3));
  ctx.font = `bold ${fontSize}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#3E2723';
  ctx.fillText(info.sieve, 0, 0);

  ctx.restore();
}

let dangerTimer = 0;

function drawGame() {
  const ctx = gCtx;

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
  ctx.moveTo(CL, 0);
  ctx.lineTo(CL, CB);
  ctx.lineTo(CR, CB);
  ctx.lineTo(CR, 0);
  ctx.stroke();

  const dAlpha = dangerTimer > 0 ? 0.7 + 0.3 * Math.abs(Math.sin(Date.now() / 130)) : 0.6;
  ctx.strokeStyle = `rgba(255,160,0,${dAlpha})`;
  ctx.lineWidth = 5;
  ctx.setLineDash([12, 6]);
  ctx.beginPath();
  ctx.moveTo(CL, DANGER_Y);
  ctx.lineTo(CR, DANGER_Y);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = `rgba(255,160,0,${dAlpha * 0.8})`;
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('DEAD LINE', CR - 4, DANGER_Y - 4);

  if (showForceChains && contactVis.length > 0) {
    const maxF = Math.max(...contactVis.map(c => c.force), 1);
    for (const c of contactVis) {
      const t = Math.min(c.force / maxF, 1);
      const baseR = 8 + t * 24;
      const coreR = baseR * 0.3;
      const midR = baseR * 0.6;
      ctx.save();
      ctx.shadowColor = t > 0.5
        ? `rgba(255,${Math.floor(60 * (1 - t))},0,0.9)`
        : `rgba(255,255,0,0.7)`;
      ctx.shadowBlur = 12 + t * 18;
      const grad = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, baseR);
      grad.addColorStop(0, `rgba(255,255,${Math.floor(220 * (1 - t))},1)`);
      grad.addColorStop(0.2, `rgba(255,${Math.floor(255 * (1 - t * 0.8))},${Math.floor(50 * (1 - t))},${0.95 - t * 0.15})`);
      grad.addColorStop(0.5, `rgba(${Math.floor(255 - 40 * t)},${Math.floor(80 * (1 - t))},0,${0.6 + t * 0.2})`);
      grad.addColorStop(1, `rgba(${Math.floor(180 * t)},0,0,0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(c.x, c.y, baseR, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = `rgba(255,255,200,${0.5 + t * 0.5})`;
      ctx.lineWidth = 2 + t * 3;
      ctx.beginPath();
      ctx.arc(c.x, c.y, coreR, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = `rgba(255,255,100,${0.3 + t * 0.4})`;
      ctx.lineWidth = 1 + t * 2;
      ctx.beginPath();
      ctx.arc(c.x, c.y, midR, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
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
  for (const p of [0, 20, 40, 60, 80, 100]) {
    const y = pad.top + pH - (p / 100) * pH;
    ctx.fillText(`${p}`, pad.left - 4, y);
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
    paramsEl.textContent = 'D\u2081\u2080=--mm  D\u2083\u2080=--mm  D\u2085\u2080=--mm  D\u2086\u2080=--mm  Uc=--  Uc\'=--';
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
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  ctx.fillStyle = 'rgba(139,111,71,0.1)';
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
    ctx.moveTo(hx, pad.top + pH);
    ctx.lineTo(hx, hy);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad.left, hy);
    ctx.lineTo(hx, hy);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.lineJoin = 'round';
    ctx.strokeText(dl.label, hx + 4, hy - 3);
    ctx.fillStyle = dl.color;
    ctx.fillText(dl.label, hx + 4, hy - 3);
    ctx.textBaseline = 'alphabetic';
    ctx.beginPath();
    ctx.arc(hx, hy, 4, 0, Math.PI * 2);
    ctx.fillStyle = dl.color;
    ctx.fill();
  }

  let txt = '';
  if (d10 !== null) txt += `D\u2081\u2080=${d10.toFixed(1)}mm  `;
  if (d30 !== null) txt += `D\u2083\u2080=${d30.toFixed(1)}mm  `;
  if (d50 !== null) txt += `D\u2085\u2080=${d50.toFixed(1)}mm  `;
  if (d60 !== null) txt += `D\u2086\u2080=${d60.toFixed(1)}mm`;
  if (d10 !== null && d60 !== null) {
    const Uc = d60 / d10;
    txt += `  Uc=${Uc.toFixed(2)}`;
    if (d30 !== null) {
      const Ucp = (d30 * d30) / (d10 * d60);
      txt += `  Uc'=${Ucp.toFixed(2)}`;
      if (Uc >= 4 && Ucp >= 1 && Ucp <= 3) {
        txt += '  \u2192 良粒度礫 (GW)';
      } else if (Uc >= 6 && Ucp >= 1 && Ucp <= 3) {
        txt += '  \u2192 良粒度砂 (SW)';
      } else {
        txt += '  \u2192 不良粒度 (GP/SP)';
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
  const s = 96;
  ctx.clearRect(0, 0, s, s);
  const r = LEVELS[nextLevel].radius;
  const sc = Math.min(40 / r, 1);
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

  const scoreRestartBtn = document.getElementById('score-restart-btn')!;
  scoreRestartBtn.addEventListener('click', restart);

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

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
