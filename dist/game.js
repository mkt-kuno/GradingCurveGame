// src/game.ts
var LEVELS = [
  { name: "細粒分", sieve: "<4.75mm", upperSieveMM: 4.75, radius: 28, color: "#F5E6CC", strokeColor: "#B8A48A", score: 1 },
  { name: "細礫", sieve: "4.75mm", upperSieveMM: 9.5, radius: 40, color: "#EAD5B8", strokeColor: "#A89070", score: 3 },
  { name: "小礫", sieve: "9.5mm", upperSieveMM: 19, radius: 54, color: "#DCC4A0", strokeColor: "#9A7C5A", score: 6 },
  { name: "中礫", sieve: "19mm", upperSieveMM: 26.5, radius: 70, color: "#D0B48E", strokeColor: "#8C6C46", score: 10 },
  { name: "粗礫", sieve: "26.5mm", upperSieveMM: 37.5, radius: 88, color: "#C2A47A", strokeColor: "#7E5E38", score: 15 },
  { name: "大礫", sieve: "37.5mm", upperSieveMM: 52, radius: 108, color: "#B49468", strokeColor: "#6E5030", score: 21 },
  { name: "巨礫", sieve: "52mm", upperSieveMM: 75, radius: 130, color: "#A28458", strokeColor: "#5E4228", score: 28 },
  { name: "転石", sieve: "75mm+", upperSieveMM: 100, radius: 154, color: "#907448", strokeColor: "#4E3420", score: 36 }
];
var SIEVE_SIZES = [4.75, 9.5, 19, 26.5, 37.5, 52, 75];
var GAME_W = 520;
var GAME_H = 560;
var WALL_T = 20;
var CL = WALL_T;
var CR = GAME_W - WALL_T;
var CB = GAME_H - WALL_T;
var CONTAINER_W = CR - CL;
var CONTAINER_H = CB - WALL_T;
var DROP_Y = 45;
var DANGER_Y = 75;
var GRAVITY = 700;
var ESTAR_PP = 1e4;
var ESTAR_PW = 1e4;
var KN_PP = 1e5;
var KN_PW = 1e5;
var MU_PP = 0.45;
var MU_PW = 0.8;
var REST_PP = 0.5;
var REST_PW = 0.5;
var MU_ROLL_PP = 0.03;
var MU_ROLL_PW = 0.15;
function beta(e) {
  if (e <= 0)
    return 1;
  if (e >= 1)
    return 0;
  const ln = Math.log(e);
  return -ln / Math.sqrt(Math.PI * Math.PI + ln * ln);
}
var BETA_PP = beta(REST_PP);
var BETA_PW = beta(REST_PW);
var SUB_STEPS = 10;
var MAX_DELTA_RATIO = 0.08;
var MAX_VEL = 3000;
var MAX_OMEGA = 80;
var VEL_DAMP = 0.9995;
var ANG_DAMP = 0.998;
var particles = [];
var nextId = 0;
var contactVis = [];
var effects = [];
var scorePopups = [];
var mergeQueue = [];
var contactedPairs = new Set;
var score = 0;
var currentLevel = 0;
var nextLevel = 0;
var dropX = GAME_W / 2;
var canDrop = true;
var gameOver = false;
var mergeCount = 0;
var contactModel = "hooke";
var showForceChains = false;
var comboCount = 0;
var comboTimer = 0;
var pendingGameOver = false;
var gameCanvas = document.getElementById("game-canvas");
var gCtx = gameCanvas.getContext("2d");
var chartCanvas = document.getElementById("chart-canvas");
var cCtx = chartCanvas.getContext("2d");
var nextCanvas = document.getElementById("next-canvas");
var nCtx = nextCanvas.getContext("2d");
var scoreEl = document.getElementById("score-value");
var mergeEl = document.getElementById("merge-count");
var nextNameEl = document.getElementById("next-name");
var gameOverEl = document.getElementById("game-over");
var finalScoreEl = document.getElementById("final-score");
var restartBtn = document.getElementById("restart-btn");
var paramsEl = document.getElementById("grading-params");
var btnHooke = document.getElementById("btn-hooke");
var btnHertz = document.getElementById("btn-hertz");
var chkForces = document.getElementById("chk-forces");
function createParticle(x, y, level) {
  const r = LEVELS[level].radius;
  const m = r * r * 0.008;
  return {
    id: nextId++,
    x,
    y,
    vx: 0,
    vy: 0,
    radius: r,
    mass: m,
    inertia: 0.5 * m * r * r,
    level,
    angle: 0,
    omega: 0,
    active: false,
    graceFrames: 0
  };
}
function pairKey(a, b) {
  return a < b ? `${a}_${b}` : `${b}_${a}`;
}
function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}
function hertzNormalForce(delta, rEff, eStar) {
  return 4 / 3 * eStar * Math.sqrt(Math.max(rEff, 0.1)) * Math.pow(Math.max(delta, 0), 1.5);
}
function hookeNormalForce(delta, kn) {
  return kn * delta;
}
function computeNormalForce(delta, rEff, mEff, vn, eStar, kn, b) {
  if (delta <= 0)
    return 0;
  let fElastic;
  let kEff;
  if (contactModel === "hertz") {
    fElastic = hertzNormalForce(delta, rEff, eStar);
    kEff = 2 * eStar * Math.sqrt(Math.max(rEff * delta, 0.01));
  } else {
    fElastic = hookeNormalForce(delta, kn);
    kEff = kn;
  }
  const eta = 2 * b * Math.sqrt(Math.max(mEff * kEff, 0.001));
  return Math.max(0, fElastic - eta * vn);
}
function physicsStep(dt) {
  const newContactVis = [];
  const newPairs = new Set;
  pendingGameOver = false;
  for (const p of particles) {
    p.vy += GRAVITY * dt;
  }
  for (let i = 0;i < particles.length; i++) {
    for (let j = i + 1;j < particles.length; j++) {
      const a = particles[i];
      const b = particles[j];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distSq = dx * dx + dy * dy;
      const minDist = a.radius + b.radius;
      if (distSq >= minDist * minDist)
        continue;
      const dist = Math.sqrt(Math.max(distSq, 0.00000001));
      let delta = minDist - dist;
      if (delta <= 0)
        continue;
      const maxDelta = MAX_DELTA_RATIO * Math.min(a.radius, b.radius);
      delta = Math.min(delta, maxDelta);
      const nx = dx / dist;
      const ny = dy / dist;
      const tx = -ny;
      const ty = nx;
      const dvx = b.vx - a.vx;
      const dvy = b.vy - a.vy;
      const vn = dvx * nx + dvy * ny;
      const vSlide = dvx * tx + dvy * ty - a.omega * a.radius - b.omega * b.radius;
      const mEff = a.mass * b.mass / (a.mass + b.mass);
      const rEff = a.radius * b.radius / (a.radius + b.radius);
      const fn = computeNormalForce(delta, rEff, mEff, vn, ESTAR_PP, KN_PP, BETA_PP);
      const ftDamp = 2 * BETA_PP * Math.sqrt(Math.max(mEff * ESTAR_PP, 0.001)) * 0.5;
      const ftMag = Math.min(MU_PP * fn, ftDamp * Math.abs(vSlide));
      const ftSign = vSlide > 0.001 ? -1 : vSlide < -0.001 ? 1 : 0;
      const fx = fn * nx + ftSign * ftMag * tx;
      const fy = fn * ny + ftSign * ftMag * ty;
      b.vx += fx / b.mass * dt;
      b.vy += fy / b.mass * dt;
      a.vx -= fx / a.mass * dt;
      a.vy -= fy / a.mass * dt;
      const torqueSign = vSlide > 0.001 ? 1 : vSlide < -0.001 ? -1 : 0;
      const torqueMag = ftMag;
      a.omega += torqueSign * torqueMag * a.radius / a.inertia * dt;
      b.omega += torqueSign * torqueMag * b.radius / b.inertia * dt;
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
        force: fMag
      });
      const key = pairKey(a.id, b.id);
      newPairs.add(key);
      if (!contactedPairs.has(key) && a.level === b.level) {
        mergeQueue.push([a.id, b.id]);
      }
      if (!pendingGameOver) {
        const aAbove = a.y < DANGER_Y;
        const bAbove = b.y < DANGER_Y;
        if (!a.active && aAbove)
          pendingGameOver = true;
        if (!b.active && bAbove)
          pendingGameOver = true;
      }
    }
  }
  for (const p of particles) {
    const oBot = p.y + p.radius - CB;
    if (oBot > 0) {
      const d = Math.min(oBot, MAX_DELTA_RATIO * p.radius);
      const vn = -p.vy;
      const fn = computeNormalForce(d, p.radius, p.mass, vn, ESTAR_PW, KN_PW, BETA_PW);
      p.vy -= fn / p.mass * dt;
      const vSlide = p.vx + p.omega * p.radius;
      if (Math.abs(vSlide) > 0.01) {
        const ftMax = MU_PW * fn;
        const ft = Math.min(ftMax, Math.abs(vSlide) * p.mass / dt * 2);
        p.vx -= Math.sign(vSlide) * (ft / p.mass) * dt;
        p.omega -= Math.sign(vSlide) * (ft * p.radius / p.inertia) * dt;
      }
      if (Math.abs(p.omega) > 0.01) {
        const tauR = MU_ROLL_PW * p.radius * fn;
        const imp = Math.min(tauR * dt, Math.abs(p.omega) * p.inertia * 0.5);
        p.omega -= Math.sign(p.omega) * imp / p.inertia;
      }
      p.omega *= 0.85;
      p.y = Math.min(p.y, CB - p.radius);
    }
    const oL = CL - (p.x - p.radius);
    if (oL > 0) {
      const d = Math.min(oL, MAX_DELTA_RATIO * p.radius);
      const vn = p.vx;
      const fn = computeNormalForce(d, p.radius, p.mass, vn, ESTAR_PW, KN_PW, BETA_PW);
      p.vx += fn / p.mass * dt;
      const vSlide = p.vy + p.omega * p.radius;
      if (Math.abs(vSlide) > 0.01) {
        const ft = Math.min(MU_PW * fn, Math.abs(vSlide) * p.mass / dt * 2);
        p.vy -= Math.sign(vSlide) * (ft / p.mass) * dt;
        p.omega -= Math.sign(vSlide) * (ft * p.radius / p.inertia) * dt;
      }
      p.omega *= 0.85;
      p.x = Math.max(p.x, CL + p.radius);
    }
    const oR = p.x + p.radius - CR;
    if (oR > 0) {
      const d = Math.min(oR, MAX_DELTA_RATIO * p.radius);
      const vn = -p.vx;
      const fn = computeNormalForce(d, p.radius, p.mass, vn, ESTAR_PW, KN_PW, BETA_PW);
      p.vx -= fn / p.mass * dt;
      const vSlide = -(p.vy - p.omega * p.radius);
      if (Math.abs(vSlide) > 0.01) {
        const ft = Math.min(MU_PW * fn, Math.abs(vSlide) * p.mass / dt * 2);
        p.vy += Math.sign(vSlide) * (ft / p.mass) * dt;
        p.omega += Math.sign(vSlide) * (ft * p.radius / p.inertia) * dt;
      }
      p.omega *= 0.85;
      p.x = Math.min(p.x, CR - p.radius);
    }
  }
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
  contactVis = newContactVis;
  contactedPairs = newPairs;
}
function checkGameOver() {
  if (pendingGameOver) {
    gameOver = true;
    finalScoreEl.textContent = score.toString();
    gameOverEl.style.display = "flex";
    return;
  }
  for (const p of particles) {
    if (p.graceFrames > 0)
      continue;
    if (p.active && p.y < DANGER_Y) {
      gameOver = true;
      finalScoreEl.textContent = score.toString();
      gameOverEl.style.display = "flex";
      return;
    }
  }
}
function processMerges() {
  const consumed = new Set;
  for (const [idA, idB] of mergeQueue) {
    if (consumed.has(idA) || consumed.has(idB))
      continue;
    const a = particles.find((p) => p.id === idA);
    const b = particles.find((p) => p.id === idB);
    if (!a || !b || a.level !== b.level)
      continue;
    const mx = (a.x + b.x) / 2;
    const my = (a.y + b.y) / 2;
    const newLevel = a.level + 1;
    consumed.add(idA);
    consumed.add(idB);
    particles = particles.filter((p) => p.id !== idA && p.id !== idB);
    if (newLevel >= LEVELS.length) {
      const pts2 = LEVELS[a.level].score * 5;
      score += pts2;
      mergeCount++;
      comboCount++;
      comboTimer = 45;
      effects.push({ x: mx, y: my, r: 10, alpha: 1, color: "#FFFFFF" });
      effects.push({ x: mx, y: my, r: 30, alpha: 0.7, color: "#F5E6CC" });
      scorePopups.push({ x: mx, y: my, text: `+${pts2} MAX!`, timer: 90 });
      scoreEl.textContent = score.toString();
      mergeEl.textContent = `合体回数: ${mergeCount}`;
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
    mergeEl.textContent = `合体回数: ${mergeCount}`;
  }
  mergeQueue = [];
  if (comboTimer > 0) {
    comboTimer--;
    if (comboTimer === 0)
      comboCount = 0;
  }
}
function getRandomLevel() {
  const r = Math.random();
  if (r < 0.35)
    return 0;
  if (r < 0.65)
    return 1;
  if (r < 0.85)
    return 2;
  return 3;
}
function drop() {
  if (!canDrop || gameOver)
    return;
  const r = LEVELS[currentLevel].radius;
  const cx = Math.max(CL + r + 2, Math.min(CR - r - 2, dropX));
  const p = createParticle(cx, DROP_Y, currentLevel);
  particles.push(p);
  canDrop = false;
  currentLevel = nextLevel;
  nextLevel = getRandomLevel();
  updateNextPreview();
  setTimeout(() => {
    if (!gameOver)
      canDrop = true;
  }, 500);
}
function restart() {
  particles = [];
  contactVis = [];
  effects = [];
  scorePopups = [];
  mergeQueue = [];
  contactedPairs.clear();
  nextId = 0;
  pendingGameOver = false;
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
  scoreEl.textContent = "0";
  mergeEl.textContent = "合体回数: 0";
  gameOverEl.style.display = "none";
  updateNextPreview();
}
function mulberry32(seed) {
  let s = seed | 0;
  return () => {
    s = s + 1831565813 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function hexToRGBA(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${alpha})`;
}
function lighten(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.min(255, (n >> 16 & 255) + amt);
  const g = Math.min(255, (n >> 8 & 255) + amt);
  const b = Math.min(255, (n & 255) + amt);
  return `#${(r << 16 | g << 8 | b).toString(16).padStart(6, "0")}`;
}
function drawParticle(ctx, x, y, angle, level) {
  const info = LEVELS[level];
  const r = info.radius;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.beginPath();
  ctx.arc(2, 3, r, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,0,0,0.15)";
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
  for (let i = 0;i < dotN; i++) {
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
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#3E2723";
  ctx.fillText(info.sieve, 0, 0);
  ctx.restore();
}
var dangerTimer = 0;
function drawGame() {
  const ctx = gCtx;
  ctx.fillStyle = "#0f0f23";
  ctx.fillRect(0, 0, GAME_W, GAME_H);
  const bgGrad = ctx.createLinearGradient(0, 0, 0, GAME_H);
  bgGrad.addColorStop(0, "#151530");
  bgGrad.addColorStop(1, "#0d0d20");
  ctx.fillStyle = bgGrad;
  ctx.fillRect(CL, 0, CONTAINER_W, GAME_H);
  ctx.fillStyle = "#1e1e45";
  ctx.fillRect(0, 0, WALL_T, GAME_H);
  ctx.fillRect(CR, 0, WALL_T, GAME_H);
  ctx.fillRect(0, CB, GAME_W, WALL_T);
  ctx.strokeStyle = "#3a3a7a";
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
  ctx.font = "9px sans-serif";
  ctx.textAlign = "right";
  ctx.fillText("DEAD LINE", CR - 4, DANGER_Y - 4);
  if (showForceChains && contactVis.length > 0) {
    const maxF = Math.max(...contactVis.map((c) => c.force), 1);
    for (const c of contactVis) {
      const t = Math.min(c.force / maxF, 1);
      const w = 1 + t * 5;
      ctx.strokeStyle = `rgba(${Math.floor(80 + 175 * t)},${Math.floor(150 * (1 - t))},${Math.floor(200 * (1 - t))},${0.3 + t * 0.5})`;
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
    ctx.textAlign = "center";
    ctx.fillText(sp.text, sp.x, sp.y - 30 + (1 - alpha) * 20);
  }
  if (comboCount > 1 && comboTimer > 0) {
    const alpha = comboTimer / 45;
    ctx.fillStyle = `rgba(255,100,100,${alpha})`;
    ctx.font = `bold ${22 + comboCount * 2}px sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText(`${comboCount} COMBO!`, GAME_W / 2, GAME_H / 2 - 60);
  }
  if (!gameOver && canDrop) {
    const r = LEVELS[currentLevel].radius;
    const cx = Math.max(CL + r + 2, Math.min(CR - r - 2, dropX));
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, DROP_Y + r);
    ctx.lineTo(cx, CB);
    ctx.stroke();
    ctx.setLineDash([]);
    drawParticle(ctx, cx, DROP_Y, 0, currentLevel);
  }
  ctx.fillStyle = "rgba(255,255,255,0.4)";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(contactModel === "hertz" ? "Hertz Contact" : "Hooke Contact", CL + 4, 14);
  ctx.fillText(`N=${particles.length}`, CL + 4, 26);
}
function drawGradingChart() {
  const ctx = cCtx;
  const W = chartCanvas.width;
  const H = chartCanvas.height;
  const pad = { top: 20, right: 16, bottom: 40, left: 46 };
  const pW = W - pad.left - pad.right;
  const pH = H - pad.top - pad.bottom;
  ctx.fillStyle = "#f5f5f5";
  ctx.fillRect(0, 0, W, H);
  const counts = new Array(LEVELS.length).fill(0);
  let totalMass = 0;
  for (const p of particles) {
    counts[p.level] += p.mass;
    totalMass += p.mass;
  }
  ctx.strokeStyle = "#555";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + pH);
  ctx.lineTo(pad.left + pW, pad.top + pH);
  ctx.stroke();
  ctx.fillStyle = "#444";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const p of [0, 20, 40, 60, 80, 100]) {
    const y = pad.top + pH - p / 100 * pH;
    ctx.fillText(`${p}`, pad.left - 4, y);
    ctx.strokeStyle = "#ddd";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + pW, y);
    ctx.stroke();
  }
  const logMin = Math.log10(1);
  const logMax = Math.log10(150);
  function toX(mm) {
    return pad.left + (Math.log10(mm) - logMin) / (logMax - logMin) * pW;
  }
  const xTicks = [1, 2, 4.75, 9.5, 19, 26.5, 37.5, 52, 75, 100];
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const s of xTicks) {
    const x = toX(s);
    const isMain = SIEVE_SIZES.includes(s);
    ctx.strokeStyle = isMain ? "#ccc" : "#e5e5e5";
    ctx.lineWidth = isMain ? 0.8 : 0.4;
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + pH);
    ctx.stroke();
    ctx.fillStyle = isMain ? "#333" : "#999";
    ctx.font = isMain ? "10px sans-serif" : "8px sans-serif";
    ctx.fillText(`${s}`, x, pad.top + pH + 2);
  }
  ctx.font = "8px sans-serif";
  ctx.fillStyle = "#555";
  ctx.textAlign = "center";
  ctx.fillText("粒径 (mm)", pad.left + pW / 2, pad.top + pH + 20);
  ctx.save();
  ctx.translate(9, pad.top + pH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("通過質量百分率 (%)", 0, 0);
  ctx.restore();
  if (totalMass < 0.01) {
    ctx.fillStyle = "#aaa";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("粒子を落としてください", W / 2, H / 2);
    paramsEl.textContent = "";
    return;
  }
  const cumMass = [];
  let cm = 0;
  for (let i = 0;i < LEVELS.length; i++) {
    cm += counts[i];
    cumMass.push(cm);
  }
  const points = [];
  points.push({ x: 150, y: 100 });
  for (let i = SIEVE_SIZES.length - 1;i >= 0; i--) {
    points.push({ x: SIEVE_SIZES[i], y: cumMass[i] / totalMass * 100 });
  }
  points.push({ x: 0.8, y: 0 });
  points.sort((a, b) => a.x - b.x);
  ctx.strokeStyle = "#8B6F47";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0;i < points.length; i++) {
    const px = toX(points[i].x);
    const py = pad.top + pH - points[i].y / 100 * pH;
    if (i === 0)
      ctx.moveTo(px, py);
    else
      ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.fillStyle = "rgba(139,111,71,0.1)";
  ctx.beginPath();
  for (let i = 0;i < points.length; i++) {
    const px = toX(points[i].x);
    const py = pad.top + pH - points[i].y / 100 * pH;
    if (i === 0)
      ctx.moveTo(px, py);
    else
      ctx.lineTo(px, py);
  }
  ctx.lineTo(pad.left + pW, pad.top + pH);
  ctx.lineTo(pad.left, pad.top + pH);
  ctx.closePath();
  ctx.fill();
  for (let i = 1;i < points.length - 1; i++) {
    const px = toX(points[i].x);
    const py = pad.top + pH - points[i].y / 100 * pH;
    ctx.beginPath();
    ctx.arc(px, py, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = "#6B5235";
    ctx.fill();
  }
  const d10 = interpD(10, points);
  const d30 = interpD(30, points);
  const d60 = interpD(60, points);
  let txt = "";
  if (d10 !== null)
    txt += `D₁₀=${d10.toFixed(1)}mm  `;
  if (d30 !== null)
    txt += `D₃₀=${d30.toFixed(1)}mm  `;
  if (d60 !== null)
    txt += `D₆₀=${d60.toFixed(1)}mm`;
  if (d10 !== null && d60 !== null) {
    const Cu = d60 / d10;
    txt += `
Cu=${Cu.toFixed(2)}`;
    if (d30 !== null) {
      const Cc = d30 * d30 / (d10 * d60);
      txt += `  Cc=${Cc.toFixed(2)}`;
      if (Cu >= 4 && Cc >= 1 && Cc <= 3) {
        txt += `
→ 良粒度礫 (GW)`;
      } else if (Cu >= 6 && Cc >= 1 && Cc <= 3) {
        txt += `
→ 良粒度砂 (SW)`;
      } else {
        txt += `
→ 不良粒度 (GP/SP)`;
      }
    }
  }
  paramsEl.textContent = txt;
}
function interpD(target, pts) {
  for (let i = 1;i < pts.length; i++) {
    const y0 = pts[i - 1].y;
    const y1 = pts[i].y;
    if (y0 <= target && y1 >= target || y0 >= target && y1 <= target) {
      const dy = y1 - y0;
      if (Math.abs(dy) < 0.001)
        continue;
      const t = (target - y0) / dy;
      if (t < 0 || t > 1)
        continue;
      const logX = Math.log10(Math.max(pts[i - 1].x, 0.01)) + t * (Math.log10(pts[i].x) - Math.log10(Math.max(pts[i - 1].x, 0.01)));
      return Math.pow(10, logX);
    }
  }
  return null;
}
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
function setupInput() {
  gameCanvas.addEventListener("mousemove", (e) => {
    const rect = gameCanvas.getBoundingClientRect();
    dropX = (e.clientX - rect.left) * (GAME_W / rect.width);
  });
  gameCanvas.addEventListener("click", () => drop());
  gameCanvas.addEventListener("touchmove", (e) => {
    e.preventDefault();
    const rect = gameCanvas.getBoundingClientRect();
    dropX = (e.touches[0].clientX - rect.left) * (GAME_W / rect.width);
  }, { passive: false });
  gameCanvas.addEventListener("touchend", (e) => {
    e.preventDefault();
    drop();
  });
  const keys = new Set;
  document.addEventListener("keydown", (e) => {
    keys.add(e.key);
    if (e.key === " ") {
      e.preventDefault();
      drop();
    }
  });
  document.addEventListener("keyup", (e) => keys.delete(e.key));
  setInterval(() => {
    if (keys.has("ArrowLeft") || keys.has("a") || keys.has("A"))
      dropX = Math.max(CL + 20, dropX - 6);
    if (keys.has("ArrowRight") || keys.has("d") || keys.has("D"))
      dropX = Math.min(CR - 20, dropX + 6);
  }, 16);
  restartBtn.addEventListener("click", restart);
  btnHooke.addEventListener("click", () => {
    contactModel = "hooke";
    btnHooke.classList.add("active");
    btnHertz.classList.remove("active");
  });
  btnHertz.addEventListener("click", () => {
    contactModel = "hertz";
    btnHertz.classList.add("active");
    btnHooke.classList.remove("active");
  });
  chkForces.addEventListener("change", () => {
    showForceChains = chkForces.checked;
  });
}
function update() {
  if (!gameOver) {
    const dt = 1 / 60 / SUB_STEPS;
    for (let i = 0;i < SUB_STEPS; i++) {
      physicsStep(dt);
    }
    processMerges();
    checkGameOver();
    for (const p of particles) {
      if (p.graceFrames > 0)
        p.graceFrames--;
    }
  }
  for (const e of effects) {
    e.r += 2.5;
    e.alpha -= 0.025;
  }
  effects = effects.filter((e) => e.alpha > 0);
  for (const sp of scorePopups)
    sp.timer--;
  scorePopups = scorePopups.filter((sp) => sp.timer > 0);
  drawGame();
  drawGradingChart();
  requestAnimationFrame(update);
}
currentLevel = getRandomLevel();
nextLevel = getRandomLevel();
updateNextPreview();
setupInput();
update();
