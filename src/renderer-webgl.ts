// ================================================================
// WebGL Renderer — instanced procedural particles + instanced additive
// Force Chains & glow effects. Used when WebGPU is missing
// (Safari iOS, older browsers). Text & overlays on Canvas2D overlay.
// ================================================================

import {
  LEVELS, GAME_W, GAME_H, WALL_T, CL, CR, CB, DANGER_Y, DROP_Y,
  COMBO_TIMEOUT_STEPS, ContactVis, getForceColor, getForceLineWidth,
} from './constants';

interface Effect { x: number; y: number; r: number; alpha: number; color: string; }
interface ScorePopup { x: number; y: number; text: string; timer: number; }
interface RenderProfile {
  forceChainStride: number;
  effectiveForceChains: boolean;
  drawParticleLabels: boolean;
  minLabelRadius: number;
  maxScorePopups: number;
  showDropGuide: boolean;
}

function hexToRGB(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) / 255, ((n >> 8) & 0xFF) / 255, (n & 0xFF) / 255];
}

// Precomputed per-level colors
const LEVEL_FILL = new Float32Array(LEVELS.length * 3);
const LEVEL_STROKE = new Float32Array(LEVELS.length * 3);
for (let i = 0; i < LEVELS.length; i++) {
  const [r, g, b] = hexToRGB(LEVELS[i].color);
  const [sr, sg, sb] = hexToRGB(LEVELS[i].strokeColor);
  LEVEL_FILL[i * 3] = r; LEVEL_FILL[i * 3 + 1] = g; LEVEL_FILL[i * 3 + 2] = b;
  LEVEL_STROKE[i * 3] = sr; LEVEL_STROKE[i * 3 + 1] = sg; LEVEL_STROKE[i * 3 + 2] = sb;
}

const VS_PARTICLE = `
attribute vec2 aCorner;
attribute vec2 aOffset;
attribute float aRadius;
attribute float aAngle;
attribute vec3 aFillColor;
attribute vec3 aStrokeColor;
attribute float aLevel;
uniform vec2 uResolution;
varying vec2 vUv;
varying vec3 vColor;
varying vec3 vStroke;
varying float vRadius;
varying float vLevel;
void main() {
  float c = cos(aAngle);
  float s = sin(aAngle);
  vec2 rot = vec2(aCorner.x * c - aCorner.y * s, aCorner.x * s + aCorner.y * c);
  vec2 world = rot * aRadius + aOffset;
  vec2 clip = (world / uResolution) * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  vUv = aCorner;
  vColor = aFillColor;
  vStroke = aStrokeColor;
  vRadius = aRadius;
  vLevel = aLevel;
}
`;

const FS_PARTICLE = `
precision mediump float;
varying vec2 vUv;
varying vec3 vColor;
varying vec3 vStroke;
varying float vRadius;
varying float vLevel;
float hash(float n) {
  return fract(sin(n) * 43758.5453123);
}
void main() {
  float dist = length(vUv);
  if (dist > 1.0) { discard; }
  float sw = 2.0 / vRadius;
  if (dist > 1.0 - sw) {
    gl_FragColor = vec4(vStroke, 1.0);
    return;
  }
  vec3 lightC = min(vColor + 0.137, 1.0);
  float ld = length(vUv - vec2(-0.25, -0.25));
  float t = clamp((ld - 0.05) / 0.95, 0.0, 1.0);
  vec3 gc = mix(lightC, vColor, t);
  int li = int(vLevel + 0.5);
  int dc = li * 6 + 4;
  if (dc > 27) dc = 27;
  for (int i = 0; i < 27; i++) {
    if (i >= dc) break;
    float seed = vLevel * 54321.0 + 7.0 + float(i) * 1337.0;
    float dx = hash(seed) * 2.0 - 1.0;
    float dy = hash(seed + 1.0) * 2.0 - 1.0;
    if (dx*dx + dy*dy < 0.49) {
      float d = length(vUv - vec2(dx * 0.7, dy * 0.7));
      if (d < 0.05) { gc = mix(gc, vStroke * 0.5, 0.3); }
    }
  }
  gl_FragColor = vec4(gc, 1.0);
}
`;

const VS_GLOW = `
attribute vec2 aCorner;
attribute vec2 aCenter;
attribute vec2 aHalfSize;
attribute float aAngle;
attribute vec3 aColor;
attribute float aAlpha;
attribute float aMode;
uniform vec2 uResolution;
varying vec2 vUv;
varying vec3 vColor;
varying float vAlpha;
varying float vMode;
void main() {
  float c = cos(aAngle);
  float s = sin(aAngle);
  vec2 scaled = aCorner * aHalfSize;
  vec2 rot = vec2(scaled.x * c - scaled.y * s, scaled.x * s + scaled.y * c);
  vec2 world = rot + aCenter;
  vec2 clip = (world / uResolution) * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  vUv = aCorner;
  vColor = aColor;
  vAlpha = aAlpha;
  vMode = aMode;
}
`;

const FS_GLOW = `
precision mediump float;
varying vec2 vUv;
varying vec3 vColor;
varying float vAlpha;
varying float vMode;
void main() {
  if (vMode > 0.5) {
    float d = length(vUv);
    if (d > 1.0) { discard; }
    float fall = (1.0 - d) * (1.0 - d);
    gl_FragColor = vec4(vColor * fall * vAlpha, fall * vAlpha);
  } else {
    float dy = abs(vUv.y);
    if (dy > 1.0) { discard; }
    float beam = (1.0 - dy * dy);
    float core = max(0.0, 1.0 - dy * 2.2);
    vec3 col = mix(vColor, vec3(1.0), core * 0.7);
    gl_FragColor = vec4(col * beam * vAlpha, beam * vAlpha);
  }
}
`;

export class WebGLRenderer {
  private gl: WebGLRenderingContext;
  private overlayCanvas: HTMLCanvasElement;
  private overlayCtx: CanvasRenderingContext2D;
  private particleProgram: WebGLProgram;
  private glowProgram: WebGLProgram;
  private quadVBO: WebGLBuffer;
  private instVBO: WebGLBuffer;
  private glowVBO: WebGLBuffer;
  private maxInst = 512;
  private maxGlowInst = 2048;
  private extInst: ANGLE_instanced_arrays | null = null;
  private vaoExt: OES_vertex_array_object | null = null;
  private particleVAO: WebGLVertexArrayObjectOES | null = null;
  private glowVAO: WebGLVertexArrayObjectOES | null = null;
  private particleBind: (() => void) | null = null;
  private glowBind: (() => void) | null = null;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
    }) as WebGLRenderingContext | null;
    if (!gl) throw new Error('WebGL unavailable');
    this.gl = gl;

    this.extInst = gl.getExtension('ANGLE_instanced_arrays');
    if (!this.extInst) throw new Error('instanced arrays unavailable');
    this.vaoExt = gl.getExtension('OES_vertex_array_object');

    this.particleProgram = this.compile(VS_PARTICLE, FS_PARTICLE);
    this.glowProgram = this.compile(VS_GLOW, FS_GLOW);

    // Unit quad [-1, -1] to [1, 1]
    const quad = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
    this.quadVBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

    // Particle instance buffer: 11 floats per instance (44 bytes)
    this.instVBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instVBO);
    gl.bufferData(gl.ARRAY_BUFFER, this.maxInst * 11 * 4, gl.DYNAMIC_DRAW);

    // Glow instance buffer: 10 floats per instance (40 bytes)
    this.glowVBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.glowVBO);
    gl.bufferData(gl.ARRAY_BUFFER, this.maxGlowInst * 10 * 4, gl.DYNAMIC_DRAW);

    this.setupVAOs();

    // Overlay canvas for text & UI
    this.overlayCanvas = document.createElement('canvas');
    this.overlayCanvas.width = GAME_W;
    this.overlayCanvas.height = GAME_H;
    this.overlayCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;border-radius:12px;z-index:10';
    this.overlayCtx = this.overlayCanvas.getContext('2d')!;
    document.getElementById('game-section')!.style.position = 'relative';
    document.getElementById('game-section')!.appendChild(this.overlayCanvas);
  }

  private compile(vsSrc: string, fsSrc: string): WebGLProgram {
    const gl = this.gl;
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vsSrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error('VS: ' + gl.getShaderInfoLog(vs));
    }
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fsSrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error('FS: ' + gl.getShaderInfoLog(fs));
    }
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('link: ' + gl.getProgramInfoLog(prog));
    }
    return prog;
  }

  private setupVAOs() {
    const gl = this.gl;
    const inst = this.extInst!;

    const bindParticleAttribs = () => {
      const p = this.particleProgram;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
      const corner = gl.getAttribLocation(p, 'aCorner');
      gl.enableVertexAttribArray(corner);
      gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.instVBO);
      const stride = 11 * 4;
      const attrs: [string, number, number][] = [
        ['aOffset', 2, 0],
        ['aRadius', 1, 8],
        ['aAngle', 1, 12],
        ['aFillColor', 3, 16],
        ['aStrokeColor', 3, 28],
        ['aLevel', 1, 40],
      ];
      for (const [name, size, offset] of attrs) {
        const loc = gl.getAttribLocation(p, name);
        if (loc >= 0) {
          gl.enableVertexAttribArray(loc);
          gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
          inst.vertexAttribDivisorANGLE(loc, 1);
        }
      }
    };

    const bindGlowAttribs = () => {
      const p = this.glowProgram;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
      const corner = gl.getAttribLocation(p, 'aCorner');
      gl.enableVertexAttribArray(corner);
      gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.glowVBO);
      const stride = 10 * 4;
      const attrs: [string, number, number][] = [
        ['aCenter', 2, 0],
        ['aHalfSize', 2, 8],
        ['aAngle', 1, 16],
        ['aColor', 3, 20],
        ['aAlpha', 1, 32],
        ['aMode', 1, 36],
      ];
      for (const [name, size, offset] of attrs) {
        const loc = gl.getAttribLocation(p, name);
        if (loc >= 0) {
          gl.enableVertexAttribArray(loc);
          gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
          inst.vertexAttribDivisorANGLE(loc, 1);
        }
      }
    };

    if (this.vaoExt) {
      this.particleVAO = this.vaoExt.createVertexArrayOES();
      this.vaoExt.bindVertexArrayOES(this.particleVAO);
      bindParticleAttribs();
      this.glowVAO = this.vaoExt.createVertexArrayOES();
      this.vaoExt.bindVertexArrayOES(this.glowVAO);
      bindGlowAttribs();
      this.vaoExt.bindVertexArrayOES(null);
    } else {
      this.particleBind = bindParticleAttribs;
      this.glowBind = bindGlowAttribs;
    }
  }

  private ensureInstanceCapacity(count: number) {
    if (count <= this.maxInst) return;
    let next = this.maxInst;
    while (next < count) next *= 2;
    this.maxInst = next;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instVBO);
    gl.bufferData(gl.ARRAY_BUFFER, this.maxInst * 11 * 4, gl.DYNAMIC_DRAW);
  }

  private ensureGlowCapacity(count: number) {
    if (count <= this.maxGlowInst) return;
    let next = this.maxGlowInst;
    while (next < count) next *= 2;
    this.maxGlowInst = next;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.glowVBO);
    gl.bufferData(gl.ARRAY_BUFFER, this.maxGlowInst * 10 * 4, gl.DYNAMIC_DRAW);
  }

  drawFrame(
    pCount: number,
    px: Float32Array, py: Float32Array, pRadius: Float32Array,
    pAngle: Float32Array, pLevel: Int32Array,
    contactVis: ContactVis[], contactVisLen: number,
    effects: Effect[],
    scorePopups: ScorePopup[],
    comboCount: number,
    comboTimer: number,
    currentLevel: number,
    dropX: number,
    canDrop: boolean,
    gameOver: boolean,
    contactModel: string,
    showForceChains: boolean,
    dangerAlpha: number,
    time: number,
    profile: RenderProfile,
  ) {
    const gl = this.gl;
    const ctx = this.overlayCtx;
    const inst = this.extInst!;

    const showPreview = !gameOver && canDrop;
    const instCount = pCount + (showPreview ? 1 : 0);
    this.ensureInstanceCapacity(instCount);

    const data = new Float32Array(instCount * 11);
    for (let i = 0; i < pCount; i++) {
      const lv = pLevel[i];
      const o = i * 11;
      data[o] = px[i]; data[o + 1] = py[i];
      data[o + 2] = pRadius[i]; data[o + 3] = pAngle[i];
      data[o + 4] = LEVEL_FILL[lv * 3]; data[o + 5] = LEVEL_FILL[lv * 3 + 1]; data[o + 6] = LEVEL_FILL[lv * 3 + 2];
      data[o + 7] = LEVEL_STROKE[lv * 3]; data[o + 8] = LEVEL_STROKE[lv * 3 + 1]; data[o + 9] = LEVEL_STROKE[lv * 3 + 2];
      data[o + 10] = lv;
    }
    let previewX = 0;
    if (showPreview) {
      const r = LEVELS[currentLevel].radius;
      previewX = Math.max(CL + r + 2, Math.min(CR - r - 2, dropX));
      const o = pCount * 11;
      data[o] = previewX; data[o + 1] = DROP_Y; data[o + 2] = r;
      data[o + 3] = 0;
      data[o + 4] = LEVEL_FILL[currentLevel * 3]; data[o + 5] = LEVEL_FILL[currentLevel * 3 + 1]; data[o + 6] = LEVEL_FILL[currentLevel * 3 + 2];
      data[o + 7] = LEVEL_STROKE[currentLevel * 3]; data[o + 8] = LEVEL_STROKE[currentLevel * 3 + 1]; data[o + 9] = LEVEL_STROKE[currentLevel * 3 + 2];
      data[o + 10] = currentLevel;
    }

    // Glow instances: Force Chain beams + Contact sparks + effects
    const chainsOn = showForceChains && profile.effectiveForceChains && contactVisLen > 0;
    const stride = Math.max(1, profile.forceChainStride);
    let glowCount = 0;
    if (chainsOn) {
      glowCount += Math.ceil(contactVisLen / stride) * 2;
    }
    glowCount += effects.length;
    this.ensureGlowCapacity(glowCount);
    const glow = new Float32Array(Math.max(glowCount, 1) * 10);
    let gi = 0;

    if (chainsOn) {
      let maxF = 1;
      for (let i = 0; i < contactVisLen; i += stride) {
        if (contactVis[i].force > maxF) maxF = contactVis[i].force;
      }
      const invMax = 1 / maxF;

      for (let i = 0; i < contactVisLen; i += stride) {
        const c = contactVis[i];
        const t = Math.min(c.force * invMax, 1);
        const col = getForceColor(t);
        const halfThick = getForceLineWidth(t) * 0.5;

        // 1. Force Chain Beam
        const dx = c.x2 - c.x1;
        const dy = c.y2 - c.y1;
        const len = Math.sqrt(dx * dx + dy * dy);
        if (len > 1e-4) {
          const angle = Math.atan2(dy, dx);
          const o = gi * 10;
          glow[o] = (c.x1 + c.x2) * 0.5;
          glow[o + 1] = (c.y1 + c.y2) * 0.5;
          glow[o + 2] = len * 0.5;
          glow[o + 3] = halfThick;
          glow[o + 4] = angle;
          glow[o + 5] = col[0]; glow[o + 6] = col[1]; glow[o + 7] = col[2];
          glow[o + 8] = 0.5 + t * 0.5;
          glow[o + 9] = 0.0; // beam line
          gi++;
        }

        // 2. Contact Point Spark
        const sparkR = 7 + t * 18;
        const so = gi * 10;
        glow[so] = c.x;
        glow[so + 1] = c.y;
        glow[so + 2] = sparkR;
        glow[so + 3] = sparkR;
        glow[so + 4] = 0.0;
        glow[so + 5] = col[0]; glow[so + 6] = col[1]; glow[so + 7] = col[2];
        glow[so + 8] = 0.6 + t * 0.4;
        glow[so + 9] = 1.0; // radial glow
        gi++;
      }
    }

    for (const e of effects) {
      const c = hexToRGB(e.color);
      const o = gi * 10;
      glow[o] = e.x; glow[o + 1] = e.y;
      glow[o + 2] = e.r * 1.4;
      glow[o + 3] = e.r * 1.4;
      glow[o + 4] = 0.0;
      glow[o + 5] = c[0]; glow[o + 6] = c[1]; glow[o + 7] = c[2];
      glow[o + 8] = e.alpha * 0.5;
      glow[o + 9] = 1.0; // radial glow
      gi++;
    }

    // ---- GL draw ----
    gl.viewport(0, 0, GAME_W, GAME_H);
    gl.clearColor(0.059, 0.059, 0.137, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);

    // 1. Draw particles
    gl.useProgram(this.particleProgram);
    gl.uniform2f(gl.getUniformLocation(this.particleProgram, 'uResolution'), GAME_W, GAME_H);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.instVBO);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    if (this.vaoExt && this.particleVAO) this.vaoExt.bindVertexArrayOES(this.particleVAO);
    else this.particleBind!();
    inst.drawArraysInstancedANGLE(gl.TRIANGLES, 0, 6, instCount);

    // 2. Draw Force Chains & Glows on top with additive blend
    if (gi > 0) {
      gl.useProgram(this.glowProgram);
      gl.uniform2f(gl.getUniformLocation(this.glowProgram, 'uResolution'), GAME_W, GAME_H);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.glowVBO);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, glow.subarray(0, gi * 10));
      if (this.vaoExt && this.glowVAO) this.vaoExt.bindVertexArrayOES(this.glowVAO);
      else this.glowBind!();
      inst.drawArraysInstancedANGLE(gl.TRIANGLES, 0, 6, gi);
    }
    if (this.vaoExt) this.vaoExt.bindVertexArrayOES(null);

    // ---- Canvas2D overlay (text, walls) ----
    ctx.clearRect(0, 0, GAME_W, GAME_H);

    ctx.fillStyle = '#1e1e45';
    ctx.fillRect(0, 0, WALL_T, GAME_H);
    ctx.fillRect(CR, 0, WALL_T, GAME_H);
    ctx.fillRect(0, CB, GAME_W, WALL_T);
    ctx.strokeStyle = '#3a3a7a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(CL, 0); ctx.lineTo(CL, CB); ctx.lineTo(CR, CB); ctx.lineTo(CR, 0);
    ctx.stroke();

    const dAlpha = dangerAlpha > 0 ? 0.7 + 0.3 * Math.abs(Math.sin(time / 130)) : 0.6;
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

    if (!gameOver) {
      for (let i = 0; i < pCount; i++) {
        const r = pRadius[i];
        if (!profile.drawParticleLabels && r < profile.minLabelRadius) continue;
        const fontSize = Math.max(8, Math.floor(r * 0.3));
        ctx.save();
        ctx.translate(px[i], py[i]);
        ctx.rotate(pAngle[i]);
        ctx.font = `bold ${fontSize}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#3E2723';
        ctx.fillText(LEVELS[pLevel[i]].sieve, 0, 0);
        ctx.restore();
      }
    }

    if (showPreview) {
      const r = LEVELS[currentLevel].radius;
      const fontSize = Math.max(8, Math.floor(r * 0.3));
      ctx.save();
      ctx.translate(previewX, DROP_Y);
      ctx.font = `bold ${fontSize}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#3E2723';
      ctx.fillText(LEVELS[currentLevel].sieve, 0, 0);
      ctx.restore();
    }

    if (profile.showDropGuide && showPreview) {
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(previewX, DROP_Y + LEVELS[currentLevel].radius);
      ctx.lineTo(previewX, CB);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (comboCount > 1) {
      const remainingRatio = Math.max(0, comboTimer / COMBO_TIMEOUT_STEPS);
      const cx = GAME_W / 2;
      const cy = 46;
      ctx.save();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = '900 24px "Trebuchet MS", sans-serif';
      ctx.fillStyle = '#ffd54f';
      ctx.shadowColor = 'rgba(255, 193, 7, 0.4)';
      ctx.shadowBlur = 12;
      ctx.fillText(`COMBO x${comboCount}`, cx, cy);
      ctx.restore();

      const bw = 140;
      const bh = 5;
      const bx = cx - bw / 2;
      const by = cy + 16;
      ctx.fillStyle = 'rgba(255,255,255,0.12)';
      ctx.fillRect(bx, by, bw, bh);
      ctx.fillStyle = '#ffd54f';
      ctx.fillRect(bx, by, bw * remainingRatio, bh);
    }

    const maxPop = Math.min(scorePopups.length, profile.maxScorePopups);
    for (let i = 0; i < maxPop; i++) {
      const sp = scorePopups[i];
      const alpha = sp.timer / 60;
      const sz = 14 + (1 - alpha) * 8;
      ctx.font = `bold ${sz}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillStyle = `rgba(255,215,0,${alpha})`;
      ctx.fillText(sp.text, sp.x, sp.y - (60 - sp.timer) * 0.8);
    }
  }
}
