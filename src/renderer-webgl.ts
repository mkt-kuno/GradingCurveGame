// ================================================================
// WebGL Renderer — instanced sprite particles + instanced additive
// glow quads (force chains / effects). Used when WebGPU is missing
// (Safari iOS, older browsers). Text & overlays on Canvas2D overlay.
// ================================================================

import {
  LEVELS, GAME_W, GAME_H, WALL_T, CL, CR, CB, DANGER_Y, DROP_Y, COMBO_TIMEOUT_STEPS,
} from './constants';

interface ContactVis { x: number; y: number; force: number; }
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

const VS_PARTICLE = `
attribute vec2 aCorner;
attribute vec2 aOffset;
attribute float aRadius;
attribute float aAngle;
attribute float aTexIndex;
uniform vec2 uResolution;
uniform float uMaxRadius;
varying vec2 vUv;
varying float vTexIndex;
void main() {
  float c = cos(aAngle);
  float s = sin(aAngle);
  float scale = aRadius / uMaxRadius;
  vec2 quad = aCorner * 2.0 - 1.0;
  vec2 local = quad * (uMaxRadius + 10.0) * scale;
  vec2 rot = vec2(local.x * c - local.y * s, local.x * s + local.y * c);
  vec2 world = rot + aOffset;
  vec2 clip = (world / uResolution) * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  vUv = aCorner;
  vTexIndex = aTexIndex;
}
`;

const FS_PARTICLE = `
precision mediump float;
varying vec2 vUv;
varying float vTexIndex;
uniform sampler2D uAtlas;
uniform float uLevels;
void main() {
  float i = floor(vTexIndex + 0.5);
  vec2 uv = vec2((i + vUv.x) / uLevels, 1.0 - vUv.y);
  vec4 c = texture2D(uAtlas, uv);
  if (c.a < 0.01) { discard; }
  gl_FragColor = c;
}
`;

const VS_GLOW = `
attribute vec2 aCorner;
attribute vec2 aOffset;
attribute float aRadius;
attribute vec3 aColor;
attribute float aAlpha;
uniform vec2 uResolution;
varying vec2 vLocal;
varying vec3 vColor;
varying float vAlpha;
void main() {
  vec2 quad = aCorner * 2.0 - 1.0;
  vec2 world = quad * aRadius + aOffset;
  vec2 clip = (world / uResolution) * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  vLocal = quad;
  vColor = aColor;
  vAlpha = aAlpha;
}
`;

const FS_GLOW = `
precision mediump float;
varying vec2 vLocal;
varying vec3 vColor;
varying float vAlpha;
void main() {
  float d = length(vLocal);
  if (d > 1.0) { discard; }
  float fall = (1.0 - d) * (1.0 - d);
  gl_FragColor = vec4(vColor * fall * vAlpha, fall * vAlpha);
}
`;

export class WebGLRenderer {
  private gl: WebGLRenderingContext;
  private overlayCanvas: HTMLCanvasElement;
  private overlayCtx: CanvasRenderingContext2D;
  private particleProgram: WebGLProgram;
  private glowProgram: WebGLProgram;
  private atlas: WebGLTexture;
  private spriteCanvases: HTMLCanvasElement[];
  private maxRadius: number;
  private quadVBO: WebGLBuffer;
  private instVBO: WebGLBuffer;
  private glowVBO: WebGLBuffer;
  private maxInst = 512;
  private maxGlowInst = 1024;
  private extInst: ANGLE_instanced_arrays | null = null;
  private vaoExt: OES_vertex_array_object | null = null;
  private particleVAO: WebGLVertexArrayObjectOES | null = null;
  private glowVAO: WebGLVertexArrayObjectOES | null = null;

  constructor(
    canvas: HTMLCanvasElement,
    spriteCanvases: HTMLCanvasElement[],
  ) {
    const gl = canvas.getContext('webgl', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
    }) as WebGLRenderingContext | null;
    if (!gl) throw new Error('WebGL unavailable');
    this.gl = gl;
    this.spriteCanvases = spriteCanvases;
    this.maxRadius = Math.max(...LEVELS.map(l => l.radius));

    this.extInst = gl.getExtension('ANGLE_instanced_arrays');
    if (!this.extInst) throw new Error('instanced arrays unavailable');
    this.vaoExt = gl.getExtension('OES_vertex_array_object');

    this.particleProgram = this.compile(VS_PARTICLE, FS_PARTICLE);
    this.glowProgram = this.compile(VS_GLOW, FS_GLOW);
    this.atlas = this.buildAtlas();
    this.pad = spriteCanvases.length > 0
      ? (spriteCanvases[0].width - LEVELS[0].radius * 2) / 2
      : 10;

    const quad = new Float32Array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1]);
    this.quadVBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

    this.instVBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instVBO);
    gl.bufferData(gl.ARRAY_BUFFER, this.maxInst * 5 * 4, gl.DYNAMIC_DRAW);

    this.glowVBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.glowVBO);
    gl.bufferData(gl.ARRAY_BUFFER, this.maxGlowInst * 7 * 4, gl.DYNAMIC_DRAW);

    this.setupVAOs();

    // Overlay canvas for text
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

  private buildAtlas(): WebGLTexture {
    const gl = this.gl;
    const cell = (this.maxRadius + this.pad) * 2;
    const levels = this.spriteCanvases.length;
    const atlasCanvas = document.createElement('canvas');
    atlasCanvas.width = cell * levels;
    atlasCanvas.height = cell;
    const actx = atlasCanvas.getContext('2d')!;
    for (let i = 0; i < levels; i++) {
      actx.drawImage(this.spriteCanvases[i], i * cell, 0, cell, cell);
    }
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, atlasCanvas);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
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
      const stride = 5 * 4;
      const attrs: [string, number, number][] = [
        ['aOffset', 2, 0], ['aRadius', 1, 8], ['aAngle', 1, 12], ['aTexIndex', 1, 16],
      ];
      for (const [name, size, offset] of attrs) {
        const loc = gl.getAttribLocation(p, name);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
        inst.vertexAttribDivisorANGLE(loc, 1);
      }
    };

    const bindGlowAttribs = () => {
      const p = this.glowProgram;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
      const corner = gl.getAttribLocation(p, 'aCorner');
      gl.enableVertexAttribArray(corner);
      gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.glowVBO);
      const stride = 7 * 4;
      const attrs: [string, number, number][] = [
        ['aOffset', 2, 0], ['aRadius', 1, 8], ['aColor', 3, 12], ['aAlpha', 1, 24],
      ];
      for (const [name, size, offset] of attrs) {
        const loc = gl.getAttribLocation(p, name);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
        inst.vertexAttribDivisorANGLE(loc, 1);
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
      // Re-bind per draw; store the binders for later
      this.particleBind = bindParticleAttribs;
      this.glowBind = bindGlowAttribs;
    }
  }

  private particleBind: (() => void) | null = null;
  private glowBind: (() => void) | null = null;
  private pad = 10;

  private ensureInstanceCapacity(count: number) {
    if (count <= this.maxInst) return;
    let next = this.maxInst;
    while (next < count) next *= 2;
    this.maxInst = next;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instVBO);
    gl.bufferData(gl.ARRAY_BUFFER, this.maxInst * 5 * 4, gl.DYNAMIC_DRAW);
  }

  private ensureGlowCapacity(count: number) {
    if (count <= this.maxGlowInst) return;
    let next = this.maxGlowInst;
    while (next < count) next *= 2;
    this.maxGlowInst = next;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.glowVBO);
    gl.bufferData(gl.ARRAY_BUFFER, this.maxGlowInst * 7 * 4, gl.DYNAMIC_DRAW);
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

    const data = new Float32Array(instCount * 5);
    for (let i = 0; i < pCount; i++) {
      const o = i * 5;
      data[o] = px[i]; data[o + 1] = py[i];
      data[o + 2] = pRadius[i]; data[o + 3] = pAngle[i]; data[o + 4] = pLevel[i];
    }
    let previewX = 0;
    if (showPreview) {
      const r = LEVELS[currentLevel].radius;
      previewX = Math.max(CL + r + 2, Math.min(CR - r - 2, dropX));
      const o = pCount * 5;
      data[o] = previewX; data[o + 1] = DROP_Y; data[o + 2] = r;
      data[o + 3] = 0; data[o + 4] = currentLevel;
    }

    // Glow instances
    const chainsOn = showForceChains && profile.effectiveForceChains && contactVisLen > 0;
    const stride = Math.max(1, profile.forceChainStride);
    let glowCount = 0;
    if (chainsOn) glowCount += Math.ceil(contactVisLen / stride);
    glowCount += effects.length;
    this.ensureGlowCapacity(glowCount);
    const glow = new Float32Array(Math.max(glowCount, 1) * 7);
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
        const o = gi * 7;
        glow[o] = c.x; glow[o + 1] = c.y;
        glow[o + 2] = 8 + t * 24;
        glow[o + 3] = 1.0;
        glow[o + 4] = 0.55 + 0.45 * (1 - t);
        glow[o + 5] = 0.25 + 0.6 * (1 - t);
        glow[o + 6] = 0.35 + t * 0.5;
        gi++;
      }
    }
    for (const e of effects) {
      const c = hexToRGB(e.color);
      const o = gi * 7;
      glow[o] = e.x; glow[o + 1] = e.y;
      glow[o + 2] = e.r * 1.4;
      glow[o + 3] = c[0]; glow[o + 4] = c[1]; glow[o + 5] = c[2];
      glow[o + 6] = e.alpha * 0.5;
      gi++;
    }

    // ---- GL draw ----
    gl.viewport(0, 0, GAME_W, GAME_H);
    gl.clearColor(0.059, 0.059, 0.137, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);

    gl.useProgram(this.particleProgram);
    gl.uniform2f(gl.getUniformLocation(this.particleProgram, 'uResolution'), GAME_W, GAME_H);
    gl.uniform1f(gl.getUniformLocation(this.particleProgram, 'uMaxRadius'), this.maxRadius);
    gl.uniform1f(gl.getUniformLocation(this.particleProgram, 'uLevels'), this.spriteCanvases.length);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.atlas);
    gl.uniform1i(gl.getUniformLocation(this.particleProgram, 'uAtlas'), 0);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.instVBO);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    if (this.vaoExt && this.particleVAO) this.vaoExt.bindVertexArrayOES(this.particleVAO);
    else this.particleBind!();
    inst.drawArraysInstancedANGLE(gl.TRIANGLES, 0, 6, instCount);

    if (glowCount > 0) {
      gl.useProgram(this.glowProgram);
      gl.uniform2f(gl.getUniformLocation(this.glowProgram, 'uResolution'), GAME_W, GAME_H);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.glowVBO);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, glow.subarray(0, glowCount * 7));
      if (this.vaoExt && this.glowVAO) this.vaoExt.bindVertexArrayOES(this.glowVAO);
      else this.glowBind!();
      inst.drawArraysInstancedANGLE(gl.TRIANGLES, 0, 6, glowCount);
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

    const popupStart = Math.max(0, scorePopups.length - profile.maxScorePopups);
    for (let i = popupStart; i < scorePopups.length; i++) {
      const sp = scorePopups[i];
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

    if (showPreview && profile.showDropGuide) {
      const r = LEVELS[currentLevel].radius;
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(previewX, DROP_Y + r);
      ctx.lineTo(previewX, CB);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}
