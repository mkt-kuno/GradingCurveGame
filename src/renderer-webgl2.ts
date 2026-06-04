// ================================================================
// WebGL2 Renderer — instanced particles, FBO glow, texture atlas text
// ================================================================

import { LEVELS, GAME_W, GAME_H, WALL_T, CL, CR, CB, DANGER_Y } from './constants';

interface ContactVis { x: number; y: number; force: number; }
interface Effect { x: number; y: number; r: number; alpha: number; color: string; }
interface ScorePopup { x: number; y: number; text: string; timer: number; }
interface Particle { x: number; y: number; vx: number; vy: number; radius: number; mass: number; inertia: number; level: number; angle: number; omega: number; active: boolean; graceFrames: number; id: number; }

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error('Shader compile:', gl.getShaderInfoLog(s));
    gl.deleteShader(s);
    throw new Error('Shader compile failed');
  }
  return s;
}

function createProgram(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compileShader(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compileShader(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    console.error('Program link:', gl.getProgramInfoLog(p));
    throw new Error('Program link failed');
  }
  return p;
}

function createFBO(gl: WebGL2RenderingContext, w: number, h: number) {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fb = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { texture: tex, framebuffer: fb };
}

function hexToRGB(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) / 255, ((n >> 8) & 0xFF) / 255, (n & 0xFF) / 255];
}

const BG_VS = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const BG_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;
uniform vec2 u_res;
uniform float u_wallT;
uniform float u_cl;
uniform float u_cr;
uniform float u_cb;
uniform float u_dangerY;
uniform float u_dangerAlpha;
uniform float u_time;
void main() {
  vec2 px = v_uv * u_res;
  vec3 wallColor = vec3(0.118, 0.118, 0.271);
  vec3 topC = vec3(0.082, 0.082, 0.188);
  vec3 botC = vec3(0.051, 0.051, 0.125);
  vec3 outsideC = vec3(0.059, 0.059, 0.137);
  vec3 color;
  if (px.x < u_wallT || px.x > u_cr || px.y > u_cb) {
    color = wallColor;
  } else {
    color = mix(topC, botC, px.y / u_res.y);
  }
  if (px.x < u_wallT || px.x > u_cr || px.y < 0.0 || px.y > u_res.y) {
    color = outsideC;
  }
  if (abs(px.y - u_dangerY) < 2.5 && px.x >= u_cl && px.x <= u_cr) {
    float dash = mod(px.x + u_time * 0.5, 18.0);
    if (dash < 12.0) {
      color = mix(color, vec3(1.0, 0.627, 0.0), u_dangerAlpha * 0.9);
    }
  }
  fragColor = vec4(color, 1.0);
}`;

const PARTICLE_VS = `#version 300 es
precision highp float;
in vec2 a_pos;
in vec2 a_offset;
in float a_radius;
in float a_angle;
in vec3 a_color;
in vec3 a_strokeColor;
in float a_level;
uniform vec2 u_res;
out vec2 v_uv;
out vec3 v_color;
out vec3 v_strokeColor;
flat out float v_radius;
flat out float v_level;
void main() {
  float c = cos(a_angle);
  float s = sin(a_angle);
  vec2 rot = vec2(a_pos.x*c - a_pos.y*s, a_pos.x*s + a_pos.y*c);
  vec2 world = rot * a_radius + a_offset;
  vec2 clip = (world / u_res) * 2.0 - 1.0;
  clip.y = -clip.y;
  gl_Position = vec4(clip, 0.0, 1.0);
  v_uv = a_pos;
  v_color = a_color;
  v_strokeColor = a_strokeColor;
  v_radius = a_radius;
  v_level = a_level;
}`;

const PARTICLE_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
in vec3 v_color;
in vec3 v_strokeColor;
flat in float v_radius;
flat in float v_level;
out vec4 fragColor;
float hash(float n) { return fract(sin(n) * 43758.5453123); }
void main() {
  float dist = length(v_uv);
  if (dist > 1.0) discard;
  float sw = 2.0 / v_radius;
  if (dist > 1.0 - sw) {
    fragColor = vec4(v_strokeColor, 1.0);
    return;
  }
  vec3 lightC = min(v_color + vec3(0.137), vec3(1.0));
  float ld = length(v_uv - vec2(-0.25, -0.25));
  float t = smoothstep(0.05, 1.0, ld);
  vec3 gc = mix(lightC, v_color, t);
  int li = int(v_level + 0.5);
  int dc = min(li * 4 + 3, 18);
  for (int i = 0; i < 18; i++) {
    if (i >= dc) break;
    float seed = v_level * 54321.0 + 7.0 + float(i) * 1337.0;
    float dx = hash(seed) * 2.0 - 1.0;
    float dy = hash(seed + 1.0) * 2.0 - 1.0;
    if (dx*dx + dy*dy < 0.49) {
      float d = length(v_uv - vec2(dx*0.7, dy*0.7));
      if (d < 0.05) gc = mix(gc, v_strokeColor * 0.5, 0.3);
    }
  }
  fragColor = vec4(gc, 1.0);
}`;

const GLOW_VS = `#version 300 es
precision highp float;
in vec2 a_offset;
in float a_intensity;
uniform vec2 u_res;
uniform float u_baseR;
out float v_int;
void main() {
  vec2 clip = (a_offset / u_res) * 2.0 - 1.0;
  clip.y = -clip.y;
  gl_Position = vec4(clip, 0.0, 1.0);
  gl_PointSize = u_baseR * (1.0 + a_intensity * 2.0);
  v_int = a_intensity;
}`;

const GLOW_FS = `#version 300 es
precision highp float;
in float v_int;
out vec4 fragColor;
void main() {
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float dist = length(uv);
  if (dist > 1.0) discard;
  float core = exp(-dist*dist*8.0);
  float outer = exp(-dist*dist*3.0);
  vec3 coreC = vec3(1.0, 1.0, mix(0.9, 0.2, v_int));
  vec3 outerC = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.2, 0.0), v_int);
  vec3 color = mix(outerC, coreC, core);
  fragColor = vec4(color, outer * v_int);
}`;

const BLUR_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
uniform vec2 u_texel;
uniform float u_radius;
uniform vec2 u_dir;
void main() {
  vec4 sum = vec4(0.0);
  float tw = 0.0;
  for (int i = -20; i <= 20; i++) {
    if (abs(float(i)) > u_radius) continue;
    float w = exp(-float(i*i) / (2.0*u_radius*u_radius));
    sum += texture(u_tex, v_uv + u_dir * float(i) * u_texel) * w;
    tw += w;
  }
  fragColor = sum / tw;
}`;

const COMPOSITE_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_scene;
uniform sampler2D u_glow;
void main() {
  vec4 scene = texture(u_scene, v_uv);
  vec4 glow = texture(u_glow, v_uv);
  fragColor = scene + glow;
}`;

const CIRCLE_SEGS = 24;

export class WebGL2Renderer {
  gl: WebGL2RenderingContext;
  private bgProg: WebGLProgram;
  private particleProg: WebGLProgram;
  private glowProg: WebGLProgram;
  private blurProg: WebGLProgram;
  private compositeProg: WebGLProgram;
  private quadVAO: WebGLVertexArrayObject;
  private particleVAO: WebGLVertexArrayObject;
  private particleBaseVBO: WebGLBuffer;
  private particleInstVBO: WebGLBuffer;
  private glowVAO: WebGLVertexArrayObject;
  private glowVBO: WebGLBuffer;
  private sceneFBO: { texture: WebGLTexture; framebuffer: WebGLFramebuffer };
  private glowFBO: { texture: WebGLTexture; framebuffer: WebGLFramebuffer };
  private blurAFBO: { texture: WebGLTexture; framebuffer: WebGLFramebuffer };
  private blurBFBO: { texture: WebGLTexture; framebuffer: WebGLFramebuffer };
  private maxInst = 512;
  private maxGlow = 512;
  private circleVertCount: number;

  overlayCanvas: HTMLCanvasElement;
  overlayCtx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: true, premultipliedAlpha: false })!;
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;

    this.bgProg = createProgram(gl, BG_VS, BG_FS);
    this.particleProg = createProgram(gl, PARTICLE_VS, PARTICLE_FS);
    this.glowProg = createProgram(gl, GLOW_VS, GLOW_FS);
    this.blurProg = createProgram(gl, BG_VS, BLUR_FS);
    this.compositeProg = createProgram(gl, BG_VS, COMPOSITE_FS);

    // Quad VAO
    this.quadVAO = gl.createVertexArray()!;
    gl.bindVertexArray(this.quadVAO);
    const qvbo = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, qvbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // Particle circle mesh
    const triVerts: number[] = [];
    for (let i = 0; i < CIRCLE_SEGS; i++) {
      const a0 = (i / CIRCLE_SEGS) * Math.PI * 2;
      const a1 = ((i + 1) / CIRCLE_SEGS) * Math.PI * 2;
      triVerts.push(0, 0, Math.cos(a0), Math.sin(a0), Math.cos(a1), Math.sin(a1));
    }
    this.circleVertCount = CIRCLE_SEGS * 3;

    this.particleVAO = gl.createVertexArray()!;
    gl.bindVertexArray(this.particleVAO);

    this.particleBaseVBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleBaseVBO);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(triVerts), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);

    this.particleInstVBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleInstVBO);
    gl.bufferData(gl.ARRAY_BUFFER, this.maxInst * 11 * 4, gl.DYNAMIC_DRAW);
    const S = 44;
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, S, 0);
    gl.vertexAttribDivisor(1, 1);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, S, 8);
    gl.vertexAttribDivisor(2, 1);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 1, gl.FLOAT, false, S, 12);
    gl.vertexAttribDivisor(3, 1);
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 3, gl.FLOAT, false, S, 16);
    gl.vertexAttribDivisor(4, 1);
    gl.enableVertexAttribArray(5);
    gl.vertexAttribPointer(5, 3, gl.FLOAT, false, S, 28);
    gl.vertexAttribDivisor(5, 1);
    gl.enableVertexAttribArray(6);
    gl.vertexAttribPointer(6, 1, gl.FLOAT, false, S, 40);
    gl.vertexAttribDivisor(6, 1);
    gl.bindVertexArray(null);

    // Glow points VAO
    this.glowVAO = gl.createVertexArray()!;
    gl.bindVertexArray(this.glowVAO);
    this.glowVBO = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.glowVBO);
    gl.bufferData(gl.ARRAY_BUFFER, this.maxGlow * 3 * 4, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 12, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 12, 8);
    gl.bindVertexArray(null);

    // FBOs
    this.sceneFBO = createFBO(gl, GAME_W, GAME_H);
    this.glowFBO = createFBO(gl, GAME_W, GAME_H);
    this.blurAFBO = createFBO(gl, GAME_W, GAME_H);
    this.blurBFBO = createFBO(gl, GAME_W, GAME_H);

    // Overlay canvas for text
    this.overlayCanvas = document.createElement('canvas');
    this.overlayCanvas.width = GAME_W;
    this.overlayCanvas.height = GAME_H;
    this.overlayCanvas.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;border-radius:12px;z-index:10';
    this.overlayCtx = this.overlayCanvas.getContext('2d')!;
    document.getElementById('game-section')!.style.position = 'relative';
    document.getElementById('game-section')!.appendChild(this.overlayCanvas);
  }

  drawFrame(
    particles: Particle[],
    contactVis: ContactVis[],
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
    rendererName: string,
  ) {
    const gl = this.gl;
    gl.viewport(0, 0, GAME_W, GAME_H);

    // Clear blur buffer to prevent stale glow
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurBFBO.framebuffer);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // === Background ===
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.sceneFBO.framebuffer);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.bgProg);
    gl.uniform2f(gl.getUniformLocation(this.bgProg, 'u_res'), GAME_W, GAME_H);
    gl.uniform1f(gl.getUniformLocation(this.bgProg, 'u_wallT'), WALL_T);
    gl.uniform1f(gl.getUniformLocation(this.bgProg, 'u_cl'), CL);
    gl.uniform1f(gl.getUniformLocation(this.bgProg, 'u_cr'), CR);
    gl.uniform1f(gl.getUniformLocation(this.bgProg, 'u_cb'), CB);
    gl.uniform1f(gl.getUniformLocation(this.bgProg, 'u_dangerY'), DANGER_Y);
    gl.uniform1f(gl.getUniformLocation(this.bgProg, 'u_dangerAlpha'), dangerAlpha);
    gl.uniform1f(gl.getUniformLocation(this.bgProg, 'u_time'), time);
    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // === Glow (force chains) ===
    if (showForceChains && contactVis.length > 0) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.glowFBO.framebuffer);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

      gl.useProgram(this.glowProg);
      gl.uniform2f(gl.getUniformLocation(this.glowProg, 'u_res'), GAME_W, GAME_H);
      gl.uniform1f(gl.getUniformLocation(this.glowProg, 'u_baseR'), 30.0);

      const cnt = Math.min(contactVis.length, this.maxGlow);
      const gd = new Float32Array(cnt * 3);
      const maxF = Math.max(...contactVis.map(c => c.force), 1);
      for (let i = 0; i < cnt; i++) {
        gd[i * 3] = contactVis[i].x;
        gd[i * 3 + 1] = contactVis[i].y;
        gd[i * 3 + 2] = Math.min(contactVis[i].force / maxF, 1);
      }
      gl.bindVertexArray(this.glowVAO);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.glowVBO);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, gd);
      gl.drawArrays(gl.POINTS, 0, cnt);
      gl.disable(gl.BLEND);

      // Blur H
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurAFBO.framebuffer);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.blurProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.glowFBO.texture);
      gl.uniform1i(gl.getUniformLocation(this.blurProg, 'u_tex'), 0);
      gl.uniform2f(gl.getUniformLocation(this.blurProg, 'u_texel'), 1 / GAME_W, 1 / GAME_H);
      gl.uniform1f(gl.getUniformLocation(this.blurProg, 'u_radius'), 10.0);
      gl.uniform2f(gl.getUniformLocation(this.blurProg, 'u_dir'), 1, 0);
      gl.bindVertexArray(this.quadVAO);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      // Blur V
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurBFBO.framebuffer);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.blurAFBO.texture);
      gl.uniform2f(gl.getUniformLocation(this.blurProg, 'u_dir'), 0, 1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      // Composite glow onto scene
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.sceneFBO.framebuffer);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      gl.useProgram(this.compositeProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.sceneFBO.texture);
      gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'u_scene'), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.blurBFBO.texture);
      gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'u_glow'), 1);
      gl.bindVertexArray(this.quadVAO);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.disable(gl.BLEND);
    }

    // === Particles ===
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.sceneFBO.framebuffer);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.particleProg);
    gl.uniform2f(gl.getUniformLocation(this.particleProg, 'u_res'), GAME_W, GAME_H);

    let allParticles = particles;
    if (!gameOver && canDrop) {
      const r = LEVELS[currentLevel].radius;
      const cx = Math.max(CL + r + 2, Math.min(CR - r - 2, dropX));
      allParticles = [...particles, { x: cx, y: DROP_Y, vx: 0, vy: 0, radius: r, mass: 0, inertia: 0, level: currentLevel, angle: 0, omega: 0, active: false, graceFrames: 0, id: -1 }];
    }

    const cnt = Math.min(allParticles.length, this.maxInst);
    const inst = new Float32Array(cnt * 11);
    for (let i = 0; i < cnt; i++) {
      const p = allParticles[i];
      const info = LEVELS[p.level];
      const c = hexToRGB(info.color);
      const sc = hexToRGB(info.strokeColor);
      const o = i * 11;
      inst[o] = p.x; inst[o+1] = p.y; inst[o+2] = p.radius;
      inst[o+3] = p.angle; inst[o+4] = c[0]; inst[o+5] = c[1]; inst[o+6] = c[2];
      inst[o+7] = sc[0]; inst[o+8] = sc[1]; inst[o+9] = sc[2]; inst[o+10] = p.level;
    }
    gl.bindVertexArray(this.particleVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleInstVBO);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, inst);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, this.circleVertCount, cnt);

    // Drop guide line
    if (!gameOver && canDrop) {
      // Drawn in overlay
    }

    gl.disable(gl.BLEND);

    // === Final blit to screen ===
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.compositeProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sceneFBO.texture);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'u_scene'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.blurBFBO.texture);
    gl.uniform1i(gl.getUniformLocation(this.compositeProg, 'u_glow'), 1);
    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // === Canvas2D overlay for text, effects, UI ===
    const ctx = this.overlayCtx;
    ctx.clearRect(0, 0, GAME_W, GAME_H);

    for (const e of effects) {
      ctx.beginPath();
      ctx.arc(e.x, e.y, e.r, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,255,255,${e.alpha})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Text for all particles
    for (const p of particles) {
      const info = LEVELS[p.level];
      const fontSize = Math.max(8, Math.floor(p.radius * 0.3));
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.angle);
      ctx.font = `bold ${fontSize}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#3E2723';
      ctx.fillText(info.sieve, 0, 0);
      ctx.restore();
    }

    // Preview particle text
    if (!gameOver && canDrop) {
      const r = LEVELS[currentLevel].radius;
      const cx = Math.max(CL + r + 2, Math.min(CR - r - 2, dropX));
      const info = LEVELS[currentLevel];
      const fontSize = Math.max(8, Math.floor(r * 0.3));
      ctx.save();
      ctx.translate(cx, DROP_Y);
      ctx.font = `bold ${fontSize}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#3E2723';
      ctx.fillText(info.sieve, 0, 0);
      ctx.restore();
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
    }

    // Status line
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    const modelText = contactModel === 'hertz' ? 'Hertz Contact' : 'Hooke Contact';
    ctx.fillText(`${modelText}  N=${particles.length}  ${rendererName}`, CL + 4, 14);
  }

  dispose() {
    // cleanup
  }
}
