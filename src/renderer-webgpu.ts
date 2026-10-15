// ================================================================
// WebGPU Renderer — instanced particles + instanced additive glow
// (force chains / effects). Text & overlays on a Canvas2D overlay.
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

// Precomputed per-level colors (f32) — no per-frame parsing
const LEVEL_FILL = new Float32Array(LEVELS.length * 3);
const LEVEL_STROKE = new Float32Array(LEVELS.length * 3);
for (let i = 0; i < LEVELS.length; i++) {
  const [r, g, b] = hexToRGB(LEVELS[i].color);
  const [sr, sg, sb] = hexToRGB(LEVELS[i].strokeColor);
  LEVEL_FILL[i * 3] = r; LEVEL_FILL[i * 3 + 1] = g; LEVEL_FILL[i * 3 + 2] = b;
  LEVEL_STROKE[i * 3] = sr; LEVEL_STROKE[i * 3 + 1] = sg; LEVEL_STROKE[i * 3 + 2] = sb;
}

const PARTICLE_WGSL = `
struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) strokeColor: vec3<f32>,
  @location(3) radius: f32,
  @location(4) level: f32,
}

@vertex
fn vs_main(
  @location(0) pos: vec2<f32>,
  @location(1) offset: vec2<f32>,
  @location(2) radius: f32,
  @location(3) angle: f32,
  @location(4) color: vec3<f32>,
  @location(5) strokeColor: vec3<f32>,
  @location(6) level: f32,
) -> VertexOut {
  var out: VertexOut;
  let c = cos(angle);
  let s = sin(angle);
  let rot = vec2<f32>(pos.x * c - pos.y * s, pos.x * s + pos.y * c);
  let world = rot * radius + offset;
  let clip = (world / vec2<f32>(${GAME_W}.0, ${GAME_H}.0)) * 2.0 - 1.0;
  out.position = vec4<f32>(clip.x, -clip.y, 0.0, 1.0);
  out.uv = pos;
  out.color = color;
  out.strokeColor = strokeColor;
  out.radius = radius;
  out.level = level;
  return out;
}

fn hash(n: f32) -> f32 {
  return fract(sin(n) * 43758.5453123);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let dist = length(in.uv);
  if (dist > 1.0) { discard; }
  let sw = 2.0 / in.radius;
  if (dist > 1.0 - sw) {
    return vec4<f32>(in.strokeColor, 1.0);
  }
  let lightC = min(in.color + vec3<f32>(0.137), vec3<f32>(1.0));
  let ld = length(in.uv - vec2<f32>(-0.25, -0.25));
  let t = smoothstep(0.05, 1.0, ld);
  var gc = mix(lightC, in.color, t);
  let li = i32(in.level + 0.5);
  let dc = min(li * 6 + 4, 27);
  for (var i = 0; i < 27; i++) {
    if (i >= dc) { break; }
    let seed = in.level * 54321.0 + 7.0 + f32(i) * 1337.0;
    let dx = hash(seed) * 2.0 - 1.0;
    let dy = hash(seed + 1.0) * 2.0 - 1.0;
    if (dx*dx + dy*dy < 0.49) {
      let d = length(in.uv - vec2<f32>(dx * 0.7, dy * 0.7));
      if (d < 0.05) { gc = mix(gc, in.strokeColor * 0.5, 0.3); }
    }
  }
  return vec4<f32>(gc, 1.0);
}
`;

// Additive glow — unit quad, radial falloff (force chains & effects)
const GLOW_WGSL = `
struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) alpha: f32,
}

@vertex
fn vs_main(
  @location(0) pos: vec2<f32>,
  @location(1) offset: vec2<f32>,
  @location(2) radius: f32,
  @location(3) color: vec3<f32>,
  @location(4) alpha: f32,
) -> VertexOut {
  var out: VertexOut;
  let world = pos * radius + offset;
  let clip = (world / vec2<f32>(${GAME_W}.0, ${GAME_H}.0)) * 2.0 - 1.0;
  out.position = vec4<f32>(clip.x, -clip.y, 0.0, 1.0);
  out.uv = pos;
  out.color = color;
  out.alpha = alpha;
  return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  let fall = (1.0 - d) * (1.0 - d);
  return vec4<f32>(in.color * fall * in.alpha, fall * in.alpha);
}
`;

const CIRCLE_SEGS = 24;
const GPU_BUFFER_USAGE_VERTEX = 0x0020;
const GPU_BUFFER_USAGE_COPY_DST = 0x0008;

const ADDITIVE_BLEND: GPUBlendState = {
  color: { srcFactor: 'src-alpha', dstFactor: 'one' },
  alpha: { srcFactor: 'one', dstFactor: 'one' },
};

export class WebGPURenderer {
  device: GPUDevice;
  context: GPUCanvasContext;
  pipeline: GPURenderPipeline;
  glowPipeline: GPURenderPipeline;
  vertexBuffer: GPUBuffer;
  quadBuffer: GPUBuffer;
  instanceBuffer: GPUBuffer;
  glowInstanceBuffer: GPUBuffer;
  overlayCanvas: HTMLCanvasElement;
  overlayCtx: CanvasRenderingContext2D;
  maxInst = 512;
  maxGlowInst = 1024;
  circleVertCount: number;

  constructor(canvas: HTMLCanvasElement, device: GPUDevice) {
    this.device = device;
    this.context = canvas.getContext('webgpu') as unknown as GPUCanvasContext;
    this.context.configure({ device, format: 'rgba8unorm', alphaMode: 'premultiplied' });

    // Circle fan mesh
    const triVerts: number[] = [];
    for (let i = 0; i < CIRCLE_SEGS; i++) {
      const a0 = (i / CIRCLE_SEGS) * Math.PI * 2;
      const a1 = ((i + 1) / CIRCLE_SEGS) * Math.PI * 2;
      triVerts.push(0, 0, Math.cos(a0), Math.sin(a0), Math.cos(a1), Math.sin(a1));
    }
    this.circleVertCount = CIRCLE_SEGS * 3;

    this.vertexBuffer = device.createBuffer({
      size: triVerts.length * 4,
      usage: GPU_BUFFER_USAGE_VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(triVerts);
    this.vertexBuffer.unmap();

    // Unit quad for glows
    const quad = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
    this.quadBuffer = device.createBuffer({
      size: quad.byteLength,
      usage: GPU_BUFFER_USAGE_VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.quadBuffer.getMappedRange()).set(quad);
    this.quadBuffer.unmap();

    this.instanceBuffer = this.createInstanceBuffer(this.maxInst);
    this.glowInstanceBuffer = this.createGlowBuffer(this.maxGlowInst);

    const shaderModule = device.createShaderModule({ code: PARTICLE_WGSL });
    const glowModule = device.createShaderModule({ code: GLOW_WGSL });

    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [
          { arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] },
          {
            arrayStride: 44,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 1, offset: 0, format: 'float32x2' },
              { shaderLocation: 2, offset: 8, format: 'float32' },
              { shaderLocation: 3, offset: 12, format: 'float32' },
              { shaderLocation: 4, offset: 16, format: 'float32x3' },
              { shaderLocation: 5, offset: 28, format: 'float32x3' },
              { shaderLocation: 6, offset: 40, format: 'float32' },
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: 'rgba8unorm', blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.glowPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: glowModule,
        entryPoint: 'vs_main',
        buffers: [
          { arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] },
          {
            arrayStride: 28,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 1, offset: 0, format: 'float32x2' },
              { shaderLocation: 2, offset: 8, format: 'float32' },
              { shaderLocation: 3, offset: 12, format: 'float32x3' },
              { shaderLocation: 4, offset: 24, format: 'float32' },
            ],
          },
        ],
      },
      fragment: {
        module: glowModule,
        entryPoint: 'fs_main',
        targets: [{ format: 'rgba8unorm', blend: ADDITIVE_BLEND }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // Overlay canvas for text
    this.overlayCanvas = document.createElement('canvas');
    this.overlayCanvas.width = GAME_W;
    this.overlayCanvas.height = GAME_H;
    this.overlayCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;border-radius:12px;z-index:10';
    this.overlayCtx = this.overlayCanvas.getContext('2d')!;
    document.getElementById('game-section')!.style.position = 'relative';
    document.getElementById('game-section')!.appendChild(this.overlayCanvas);
  }

  private createInstanceBuffer(maxInst: number): GPUBuffer {
    return this.device.createBuffer({
      size: maxInst * 11 * 4,
      usage: GPU_BUFFER_USAGE_VERTEX | GPU_BUFFER_USAGE_COPY_DST,
    });
  }

  private createGlowBuffer(maxInst: number): GPUBuffer {
    return this.device.createBuffer({
      size: maxInst * 7 * 4,
      usage: GPU_BUFFER_USAGE_VERTEX | GPU_BUFFER_USAGE_COPY_DST,
    });
  }

  private ensureInstanceCapacity(count: number) {
    if (count <= this.maxInst) return;
    let next = this.maxInst;
    while (next < count) next *= 2;
    this.instanceBuffer.destroy();
    this.maxInst = next;
    this.instanceBuffer = this.createInstanceBuffer(this.maxInst);
  }

  private ensureGlowCapacity(count: number) {
    if (count <= this.maxGlowInst) return;
    let next = this.maxGlowInst;
    while (next < count) next *= 2;
    this.glowInstanceBuffer.destroy();
    this.maxGlowInst = next;
    this.glowInstanceBuffer = this.createGlowBuffer(this.maxGlowInst);
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
    const device = this.device;
    const ctx = this.overlayCtx;

    const showPreview = !gameOver && canDrop;
    const instCount = pCount + (showPreview ? 1 : 0);
    this.ensureInstanceCapacity(instCount);

    const inst = new Float32Array(instCount * 11);
    for (let i = 0; i < pCount; i++) {
      const lv = pLevel[i];
      const o = i * 11;
      inst[o] = px[i]; inst[o + 1] = py[i]; inst[o + 2] = pRadius[i];
      inst[o + 3] = pAngle[i];
      inst[o + 4] = LEVEL_FILL[lv * 3]; inst[o + 5] = LEVEL_FILL[lv * 3 + 1]; inst[o + 6] = LEVEL_FILL[lv * 3 + 2];
      inst[o + 7] = LEVEL_STROKE[lv * 3]; inst[o + 8] = LEVEL_STROKE[lv * 3 + 1]; inst[o + 9] = LEVEL_STROKE[lv * 3 + 2];
      inst[o + 10] = lv;
    }
    let previewX = 0;
    if (showPreview) {
      const r = LEVELS[currentLevel].radius;
      previewX = Math.max(CL + r + 2, Math.min(CR - r - 2, dropX));
      const o = pCount * 11;
      inst[o] = previewX; inst[o + 1] = DROP_Y; inst[o + 2] = r;
      inst[o + 3] = 0;
      inst[o + 4] = LEVEL_FILL[currentLevel * 3]; inst[o + 5] = LEVEL_FILL[currentLevel * 3 + 1]; inst[o + 6] = LEVEL_FILL[currentLevel * 3 + 2];
      inst[o + 7] = LEVEL_STROKE[currentLevel * 3]; inst[o + 8] = LEVEL_STROKE[currentLevel * 3 + 1]; inst[o + 9] = LEVEL_STROKE[currentLevel * 3 + 2];
      inst[o + 10] = currentLevel;
    }
    device.queue.writeBuffer(this.instanceBuffer, 0, inst);

    // Glow instances: force chains + effect rings
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
    if (glowCount > 0) {
      device.queue.writeBuffer(this.glowInstanceBuffer, 0, glow.subarray(0, glowCount * 7));
    }

    // Single render pass: particles, then additive glows on top
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.059, g: 0.059, b: 0.137, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setVertexBuffer(1, this.instanceBuffer);
    pass.draw(this.circleVertCount, instCount);
    if (glowCount > 0) {
      pass.setPipeline(this.glowPipeline);
      pass.setVertexBuffer(0, this.quadBuffer);
      pass.setVertexBuffer(1, this.glowInstanceBuffer);
      pass.draw(6, glowCount);
    }
    pass.end();
    device.queue.submit([encoder.finish()]);

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

    // Particle labels
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

    // Preview label
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

    // Score popups
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

    // Combo
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

    // Drop guide
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

  dispose() {
    // cleanup
  }
}
