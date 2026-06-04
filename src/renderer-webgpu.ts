// ================================================================
// WebGPU Renderer — instanced particle rendering with WGSL
// Background + text handled by Canvas2D overlay
// ================================================================

import {
  LEVELS, GAME_W, GAME_H, WALL_T, CL, CR, CB, DANGER_Y, DROP_Y,
} from './constants';

interface ContactVis { x: number; y: number; force: number; }
interface Effect { x: number; y: number; r: number; alpha: number; color: string; }
interface ScorePopup { x: number; y: number; text: string; timer: number; }
interface Particle { x: number; y: number; vx: number; vy: number; radius: number; mass: number; inertia: number; level: number; angle: number; omega: number; active: boolean; graceFrames: number; id: number; }

function hexToRGB(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) / 255, ((n >> 8) & 0xFF) / 255, (n & 0xFF) / 255];
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
fn hash2(n: f32) -> f32 {
  return fract(cos(n * 1.234) * 35791.234);
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
  let dc = min(li * 10 + 12, 80);
  for (var i = 0; i < 80; i++) {
    if (i >= dc) { break; }
    let seed = in.level * 54321.0 + 7.0 + f32(i) * 1337.0;
    let dx = hash(seed) * 2.0 - 1.0;
    let dy = hash(seed + 1.0) * 2.0 - 1.0;
    let inside = hash2(seed + 2.0);
    if (dx*dx + dy*dy < 0.55 && inside < 0.85) {
      let dotR = 0.015 + hash2(seed + 3.0) * 0.05;
      let d = length(in.uv - vec2<f32>(dx * 0.75, dy * 0.75));
      if (d < dotR) {
        let intensity = 0.15 + hash2(seed + 4.0) * 0.35;
        let gray = 0.3 + hash2(seed + 5.0) * 0.4;
        let dotCol = vec3<f32>(gray) * in.strokeColor;
        gc = mix(gc, dotCol, intensity);
      }
    }
  }
  return vec4<f32>(gc, 1.0);
}
`;

const CIRCLE_SEGS = 24;

export class WebGPURenderer {
  device: GPUDevice;
  context: GPUCanvasContext;
  pipeline: GPURenderPipeline;
  vertexBuffer: GPUBuffer;
  instanceBuffer: GPUBuffer;
  overlayCanvas: HTMLCanvasElement;
  overlayCtx: CanvasRenderingContext2D;
  maxInst = 512;
  circleVertCount: number;

  constructor(canvas: HTMLCanvasElement, device: GPUDevice) {
    this.device = device;
    this.context = canvas.getContext('webgpu')!;
    this.context.configure({ device, format: 'rgba8unorm', alphaMode: 'premultiplied' });

    // Circle mesh
    const triVerts: number[] = [];
    for (let i = 0; i < CIRCLE_SEGS; i++) {
      const a0 = (i / CIRCLE_SEGS) * Math.PI * 2;
      const a1 = ((i + 1) / CIRCLE_SEGS) * Math.PI * 2;
      triVerts.push(0, 0, Math.cos(a0), Math.sin(a0), Math.cos(a1), Math.sin(a1));
    }
    this.circleVertCount = CIRCLE_SEGS * 3;

    this.vertexBuffer = device.createBuffer({
      size: triVerts.length * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(triVerts);
    this.vertexBuffer.unmap();

    this.instanceBuffer = device.createBuffer({
      size: this.maxInst * 11 * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    const shaderModule = device.createShaderModule({ code: PARTICLE_WGSL });

    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [
          {
            arrayStride: 8,
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
          },
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

    // Overlay canvas
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
    const device = this.device;
    const ctx = this.overlayCtx;

    // Update instance buffer
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
    device.queue.writeBuffer(this.instanceBuffer, 0, inst);

    // Render particles
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
    pass.draw(this.circleVertCount, cnt);
    pass.end();
    device.queue.submit([encoder.finish()]);

    // Canvas2D overlay (text, effects, walls on top)
    ctx.clearRect(0, 0, GAME_W, GAME_H);

    // Walls + container outline (on top of particles)
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

    // Force chains (simplified, no shadowBlur)
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
      }
    }

    // Effects
    for (const e of effects) {
      ctx.beginPath();
      ctx.arc(e.x, e.y, e.r, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,255,255,${e.alpha})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Text for all particles
    if (!gameOver) {
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

    // Drop guide
    if (!gameOver && canDrop) {
      const r = LEVELS[currentLevel].radius;
      const cx = Math.max(CL + r + 2, Math.min(CR - r - 2, dropX));
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(cx, DROP_Y + r); ctx.lineTo(cx, CB);
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
