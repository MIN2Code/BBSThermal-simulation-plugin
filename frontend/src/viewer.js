// 3D 视图：G-code 刀路渲染、逐层显示、TQI 热图着色。
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export const SEG_STRIDE = 40;

export const FEATURE_NAMES = {
  0: '未知', 1: '裙边', 2: '引入线', 3: '清料', 4: '边缘(Brim)',
  5: '支撑', 6: '支撑界面', 7: '内墙', 8: '外墙', 9: '间隙填充',
  10: '实心填充', 11: '稀疏填充', 12: '桥接', 13: '内部桥接', 14: '熨烫', 15: '自定义',
};

const FEATURE_COLORS = {
  0: 0x888888, 1: 0x557799, 2: 0x446644, 3: 0x446644, 4: 0x557799,
  5: 0x3aa0a8, 6: 0x3aa0a8, 7: 0xd8b34a, 8: 0xe8913d, 9: 0x9a6ad8,
  10: 0xc46a9e, 11: 0x7a8a5a, 12: 0xff8ad8, 13: 0xd86ad8, 14: 0x88d8ff, 15: 0xcccccc,
};

// TQI(-100..100) → RGB，蓝(冷)→绿(理想)→红(热)，与后端 tqi_color 一致
export function tqiColor(t) {
  const x = Math.max(-100, Math.min(100, t)) / 100;
  const r = Math.max(0, Math.min(1, 1.5 * x + 0.5));
  const g = Math.max(0, Math.min(1, 1.0 - 2.0 * Math.abs(x)));
  const b = Math.max(0, Math.min(1, 0.5 - 1.5 * x));
  return [r, g, b];
}

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    // 性能取向：关抗锯齿（密排线条视觉差异极小）、高性能偏好、限制像素比
    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: false, powerPreference: 'high-performance',
    });
    this._maxPixelRatio = 1.5;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this._maxPixelRatio));
    this.renderer.setClearColor(0x14171c);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
    this.camera.position.set(160, -160, 140);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;

    // 帧率自适应降档：持续低于 30fps 时把渲染分辨率降到 1x
    this._frameTimes = [];
    this._degraded = false;

    // 热床网格
    const grid = new THREE.GridHelper(400, 40, 0x2e3745, 0x222935);
    grid.position.z = 0;
    this.scene.add(grid);

    this.lines = null;          // THREE.LineSegments
    this.colorAttr = null;
    this.positions = null;      // Float32Array 原始数据缓存
    this.layerRanges = [];
    this.offset = new THREE.Vector3();

    window.addEventListener('resize', () => this._resize());
    this._resize();
    this._lastT = performance.now();
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  _resize() {
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this._maxPixelRatio));
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _loop() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    // 帧率监控（每 60 帧评估一次；卡顿则降为 1x 渲染分辨率）
    const now = performance.now();
    const dt = now - this._lastT;
    this._lastT = now;
    if (!this._degraded) {
      this._frameTimes.push(dt);
      if (this._frameTimes.length >= 60) {
        const avg = this._frameTimes.reduce((a, b) => a + b, 0) / this._frameTimes.length;
        if (avg > 33 && this.renderer.getPixelRatio() > 1.01) {
          this._maxPixelRatio = 1.0;
          this._degraded = true;
          this._resize();
        }
        this._frameTimes = [];
      }
    }
    requestAnimationFrame(this._loop);
  }

  /** 加载结果二进制（行结构与后端 pack_result 对齐）。 */
  loadData(buffer, meta) {
    this.disposeLines();
    const n = meta.num_segments;
    const u8 = new Uint8Array(buffer);
    const f32 = new Float32Array(buffer);
    const i32 = new Int32Array(buffer);
    this.count = n;
    this._u8 = u8; this._f32 = f32; this._i32 = i32;

    // 顶点位置：每段 2 端点，整体平移使模型中心在原点、底部贴床
    const stride4 = SEG_STRIDE / 4;
    this.positions = new Float32Array(n * 6);
    const off = meta.summary.bbox_min, top = meta.summary.bbox_max;
    const ox = (off[0] + top[0]) / 2, oy = (off[1] + top[1]) / 2;
    this.offset.set(-ox, -oy, -off[2]);
    for (let i = 0; i < n; i++) {
      const b = i * stride4;
      this.positions[i * 6 + 0] = f32[b + 0] - ox;
      this.positions[i * 6 + 1] = f32[b + 1] - oy;
      this.positions[i * 6 + 2] = f32[b + 2] - off[2];
      this.positions[i * 6 + 3] = f32[b + 3] - ox;
      this.positions[i * 6 + 4] = f32[b + 4] - oy;
      this.positions[i * 6 + 5] = f32[b + 5] - off[2];
    }
    const posAttr = new THREE.BufferAttribute(this.positions, 3);
    const colors = new Float32Array(n * 6).fill(0.6);
    this.colorAttr = new THREE.BufferAttribute(colors, 3);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', posAttr);
    geo.setAttribute('color', this.colorAttr);
    this.lines = new THREE.LineSegments(
      geo, new THREE.LineBasicMaterial({ vertexColors: true })
    );
    this.lines.frustumCulled = false;
    // G-code 为 Z-up，three.js 默认 Y-up：绕 X 轴 -90° 使打印件直立于热床上
    this.lines.rotation.x = -Math.PI / 2;
    this.scene.add(this.lines);

    this.layerRanges = meta.layer_ranges;
    this._frameCamera(top, off);
  }

  _frameCamera(top, off) {
    const w = Math.max(top[0] - off[0], top[1] - off[1], 40);
    const h = Math.max(top[2] - off[2], 10);
    const d = Math.max(w, h) * 1.5;
    this.controls.target.set(0, h / 2, 0);
    this.camera.position.set(d * 0.72, d * 0.62, d * 0.72);
    this.camera.updateProjectionMatrix();
  }

  disposeLines() {
    if (this.lines) {
      this.scene.remove(this.lines);
      this.lines.geometry.dispose();
      this.lines.material.dispose();
      this.lines = null;
    }
  }

  /** 重新着色（昂贵：全量重写颜色缓冲）。仅在加载完成或切换色彩模式时调用。 */
  applyColors(mode) {
    if (!this.lines) return;
    this._mode = mode;
    const { count, _f32, _u8 } = this;
    const colors = this.colorAttr.array;
    for (let i = 0; i < count; i++) {
      const b = i * 10;
      let r, g, bl;
      if (mode === 'tqi') {
        const valid = _u8[i * SEG_STRIDE + 37] === 1;
        if (valid) {
          [r, g, bl] = tqiColor(_f32[b + 6]);
        } else {
          const fc = FEATURE_COLORS[_u8[i * SEG_STRIDE + 36]] || 0x666666;
          r = ((fc >> 16) & 255) / 255 * 0.45;
          g = ((fc >> 8) & 255) / 255 * 0.45;
          bl = (fc & 255) / 255 * 0.45;
        }
      } else {
        const fc = FEATURE_COLORS[_u8[i * SEG_STRIDE + 36]] || 0x888888;
        r = ((fc >> 16) & 255) / 255; g = ((fc >> 8) & 255) / 255; bl = (fc & 255) / 255;
      }
      const v = i * 6;
      colors[v] = r; colors[v + 1] = g; colors[v + 2] = bl;
      colors[v + 3] = r; colors[v + 4] = g; colors[v + 5] = bl;
    }
    this.colorAttr.needsUpdate = true;
  }

  /** 改可见层（便宜：只改 drawRange，不碰颜色）。滑块拖动专用。
   * 注意：非索引几何的 drawRange 计数单位是顶点（每段 2 顶点），需 ×2。 */
  setLayerRange(upToLayer, onlyLayer) {
    if (!this.lines) return;
    let start = 0, countDraw = 0;
    if (onlyLayer != null) {
      const rg = this.layerRanges.find((x) => x.layer === onlyLayer);
      if (rg) { start = rg.start; countDraw = rg.count; }
    } else {
      for (const rg of this.layerRanges) {
        if (rg.layer <= upToLayer) countDraw = Math.max(countDraw, rg.start + rg.count);
      }
    }
    this.lines.geometry.setDrawRange(start * 2, countDraw * 2);
  }
}
