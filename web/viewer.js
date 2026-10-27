// A WebGL2 view of the registered point clouds and the trajectory.
//
// Every scan arrives at full resolution in world coordinates with 4 values per
// point (x, y, z, intensity). The newest scan is drawn complete. Every
// HISTORY_STRIDE-th point of each scan goes into a GPU buffer that works as a
// ring, so the view also shows a rolling map of the recent scans. Thinning
// along the scan order keeps the sensor's ring pattern. The camera follows the
// platform until the user moves it. Points are coloured by intensity when the
// scans carry varying intensities and `colorBy` is "intensity", by height
// otherwise. One finger or the mouse rotates, two
// fingers pinch to zoom and drag to pan, as does the right mouse button or
// Shift. Double-click, or set `follow`, to follow again.

const POINT_FLOATS = 4;
const POINT_BYTES = POINT_FLOATS * 4;
/** Points the rolling map holds before the oldest scans are overwritten. */
const HISTORY_CAPACITY = 3_000_000;
/** The rolling map keeps every n-th point of a scan. */
const HISTORY_STRIDE = 4;
/** Most points of the newest scan. */
const LIVE_CAPACITY = 400_000;
const MAX_TRAJECTORY = 20_000;
/** Height range the colour map spans, around the current position. */
export const HEIGHT_BELOW_M = 3;
export const HEIGHT_ABOVE_M = 8;
/** Share of each scan's intensities below and above the colour range. */
const INTENSITY_CLIP = 0.02;
/** How quickly the intensity range follows new scans. */
const INTENSITY_SMOOTHING = 0.2;
/** Points per scan the intensity range is estimated from. */
const INTENSITY_SAMPLES = 2000;

const POINT_VERTEX = `#version 300 es
in vec4 a_point;
uniform mat4 u_view_proj;
uniform float u_z_min;
uniform float u_z_max;
uniform bool u_by_intensity;
uniform vec2 u_intensity_range;
uniform bool u_live;
uniform float u_point_size;
out vec3 v_color;

// Turbo (Google, 2019) in its published polynomial approximation.
vec3 turbo(float x) {
  x = clamp(x, 0.0, 1.0);
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  const vec4 kRed4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec2 kRed2 = vec2(-152.94239396, 59.28637943);
  const vec4 kGreen4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec2 kGreen2 = vec2(4.27729857, 2.82956604);
  const vec4 kBlue4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 kBlue2 = vec2(-89.90310912, 27.34824973);
  return clamp(vec3(dot(v4, kRed4) + dot(v2, kRed2),
                    dot(v4, kGreen4) + dot(v2, kGreen2),
                    dot(v4, kBlue4) + dot(v2, kBlue2)), 0.0, 1.0);
}

void main() {
  gl_Position = u_view_proj * vec4(a_point.xyz, 1.0);
  vec3 color = turbo(u_by_intensity
    ? (a_point.w - u_intensity_range.x) / (u_intensity_range.y - u_intensity_range.x)
    : (a_point.z - u_z_min) / (u_z_max - u_z_min));
  // The newest scan is lighter, which sets it apart from the map.
  v_color = u_live ? mix(color, vec3(1.0), 0.35) : color * 0.85;
  gl_PointSize = u_live ? u_point_size * 1.25 : u_point_size;
}`;

const POINT_FRAGMENT = `#version 300 es
precision mediump float;
in vec3 v_color;
out vec4 color;
void main() {
  color = vec4(v_color, 1.0);
}`;

const LINE_VERTEX = `#version 300 es
in vec3 a_position;
uniform mat4 u_view_proj;
void main() {
  gl_Position = u_view_proj * vec4(a_position, 1.0);
}`;

const LINE_FRAGMENT = `#version 300 es
precision mediump float;
uniform vec4 u_color;
out vec4 color;
void main() {
  color = u_color;
}`;

function compile(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  for (const [type, source] of [
    [gl.VERTEX_SHADER, vertexSource],
    [gl.FRAGMENT_SHADER, fragmentSource],
  ]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(`shader: ${gl.getShaderInfoLog(shader)}`);
    }
    gl.attachShader(program, shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`program: ${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
}

function lookAt(eye, target, up) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const normalize = (v) => {
    const n = Math.hypot(...v) || 1;
    return [v[0] / n, v[1] / n, v[2] / n];
  };
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const z = normalize(sub(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
}

function multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let column = 0; column < 4; column++) {
    for (let row = 0; row < 4; row++) {
      for (let k = 0; k < 4; k++) out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
    }
  }
  return out;
}

/** Turbo at `x` from 0 to 1 as [r, g, b] from 0 to 255, the same as the shader. */
function turboRgb(x) {
  const t = Math.min(1, Math.max(0, x));
  const v4 = [1, t, t * t, t * t * t];
  const v2 = [v4[2] * v4[2], v4[3] * v4[2]];
  const channel = (k4, k2) => {
    const value = k4.reduce((sum, k, i) => sum + k * v4[i], 0) + k2[0] * v2[0] + k2[1] * v2[1];
    return Math.round(Math.min(1, Math.max(0, value)) * 255);
  };
  return [
    channel([0.13572138, 4.6153926, -42.66032258, 132.13108234], [-152.94239396, 59.28637943]),
    channel([0.09140261, 2.19418839, 4.84296658, -14.18503333], [4.27729857, 2.82956604]),
    channel([0.1066733, 12.64194608, -60.58204836, 110.36276771], [-89.90310912, 27.34824973]),
  ];
}

/** The colour map as a CSS gradient from left to right, for the legend. */
export function turboGradient(stops = 16) {
  const colors = Array.from({ length: stops + 1 }, (_, i) => `rgb(${turboRgb(i / stops).join(" ")})`);
  return `linear-gradient(to right, ${colors.join(", ")})`;
}

export class PointCloudView {
  /** Draw into `canvas`. Throws when the browser has no WebGL2. */
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { antialias: false });
    if (!gl) throw new Error("WebGL2 is not available");
    this.gl = gl;

    this.pointProgram = compile(gl, POINT_VERTEX, POINT_FRAGMENT);
    this.lineProgram = compile(gl, LINE_VERTEX, LINE_FRAGMENT);

    const pointAttribute = gl.getAttribLocation(this.pointProgram, "a_point");
    const pointBuffer = (capacity) => {
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, capacity * POINT_BYTES, gl.DYNAMIC_DRAW);
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      gl.enableVertexAttribArray(pointAttribute);
      gl.vertexAttribPointer(pointAttribute, 4, gl.FLOAT, false, POINT_BYTES, 0);
      return { buffer, vao };
    };
    this.history = pointBuffer(HISTORY_CAPACITY);
    this.live = pointBuffer(LIVE_CAPACITY);
    this.liveCount = 0;
    // Reused for thinning a scan into the history.
    this.thinned = new Float32Array(Math.ceil(LIVE_CAPACITY / HISTORY_STRIDE) * POINT_FLOATS);

    this.trajectory = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.trajectory);
    gl.bufferData(gl.ARRAY_BUFFER, MAX_TRAJECTORY * 12, gl.DYNAMIC_DRAW);
    this.lineVao = gl.createVertexArray();
    gl.bindVertexArray(this.lineVao);
    const lineAttribute = gl.getAttribLocation(this.lineProgram, "a_position");
    gl.enableVertexAttribArray(lineAttribute);
    gl.vertexAttribPointer(lineAttribute, 3, gl.FLOAT, false, 12, 0);
    gl.bindVertexArray(null);

    // Colouring: by intensity once the scans carry varying intensities, which
    // `intensityRange` tracks, by height otherwise or when chosen.
    this.colorBy = "intensity";
    this.intensityRange = null;
    // History ring: the next write position and how far the buffer is filled.
    this.head = 0;
    this.filled = 0;
    this.trajectoryCount = 0;
    this.position = [0, 0, 0];

    // Orbit camera around `target`, Z up.
    this.target = [0, 0, 0];
    this.yaw = -Math.PI / 2;
    this.pitch = 0.9;
    this.distance = 45;
    this.follow = true;

    this.attachControls();
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    const frame = () => {
      this.draw();
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  /** Add one scan. `points` holds x, y, z and intensity per point in the world frame. */
  addScan(points, position) {
    const gl = this.gl;
    const count = Math.min(points.length / POINT_FLOATS, LIVE_CAPACITY);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.live.buffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, points, 0, count * POINT_FLOATS);
    this.liveCount = count;

    let kept = 0;
    for (let i = 0; i < count; i += HISTORY_STRIDE) {
      const from = i * POINT_FLOATS;
      this.thinned.set(points.subarray(from, from + POINT_FLOATS), kept * POINT_FLOATS);
      kept++;
    }
    if (this.head + kept > HISTORY_CAPACITY) this.head = 0;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.history.buffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, this.head * POINT_BYTES, this.thinned, 0, kept * POINT_FLOATS);
    this.head += kept;
    this.filled = Math.max(this.filled, this.head);
    this.trackIntensity(points, count);

    this.position = position;
    if (this.trajectoryCount < MAX_TRAJECTORY) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.trajectory);
      gl.bufferSubData(gl.ARRAY_BUFFER, this.trajectoryCount * 12, new Float32Array(position));
      this.trajectoryCount++;
    }
  }

  /** Whether the scans carry intensities that vary, which colouring by intensity needs. */
  get hasIntensity() {
    return this.intensityRange !== null && this.intensityRange[1] > this.intensityRange[0];
  }

  /** Follow the intensity range of the scans, clipped at the extremes. */
  trackIntensity(points, count) {
    const step = Math.max(1, Math.floor(count / INTENSITY_SAMPLES));
    const sample = [];
    for (let i = 0; i < count; i += step) sample.push(points[i * POINT_FLOATS + 3]);
    if (sample.length === 0) return;
    sample.sort((a, b) => a - b);
    const low = sample[Math.floor(INTENSITY_CLIP * (sample.length - 1))];
    const high = sample[Math.floor((1 - INTENSITY_CLIP) * (sample.length - 1))];
    if (!this.intensityRange) {
      this.intensityRange = [low, high];
      return;
    }
    this.intensityRange[0] += (low - this.intensityRange[0]) * INTENSITY_SMOOTHING;
    this.intensityRange[1] += (high - this.intensityRange[1]) * INTENSITY_SMOOTHING;
  }

  /** Forget every scan and the trajectory, for a replay from the start. */
  clear() {
    this.head = 0;
    this.filled = 0;
    this.liveCount = 0;
    this.trajectoryCount = 0;
    this.intensityRange = null;
    this.position = [0, 0, 0];
    this.follow = true;
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(this.canvas.clientWidth * ratio));
    this.canvas.height = Math.max(1, Math.round(this.canvas.clientHeight * ratio));
  }

  attachControls() {
    const canvas = this.canvas;
    // Active pointers by id, for one-finger rotation and two-finger pinch.
    const pointers = new Map();
    let pinch = null;

    const pan = (dx, dy) => {
      this.follow = false;
      // The scene moves with the pointer.
      const scale = this.distance / canvas.clientHeight;
      const right = [Math.sin(this.yaw), -Math.cos(this.yaw)];
      const forward = [Math.cos(this.yaw), Math.sin(this.yaw)];
      this.target[0] += (dy * forward[0] - dx * right[0]) * scale;
      this.target[1] += (dy * forward[1] - dx * right[1]) * scale;
    };
    const zoom = (factor) => {
      this.distance = Math.min(500, Math.max(2, this.distance * factor));
    };
    const pinchState = () => {
      const [a, b] = [...pointers.values()];
      return {
        spread: Math.hypot(a.x - b.x, a.y - b.y),
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2,
      };
    };

    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      pointers.set(event.pointerId, {
        x: event.clientX,
        y: event.clientY,
        pan: event.button !== 0 || event.shiftKey,
      });
      pinch = pointers.size === 2 ? pinchState() : null;
    });
    canvas.addEventListener("pointermove", (event) => {
      const pointer = pointers.get(event.pointerId);
      if (!pointer) return;
      const dx = event.clientX - pointer.x;
      const dy = event.clientY - pointer.y;
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      if (pointers.size === 2 && pinch) {
        const next = pinchState();
        if (next.spread > 0 && pinch.spread > 0) zoom(pinch.spread / next.spread);
        pan(next.x - pinch.x, next.y - pinch.y);
        pinch = next;
      } else if (pointers.size === 1) {
        if (pointer.pan) {
          pan(dx, dy);
        } else {
          this.yaw -= dx * 0.005;
          this.pitch = Math.min(1.55, Math.max(0.05, this.pitch + dy * 0.005));
        }
      }
    });
    const release = (event) => {
      pointers.delete(event.pointerId);
      pinch = pointers.size === 2 ? pinchState() : null;
    };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        zoom(Math.exp(event.deltaY * 0.001));
      },
      { passive: false },
    );
    canvas.addEventListener("dblclick", () => (this.follow = true));
  }

  draw() {
    const gl = this.gl;
    if (this.follow) this.target = [...this.position];
    const eye = [
      this.target[0] - Math.cos(this.yaw) * Math.cos(this.pitch) * this.distance,
      this.target[1] - Math.sin(this.yaw) * Math.cos(this.pitch) * this.distance,
      this.target[2] + Math.sin(this.pitch) * this.distance,
    ];
    const aspect = this.canvas.width / this.canvas.height;
    const viewProj = multiply(
      perspective(Math.PI / 3, aspect, 0.1, 2000),
      lookAt(eye, this.target, [0, 0, 1]),
    );

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(11 / 255, 12 / 255, 15 / 255, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);

    gl.useProgram(this.pointProgram);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.pointProgram, "u_view_proj"), false, viewProj);
    gl.uniform1f(gl.getUniformLocation(this.pointProgram, "u_z_min"), this.position[2] - HEIGHT_BELOW_M);
    gl.uniform1f(gl.getUniformLocation(this.pointProgram, "u_z_max"), this.position[2] + HEIGHT_ABOVE_M);
    gl.uniform1f(
      gl.getUniformLocation(this.pointProgram, "u_point_size"),
      Math.max(1, 1.5 * (window.devicePixelRatio || 1)),
    );
    const byIntensity = this.colorBy === "intensity" && this.hasIntensity;
    gl.uniform1i(gl.getUniformLocation(this.pointProgram, "u_by_intensity"), byIntensity ? 1 : 0);
    gl.uniform2f(
      gl.getUniformLocation(this.pointProgram, "u_intensity_range"),
      ...(this.intensityRange ?? [0, 1]),
    );
    const live = gl.getUniformLocation(this.pointProgram, "u_live");
    gl.uniform1i(live, 0);
    gl.bindVertexArray(this.history.vao);
    gl.drawArrays(gl.POINTS, 0, this.filled);
    gl.uniform1i(live, 1);
    gl.bindVertexArray(this.live.vao);
    gl.drawArrays(gl.POINTS, 0, this.liveCount);

    if (this.trajectoryCount > 1) {
      // On top of the points, which would hide it otherwise.
      gl.disable(gl.DEPTH_TEST);
      gl.useProgram(this.lineProgram);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.lineProgram, "u_view_proj"), false, viewProj);
      // The gold of the logo, #e5c65c.
      gl.uniform4f(gl.getUniformLocation(this.lineProgram, "u_color"), 0.9, 0.78, 0.36, 1);
      gl.bindVertexArray(this.lineVao);
      gl.drawArrays(gl.LINE_STRIP, 0, this.trajectoryCount);
    }
    gl.bindVertexArray(null);
  }
}
