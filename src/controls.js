/**
 * Orbit / pan / zoom camera with damping.
 *
 * Built on pointer events so mouse, pen and touch share one code path: one
 * pointer orbits, two pointers pinch to zoom, right button or two-finger drag
 * pans. Values are eased toward targets each frame, which keeps the motion
 * smooth without needing an animation library.
 */

const EPS = 1e-4;

function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[14] = 2 * far * near * nf;
  return out;
}

function lookAt(eye, target, up) {
  let zx = eye[0] - target[0];
  let zy = eye[1] - target[1];
  let zz = eye[2] - target[2];
  let len = Math.hypot(zx, zy, zz) || 1;
  zx /= len; zy /= len; zz /= len;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  len = Math.hypot(xx, xy, xz) || 1;
  xx /= len; xy /= len; xz /= len;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  const out = new Float32Array(16);
  out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
  out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
  out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1;
  return out;
}

export class OrbitControls {
  constructor(canvas, { fov = 0.8 } = {}) {
    this.canvas = canvas;
    this.fov = fov;

    this.theta = 0;
    this.phi = 0.22;
    this.distance = 3;
    this.target = [0, 0, 0];

    this.thetaGoal = this.theta;
    this.phiGoal = this.phi;
    this.distanceGoal = this.distance;
    this.targetGoal = [...this.target];

    this.dirty = true;
    this.minDistance = 0.02;
    this.maxDistance = 5000;

    this.eye = [0, 0, 3];
    this.view = lookAt(this.eye, this.target, [0, 1, 0]);
    this.projection = perspective(this.fov, 1, 0.01, 20000);

    this.#bind();
  }

  #bind() {
    const canvas = this.canvas;
    const pointers = new Map();
    let pinchDistance = 0;

    const onDown = (event) => {
      canvas.setPointerCapture?.(event.pointerId);
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY, button: event.button });
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinchDistance = Math.hypot(a.x - b.x, a.y - b.y);
      }
    };

    const onMove = (event) => {
      const previous = pointers.get(event.pointerId);
      if (!previous) return;
      const dx = event.clientX - previous.x;
      const dy = event.clientY - previous.y;
      previous.x = event.clientX;
      previous.y = event.clientY;

      if (pointers.size >= 2) {
        const [a, b] = [...pointers.values()];
        const distance = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinchDistance > 0 && distance > 0) {
          this.distanceGoal = Math.min(this.maxDistance, Math.max(this.minDistance, this.distanceGoal * (pinchDistance / distance)));
          this.dirty = true;
        }
        pinchDistance = distance;
        this.#pan(dx * 0.5, dy * 0.5);
        return;
      }

      if (previous.button === 2 || event.shiftKey) {
        this.#pan(dx, dy);
      } else {
        this.thetaGoal -= dx * 0.005;
        this.phiGoal = Math.max(-1.5, Math.min(1.5, this.phiGoal + dy * 0.005));
        this.dirty = true;
      }
    };

    const onUp = (event) => {
      pointers.delete(event.pointerId);
      if (pointers.size < 2) pinchDistance = 0;
    };

    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointercancel', onUp);
    canvas.addEventListener('contextmenu', (event) => event.preventDefault());

    canvas.addEventListener(
      'wheel',
      (event) => {
        event.preventDefault();
        const factor = Math.exp(event.deltaY * 0.0012);
        this.distanceGoal = Math.min(this.maxDistance, Math.max(this.minDistance, this.distanceGoal * factor));
        this.dirty = true;
      },
      { passive: false }
    );
  }

  #pan(dx, dy) {
    const scale = this.distance * 0.0022;
    const cos = Math.cos(this.theta);
    const sin = Math.sin(this.theta);
    const sinPhi = Math.sin(this.phi);
    const cosPhi = Math.cos(this.phi);
    this.targetGoal[0] -= (dx * cos + dy * sin * sinPhi) * scale;
    this.targetGoal[1] += dy * cosPhi * scale;
    this.targetGoal[2] -= (dx * sin - dy * cos * sinPhi) * scale;
    this.dirty = true;
  }

  /**
   * Place the camera so that an axis-aligned box fills the viewport.
   * Fitting against a percentile box rather than a raw min/max avoids the
   * stray floaters that trained scenes almost always contain.
   */
  frame(min, max, aspect) {
    const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    const halfTan = Math.tan(this.fov / 2);
    const vertical = size[1] * 0.5 / halfTan;
    const horizontal = size[0] * 0.5 / (halfTan * Math.max(aspect, EPS));
    this.target = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    this.targetGoal = [...this.target];
    this.distance = (Math.max(vertical, horizontal) + size[2] * 0.5) * 1.08;
    this.distanceGoal = this.distance;
    this.dirty = true;
  }

  update(dt) {
    const ease = 1 - Math.pow(0.001, Math.min(dt, 0.1) / 0.25);
    const before = `${this.theta},${this.phi},${this.distance},${this.target}`;

    this.theta += (this.thetaGoal - this.theta) * ease;
    this.phi += (this.phiGoal - this.phi) * ease;
    this.distance += (this.distanceGoal - this.distance) * ease;
    for (let i = 0; i < 3; i++) {
      this.target[i] += (this.targetGoal[i] - this.target[i]) * ease;
    }

    const cosPhi = Math.cos(this.phi);
    this.eye = [
      this.target[0] + this.distance * Math.sin(this.theta) * cosPhi,
      this.target[1] + this.distance * Math.sin(this.phi),
      this.target[2] + this.distance * Math.cos(this.theta) * cosPhi
    ];
    this.view = lookAt(this.eye, this.target, [0, 1, 0]);

    const changed = before !== `${this.theta},${this.phi},${this.distance},${this.target}`;
    if (changed) this.dirty = true;
    return changed;
  }

  setViewport(width, height) {
    this.projection = perspective(this.fov, width / Math.max(height, 1), 0.01, 20000);
  }
}
