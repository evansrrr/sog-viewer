/**
 * Depth sort worker.
 *
 * Gaussian splatting needs back-to-front alpha compositing, so the draw order
 * has to be rebuilt whenever the camera moves. Doing that on the main thread
 * costs tens of milliseconds for a few hundred thousand splats, which is why it
 * lives here.
 *
 * Sorting is a counting sort over quantised view-space depth: O(n + bins), no
 * comparisons, and it also culls splats that fall outside the frustum so the GPU
 * never touches them.
 *
 * The view transform is the expensive part, so it runs exactly once per splat in
 * the first pass. Splats that fail culling are marked with a sentinel depth and
 * skipped by the remaining two passes, which only read the array back.
 */

const BINS = 1 << 16;

/** Conservative screen margin, wide enough to keep splats whose quad straddles the edge. */
const MARGIN = 1.25;

/** Larger than any real view depth; marks a culled splat. */
const CULLED = 1e30;

let position = null;
let count = 0;
let capacity = 0;
let order = null;
let depth = null;
const hist = new Uint32Array(BINS);

self.onmessage = (event) => {
  const message = event.data;

  if (message.type === 'init') {
    position = new Float32Array(message.position);
    count = message.count;
    capacity = message.capacity;
    order = new Uint32Array(capacity);
    depth = new Float32Array(count);
    self.postMessage({ type: 'ready', count });
    return;
  }

  if (message.type === 'free') {
    if (!order) order = new Uint32Array(message.order);
    return;
  }

  if (message.type !== 'sort') return;
  if (!order) {
    // The previous order buffer is still in flight; skip this request.
    self.postMessage({ type: 'skipped', seq: message.seq });
    return;
  }

  const started = performance.now();
  const view = message.view;
  const proj = message.proj;

  const m0 = view[0], m1 = view[1], m2 = view[2];
  const m4 = view[4], m5 = view[5], m6 = view[6];
  const m8 = view[8], m9 = view[9], m10 = view[10];
  const m12 = view[12], m13 = view[13], m14 = view[14];
  const px = proj[0];
  const py = proj[5];

  // Pass 1: transform once, reject what cannot reach the screen, record depth range.
  let lo = Infinity;
  let hi = -Infinity;

  for (let i = 0; i < count; i++) {
    const t = i * 4;
    const x = position[t];
    const y = position[t + 1];
    const z = position[t + 2];

    const wz = m2 * x + m6 * y + m10 * z + m14;
    const w = -wz;
    if (w <= 0.02) {
      depth[i] = CULLED;
      continue;
    }

    const cx = m0 * x + m4 * y + m8 * z + m12;
    const cy = m1 * x + m5 * y + m9 * z + m13;
    const nx = (px * cx) / w;
    const ny = (py * cy) / w;
    if (nx < -MARGIN || nx > MARGIN || ny < -MARGIN || ny > MARGIN) {
      depth[i] = CULLED;
      continue;
    }

    depth[i] = wz;
    if (wz < lo) lo = wz;
    if (wz > hi) hi = wz;
  }

  const span = hi - lo;
  const scale = span > 1e-6 ? (BINS - 1) / span : 0;

  // Pass 2: histogram of visible splats only.
  hist.fill(0);
  let visible = 0;
  for (let i = 0; i < count; i++) {
    const d = depth[i];
    if (d === CULLED) continue;
    hist[((d - lo) * scale) | 0]++;
    visible++;
  }

  if (visible === 0) {
    self.postMessage(
      { type: 'sorted', order: order.buffer, visible: 0, ms: performance.now() - started, seq: message.seq },
      [order.buffer]
    );
    order = null;
    return;
  }

  // Prefix sum turns the histogram into write cursors.
  let acc = 0;
  for (let b = 0; b < BINS; b++) {
    const c = hist[b];
    hist[b] = acc;
    acc += c;
  }

  // Pass 3: scatter. Back-to-front, so the far end is written first.
  for (let i = 0; i < count; i++) {
    const d = depth[i];
    if (d === CULLED) continue;
    order[hist[((d - lo) * scale) | 0]++] = i;
  }

  self.postMessage(
    { type: 'sorted', order: order.buffer, visible, ms: performance.now() - started, seq: message.seq },
    [order.buffer]
  );
  order = null;
};
