/**
 * Depth sort worker.
 *
 * Gaussian splatting needs back-to-front alpha compositing, so the draw order
 * has to be rebuilt whenever the camera moves. Doing that on the main thread
 * costs tens of milliseconds for a few hundred thousand splats, which is why it
 * lives here.
 *
 * Sorting is a counting sort over quantised view-space depth: O(n + bins), no
 * comparisons, and it also culls splats that fall outside the frustum so the
 * GPU never touches them.
 */

const BINS = 1 << 16;

/** Conservative screen margin, wide enough to keep splats whose quad straddles the edge. */
const MARGIN = 1.25;

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

  // View-space depth drives the ordering; the two other axes drive culling.
  const m2 = view[2];
  const m6 = view[6];
  const m10 = view[10];
  const m14 = view[14];
  const px = proj[0];
  const py = proj[5];

  let lo = Infinity;
  let hi = -Infinity;

  for (let i = 0; i < count; i++) {
    const t = i * 4;
    const x = position[t];
    const y = position[t + 1];
    const z = position[t + 2];
    const vz = m2 * x + m6 * y + m10 * z + m14;
    depth[i] = vz;
    if (vz < lo) lo = vz;
    if (vz > hi) hi = vz;
  }

  const span = hi - lo;
  const scale = span > 1e-6 ? (BINS - 1) / span : 0;

  hist.fill(0);

  // Count only splats that survive culling.
  let visible = 0;
  for (let i = 0; i < count; i++) {
    const wz = depth[i];
    const t = i * 4;
    const x = position[t];
    const y = position[t + 1];
    const z = position[t + 2];
    const cx = view[0] * x + view[4] * y + view[8] * z + view[12];
    const cy = view[1] * x + view[5] * y + view[9] * z + view[13];

    const w = -wz;
    if (w > 0.02) {
      const nx = (px * cx) / w;
      const ny = (py * cy) / w;
      if (nx < -MARGIN || nx > MARGIN || ny < -MARGIN || ny > MARGIN) continue;
    } else {
      // Behind the camera, or close enough that projection is meaningless.
      continue;
    }

    hist[(wz - lo) * scale | 0]++;
    visible++;
  }

  if (visible === 0) {
    self.postMessage({ type: 'sorted', order: order.buffer, visible: 0, ms: performance.now() - started, seq: message.seq }, [order.buffer]);
    order = null;
    return;
  }

  let acc = 0;
  for (let b = 0; b < BINS; b++) {
    const c = hist[b];
    hist[b] = acc;
    acc += c;
  }

  for (let i = 0; i < count; i++) {
    const wz = depth[i];
    const t = i * 4;
    const x = position[t];
    const y = position[t + 1];
    const z = position[t + 2];
    const cx = view[0] * x + view[4] * y + view[8] * z + view[12];
    const cy = view[1] * x + view[5] * y + view[9] * z + view[13];

    const w = -wz;
    if (w <= 0.02) continue;
    const nx = (px * cx) / w;
    const ny = (py * cy) / w;
    if (nx < -MARGIN || nx > MARGIN || ny < -MARGIN || ny > MARGIN) continue;

    order[hist[(wz - lo) * scale | 0]++] = i;
  }

  self.postMessage(
    { type: 'sorted', order: order.buffer, visible, ms: performance.now() - started, seq: message.seq },
    [order.buffer]
  );
  order = null;
};
