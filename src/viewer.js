/**
 * SOG Viewer — a small, dependency-free WebGL2 viewer for SOG Gaussian splats.
 *
 * Orchestration only: decoding lives in sog.js, drawing in renderer.js, camera
 * in controls.js, and depth sorting in sort.worker.js. Nothing here blocks the
 * main thread for longer than a frame.
 */

import { decodeSog } from './sog.js';
import { SplatRenderer } from './renderer.js';
import { OrbitControls } from './controls.js';

function createEmitter() {
  const handlers = new Map();
  return {
    on(event, handler) {
      if (!handlers.has(event)) handlers.set(event, new Set());
      handlers.get(event).add(handler);
      return () => handlers.get(event)?.delete(handler);
    },
    emit(event, payload) {
      for (const handler of handlers.get(event) ?? []) {
        try {
          handler(payload);
        } catch (error) {
          console.error(`[sog-viewer] handler for "${event}" failed`, error);
        }
      }
    }
  };
}

/**
 * Percentile bounds. Trained scenes carry stray floaters far outside the
 * subject, so a raw min/max makes the camera frame empty space.
 */
function percentileBounds(position, count) {
  const stride = Math.max(1, Math.floor(count / 24000));
  const xs = [];
  const ys = [];
  const zs = [];
  for (let i = 0; i < count; i += stride) {
    const t = i * 4;
    xs.push(position[t]);
    ys.push(position[t + 1]);
    zs.push(position[t + 2]);
  }
  const quantile = (values, q) => {
    values.sort((a, b) => a - b);
    return values[Math.min(values.length - 1, Math.max(0, Math.round(q * (values.length - 1))))];
  };
  const lo = [quantile(xs, 0.01), quantile(ys, 0.01), quantile(zs, 0.01)];
  const hi = [quantile(xs, 0.99), quantile(ys, 0.99), quantile(zs, 0.99)];
  return { lo, hi };
}

/**
 * @param {HTMLCanvasElement} canvas
 * @param {{ pixelRatio?: number, exposure?: number, background?: number[], maxPixelRatio?: number }} [options]
 */
export function createViewer(canvas, options = {}) {
  const emitter = createEmitter();
  const renderer = new SplatRenderer(canvas, { maxPixelRatio: options.maxPixelRatio ?? 2 });
  const controls = new OrbitControls(canvas);

  if (options.background) renderer.background = options.background;
  if (options.exposure) renderer.exposure = options.exposure;
  if (options.pixelRatio) {
    renderer.maxPixelRatio = options.pixelRatio;
    renderer.resize();
  }

  let worker = null;
  let orderInFlight = false;
  let pendingSort = false;
  let sequence = 0;
  let sortMs = 0;
  let visible = 0;
  let frameMs = 0;
  let lastFrameTime = performance.now();
  let statsTime = performance.now();
  let ready = false;

  function handleWorkerMessage(event) {
    const message = event.data;

    if (message.type === 'ready') {
      controls.dirty = true;
      requestSort();
      return;
    }

    if (message.type === 'sorted') {
      const order = new Uint32Array(message.order);
      renderer.updateOrder(order, message.visible);
      sortMs = message.ms;
      visible = message.visible;
      worker.postMessage({ type: 'free', order: message.order }, [message.order]);
      orderInFlight = false;
      if (pendingSort) {
        pendingSort = false;
        requestSort();
      }
    }
  }

  function requestSort() {
    if (!worker || !ready) return;
    if (orderInFlight) {
      pendingSort = true;
      return;
    }
    orderInFlight = true;
    worker.postMessage({
      type: 'sort',
      view: controls.view,
      proj: controls.projection,
      seq: ++sequence
    });
  }

  function spawnWorker(position, count, capacity) {
    worker?.terminate();
    worker = new Worker(new URL('./sort.worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = handleWorkerMessage;
    worker.onerror = (event) => emitter.emit('error', `Sort worker failed: ${event.message || 'unknown error'}`);
    worker.postMessage({ type: 'init', position: position.buffer, count, capacity }, [position.buffer]);
    orderInFlight = false;
    pendingSort = false;
  }

  async function ingest(buffer, meta, name, bytes) {
    const started = performance.now();
    ready = false;

    emitter.emit('progress', { phase: 'decode', ratio: 0.05 });
    const data = await decodeSog(buffer, ({ phase, ratio }) => {
      emitter.emit('progress', { phase, ratio });
    });

    emitter.emit('progress', { phase: 'upload', ratio: 0.94 });

    const { lo, hi } = percentileBounds(data.position, data.count);

    const capacity = data.width * data.height;
    const order = new Uint32Array(capacity);
    for (let i = 0; i < capacity; i++) order[i] = i;

    renderer.setSplatData(data, order);
    renderer.resize();
    controls.setViewport(canvas.width, canvas.height);
    controls.frame(lo, hi, canvas.width / Math.max(canvas.height, 1));

    spawnWorker(data.position, data.count, capacity);
    visible = data.count;
    ready = true;
    controls.dirty = true;

    const loadMs = performance.now() - started;
    emitter.emit('progress', { phase: 'ready', ratio: 1 });
    emitter.emit('load', {
      name: name ?? 'scene.sog',
      bytes: bytes ?? buffer.byteLength,
      splats: data.count,
      loadMs,
      meta
    });
  }

  async function loadUrl(url) {
    emitter.emit('progress', { phase: 'fetch', ratio: 0.02 });
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch ${url} (HTTP ${response.status})`);
    const buffer = await response.arrayBuffer();
    const name = url.split('/').pop() || 'scene.sog';
    await ingest(buffer, null, name, buffer.byteLength);
  }

  async function loadFile(file) {
    if (!file) throw new Error('No file provided');
    const buffer = await file.arrayBuffer();
    await ingest(buffer, null, file.name, file.size);
  }

  function onResize() {
    renderer.resize();
    controls.setViewport(canvas.width, canvas.height);
    controls.dirty = true;
    requestSort();
  }

  window.addEventListener('resize', onResize);

  /** Advance one frame. Called by the internal rAF loop; exposed for embedding and tests. */
  function tick() {
    const now = performance.now();
    let dt = (now - lastFrameTime) / 1000;
    lastFrameTime = now;
    if (!(dt > 0) || dt > 0.5) dt = 1 / 60;
    frameMs = frameMs * 0.9 + dt * 1000 * 0.1;

    controls.update(dt);
    if (controls.dirty) {
      requestSort();
      controls.dirty = false;
    }
    renderer.render(controls);

    if (now - statsTime > 400) {
      statsTime = now;
      emitter.emit('stats', {
        frameMs,
        sortMs,
        visible,
        total: renderer.count
      });
    }
  }

  function loop() {
    requestAnimationFrame(loop);
    tick();
  }

  requestAnimationFrame(loop);

  return {
    on: emitter.on,
    loadUrl,
    loadFile,
    tick,
    setExposure(value) {
      renderer.exposure = value;
    },
    setPixelRatio(value) {
      renderer.maxPixelRatio = value;
      onResize();
    },
    get renderer() {
      return renderer;
    },
    get controls() {
      return controls;
    },
    dispose() {
      worker?.terminate();
      worker = null;
      window.removeEventListener('resize', onResize);
      renderer.dispose();
    }
  };
}
