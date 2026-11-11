/**
 * SOG decoder.
 *
 * Reads a `.sog` container (a ZIP holding `meta.json` plus lossless WebP
 * attribute planes) and dequantises it into flat planes ready for GPU upload.
 *
 * Format reference: https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/
 *
 * The decoder has no DOM dependency beyond `createImageBitmap` / canvas, so it
 * runs identically on the main thread and inside a worker.
 */

const SH_C0 = 0.28209479177387814;
const SQRT2 = Math.SQRT2;

const lerp = (a, b, t) => a + (b - a) * t;

/** Symmetric log transform used by SOG v2 for positions. */
const unlog = (n) => Math.sign(n) * (Math.exp(Math.abs(n)) - 1);

const createCanvas = (w, h) => {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  return canvas;
};

/**
 * Parse a ZIP central directory and inflate every entry.
 * ZIP method 8 is raw deflate, which maps to DecompressionStream('deflate-raw')
 * and *not* to 'deflate' (that one expects a zlib wrapper).
 */
export async function unzip(buffer) {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);

  let eocd = -1;
  const floor = Math.max(0, bytes.length - 65557);
  for (let i = bytes.length - 22; i >= floor; i--) {
    if (view.getUint32(i, true) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error('Not a SOG container (no ZIP end-of-directory record)');

  const entries = view.getUint16(eocd + 10, true);
  let offset = view.getUint32(eocd + 16, true);
  const files = {};
  const pending = [];

  for (let i = 0; i < entries; i++) {
    if (view.getUint32(offset, true) !== 0x02014b50) break;

    const method = view.getUint16(offset + 10, true);
    const compressedSize = view.getUint32(offset + 20, true);
    const nameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true);
    const commentLength = view.getUint16(offset + 32, true);
    const localOffset = view.getUint32(offset + 42, true);
    const name = new TextDecoder().decode(bytes.subarray(offset + 46, offset + 46 + nameLength));

    const localNameLength = view.getUint16(localOffset + 26, true);
    const localExtraLength = view.getUint16(localOffset + 28, true);
    const dataStart = localOffset + 30 + localNameLength + localExtraLength;
    const raw = bytes.subarray(dataStart, dataStart + compressedSize);

    if (method === 0) {
      files[name] = Promise.resolve(raw);
    } else if (method === 8) {
      pending.push(
        (async () => {
          const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
          files[name] = Promise.resolve(new Uint8Array(await new Response(stream).arrayBuffer()));
        })()
      );
    } else {
      throw new Error(`Unsupported ZIP compression method ${method} for ${name}`);
    }

    offset += 46 + nameLength + extraLength + commentLength;
  }

  await Promise.all(pending);

  const resolved = {};
  for (const [name, promise] of Object.entries(files)) resolved[name] = await promise;
  return resolved;
}

/** Decode a WebP plus its alpha channel into raw RGBA bytes. */
export async function decodeWebP(data) {
  const bitmap = await createImageBitmap(new Blob([data], { type: 'image/webp' }));
  const canvas = createCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const { data: pixels } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close?.();
  return { pixels, width: bitmap.width, height: bitmap.height };
}

/**
 * Smallest-three quaternion decode (26 bit: three signed components + 2 bit mode).
 * Components are ordered (w, x, y, z); the mode names the omitted one.
 */
function decodeQuat(r, g, b, modeByte, out, o) {
  const toComp = (v) => (v / 255 - 0.5) * 2.0 / SQRT2;
  const a = toComp(r);
  const bq = toComp(g);
  const c = toComp(b);
  const mode = modeByte - 252;
  const d = Math.sqrt(Math.max(0, 1 - (a * a + bq * bq + c * c)));

  switch (mode) {
    case 0: out[o] = a; out[o + 1] = bq; out[o + 2] = c; out[o + 3] = d; break;
    case 1: out[o] = d; out[o + 1] = bq; out[o + 2] = c; out[o + 3] = a; break;
    case 2: out[o] = bq; out[o + 1] = d; out[o + 2] = c; out[o + 3] = a; break;
    case 3: out[o] = bq; out[o + 1] = c; out[o + 2] = d; out[o + 3] = a; break;
    default: out[o] = 0; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 1;
  }
}

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const sigmoid = (v) => 1 / (1 + Math.exp(-v));

/**
 * Fold rotation and non-uniform scale into a world-space covariance matrix.
 *
 * Sigma = M M^T where M = R S. Only the six independent components are kept,
 * because the renderer never needs the factors again: it projects this matrix
 * to screen space and reads off the ellipse, which is invariant to the
 * decomposition. Doing it here removes a quaternion-to-matrix conversion, a
 * 3x3 product and a second 3x3 product from every vertex.
 */
function bakeCovariance(qx, qy, qz, qw, sx, sy, sz, covA, covB, t, t2) {
  const x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
  const xx = qx * x2, xy = qx * y2, xz = qx * z2;
  const yy = qy * y2, yz = qy * z2, zz = qz * z2;
  const wx = qw * x2, wy = qw * y2, wz = qw * z2;

  const r00 = 1 - (yy + zz), r01 = xy - wz, r02 = xz + wy;
  const r10 = xy + wz, r11 = 1 - (xx + zz), r12 = yz - wx;
  const r20 = xz - wy, r21 = yz + wx, r22 = 1 - (xx + yy);

  const m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz;
  const m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
  const m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;

  covA[t] = m00 * m00 + m01 * m01 + m02 * m02;
  covA[t + 1] = m00 * m10 + m01 * m11 + m02 * m12;
  covA[t + 2] = m00 * m20 + m01 * m21 + m02 * m22;
  covA[t + 3] = m10 * m10 + m11 * m11 + m12 * m12;
  covB[t2] = m10 * m20 + m11 * m21 + m12 * m22;
  covB[t2 + 1] = m20 * m20 + m21 * m21 + m22 * m22;
}

const quatTmp = new Float32Array(4);

/**
 * Decode a SOG buffer.
 *
 * @param {ArrayBuffer} buffer raw `.sog` bytes
 * @param {(p: {phase: string, ratio: number}) => void} [onProgress]
 * @returns {Promise<{count: number, width: number, height: number, position: Float32Array,
 *   covarianceA: Float32Array, covarianceB: Float32Array, color: Uint8Array, meta: object}>}
 */
export async function decodeSog(buffer, onProgress = () => {}) {
  const report = (phase, ratio) => onProgress({ phase, ratio });

  report('unpack', 0.05);
  const files = await unzip(buffer);
  if (!files['meta.json']) throw new Error('SOG container is missing meta.json');

  const meta = JSON.parse(new TextDecoder().decode(files['meta.json']));
  const width = meta.width;
  const height = meta.height;
  const count = meta.count;
  const texels = width * height;

  if (!(width > 0 && height > 0 && count > 0)) throw new Error('SOG meta.json has invalid dimensions');

  const needs = [
    meta.means?.files?.[0],
    meta.means?.files?.[1],
    meta.scales?.files?.[0],
    meta.quats?.files?.[0],
    meta.sh0?.files?.[0]
  ];
  for (const name of needs) {
    if (!name) throw new Error('SOG meta.json is missing a required attribute plane');
    if (!files[name]) throw new Error(`SOG container is missing ${name}`);
  }

  report('image', 0.15);
  const decoded = await Promise.all(needs.map((name) => decodeWebP(files[name])));
  const [low, high, scalePlane, quatPlane, colorPlane] = decoded.map((d) => d.pixels);

  report('decode', 0.45);

  const position = new Float32Array(texels * 4);
  const covarianceA = new Float32Array(texels * 4);
  const covarianceB = new Float32Array(texels * 2);
  const color = new Uint8Array(texels * 4);

  const version = meta.version ?? 2;
  const scaleBook = meta.scales.codebook;
  const colorBook = meta.sh0.codebook;
  const means = meta.means;

  const chunk = Math.max(1024, Math.floor(count / 40));
  let nextReport = chunk;

  for (let i = 0; i < count; i++) {
    const t = i * 4;

    const qx = (high[t] << 8) | low[t];
    const qy = (high[t + 1] << 8) | low[t + 1];
    const qz = (high[t + 2] << 8) | low[t + 2];
    position[t] = unlog(lerp(means.mins[0], means.maxs[0], qx / 65535));
    position[t + 1] = unlog(lerp(means.mins[1], means.maxs[1], qy / 65535));
    position[t + 2] = unlog(lerp(means.mins[2], means.maxs[2], qz / 65535));
    decodeQuat(quatPlane[t], quatPlane[t + 1], quatPlane[t + 2], quatPlane[t + 3], quatTmp, 0);

    let sx, sy, sz, opacity;
    if (version === 2) {
      sx = Math.exp(scaleBook[scalePlane[t]]);
      sy = Math.exp(scaleBook[scalePlane[t + 1]]);
      sz = Math.exp(scaleBook[scalePlane[t + 2]]);
      opacity = colorPlane[t + 3] / 255;
    } else {
      sx = lerp(meta.scales.mins[0], meta.scales.maxs[0], scalePlane[t] / 255);
      sy = lerp(meta.scales.mins[1], meta.scales.maxs[1], scalePlane[t + 1] / 255);
      sz = lerp(meta.scales.mins[2], meta.scales.maxs[2], scalePlane[t + 2] / 255);
      opacity = sigmoid(lerp(meta.sh0.mins[3], meta.sh0.maxs[3], colorPlane[t + 3] / 255));
    }
    position[t + 3] = opacity;
    bakeCovariance(quatTmp[0], quatTmp[1], quatTmp[2], quatTmp[3], sx, sy, sz, covarianceA, covarianceB, t, i * 2);

    if (version === 2) {
      color[t] = clamp01(0.5 + colorBook[colorPlane[t]] * SH_C0) * 255;
      color[t + 1] = clamp01(0.5 + colorBook[colorPlane[t + 1]] * SH_C0) * 255;
      color[t + 2] = clamp01(0.5 + colorBook[colorPlane[t + 2]] * SH_C0) * 255;
    } else {
      const r = lerp(meta.sh0.mins[0], meta.sh0.maxs[0], colorPlane[t] / 255);
      const g = lerp(meta.sh0.mins[1], meta.sh0.maxs[1], colorPlane[t + 1] / 255);
      const b = lerp(meta.sh0.mins[2], meta.sh0.maxs[2], colorPlane[t + 2] / 255);
      color[t] = clamp01(0.5 + r * SH_C0) * 255;
      color[t + 1] = clamp01(0.5 + g * SH_C0) * 255;
      color[t + 2] = clamp01(0.5 + b * SH_C0) * 255;
    }
    color[t + 3] = 255;

    if (i >= nextReport) {
      nextReport += chunk;
      report('decode', 0.45 + 0.5 * (i / count));
    }
  }

  report('done', 1);
  return { count, width, height, position, covarianceA, covarianceB, color, meta };
}
