# SOG Viewer

A lightweight, dependency-free WebGL2 viewer for [SOG](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/) Gaussian splats.

No build step, no framework, no runtime dependencies. Clone it, serve the folder, open the page.

```
~1,300 lines of plain ES modules   ·   one static folder   ·   MIT
```

## Why another splat viewer

Most web splat viewers are either a large engine you have to adopt wholesale, or a demo that only handles `.ply`. This one is deliberately narrow: read the SOG container correctly, project the Gaussians properly, keep the main thread free, and stay small enough to read in an afternoon.

It exists because a hand-rolled SOG path is easy to get subtly wrong — a ZIP stored with raw deflate handed to `DecompressionStream('deflate')`, a vertex buffer described with the wrong type, premultiplied output blended with `SRC_ALPHA`. Each of those produces a picture that is *almost* right, which is worse than an error. This implementation follows the SOG specification item by item, and the renderer follows the standard 3D Gaussian rasterisation path.

## Features

- **SOG v2 and v1** — ZIP container, WebP attribute planes, codebook and smallest-three quaternion decoding
- **Anisotropic rasterisation** — each splat is an instanced quad sized by projecting its 3×3 covariance and eigendecomposing the resulting 2×2 screen-space matrix
- **Depth sorting off the main thread** — counting sort over quantised view depth, running in a Worker, with double-buffered order hand-off
- **Frustum culling** — splats outside the view are dropped before they reach the GPU, so the draw call shrinks with the view
- **Attributes stay on the GPU** — positions, rotations, scales and colours live in float textures; nothing is repacked per frame
- **Pointer-native controls** — orbit, pan, pinch and wheel share a single code path, with easing
- **Drop-in** — drag a `.sog` onto the window, or pass a URL

## Quick start

The viewer is static files, so any HTTP server works.

```bash
git clone <this repo> sog-viewer
cd sog-viewer

# put a scene next to index.html
# demo.sog is not committed (see "Making a .sog file")

python -m http.server 8080
# or: npx serve .
```

Then open <http://localhost:8080>. Append `?src=/other.sog` to load a different scene.

Opening `index.html` from `file://` will not work: ES modules and `fetch` both require HTTP.

## Use as a library

`src/viewer.js` has no DOM assumptions beyond the canvas you hand it.

```js
import { createViewer } from './src/viewer.js';

const canvas = document.querySelector('canvas');
const viewer = createViewer(canvas, { pixelRatio: 1.5, exposure: 1 });

viewer.on('load', ({ splats, loadMs }) => console.log(`${splats} splats in ${loadMs} ms`));
viewer.on('stats', ({ frameMs, sortMs, visible }) => {
  console.log(`frame ${frameMs.toFixed(1)} ms, sort ${sortMs.toFixed(1)} ms, ${visible} visible`);
});

await viewer.loadUrl('./scene.sog');
// or: await viewer.loadFile(fileFromInput);
```

`createViewer` returns `loadUrl`, `loadFile`, `tick`, `setExposure`, `setPixelRatio`, `dispose`, plus the underlying `renderer` and `controls` for advanced use.

`tick()` advances one frame. The internal loop already calls it via `requestAnimationFrame`; it is exposed so the viewer can be driven manually in tests or embedded renderers.

## Performance

Measured on one synthetic fixture — 338,964 splats, SOG v2 with 3 SH bands, on a desktop browser at a 1.0 pixel ratio. Numbers are indicative, not a benchmark suite.

| Stage | Time | Notes |
|---|---|---|
| Load (bytes → GPU) | ~135–310 ms | container unpack, 8 WebP planes, dequantisation |
| Depth sort | ~13 ms | in a Worker; does not block the frame |
| Frame (render) | ~5.7 ms | 4 instanced vertices per splat |

The sort is the dominant cost per moving frame, which is exactly why it is not on the main thread. Sorting is `O(n + bins)` with 65,536 depth bins — no comparisons, no allocation after setup.

## How it works

```
.sog  →  unzip (deflate-raw)  →  decode WebP planes  →  dequantise
                                                          │
                        ┌─────────────────────────────────┴─────────────┐
                        ▼                                               ▼
            float textures (pos / rot / scale+opacity)      positions → Worker
            RGBA8 texture (DC colour)                              │
                        │                                    bucket sort + cull
                        │                                              │
                        │                          order texture (R32UI, W×H)
                        │                                              │
                        └──────────────► drawArraysInstanced(TRIANGLE_STRIP, 4, N)
```

Sorting is back-to-front, which is what alpha compositing requires. `gl_InstanceID` indexes the order texture, the order texture indexes the attribute textures, and the vertex shader expands each splat into a screen-aligned ellipse. Fragments outside the unit disc are discarded and alpha falls off as `exp(-4.5 r²)`, so 3σ lands on the quad edge.

Blending is premultiplied: the fragment writes `(rgb * alpha, alpha)` and the blend func is `ONE, ONE_MINUS_SRC_ALPHA`.

### Deliberate limits

- **SH degree 0 only.** Base colour comes from `sh0`. The `shN` palette planes are skipped, so there is no view-dependent highlight. This keeps the attribute set at three float textures; adding bands means attaching the palette lookup, which is a contained change but not free.
- **No LOD.** Every visible splat is submitted. PlayCanvas ships streamed SOG with a Gaussian budget for scenes past ~1M splats; that is the right tool once a scene outgrows one buffer.
- **No tone mapping.** SOG stores sRGB-encoded colour, so it is written straight to the framebuffer. An exposure multiplier is offered; a filmic curve would change the intended look.

## Making a .sog file

SOG is produced by the PlayCanvas CLI, which accepts PLY, SPZ, SPLAT and KSPLAT input:

```bash
npx @playcanvas/splat-transform input.ply output.sog
```

See the [format specification](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/) for the container layout.

## Project layout

```
index.html          demo page and UI wiring
src/viewer.js       orchestration, worker hand-off, event emitter
src/sog.js          container parsing and attribute decoding
src/renderer.js     WebGL2 program, textures, instanced draw
src/sort.worker.js  counting sort and frustum culling
src/controls.js     pointer controls and camera fitting
src/style.css       UI
```

Each module is independent: `sog.js` has no WebGL, `renderer.js` has no file I/O, `sort.worker.js` has no DOM.

## Browser support

WebGL2 and ES modules with module Workers. Verified on current Chromium. Safari needs 16.4+ for module Workers; `OffscreenCanvas` is used when present and falls back to a regular canvas otherwise.

## License

MIT — see [LICENSE](./LICENSE).

SOG is a format designed and open-sourced by [PlayCanvas](https://playcanvas.com). This project is an independent implementation of that specification.
