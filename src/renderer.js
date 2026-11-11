/**
 * WebGL2 renderer for Gaussian splats.
 *
 * Attribute planes live in float textures, so nothing is repacked per frame.
 * Each splat is drawn as an instanced quad whose extent comes from projecting
 * its 3D covariance: the 3x3 covariance is flattened to a 2x2 screen-space
 * matrix, then eigendecomposed to get the ellipse axes. This is what keeps
 * thin, anisotropic splats from collapsing into round dots.
 */

const VERT = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;

uniform mat4 uView;
uniform mat4 uProj;
uniform vec2 uFocal;
uniform vec2 uRes;
uniform int uW;
uniform usampler2D uOrder;
uniform sampler2D uPos;
uniform sampler2D uCovA;
uniform sampler2D uCovB;
uniform sampler2D uCol;

layout(location = 0) in vec2 aCorner;

out vec2 vLocal;
out float vOpacity;
out vec4 vColor;

void main() {
  uint id = texelFetch(uOrder, ivec2(gl_InstanceID % uW, gl_InstanceID / uW), 0).r;
  ivec2 uv = ivec2(int(id) % uW, int(id) / uW);

  vec4 p = texelFetch(uPos, uv, 0);
  vec4 ca = texelFetch(uCovA, uv, 0);
  vec2 cb = texelFetch(uCovB, uv, 0).xy;

  vec4 vp = uView * vec4(p.xyz, 1.0);
  vec4 clip = uProj * vp;

  vLocal = aCorner;
  vOpacity = p.w;
  vColor = texelFetch(uCol, uv, 0);

  if (clip.w <= 0.02 || p.w < 0.004) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  float az = max(-vp.z, 0.02);
  float invz = 1.0 / az;
  float invz2 = invz * invz;

  vec3 j0 = vec3(uFocal.x * invz, 0.0, -uFocal.x * vp.x * invz2);
  vec3 j1 = vec3(0.0, uFocal.y * invz, -uFocal.y * vp.y * invz2);

  mat3 rv = mat3(uView);
  vec3 b0 = j0 * rv;
  vec3 b1 = j1 * rv;

  float s00 = ca.x, s01 = ca.y, s02 = ca.z, s11 = ca.w;
  float s12 = cb.x, s22 = cb.y;

  float a = b0.x * b0.x * s00 + 2.0 * b0.x * b0.y * s01 + 2.0 * b0.x * b0.z * s02
          + b0.y * b0.y * s11 + 2.0 * b0.y * b0.z * s12 + b0.z * b0.z * s22 + 0.3;
  float b = b0.x * b1.x * s00 + (b0.x * b1.y + b0.y * b1.x) * s01
          + (b0.x * b1.z + b0.z * b1.x) * s02 + b0.y * b1.y * s11
          + (b0.y * b1.z + b0.z * b1.y) * s12 + b0.z * b1.z * s22;
  float d = b1.x * b1.x * s00 + 2.0 * b1.x * b1.y * s01 + 2.0 * b1.x * b1.z * s02
          + b1.y * b1.y * s11 + 2.0 * b1.y * b1.z * s12 + b1.z * b1.z * s22 + 0.3;

  float mid = 0.5 * (a + d);
  float disc = sqrt(max(0.0, mid * mid - (a * d - b * b)));
  float l1 = max(mid + disc, 0.25);
  float l2 = max(mid - disc, 0.25);

  vec2 e1 = normalize(vec2(b, l1 - a) + vec2(1e-7, 0.0));
  vec2 e2 = vec2(-e1.y, e1.x);

  vec2 offset = (e1 * (aCorner.x * sqrt(l1)) + e2 * (aCorner.y * sqrt(l2))) * 3.0;

  gl_Position = clip;
  gl_Position.xy += offset * (2.0 / uRes) * clip.w;
}`;

const FRAG = `#version 300 es
precision highp float;

uniform float uExposure;

in vec2 vLocal;
in float vOpacity;
in vec4 vColor;

out vec4 fragColor;

void main() {
  float r2 = dot(vLocal, vLocal);
  if (r2 > 1.0) discard;

  float alpha = exp(-4.5 * r2) * vOpacity;
  if (alpha < 0.0039) discard;

  fragColor = vec4(vColor.rgb * uExposure * alpha, alpha);
}`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile failed: ${log}`);
  }
  return shader;
}

export class SplatRenderer {
  constructor(canvas, { maxPixelRatio = 2 } = {}) {
    this.canvas = canvas;
    this.maxPixelRatio = maxPixelRatio;

    const gl = canvas.getContext('webgl2', {
      antialias: false,
      alpha: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser');
    this.gl = gl;

    this.program = gl.createProgram();
    gl.attachShader(this.program, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(this.program, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error(`Program link failed: ${gl.getProgramInfoLog(this.program)}`);
    }

    this.uniforms = {};
    for (const name of ['uView', 'uProj', 'uFocal', 'uRes', 'uW', 'uOrder', 'uPos', 'uCovA', 'uCovB', 'uCol', 'uExposure']) {
      this.uniforms[name] = gl.getUniformLocation(this.program, name);
    }

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const corner = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, corner, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(0, 0);
    gl.bindVertexArray(null);

    this.textures = {};
    this.width = 0;
    this.height = 0;
    this.count = 0;
    this.visible = 0;
    this.background = [0.035, 0.035, 0.043];
    this.exposure = 1;

    this.resize();
  }

  resize() {
    const { canvas, gl } = this;
    const ratio = Math.min(window.devicePixelRatio || 1, this.maxPixelRatio);
    const width = Math.max(1, Math.floor((canvas.clientWidth || window.innerWidth || 1) * ratio));
    const height = Math.max(1, Math.floor((canvas.clientHeight || window.innerHeight || 1) * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
  }

  #texture(unit, internalFormat, format, type, width, height, data) {
    const { gl } = this;
    const texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
    return texture;
  }

  /**
   * Upload decoded SOG planes. `order` is a Uint32Array of splat indices laid out
   * row-major over the same W x H grid as the attribute planes.
   *
   * The covariance arrives pre-baked as six independent components spread over a
   * vec4 and a vec2 plane. Storing the factors instead would cost two extra floats
   * per splat and force the shader to rebuild the matrix every vertex.
   */
  setSplatData(data, order) {
    const { gl } = this;
    const { width, height, position, covarianceA, covarianceB, color, count } = data;

    this.disposeTextures();

    this.textures.order = this.#texture(0, gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT, width, height, order);
    this.textures.position = this.#texture(1, gl.RGBA32F, gl.RGBA, gl.FLOAT, width, height, position);
    this.textures.covarianceA = this.#texture(2, gl.RGBA32F, gl.RGBA, gl.FLOAT, width, height, covarianceA);
    this.textures.covarianceB = this.#texture(3, gl.RG32F, gl.RG, gl.FLOAT, width, height, covarianceB);
    this.textures.color = this.#texture(4, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, width, height, color);

    this.width = width;
    this.height = height;
    this.count = count;
    this.visible = count;
  }

  /** Replace only the draw order, keeping attribute textures untouched. */
  updateOrder(order, visible) {
    const { gl } = this;
    if (!this.textures.order) return;
    // Only the first `visible` entries are read, so upload just enough rows.
    // Culling is what makes this cheaper: a zoomed-in view can cut the transfer
    // by an order of magnitude.
    const rows = Math.max(1, Math.min(this.height, Math.ceil(visible / this.width)));
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.textures.order);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.width, rows, gl.RED_INTEGER, gl.UNSIGNED_INT, order);
    this.visible = visible;
  }

  clear() {
    const { gl } = this;
    const [r, g, b] = this.background;
    gl.clearColor(r, g, b, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  render(camera) {
    const { gl } = this;
    if (!this.visible || !this.textures.order) {
      this.clear();
      return;
    }

    this.clear();

    const width = this.canvas.width;
    const height = this.canvas.height;
    const focal = 1 / Math.tan(camera.fov / 2) * height * 0.5;

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uniforms.uView, false, camera.view);
    gl.uniformMatrix4fv(this.uniforms.uProj, false, camera.projection);
    gl.uniform2f(this.uniforms.uFocal, focal, focal);
    gl.uniform2f(this.uniforms.uRes, width, height);
    gl.uniform1i(this.uniforms.uW, this.width);
    gl.uniform1f(this.uniforms.uExposure, this.exposure);
    gl.uniform1i(this.uniforms.uOrder, 0);
    gl.uniform1i(this.uniforms.uPos, 1);
    gl.uniform1i(this.uniforms.uCovA, 2);
    gl.uniform1i(this.uniforms.uCovB, 3);
    gl.uniform1i(this.uniforms.uCol, 4);

    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.visible);
    gl.bindVertexArray(null);
  }

  disposeTextures() {
    const { gl } = this;
    for (const key of Object.keys(this.textures)) {
      if (this.textures[key]) gl.deleteTexture(this.textures[key]);
    }
    this.textures = {};
  }

  dispose() {
    const { gl } = this;
    this.disposeTextures();
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
  }
}
