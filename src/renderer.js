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
uniform sampler2D uRot;
uniform sampler2D uScl;
uniform sampler2D uCol;

layout(location = 0) in vec2 aCorner;

out vec2 vLocal;
out float vOpacity;
out vec4 vColor;

mat3 quatToMat(vec4 q) {
  float x = q.x, y = q.y, z = q.z, w = q.w;
  float x2 = x + x, y2 = y + y, z2 = z + z;
  float xx = x * x2, xy = x * y2, xz = x * z2;
  float yy = y * y2, yz = y * z2, zz = z * z2;
  float wx = w * x2, wy = w * y2, wz = w * z2;
  return mat3(1.0 - (yy + zz), xy + wz, xz - wy,
              xy - wz, 1.0 - (xx + zz), yz + wx,
              xz + wy, yz - wx, 1.0 - (xx + yy));
}

void main() {
  uint id = texelFetch(uOrder, ivec2(gl_InstanceID % uW, gl_InstanceID / uW), 0).r;
  ivec2 uv = ivec2(int(id) % uW, int(id) / uW);

  vec4 p = texelFetch(uPos, uv, 0);
  vec4 q = texelFetch(uRot, uv, 0);
  vec4 s = texelFetch(uScl, uv, 0);

  vec4 vp = uView * vec4(p.xyz, 1.0);
  vec4 clip = uProj * vp;

  vLocal = aCorner;
  vOpacity = s.w;
  vColor = texelFetch(uCol, uv, 0);

  if (clip.w <= 0.02 || s.w < 0.004) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  mat3 rv = mat3(uView);
  mat3 m = quatToMat(q) * mat3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
  mat3 sigma = rv * (m * transpose(m)) * transpose(rv);

  float az = max(-vp.z, 0.02);
  float invz = 1.0 / az;
  mat3 j = mat3(
    uFocal.x * invz, 0.0, -uFocal.x * vp.x * invz * invz,
    0.0, uFocal.y * invz, -uFocal.y * vp.y * invz * invz,
    0.0, 0.0, 0.0
  );
  mat3 t = j * sigma * transpose(j);

  float a = t[0][0] + 0.3;
  float b = t[0][1];
  float d = t[1][1] + 0.3;

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
    for (const name of ['uView', 'uProj', 'uFocal', 'uRes', 'uW', 'uOrder', 'uPos', 'uRot', 'uScl', 'uCol', 'uExposure']) {
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
   */
  setSplatData(data, order) {
    const { gl } = this;
    const { width, height, position, rotation, scaleOpacity, color, count } = data;

    this.disposeTextures();

    this.textures.order = this.#texture(0, gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT, width, height, order);
    this.textures.position = this.#texture(1, gl.RGBA32F, gl.RGBA, gl.FLOAT, width, height, position);
    this.textures.rotation = this.#texture(2, gl.RGBA32F, gl.RGBA, gl.FLOAT, width, height, rotation);
    this.textures.scale = this.#texture(3, gl.RGBA32F, gl.RGBA, gl.FLOAT, width, height, scaleOpacity);
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
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.textures.order);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.width, this.height, gl.RED_INTEGER, gl.UNSIGNED_INT, order);
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
    gl.uniform1i(this.uniforms.uRot, 2);
    gl.uniform1i(this.uniforms.uScl, 3);
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
