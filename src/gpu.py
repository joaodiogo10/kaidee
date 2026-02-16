"""GPU post-processing pipeline via ModernGL (OpenGL 3.3 core)."""

from __future__ import annotations

import numpy as np

try:
    import moderngl
except ImportError:
    moderngl = None

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    uv = in_uv;
}
"""

BLIT_FRAGMENT = """
#version 330
in vec2 uv;
out vec4 fragColor;
uniform sampler2D tex_src;
void main() {
    fragColor = texture(tex_src, uv);
}
"""

FRAGMENT_SHADER = """
#version 330
precision highp float;
in vec2 uv;
out vec4 fragColor;

uniform sampler2D tex_frame;
uniform sampler2D tex_secondary;
uniform sampler2D tex_vignette;
uniform sampler2D tex_trail;
uniform sampler2D tex_hud;

uniform float u_zoom;
uniform vec2  u_pan;
uniform vec2  u_rot_shift;
uniform float u_saturation;
uniform float u_contrast;
uniform vec3  u_hue_shift;
uniform float u_brightness;
uniform float u_chroma;
uniform float u_chroma_v;
uniform float u_flash;
uniform float u_chaos;
uniform int   u_chaos_spacing;
uniform float u_blend_alpha;
uniform float u_trail_keep;
uniform float u_vig_strength;
uniform int   u_has_vignette;
uniform int   u_has_secondary;
uniform int   u_has_trail;
uniform int   u_hud_only;
uniform float u_time;
uniform vec2  u_resolution;

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    if (u_hud_only > 0) {
        vec4 hud = texture(tex_hud, uv);
        fragColor = hud;
        return;
    }

    vec2 coord = uv;

    // Zoom (in and out)
    if (abs(u_zoom - 1.0) > 0.001) {
        vec2 center = vec2(0.5);
        coord = center + (coord - center) / u_zoom;
    }

    // Pan + rotation shift
    coord += u_pan + u_rot_shift;
    coord = fract(coord);

    // Chromatic aberration
    float r, g, b;
    if (u_chroma > 0.0001) {
        vec2 ch = vec2(u_chroma, 0.0);
        vec2 cv = vec2(0.0, u_chroma_v);
        r = texture(tex_frame, fract(coord + ch)).r;
        g = texture(tex_frame, fract(coord - cv)).g;
        b = texture(tex_frame, fract(coord - ch)).b;
    } else {
        vec3 c = texture(tex_frame, coord).rgb;
        r = c.r; g = c.g; b = c.b;
    }
    vec3 color = vec3(r, g, b);

    // Secondary mode blend
    if (u_has_secondary > 0 && u_blend_alpha > 0.01) {
        vec3 sec = texture(tex_secondary, coord).rgb;
        color = mix(color, sec, u_blend_alpha);
    }

    // Trail persistence
    if (u_has_trail > 0 && u_trail_keep > 0.01) {
        vec3 trail_color = texture(tex_trail, uv).rgb;
        color = mix(color, trail_color, u_trail_keep);
    }

    // Saturation
    if (abs(u_saturation - 1.0) > 0.05) {
        float grey = dot(color, vec3(0.333));
        color = mix(vec3(grey), color, u_saturation);
    }

    // Contrast
    if (u_contrast > 0.01) {
        color = mix(vec3(0.5), color, 1.0 + u_contrast);
    }

    // Hue/warmth shift
    color += u_hue_shift;

    // Beat flash
    color += vec3(u_flash);

    // Chaos noise
    if (u_chaos > 0.0) {
        float n = hash(uv * u_resolution + vec2(u_time * 100.0)) * 2.0 - 1.0;
        color += vec3(n * u_chaos);
    }

    // Chaos scanlines
    if (u_chaos_spacing > 0) {
        int row = int(gl_FragCoord.y);
        if (row - (row / u_chaos_spacing) * u_chaos_spacing == 0) {
            color *= 0.75;
        }
    }

    // Brightness
    color *= u_brightness;

    // Vignette
    if (u_has_vignette > 0) {
        float vig_mask = texture(tex_vignette, uv).r;
        float vig = 1.0 - (1.0 - vig_mask) * u_vig_strength;
        color *= vig;
    }

    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""


class GPUPipeline:
    """Manages ModernGL context, shader, textures, and FBO ping-pong."""

    def __init__(self, render_w: int, render_h: int, display_w: int, display_h: int,
                 vignette: np.ndarray | None):
        if moderngl is None:
            raise RuntimeError("moderngl not installed")

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.NOTHING)

        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )

        vertices = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1,
        ], dtype='f4')
        vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(vbo, '2f 2f', 'in_pos', 'in_uv')])

        self.render_w, self.render_h = render_w, render_h
        self.display_w, self.display_h = display_w, display_h

        # Frame textures
        self.tex_frame = self._make_tex(render_w, render_h, 3)
        self.tex_secondary = self._make_tex(render_w, render_h, 3)

        # FBO ping-pong for trail
        self._fbo_texs = [
            self._make_tex(display_w, display_h, 3),
            self._make_tex(display_w, display_h, 3),
        ]
        self._fbos = [
            self.ctx.framebuffer(color_attachments=[self._fbo_texs[0]]),
            self.ctx.framebuffer(color_attachments=[self._fbo_texs[1]]),
        ]
        self._fbo_idx = 0

        # Vignette texture
        if vignette is not None:
            vig = self._resize_vig(vignette, render_w, render_h)
            vig_u8 = (vig * 255).clip(0, 255).astype(np.uint8)
            self.tex_vignette = self.ctx.texture((render_w, render_h), 1, data=vig_u8.tobytes())
            self._has_vignette = True
        else:
            self.tex_vignette = self.ctx.texture((1, 1), 1, data=b'\xff')
            self._has_vignette = False

        # HUD texture
        self.tex_hud = self.ctx.texture((display_w, display_h), 4, dtype='f1')
        self.tex_hud.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Blit program for fullscreen output (handles any screen resolution)
        self._blit_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=BLIT_FRAGMENT,
        )
        blit_vbo = self.ctx.buffer(vertices.tobytes())
        self._blit_vao = self.ctx.vertex_array(
            self._blit_prog, [(blit_vbo, '2f 2f', 'in_pos', 'in_uv')]
        )

        self.trail_active = False
        self.cached_has_secondary = False

        # Bind texture units
        self.tex_frame.use(location=0)
        self.tex_secondary.use(location=1)
        self.tex_vignette.use(location=2)
        self._fbo_texs[1].use(location=3)
        self.tex_hud.use(location=4)

        self.prog['tex_frame'].value = 0
        self.prog['tex_secondary'].value = 1
        self.prog['tex_vignette'].value = 2
        self.prog['tex_trail'].value = 3
        self.prog['tex_hud'].value = 4
        self.prog['u_resolution'].value = (float(render_w), float(render_h))
        self.prog['u_has_vignette'].value = 1 if self._has_vignette else 0

        # Auto-detect actual GL framebuffer size for screen viewport
        try:
            fb = self.ctx.detect_framebuffer()
            screen_w, screen_h = fb.size
        except Exception:
            screen_w, screen_h = self.ctx.screen.size
        self._screen_viewport = (0, 0, screen_w, screen_h)

        print(f"GPU: render={render_w}x{render_h} display={display_w}x{display_h} "
              f"screen_viewport={screen_w}x{screen_h} ctx.screen={self.ctx.screen.size}")

    def _make_tex(self, w, h, components):
        tex = self.ctx.texture((w, h), components, dtype='f1')
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex

    @staticmethod
    def _resize_vig(vig, w, h):
        if vig.shape[0] != h or vig.shape[1] != w:
            y_idx = np.linspace(0, vig.shape[0] - 1, h).astype(int)
            x_idx = np.linspace(0, vig.shape[1] - 1, w).astype(int)
            return vig[y_idx][:, x_idx]
        return vig

    def recreate_textures(self, render_w, render_h, vignette=None):
        """Recreate textures at new render resolution. FBOs stay at display size."""
        self.render_w, self.render_h = render_w, render_h
        self.tex_frame.release()
        self.tex_secondary.release()
        self.tex_frame = self._make_tex(render_w, render_h, 3)
        self.tex_secondary = self._make_tex(render_w, render_h, 3)
        self.tex_frame.use(location=0)
        self.tex_secondary.use(location=1)
        if vignette is not None:
            self.tex_vignette.release()
            vig = self._resize_vig(vignette, render_w, render_h)
            vig_u8 = (vig * 255).clip(0, 255).astype(np.uint8)
            self.tex_vignette = self.ctx.texture((render_w, render_h), 1, data=vig_u8.tobytes())
            self.tex_vignette.use(location=2)
        self.prog['u_resolution'].value = (float(render_w), float(render_h))
        self.trail_active = False
        print(f"GPU textures recreated: {render_w}x{render_h}")

    def upload_frame(self, frame: np.ndarray) -> None:
        data = np.ascontiguousarray(np.flipud(frame))
        self.tex_frame.write(data)

    def upload_secondary(self, frame: np.ndarray) -> None:
        data = np.ascontiguousarray(np.flipud(frame))
        self.tex_secondary.write(data)
        self.cached_has_secondary = True

    def upload_hud(self, hud_rgba: np.ndarray) -> None:
        data = np.ascontiguousarray(np.flipud(hud_rgba))
        self.tex_hud.write(data)

    def render(self, uniforms: dict) -> None:
        """Set uniforms and render to FBO, then blit to screen."""
        # Bind previous FBO texture as trail input
        prev_fbo_tex = self._fbo_texs[1 - self._fbo_idx]
        prev_fbo_tex.use(location=3)

        prog = self.prog
        for key, val in uniforms.items():
            prog[key].value = val

        prog['u_hud_only'].value = 0

        current_fbo = self._fbos[self._fbo_idx]
        current_fbo.use()
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # Blit FBO to screen via fullscreen quad (scales to any resolution)
        self._blit_to_screen(self._fbo_texs[self._fbo_idx])
        self._fbo_idx = 1 - self._fbo_idx

    def set_screen_size(self, w: int, h: int) -> None:
        """Update the screen viewport size (call after fullscreen toggle)."""
        self._screen_viewport = (0, 0, w, h)
        # Recreate HUD texture at new screen size for crisp text
        if self.tex_hud.size != (w, h):
            self.tex_hud.release()
            self.tex_hud = self.ctx.texture((w, h), 4, dtype='f1')
            self.tex_hud.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.tex_hud.use(location=4)
        print(f"GPU screen viewport: {w}x{h}")

    def _blit_to_screen(self, tex) -> None:
        """Render a texture to the screen framebuffer via fullscreen quad.

        Explicitly sets viewport to handle fullscreen / window size changes.
        """
        tex.use(location=5)
        self._blit_prog['tex_src'].value = 5
        self.ctx.screen.use()
        self.ctx.viewport = self._screen_viewport
        self._blit_vao.render(moderngl.TRIANGLE_STRIP)

    def render_hud_overlay(self) -> None:
        """Render HUD as alpha-blended overlay on top of current screen."""
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.prog['u_hud_only'].value = 1
        self.tex_hud.use(location=4)
        self.ctx.screen.use()
        self.ctx.viewport = self._screen_viewport
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.prog['u_hud_only'].value = 0
        self.ctx.disable(moderngl.BLEND)

    def readback(self) -> np.ndarray:
        """Read current screen pixels as (H, W, 3) uint8 array (no HUD)."""
        # Read from the most recently rendered FBO (before HUD overlay)
        prev_idx = 1 - self._fbo_idx  # render() already flipped the index
        raw = self._fbos[prev_idx].read(components=3)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.display_h, self.display_w, 3)
        return np.flipud(arr)

    def release(self) -> None:
        """Release all GPU resources."""
        for tex in (self.tex_frame, self.tex_secondary, self.tex_vignette,
                    self.tex_hud, *self._fbo_texs):
            tex.release()
        for fbo in self._fbos:
            fbo.release()
        self.vao.release()
        self._blit_vao.release()
        self.prog.release()
        self._blit_prog.release()
        self.ctx.release()
