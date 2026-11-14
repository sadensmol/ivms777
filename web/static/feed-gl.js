// The memory film player (design §11.1). WebGL2, no dependency.
//
// Why not CSS: the film needs true cross-dissolves and dips to black, a focus pull, a
// beat-locked exposure pulse, bloom, film weave, drifting bokeh, light leaks and a
// colour grade — none of which the DOM can express. All of it is ONE fragment shader
// over the two textures already bound, so no effect costs a pass, a framebuffer or a
// layout, and the film holds 60 fps.
//
// THE CLOCK IS THE SOUNDTRACK. `performance.now()` places no cut: every shot carries an
// absolute `at`, and the current shot is found by looking up the audio element's own
// `currentTime`. A duration chain drifts against audio within seconds — and a drifting
// cut is exactly the "three things playing at once" this design exists to kill. Seeking
// and buffering stay in sync for free, because there is nothing to keep in sync: there
// is one clock.
//
// Every shot change is PLAYER STATE, never a history entry (§13.1 rule 3): the URL
// does not move while a film plays, so close is always one step back to the grid.
(() => {
  const body = document.body;
  const key = body.dataset.key;
  const lang = body.dataset.lang || 'en';
  const canvas = document.getElementById('stage');
  const caption = document.getElementById('feed-text');
  const start = document.getElementById('start');
  const playLabel = start.querySelector('.feed-start-play');
  const music = document.getElementById('music');
  const narration = document.getElementById('narration');

  const MUSIC_LEVEL = 0.55;
  const MUSIC_DUCKED = 0.16;
  const DUCK_SECONDS = 0.6;
  // How hard each transition smears both layers radially. A whip, a zoom blur and a
  // match zoom are the same optics at three strengths.
  const SMEAR = { whip: 0.09, zoom_blur: 0.16, match_zoom: 0.05 };

  const VERT = `#version 300 es
  in vec2 aPos; out vec2 vUv;
  void main() { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

  // One shader, every effect. `uPrevFx`/`uNextFx` are the ids the TIMELINE carries
  // (feed/effects.py `payload()`), so this file holds no copy of that table — an id
  // here and a name there can never drift apart.
  const FRAG = `#version 300 es
  precision highp float;
  in vec2 vUv; out vec4 outColor;
  uniform sampler2D uPrev, uNext;
  uniform vec4 uPrevXf, uNextXf;       // scale, offsetX, offsetY, blurAmount
  uniform vec2 uPrevFit, uNextFit;     // CONTAIN fit — the whole photo, letterboxed
  uniform vec2 uPrevFill, uNextFill;   // COVER fit — the blurred backdrop
  // TWO effect slots per texture layer. A shot is an arc, not a filter: effects
  // arrive, overlap and leave over its length (feed/effects.py MAX_ACTIVE), so each
  // layer carries two of them plus the weight each currently has.
  uniform vec4 uPrevPA, uPrevPB, uNextPA, uNextPB;
  uniform ivec2 uPrevFx, uNextFx;      // the two effect ids, A and B
  uniform vec2 uPrevW, uNextW;         // their weights, 0..1, cross-faded in JS
  uniform vec2 uPrevClock, uNextClock; // seconds into that shot, and its length
  uniform float uMix, uTime, uVignette, uGrain, uWhip;
  // Which way a slide travels, in screen space. Zero for every other transition.
  uniform vec2 uSlide;
  // The one free direction a transition may ask for, with a per-mode meaning, and zero
  // for every mode that does not read one — exactly like uSlide.
  //   iris      — the SIGN of x: negative closes the outgoing picture into a circle,
  //               positive opens the incoming one out of one.
  //   page_turn — the direction the sheet peels toward, in screen space.
  uniform vec2 uAxis;
  uniform int uMode;
  uniform vec3 uLift, uGamma, uGain;

  float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
  float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
  }
  float fbm(vec2 p) {
    return noise(p) * 0.55 + noise(p * 2.1) * 0.3 + noise(p * 4.3) * 0.15;
  }

  vec2 frame(vec2 uv, vec4 xf, vec2 fit) {
    vec2 c = (uv - 0.5) / max(xf.x, 0.0001) * fit + 0.5;
    return c + vec2(xf.y, xf.z);
  }
  // Radial blur toward an arbitrary centre. A camera push blurs toward the frame
  // centre; a focus pull blurs toward the point being focused ON — same optics, so
  // one function takes the centre rather than assuming it.
  vec3 blurToward(sampler2D tex, vec2 uv, vec2 centre, float amount) {
    if (amount < 0.001) return texture(tex, uv).rgb;
    vec3 sum = vec3(0.0); vec2 dir = (uv - centre) * amount;
    for (int i = 0; i < 8; i++) sum += texture(tex, uv - dir * (float(i) / 8.0)).rgb;
    return sum / 8.0;
  }
  // A heavy blur, 26 taps on a wide radius — soft colour rather than a blurry photo.
  // Serves both the letterbox backdrop and bloom's glow source.
  vec3 wide(sampler2D tex, vec2 uv, float radius) {
    vec3 sum = vec3(0.0);
    for (int i = -6; i <= 6; i++) {
      float o = float(i) * 0.012 * radius;
      sum += texture(tex, clamp(uv + vec2(o, o * 0.6), 0.0, 1.0)).rgb;
      sum += texture(tex, clamp(uv + vec2(-o, o * 0.6), 0.0, 1.0)).rgb;
    }
    return sum / 26.0;
  }

  // A beat is an ARRIVAL and a DEPARTURE, never a switch. Every effect added after the
  // first nine multiplies its WHOLE contribution by this, so it is exactly zero at
  // t = 0 and exactly zero once the shot's clock has run out. Without it an effect is
  // full strength on the shot's first frame and dies on its last one — which is what a
  // viewer sees as a jump when one beat hands over to the next.
  float beatEnv(float t, float dur) {
    float ramp = clamp(dur * 0.22, 0.12, 0.9);
    return smoothstep(0.0, ramp, t) * (1.0 - smoothstep(dur - ramp, dur, t));
  }

  // --- effects that move where we SAMPLE ------------------------------------------
  vec2 warp(vec2 uv, int fx, vec4 p, float t, float dur) {
    if (fx == 6) {                                    // breeze — the print hangs in air
      float amp = p.x * 0.03, speed = p.y, scale = p.z * 6.0, twist = p.w;
      vec2 drift = vec2(sin(t * speed + uv.y * scale), cos(t * speed * 0.8 + uv.x * scale));
      vec2 curl = vec2(-drift.y, drift.x) * twist;
      return uv + (drift + curl) * amp;
    }
    if (fx == 3) {                                    // weave — projector gate wander
      // 0.006 put this at HALF A PIXEL on a 1080p frame — a bed nobody could see, on
      // the effect used most often, which is most of why the film read as "two
      // effects". A projector gate really does wander a visible amount.
      float s = t * p.z;
      return uv + (vec2(noise(vec2(s, 0.0)), noise(vec2(0.0, s))) - 0.5) * p.x * 0.022;
    }
    if (fx == 20) {                                   // pixelate — the frame breaks up
      float ease = min(0.45, dur * 0.25);
      float env = smoothstep(0.0, ease, t) * (1.0 - smoothstep(dur - ease, dur, t));
      float span = max(p.z, 0.05) * dur;
      float k = clamp(t / span, 0.0, 1.0);
      float march = p.y < 0.5 ? 1.0 - k : k;          // resolve out of / crumble into
      float amt = env * march * march;
      float cell = amt * (0.012 + p.x * 0.13);
      if (cell < 0.0015) return uv;
      vec2 grid = cell * vec2(1.0 + p.w * 3.0, 1.0);
      return (floor(uv / grid) + 0.5) * grid;         // CELL CENTRES, so it never shifts
    }
    if (fx == 21) {                                   // zoom_punch — a snap on the beat
      float ease = min(0.4, dur * 0.25);
      float env = smoothstep(0.0, ease, t) * (1.0 - smoothstep(dur - ease, dur, t));
      float since = fract(t * max(p.x, 20.0) / 60.0 + p.z);
      float x = since / mix(0.03, 0.25, p.w);
      // attack-decay: exactly 0 ON the beat, 1 just after, back to 0 before the next.
      float hit = x * exp(1.0 - x) * (1.0 - smoothstep(0.7, 1.0, since)) * env;
      return (uv - 0.5) * (1.0 - hit * p.y * 0.16) + 0.5;
    }
    if (fx == 22) {                                   // shake — handheld, harder on beat
      float ease = min(0.4, dur * 0.25);
      float env = smoothstep(0.0, ease, t) * (1.0 - smoothstep(dur - ease, dur, t));
      float since = fract(t * max(p.z, 20.0) / 60.0);
      float x = since / 0.22;
      float beat = x * exp(1.0 - x) * (1.0 - smoothstep(0.7, 1.0, since));
      float accent = mix(1.0, beat, p.w);
      float sp = t * mix(2.0, 13.0, p.y);
      float amp = p.x * 0.04 * accent * env;
      vec2 jitter = (vec2(noise(vec2(sp, 11.3)), noise(vec2(31.7, sp))) - 0.5) * amp;
      float roll = (noise(vec2(sp * 0.61, 5.7)) - 0.5) * amp * 0.9;
      vec2 d = uv - 0.5;
      return 0.5 + vec2(d.x * cos(roll) - d.y * sin(roll),
                        d.x * sin(roll) + d.y * cos(roll)) + jitter;
    }
    if (fx == 23) {                                   // dolly_swing — push plus a sway
      float k = clamp(t / max(dur, 0.001), 0.0, 1.0);
      float arc = sin(3.1415927 * k);                 // in and back out, zero at both ends
      float n = floor(1.0 + p.w * 3.0);               // whole half-swings only
      float sway = sin(3.1415927 * n * k);
      float a = p.z * 6.2831853;
      vec2 lat = vec2(cos(a), sin(a)) * p.y * 0.035 * sway;
      return (uv - 0.5) * (1.0 - p.x * 0.12 * arc) + 0.5 + lat;
    }
    if (fx == 24) {                                   // mirror — a fold sweeps in
      float ramp = mix(0.5, 0.08, p.w) * dur;
      float on = smoothstep(0.0, ramp, t) * (1.0 - smoothstep(dur - ramp, dur, t));
      if (on < 0.002) return uv;
      // The AXIS travels in from the edge rather than the fold fading up: a fade passes
      // through a degenerate half-frame smear at 50%, a travelling axis never does.
      float keepFar = step(0.5, p.z);
      float ax = mix(mix(1.0, 0.5, p.x * on), mix(0.0, 0.5, p.x * on), keepFar);
      float ay = mix(mix(1.0, 0.5, p.y * on), mix(0.0, 0.5, p.y * on), keepFar);
      vec2 m = uv;
      if (keepFar < 0.5) {
        if (m.x > ax) m.x = 2.0 * ax - m.x;
        if (m.y > ay) m.y = 2.0 * ay - m.y;
      } else {
        if (m.x < ax) m.x = 2.0 * ax - m.x;
        if (m.y < ay) m.y = 2.0 * ay - m.y;
      }
      return m;
    }
    if (fx == 25) {                                   // kaleidoscope — a radial fold
      float ramp = mix(0.6, 0.1, p.w) * dur;
      float on = smoothstep(0.0, ramp, t) * (1.0 - smoothstep(dur - ramp, dur, t)) * p.y;
      if (on < 0.002) return uv;
      vec2 d = uv - 0.5;
      float r = min(length(d), 0.49);                 // clamped: the fold stays in frame
      // 2*PI/segments divides 2*PI exactly, so mod() is continuous across atan's seam.
      float w = 6.2831853 / floor(2.0 + p.x * 10.0);
      float a = mod(atan(d.y, d.x) + t * 0.12, w);
      a = min(a, w - a);
      float bite = smoothstep(p.z * 0.45, p.z * 0.45 + 0.14, length(d)) * on;
      return mix(uv, 0.5 + vec2(cos(a), sin(a)) * r, bite);
    }
    if (fx == 26) {                                   // tilt — a slow dutch roll
      float ramp = mix(0.5, 0.12, p.z) * dur;
      float on = smoothstep(0.0, ramp, t) * (1.0 - smoothstep(dur - ramp, dur, t));
      float ang = p.x * 0.20 * on * (1.0 + sin(t * 0.7) * p.w * 0.35);
      vec2 d = (uv - 0.5) * (1.0 - p.y * 0.10 * on);  // push in to cover the corners
      return 0.5 + vec2(d.x * cos(ang) - d.y * sin(ang),
                        d.x * sin(ang) + d.y * cos(ang));
    }
    if (fx == 27) {                                   // parallax — the near plane leads
      float k = clamp(t / max(dur, 0.001), 0.0, 1.0);
      float sway = sin(3.1415927 * floor(1.0 + p.w * 2.0) * k);
      float a = p.y * 6.2831853;
      float rad = 0.12 + p.z * 0.4;
      float near = 1.0 - smoothstep(rad, rad + 0.22, length(uv - 0.5));
      // The EDGES barely move (0.12), so the frame boundary stays put and no void opens.
      return uv + vec2(cos(a), sin(a)) * p.x * 0.045 * sway * mix(0.12, 1.0, near);
    }
    if (fx == 28) {                                   // film_burn — gate jitter and hop
      float e = beatEnv(t, dur);
      float sp = t * 2.4;                             // a projector runs at ONE speed
      vec2 wander = vec2(noise(vec2(sp * 0.8, 4.2)) - 0.5,
                         noise(vec2(9.1, sp)) - 0.5) * vec2(0.45, 1.0);
      // Now and then the sprocket loses its hold: a fast hop that settles back. The cell
      // is entered and left at exactly zero, so the hop has no edge to snap on.
      float cell = floor(sp), into = fract(sp);
      float hop = step(0.62, hash(vec2(cell, 7.3)))
                * pow(max(sin(3.14159265 * clamp(into * 2.2, 0.0, 1.0)), 0.0), 6.0);
      return uv + (wander * 0.05 + vec2(0.0, hop * 0.055)) * p.x * e;
    }
    if (fx == 29) {                                   // vhs — tape tracking
      float e = beatEnv(t, dur) * p.x;
      // A tape head does not hold a line, so the picture SHEARS: every scanline is
      // displaced by its own wander rather than the frame sliding as one piece.
      float line = floor(uv.y * 240.0) / 240.0;
      float wob = (noise(vec2(line * 12.0, t * 2.1)) - 0.5) * 0.65
                + (noise(vec2(line * 90.0, t * 9.0)) - 0.5) * 0.35;
      float head = smoothstep(0.13, 0.0, uv.y) * (noise(vec2(t * 6.0, 3.0)) - 0.5) * 2.4;
      float creep = (noise(vec2(t * 0.7, 12.0)) - 0.5) * 0.008;
      return uv + vec2(wob * 0.014 + head * 0.02, creep) * e;
    }
    if (fx == 30) {                                   // crt — the tube's own geometry
      float e = beatEnv(t, dur);
      vec2 q = uv * 2.0 - 1.0;
      // Bent INWARD. Pushing outward opens black corners; dividing pulls every screen
      // pixel onto a point already inside the picture, so the bulge costs a hair off
      // the edge instead of a hole at the corner.
      q /= 1.0 + p.x * 0.15 * e * dot(q, q);
      return q * 0.5 + 0.5 + vec2(0.0, sin(t * 0.9) * 0.0015 * p.z * e);
    }
    if (fx == 32) {                                   // paper — the print curls
      float e = beatEnv(t, dur);
      float band = 0.30;                              // the right third is what lifts
      float u = clamp((uv.x - (1.0 - band)) / band, 0.0, 1.0);
      float k = p.x * 0.6 * e * mix(1.0, 0.55, uv.y);
      // Foreshortening WITHOUT a hole: the far edge stays pinned to the frame and the
      // picture crowds toward it, so a curling print never uncovers what is behind it.
      float bend = (u * u - u) * smoothstep(0.0, 0.35, u);
      float bow = k * u * u * 0.02 * sin(3.14159265 * uv.y);
      return uv + vec2(band * bend * k, bow);
    }
    if (fx == 34) {                                   // freeze_ripple — one wave, once
      // The wave is given a span that ALWAYS fits inside the shot, then released at
      // p.w of whatever room is left, so it can never be half-way across at the cut.
      float span = clamp(dur * 0.28, 0.35, 1.2);
      float begin = p.w * max(dur - span, 0.0);
      float k = clamp((t - begin) / span, 0.0, 1.0);
      vec2 centre = vec2(p.x, 1.0 - p.y);             // y is authored top-down
      vec2 d = uv - centre;
      float r = length(d * vec2(1.0, 0.62));          // distance in frame WIDTHS
      float front = k * 1.25;                         // the crest travels past the corners
      float dr = r - front;
      float wave = sin(dr * 31.4159) * exp(-abs(dr) * 16.0);
      float fade = sin(k * 3.14159);                  // zero at release AND at the end
      return uv + d / max(length(d), 0.0001) * wave * fade * p.z * 0.07;
    }
    if (fx == 7) {                                    // glitch — blocks and slip
      float window = max(p.w * dur, 0.001);
      if (t > window) return uv;
      float fall = 1.0 - t / window;                  // fiercest at the cut, then gone
      float burst = step(0.55, hash(vec2(floor(t * p.z), 3.7))) * fall;
      float grid = mix(220.0, 14.0, p.y);
      vec2 blocky = floor(uv * grid) / grid;
      float slip = (hash(vec2(floor(uv.y * 24.0), floor(t * p.z))) - 0.5) * p.x * 0.12;
      return mix(uv, blocky + vec2(slip, 0.0), burst);
    }
    return uv;
  }

  // A warp at partial weight is the SAME warp, moved less far. Blending the effect
  // rather than switching it is what lets one beat hand over to the next without a pop.
  vec2 warpW(vec2 uv, int fx, vec4 p, float w, float t, float dur) {
    return w < 0.001 ? uv : uv + (warp(uv, fx, p, t, dur) - uv) * w;
  }

  float extraBlur(int fx, vec4 p, float t, float dur) {
    if (fx == 1) {                                    // rack_focus
      float settle = max(p.w * dur, 0.001);
      return p.z * (1.0 - smoothstep(0.0, settle, t)) * 0.35;
    }
    return 0.0;
  }
  vec2 focusPoint(int fx, vec4 p) {
    return fx == 1 ? vec2(p.x, 1.0 - p.y) : vec2(0.5);   // y is authored top-down
  }

  float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

  // Hue work needs a round trip. It is not bit-exact — note the epsilons — so every
  // effect below blends FROM the original colour rather than returning the round trip
  // directly. At zero strength the mix collapses to the original and the conversion
  // error cannot survive.
  vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 a = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 b = mix(vec4(a.xyw, c.r), vec4(c.r, a.yzx), step(a.x, c.r));
    float d = b.x - min(b.w, b.y);
    return vec3(abs(b.z + (b.w - b.y) / (6.0 * d + 1e-10)), d / (b.x + 1e-10), b.x);
  }
  vec3 hsv2rgb(vec3 c) {
    vec3 m = abs(fract(c.xxx + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(m - 1.0, 0.0, 1.0), c.y);
  }

  // --- effects that change the COLOUR ---------------------------------------------
  vec3 tint(float warmth) {
    return warmth >= 0.0 ? mix(vec3(1.0), vec3(1.25, 1.02, 0.72), warmth)
                         : mix(vec3(1.0), vec3(0.72, 0.94, 1.3), -warmth);
  }
  vec3 colourFx(vec3 c, sampler2D tex, vec2 sampled, vec2 uv, int fx, vec4 p,
                float t, float dur) {
    if (fx == 2) {                                    // bloom — halation
      vec3 soft = wide(tex, sampled, 0.4 + p.w * 1.6);
      float lum = dot(soft, vec3(0.299, 0.587, 0.114));
      vec3 glow = soft * smoothstep(p.x, min(p.x + 0.25, 1.0), lum);
      return c + glow * p.y * 1.4 * tint(p.z);
    }
    if (fx == 4) {                                    // bokeh — drifting motes
      float count = 6.0 + floor(p.x * 22.0);
      float radius = 0.02 + p.y * 0.07;
      vec3 motes = vec3(0.0);
      for (int i = 0; i < 28; i++) {
        if (float(i) >= count) break;
        vec2 seed = vec2(float(i) * 1.37, float(i) * 2.71);
        vec2 at = vec2(hash(seed), hash(seed + 5.1));
        at += vec2(sin(t * 0.15 + at.y * 6.0), t * 0.02 + cos(t * 0.11 + at.x * 6.0))
              * p.z * 0.35;
        at = fract(at);
        float d = length((uv - at) * vec2(1.0, 0.62));
        motes += vec3(1.0 - smoothstep(radius * 0.35, radius, d))
                 * (0.35 + hash(seed) * 0.4);
      }
      return c + motes * 0.35 * tint(p.w);
    }
    if (fx == 5) {                                    // pulse — breathe on the beat
      float beats = t * max(p.x, 1.0) / 60.0 + p.z;
      float pulse = pow(max(0.0, sin(beats * 6.2831)), 6.0);
      float d = distance(uv, vec2(0.5));
      return c * (1.0 + pulse * p.y * 0.45)
               * (1.0 - pulse * p.w * 0.35 * smoothstep(0.25, 0.9, d));
    }
    if (fx == 9) {                                    // anamorphic_flare — blue streak
      float env = beatEnv(t, dur);
      if (env <= 0.0) return c;
      float taps = 4.0 + floor(p.w * 8.0);
      float reach = 0.05 + p.w * 0.30;
      float hi = min(p.x + 0.18, 1.0);
      vec3 streak = vec3(0.0);
      float total = 0.0;
      for (int i = 1; i <= 12; i++) {
        float f = float(i);
        if (f > taps) break;
        float k = f / taps;
        vec2 off = vec2(k * reach, 0.0);
        float fade = (1.0 - k) * (1.0 - k);
        vec3 l = texture(tex, clamp(sampled + off, 0.0, 1.0)).rgb;
        vec3 r = texture(tex, clamp(sampled - off, 0.0, 1.0)).rgb;
        streak += l * smoothstep(p.x, hi, dot(l, vec3(0.299, 0.587, 0.114))) * fade;
        streak += r * smoothstep(p.x, hi, dot(r, vec3(0.299, 0.587, 0.114))) * fade;
        total += 2.0 * fade;
      }
      streak /= max(total, 0.0001);
      vec3 hue = mix(vec3(0.45, 0.72, 1.7), vec3(1.6, 1.0, 0.55), p.z);
      return c + streak * p.y * 2.4 * hue * env;
    }
    if (fx == 10) {                                   // god_rays — shafts from one light
      float env = beatEnv(t, dur);
      if (env <= 0.0) return c;
      vec2 src = vec2(p.x, 1.0 - p.y);                // y authored top-down
      vec2 march = (sampled - src) * (0.9 / 18.0);
      vec2 walk = sampled;
      vec3 shaft = vec3(0.0);
      float decay = 1.0, total = 0.0;
      for (int i = 0; i < 18; i++) {
        if (decay < 0.06) break;
        walk -= march;
        vec3 sm = texture(tex, clamp(walk, 0.0, 1.0)).rgb;
        shaft += sm * smoothstep(0.58, 0.92, dot(sm, vec3(0.299, 0.587, 0.114))) * decay;
        total += decay;
        decay *= 0.8;
      }
      shaft /= max(total, 0.0001);
      float fall = 1.0 - smoothstep(0.08, 1.05, length(sampled - src));
      float pump = 1.0;
      if (p.w >= 1.0) {
        pump = 0.5 + 0.5 * pow(max(0.0, sin(t * p.w / 60.0 * 6.2831)), 4.0);
      }
      return c + shaft * fall * pump * p.z * 2.0 * vec3(1.12, 1.04, 0.88) * env;
    }
    if (fx == 11) {                                   // prism — the lens splits at edges
      float env = beatEnv(t, dur);
      if (env <= 0.0) return c;
      vec2 radial = uv - vec2(0.5);
      float edge = smoothstep(p.y, 1.0, length(radial) * 1.42);
      vec2 axis = normalize(radial + vec2(0.0001, 0.00013));
      axis = normalize(mix(axis, vec2(-axis.y, axis.x), p.z));
      float breathe = mix(1.0, 0.55 + 0.45 * cos(t * 1.9), p.w);
      vec2 off = axis * p.x * 0.028 * edge * breathe * env;
      // A DIFFERENCE, not a replacement: the colour may already carry another
      // beat's glow, and replacing two channels would throw that away.
      vec3 base = texture(tex, sampled).rgb;
      vec3 split = vec3(texture(tex, clamp(sampled + off, 0.0, 1.0)).r,
                        base.g,
                        texture(tex, clamp(sampled - off, 0.0, 1.0)).b);
      return c + (split - base);
    }
    if (fx == 12) {                                   // soft_focus — a stocking on the lens
      float env = beatEnv(t, dur);
      if (env <= 0.0) return c;
      vec3 soft = clamp(wide(tex, sampled, 0.6 + p.z * 2.2) * tint(p.w), 0.0, 1.0);
      vec3 hazed = mix(c, soft, p.x * 0.6);
      // Screened back over it: screen lifts the DARKS hardest, which is the whole
      // difference from bloom — the glow bleeds into shadow instead of off the lamps.
      vec3 veil = 1.0 - (1.0 - hazed) * (1.0 - soft * p.y * 0.55);
      return mix(c, veil, env);
    }
    if (fx == 13) {                                   // spotlight — the room goes dark
      float env = beatEnv(t, dur);
      if (env <= 0.0) return c;
      // The pool WANDERS, from exactly where it was placed: both sines are zero at t=0.
      vec2 at = vec2(p.x, 1.0 - p.y) + vec2(sin(t * 0.23), sin(t * 0.37)) * 0.05;
      float r = max(p.z, 0.05);
      float d = length((uv - at) / vec2(r, r * 0.72));
      float pool = 1.0 - smoothstep(0.55, 1.3, d);
      vec3 lit = c * (1.0 - p.w * 0.85 * (1.0 - pool)) * (1.0 + pool * p.w * 0.16);
      return mix(c, lit, env);
    }
    if (fx == 14) {                                   // duotone — two inks over luminance
      float k = beatEnv(t, dur);
      float g = clamp(luma(c), 0.0, 1.0);             // a bloom beat under us overshoots
      vec3 dark  = hsv2rgb(vec3(p.x, p.w * 0.85, 0.12 + g * 0.30));
      vec3 light = hsv2rgb(vec3(p.y, p.w * 0.55, 0.55 + g * 0.45));
      return mix(c, mix(dark, light, smoothstep(0.15, 0.85, g)), p.z * 0.85 * k);
    }
    if (fx == 15) {                                   // bleach_bypass — silver retention
      float k = beatEnv(t, dur);
      vec3 b = clamp(c, 0.0, 1.0);
      float g = luma(b);
      vec3 grey = vec3(g);
      // The picture overlaid on its OWN luminance: the silver never bleached away.
      vec3 hard = mix(2.0 * b * grey, 1.0 - 2.0 * (1.0 - b) * (1.0 - grey), step(0.5, g));
      vec3 silver = mix(b, hard, p.z);
      silver = mix(silver, vec3(luma(silver)), p.y);
      silver = clamp((silver - p.w * 0.12) / (1.0 - p.w * 0.12), 0.0, 1.0);
      return mix(c, silver, p.x * k);
    }
    if (fx == 16) {                                   // cross_process — wrong chemistry
      float k = beatEnv(t, dur);
      vec3 b = clamp(c, 0.0, 1.0);
      float g = luma(b);
      vec3 sc = mix(b, b * b * (3.0 - 2.0 * b), p.y);
      // The two ends pulled in OPPOSITE directions — cyan into the shadows, yellow into
      // the highlights. That split IS cross processing; the rest is contrast.
      sc += (vec3(-0.07, 0.02, 0.11) * (1.0 - smoothstep(0.0, 0.55, g))
           + vec3(0.11, 0.06, -0.10) * smoothstep(0.45, 1.0, g)) * p.z;
      sc = mix(vec3(luma(sc)), sc, 1.0 + p.w);
      return mix(c, clamp(sc, 0.0, 1.0), p.x * k);
    }
    if (fx == 17) {                                   // infrared — the Wood effect
      float k = beatEnv(t, dur);
      vec3 h = rgb2hsv(c);
      // Green is the only proxy for near-infrared this shader has, so foliage is found
      // by hue and trusted more the more saturated it is. Both windows sit far from the
      // red seam at 0/1, so neither needs to wrap.
      float leaf = (1.0 - smoothstep(0.0, 0.16, abs(h.x - 0.30)))
                 * smoothstep(0.08, 0.30, h.y);
      float sky  = (1.0 - smoothstep(0.0, 0.14, abs(h.x - 0.60)))
                 * smoothstep(0.08, 0.30, h.y);
      float ir = clamp(luma(c) + leaf * p.y * 1.1 - sky * p.z * 0.75, 0.0, 1.0);
      return mix(c, mix(vec3(ir), hsv2rgb(vec3(p.w, 0.55, ir)), leaf * 0.6), p.x * k);
    }
    if (fx == 18) {                                   // colour_isolate — one hue survives
      float k = beatEnv(t, dur);
      vec3 h = rgb2hsv(c);
      float d = abs(h.x - p.x);
      d = min(d, 1.0 - d);                            // hue is a circle: 0.98 sits by 0.02
      // Gated on SATURATION as well as hue: without it, skin and concrete drift in and
      // out of the band as they darken and the kept colour crawls across the frame.
      float keep = (1.0 - smoothstep(p.y * 0.5, p.y * 0.5 + 0.07, d))
                 * smoothstep(0.05, 0.20, h.y);
      vec3 lit = mix(c, hsv2rgb(vec3(h.x, min(h.y * (1.0 + p.w), 1.0), h.z)), p.w * k);
      return mix(mix(c, vec3(luma(c)), p.z * k), lit, keep);
    }
    if (fx == 19) {                                   // flash — one blowout, then recovery
      // NO envelope: this is an EVENT, and it is already identity on both sides. A 50 ms
      // ignition rather than a raw step, so the peak is the same height at 30 and at 60
      // fps, then an exponential recovery that is spent long before the beat ends.
      float since = t - p.x * dur;
      float burst = smoothstep(-0.05, 0.0, since) * exp(-max(since, 0.0) / max(p.z, 0.05));
      return mix(c, tint(p.w) * 1.15, clamp(burst * p.y, 0.0, 1.0));
    }
    if (fx == 28) {                                   // film_burn — lamp, dust, scorch
      float e = beatEnv(t, dur);
      float lamp = (noise(vec2(t * 11.0, 1.7)) - 0.5) * 1.3
                 + (noise(vec2(t * 2.3, 8.4)) - 0.5) * 0.7;
      c *= 1.0 + lamp * p.y * 0.45 * e;
      // Dust is redrawn on a 16 Hz FILM clock, not per display frame: at 60 Hz it stops
      // reading as dirt on a print and becomes a shimmer.
      float fr = floor(t * 16.0);
      vec2 g = uv * vec2(96.0, 54.0);
      vec2 gi = floor(g), gf = fract(g) - 0.5;
      float speck = step(1.0 - p.z * 0.10, hash(gi + fr * 3.31))
                  * (1.0 - smoothstep(0.10, 0.40, length(gf)));
      float sgn = step(0.5, hash(gi + fr * 7.77)) * 2.0 - 1.0;   // dust AND scratches
      c += vec3(speck * sgn * 0.5 * e);
      float hc = floor(t * 1.5), hi = fract(t * 1.5);
      float hair = (1.0 - smoothstep(0.0008, 0.0035, abs(uv.x - hash(vec2(hc, 21.0)))))
                 * step(0.6, hash(vec2(hc, 4.0))) * sin(3.14159265 * hi);
      c -= vec3(hair * 0.30 * p.z * e);
      float b = p.w * e;
      float rr = length((uv - 0.5) * vec2(1.0, 1.15));
      float edge = 0.52 + fbm(uv * 2.6 + vec2(t * 0.05, 0.0)) * 0.20;
      float scorch = smoothstep(edge - 0.18, edge + 0.04, rr);
      c = mix(c, c * vec3(0.40, 0.22, 0.10), scorch * b * 0.9);
      c += vec3(1.0, 0.42, 0.10) * scorch * (1.0 - scorch) * 2.2 * b;
      return c;
    }
    if (fx == 29) {                                   // vhs — lines, hiss, dropouts
      float e = beatEnv(t, dur);
      // Scanlines belong to the DISPLAY, not the photograph, so they are screen space
      // and must not travel with the picture when the camera pushes.
      float lines = 0.5 + 0.5 * cos(uv.y * 900.0);
      c *= 1.0 - lines * p.z * 0.35 * e;
      float hiss = hash(vec2(floor(uv.y * 480.0), floor(t * 30.0))) - 0.5;
      float snow = hash(uv * vec2(700.0, 400.0) + floor(t * 30.0)) - 0.5;
      float drop = smoothstep(0.88, 1.0, noise(vec2(uv.y * 40.0 - t * 3.0, t * 0.7)));
      c += vec3(hiss * 0.10 + snow * 0.10 + drop * 0.30) * p.w * e;
      // Chroma carries a fraction of luma's bandwidth, so colour goes soft and grey
      // while brightness stays sharp.
      c = mix(c, vec3(luma(c)), 0.20 * p.y * e);
      return c;
    }
    if (fx == 30) {                                   // crt — phosphors and a roll bar
      float e = beatEnv(t, dur);
      float col = fract(uv.x * 320.0);
      vec3 stripe = vec3(1.0 - smoothstep(0.0, 0.42, abs(col - 0.167)),
                         1.0 - smoothstep(0.0, 0.42, abs(col - 0.500)),
                         1.0 - smoothstep(0.0, 0.42, abs(col - 0.833)));
      c *= mix(vec3(1.0), 0.35 + stripe * 1.4, p.y * e);
      c *= 1.0 - (0.5 + 0.5 * cos(uv.y * 1100.0)) * p.y * 0.25 * e;
      c += c * c * p.w * 0.9 * e;                     // phosphor bleed
      // The roll bar, periodic by construction so it wraps with no seam.
      float d = abs(fract(uv.y - t * 0.13) - 0.5);
      c *= 1.0 + (1.0 - smoothstep(0.0, 0.17, d)) * p.z * 0.20 * e;
      vec2 gg = abs(uv * 2.0 - 1.0);
      float corner = length(max(gg - vec2(0.80, 0.76), vec2(0.0)));
      c *= 1.0 - smoothstep(0.06, 0.22, corner) * p.x * e;
      return c;
    }
    if (fx == 31) {                                   // halftone — the picture as print
      float e = beatEnv(t, dur);
      float ruling = mix(190.0, 42.0, p.x);           // p.x is DOT SIZE: high = coarse
      // Locked to the CANVAS, not the photograph: a screen that scaled with the camera
      // push would crawl and moire.
      vec2 sc = uv * vec2(1.0, 0.62) * ruling;
      float a = p.y * 1.5707963;
      vec2 rr = vec2(sc.x * cos(a) - sc.y * sin(a), sc.x * sin(a) + sc.y * cos(a));
      vec2 cell = fract(rr) - 0.5;
      float lm = luma(c);
      // sqrt, because ink coverage goes as the dot's AREA.
      float radius = sqrt(clamp(1.0 - lm, 0.0, 1.0)) * 0.70;
      float disc = 1.0 - smoothstep(max(radius - 0.07, 0.0), radius + 0.07, length(cell));
      disc *= smoothstep(0.0, 0.06, radius);          // a white cell prints NOTHING
      vec3 ink = mix(vec3(0.05, 0.05, 0.06), c * 0.9, p.w);
      vec3 paper = vec3(0.96, 0.95, 0.92);
      return mix(c, mix(paper, ink, disc), p.z * e);
    }
    if (fx == 32) {                                   // paper — stock, tear, curl shade
      float e = beatEnv(t, dur);
      vec3 stock  = mix(vec3(0.94, 0.92, 0.87), vec3(0.78, 0.68, 0.50), p.w);
      vec3 scorch = mix(vec3(0.52, 0.40, 0.27), vec3(0.26, 0.15, 0.07), p.w);
      vec3 print = c;
      // Fibre, and it does NOT crawl: grain is a property of the sheet, where the
      // film's own grain moves every frame. That difference is the whole tell.
      float fibre = fbm(uv * vec2(210.0, 190.0)) - 0.5;
      float laid  = fbm(uv * vec2(6.0, 320.0)) - 0.5;
      print *= 1.0 + (fibre * 0.7 + laid * 0.3) * p.z * 0.5;
      print = mix(print, stock, p.z * 0.10);
      // The border is the PRINT'S border, not the canvas: the sample point is
      // clamped, so its distance to the edge is zero outside the photograph and
      // rises inside it — the tear follows the photo through the letterbox and
      // the bars become the paper it lies on.
      float dEdge = min(min(sampled.x, 1.0 - sampled.x),
                        min(sampled.y, 1.0 - sampled.y));
      float along = sampled.x * 3.0 + sampled.y * 5.0;
      float rag = fbm(vec2(along * 3.0, 1.7)) * p.y * 0.045;
      float inside = smoothstep(rag, rag + 0.028, dEdge);
      print = mix(stock, print, inside);              // past the tear there is only paper
      print = mix(print, scorch, inside * (1.0 - inside) * 4.0 * p.y * 0.8);
      float u = clamp((uv.x - 0.70) / 0.30, 0.0, 1.0);
      print *= 1.0 - u * u * p.x * 0.5;
      print += vec3(1.0, 0.98, 0.94)
             * (1.0 - smoothstep(0.0, 0.16, abs(u - 0.62))) * p.x * 0.16;
      return mix(c, print, e);
    }
    if (fx == 33) {                                   // strobe_cut — struck on the beat
      float bps = max(p.x, 30.0) / 60.0;              // p.x is the BRIEF's tempo
      float period = 1.0 / bps;
      // The grid is ANCHORED at the first hit, not at the shot's start, so a shot that
      // begins off-beat still flashes with the music instead of against it.
      float begin = p.w * dur;
      float since = t - begin;
      // Silent before that first hit and faded across the last beat, so a cut can never
      // catch the frame mid-flash.
      float armed = step(0.0, since) * (1.0 - smoothstep(dur - period, dur, t));
      float frames = clamp(p.y, 1.0, 4.0);
      float cellw = period * 0.055;
      float within = fract(since * bps) * period;
      float slot = floor(within / cellw);
      float lit = step(slot, frames - 0.5) * step(fract(within / cellw), 0.5);
      float flash = lit * armed * (1.0 - slot / (frames + 1.0));   // each hit weaker
      return c + (mix(c, vec3(1.0), 0.4) * 1.7 + 0.1) * flash * p.z;
    }
    if (fx == 35) {                                   // echo — a decaying ghost trail
      float ramp = smoothstep(0.0, 0.5, t);           // never full on frame one
      float a = p.z * 6.28319;
      vec2 slide = vec2(cos(a), sin(a)) * p.x * 0.055;
      float push = p.w * p.x * 0.05;                  // the radial half of the trail
      vec3 trail = vec3(0.0);
      float total = 0.0;
      for (int i = 1; i <= 6; i++) {
        float f = float(i);
        float fall = pow(clamp(1.0 - p.y, 0.05, 1.0), f);
        if (fall < 0.01) break;
        vec2 at = 0.5 + (sampled - 0.5) * (1.0 + push * f) + slide * f;
        trail += texture(tex, clamp(at, 0.0, 1.0)).rgb * fall;
        total += fall;
      }
      if (total < 0.0001) return c;
      // MAX, not mix: only the lights trail, so the picture underneath stays readable
      // and it reads as video feedback rather than a smeared photograph.
      return mix(c, max(c, trail / total), ramp * 0.8);
    }
    if (fx == 8) {                                    // light_leak — a warm sweep
      float a = p.x * 6.2831;
      vec2 axis = vec2(cos(a), sin(a));
      float travel = fract(t / max(dur, 0.001)) * 2.0 - 0.5;
      float band = 1.0 - smoothstep(0.0, p.z, abs(dot(uv - 0.5, axis) - travel + 0.5));
      return c + band * p.y * 0.6 * tint(p.w);
    }
    return c;
  }

  // The whole photo, letterboxed over a blurred copy of itself. Cropping to FILL the
  // frame is what cut people's heads off: a 4:3 photo on a 21:9 canvas loses a third
  // of its height, and the camera push then took more. Nothing is cropped now.
  //
  // Two effects run over it, A then B, each at its own weight. They COMPOSE: A's warp
  // feeds B's, and B's colour pass sees A's output — so "focus pull settling into a
  // breeze" is one continuous thing rather than two filters fighting.
  vec3 layer(sampler2D tex, vec2 uv, vec4 xf, vec2 fit, vec2 fill,
             ivec2 fx, vec4 pa, vec4 pb, vec2 w, vec2 clock) {
    float t = clock.x, dur = clock.y;
    vec2 moved = warpW(warpW(uv, fx.x, pa, w.x, t, dur), fx.y, pb, w.y, t, dur);
    vec2 c = frame(moved, xf, fit);
    // The camera's motion blur, both focus pulls, and a whip through the cut all land
    // on the same radial sample — optically they are the same thing.
    float amount = xf.w + uWhip
                 + extraBlur(fx.x, pa, t, dur) * w.x
                 + extraBlur(fx.y, pb, t, dur) * w.y;
    // Whichever slot is actually pulling focus owns the centre; failing that, the frame.
    vec2 centre = fx.x == 1 ? focusPoint(fx.x, pa)
                : fx.y == 1 ? focusPoint(fx.y, pb) : vec2(0.5);
    vec3 sharp = blurToward(tex, clamp(c, 0.0, 1.0), centre, amount);
    // Weave's colour fringing needs the texture, so it is applied here rather than in
    // colourFx. Either slot may be carrying it.
    float weaveW = (fx.x == 3 ? w.x : 0.0) + (fx.y == 3 ? w.y : 0.0);
    vec4 weaveP = fx.x == 3 ? pa : pb;
    if (weaveW > 0.001) {
      vec2 off = (c - 0.5) * weaveP.y * 0.024 * weaveW;
      sharp.r = blurToward(tex, clamp(c + off, 0.0, 1.0), vec2(0.5), amount).r;
      sharp.b = blurToward(tex, clamp(c - off, 0.0, 1.0), vec2(0.5), amount).b;
      sharp *= 1.0 + (noise(vec2(t * weaveP.z * 3.0, 0.0)) - 0.5) * weaveP.w * 0.25 * weaveW;
    }
    // VHS records colour at a fraction of luma's bandwidth AND the chroma lags, so it
    // smears sideways off every edge. That needs the source texture, so it lives here
    // beside weave's fringing. Either slot may carry it; if both do, the weights add
    // and slot A's parameters win, exactly as weave behaves.
    float vhsW = (fx.x == 29 ? w.x : 0.0) + (fx.y == 29 ? w.y : 0.0);
    if (vhsW > 0.001) {
      vec4 vp = fx.x == 29 ? pa : pb;
      float bleed = beatEnv(t, dur) * vhsW * vp.y;
      float lag = bleed * 0.014;
      float taps = 1.0 + floor(vp.y * 3.0 + 0.5);
      vec3 smear = vec3(0.0);
      for (int i = 0; i < 4; i++) {
        if (float(i) >= taps) break;
        smear += texture(tex, clamp(c + vec2(lag * float(i), 0.0), 0.0, 1.0)).rgb;
      }
      smear /= taps;
      float kv = clamp(bleed * 2.5, 0.0, 1.0);
      sharp.r = mix(sharp.r, smear.r, kv);
      sharp.b = mix(sharp.b, smear.b, kv);
    }
    vec3 back = wide(tex, frame(moved, xf, fill), 1.0) * 0.45;
    float inside = step(0.0, c.x) * step(c.x, 1.0) * step(0.0, c.y) * step(c.y, 1.0);
    vec3 colour = mix(back, sharp, inside);
    vec2 sampled = clamp(c, 0.0, 1.0);
    colour = mix(colour, colourFx(colour, tex, sampled, uv, fx.x, pa, t, dur), w.x);
    colour = mix(colour, colourFx(colour, tex, sampled, uv, fx.y, pb, t, dur), w.y);
    return colour;
  }

  vec3 grade(vec3 c) {
    c = clamp(c + uLift, 0.0, 1.0);
    c = pow(c, 1.0 / max(uGamma, vec3(0.0001)));
    return clamp(c * uGain, 0.0, 1.0);
  }

  void main() {
    // A SLIDE moves both pictures instead of blending them: the outgoing one is
    // pushed out of frame while the incoming one arrives behind it, the way frames on
    // a strip move together. Sampling each layer at a shifted coordinate is all it
    // takes — no extra pass, and every effect on both layers keeps running while they
    // travel.
    vec2 prevUv = vUv + uSlide * uMix;
    vec2 nextUv = vUv - uSlide * (1.0 - uMix);

    // Two numbers every mode below shares, and both are continuity devices.
    //
    // The eased mix, k. Easing is not taste: smoothstep's DERIVATIVE is zero at 0
    // and at 1, so a mode that moves a picture is not merely in the right place when
    // the transition ends — it has already stopped moving.
    //
    // bell is exactly zero at both ends. Every mid-transition flourish below — a
    // tear, a curl, a rim light, a cast shadow — is multiplied by it, which makes it
    // ARITHMETICALLY impossible for one to survive the join. That is the dip bug above,
    // prevented by construction rather than by remembering it.
    float k = smoothstep(0.0, 1.0, uMix);
    float bell = sin(uMix * 3.14159265);

    // Anything that MOVES a picture moves the point it is SAMPLED at, exactly as the
    // slide does, so each layer is still built once and every effect on it keeps
    // running while it travels. Where a coordinate leaves [0,1] the blurred backdrop
    // that layer() already produces fills it — no geometry mode reveals emptiness.
    vec2 axis = uAxis / max(length(uAxis), 0.0001);   // zero stays zero, and no NaN
    float foldFlat = 1.0, foldStrip = 0.0, foldLift = 0.0, foldShade = 0.0;
    float shardMask = 0.0, shardFall = 0.0;
    if (uMode == 7) {                    // zoom_blur — one hard push through the cut
      prevUv = 0.5 + (vUv - 0.5) / (1.0 + 0.22 * k);
      nextUv = 0.5 + (vUv - 0.5) / (1.0 + 0.22 * (1.0 - k));
    } else if (uMode == 12) {            // match_zoom — one move, carried by two photos
      // Both frames magnify by the same FACTOR PER UNIT OF MIX, so through the
      // crossover they travel at identical relative speed and the eye reads a single
      // camera rather than a change of picture.
      prevUv = 0.5 + (vUv - 0.5) * pow(1.7, -k);
      nextUv = 0.5 + (vUv - 0.5) * pow(1.7, 1.0 - k);
    } else if (uMode == 8) {             // glitch_cut — block tearing across the join
      float bite = bell * bell;
      float row = floor(vUv.y * 18.0);
      float roll = floor(uMix * 6.0);    // re-rolled a few times, not every frame
      float torn = step(0.55, hash(vec2(row, roll)));
      float slip = (hash(vec2(roll, row) + 1.7) - 0.5) * 0.30 * bite * torn;
      vec2 shifted = mix(vUv, floor(vUv * 44.0) / 44.0, bite * 0.6) + vec2(slip, 0.0);
      prevUv = shifted; nextUv = shifted;
    } else if (uMode == 10) {            // page_turn — the outgoing sheet peels away
      float u = dot(vUv - 0.5, axis);    // distance along the peel; 0.71 at the very most
      // The fold starts PAST the free edge and finishes PAST the far one: at uMix = 0
      // no pixel is behind it, at uMix = 1 every pixel is.
      float f = mix(-0.78, 0.78, k);
      float w = 0.18 * bell + 0.0005;    // the curl, closing to nothing at both ends
      foldFlat = step(f, u);                            // still lying flat
      foldStrip = step(f - w, u) * (1.0 - foldFlat);    // the curl itself
      foldLift = clamp(1.0 - (f - u) / w, 0.0, 1.0);    // 1 at the crease, 0 at the edge
      foldShade = (1.0 - foldFlat) * (1.0 - foldStrip) * 0.55 * bell
                * (1.0 - smoothstep(0.0, 0.13, (f - w) - u));
      // A sheet folded along u = f lays the material from f + a over f - a, so the curl
      // samples its own MIRROR about the fold line. Nothing outside the curl moves.
      prevUv = vUv + axis * (2.0 * (f - u)) * foldStrip;
    } else if (uMode == 11) {            // shatter — the outgoing frame lets go in cells
      // A cell only falls DOWN, so the one covering this fragment started in this
      // column at this row or one above: walk up and take the first that lands here.
      // SEARCHING for the tile is what makes whole tiles move — a fixed grid with a
      // shifted sample slides content around inside a stationary hole instead.
      float cx = floor(vUv.x * 9.0);
      for (int i = 0; i < 8; i++) {
        float row = floor(vUv.y * 6.0) + float(i);
        if (row > 5.0) break;
        float wait = hash(vec2(cx, row) + 0.5) * 0.35;
        float go = clamp((uMix - wait) / max(1.0 - wait, 0.0001), 0.0, 1.0);
        float drop = go * go * 1.7;                      // gravity, in frame heights
        float slip = (hash(vec2(row, cx) + 3.3) - 0.5) * 0.22 * go * go;
        if (vUv.y >= row / 6.0 - drop && vUv.y < (row + 1.0) / 6.0 - drop
            && vUv.x >= cx / 9.0 + slip && vUv.x < (cx + 1.0) / 9.0 + slip) {
          shardMask = 1.0; shardFall = drop;
          prevUv = vUv + vec2(-slip, drop);   // the cell carries its OWN piece of photo
          break;
        }
      }
    }

    vec3 prev = layer(uPrev, prevUv, uPrevXf, uPrevFit, uPrevFill,
                      uPrevFx, uPrevPA, uPrevPB, uPrevW, uPrevClock);
    vec3 next = layer(uNext, nextUv, uNextXf, uNextFit, uNextFill,
                      uNextFx, uNextPA, uNextPB, uNextW, uNextClock);
    vec3 c;
    if (uMode == 1) {                       // cut — no blend at all
      c = uMix < 0.5 ? prev : next;
    } else if (uMode == 2) {                // dip to black, out then in
      // FULL at both ends, black at the midpoint. The previous form was inverted:
      // uMix * 2 starts at ZERO, so a dip opened black, peaked in the middle, and
      // ended black. A finished transition sits at uMix = 1 for the whole rest of the
      // shot, so that last zero held the screen black for 35 seconds straight.
      // (No backticks in this file's GLSL: the shader is a JS template literal.)
      float level = abs(uMix - 0.5) * 2.0;
      c = (uMix < 0.5 ? prev : next) * level;
    } else if (uMode == 5 || uMode == 6) {  // slide — a hard edge, not a blend
      // The boundary between them is exactly where the incoming picture's leading edge
      // has reached, so the two never overlap and neither ghosts through the other.
      float along = uMode == 5 ? vUv.x : 1.0 - vUv.y;
      c = along > (1.0 - uMix) ? next : prev;
    } else if (uMode == 4) {                // burn — dissolves in patches
      float n = fbm(vUv * 3.5);
      c = mix(prev, next, smoothstep(n - 0.18, n + 0.18, uMix * 1.36 - 0.18));
    } else if (uMode == 7) {                // zoom_blur — the smear does the cutting
      c = mix(prev, next, k);
    } else if (uMode == 8) {                // glitch_cut — a cut, damaged for two frames
      c = uMix < 0.5 ? prev : next;
      // In the torn bands the OTHER photograph punches through for an instant.
      float band = step(0.62, hash(vec2(floor(vUv.y * 14.0), floor(uMix * 6.0) + 9.0)));
      c = mix(c, uMix < 0.5 ? next : prev, band * bell * bell * 0.9);
    } else if (uMode == 9) {                // iris — a circle over the join
      float d = length((vUv - 0.5) * vec2(1.0, 0.62));   // distance in frame WIDTHS
      float edge = 0.035, reach = 0.62;     // 0.62 clears the far corner of any canvas
      // OPEN grows the circle and puts next inside it; CLOSE shrinks it around prev.
      // Both run the SAME radius from before the centre to past the corner, so the two
      // directions cannot have different end states.
      float grow = uAxis.x < 0.0 ? 1.0 - k : k;
      float r = mix(-edge, reach + edge, grow);
      float inside = 1.0 - smoothstep(r - edge, r + edge, d);
      c = uAxis.x < 0.0 ? mix(next, prev, inside) : mix(prev, next, inside);
      c += vec3(1.0, 0.98, 0.94)
         * (1.0 - smoothstep(0.0, edge * 1.8, abs(d - r))) * 0.10 * bell;
    } else if (uMode == 10) {               // page_turn — three regions, exclusive
      c = next * (1.0 - foldShade);                           // under the sheet's shadow
      c = mix(c, prev * (0.22 + 0.50 * foldLift), foldStrip); // the sheet's dim back
      c = mix(c, prev, foldFlat);                             // still lying flat
    } else if (uMode == 11) {               // shatter — a cell darkens as it drops away
      c = mix(next, prev * (1.0 - 0.55 * clamp(shardFall * 1.4, 0.0, 1.0)), shardMask);
    } else if (uMode == 12) {               // match_zoom — a late, quick crossover, so
      c = mix(prev, next, smoothstep(0.2, 0.8, uMix));   // the single move reads first
    } else {                                // dissolve — and whip, which is a dissolve
      c = mix(prev, next, smoothstep(0.0, 1.0, uMix));   // with uWhip on both layers
    }
    c = grade(c);
    float d = distance(vUv, vec2(0.5));
    c *= 1.0 - uVignette * smoothstep(0.35, 0.85, d);
    c += (hash(vUv * 1024.0 + uTime) - 0.5) * uGrain;
    outColor = vec4(c, 1.0);
  }`;

  const gl = canvas && canvas.getContext('webgl2', { antialias: false, alpha: false });
  if (!gl) { body.classList.add('no-webgl'); return; }

  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('[feed] shader', gl.getShaderInfoLog(s));
    }
    return s;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('[feed] link', gl.getProgramInfoLog(program));
  }
  gl.useProgram(program);

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(program, 'aPos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  const U = (n) => gl.getUniformLocation(program, n);
  const uni = {
    prev: U('uPrev'), next: U('uNext'), prevXf: U('uPrevXf'), nextXf: U('uNextXf'),
    prevFit: U('uPrevFit'), nextFit: U('uNextFit'),
    prevFill: U('uPrevFill'), nextFill: U('uNextFill'),
    prevPA: U('uPrevPA'), prevPB: U('uPrevPB'),
    nextPA: U('uNextPA'), nextPB: U('uNextPB'),
    prevFx: U('uPrevFx'), nextFx: U('uNextFx'),
    prevW: U('uPrevW'), nextW: U('uNextW'),
    prevClock: U('uPrevClock'), nextClock: U('uNextClock'),
    mix: U('uMix'), time: U('uTime'), whip: U('uWhip'), slide: U('uSlide'),
    axis: U('uAxis'),
    vignette: U('uVignette'), grain: U('uGrain'), mode: U('uMode'),
    lift: U('uLift'), gamma: U('uGamma'), gain: U('uGain'),
  };

  // WebGL's texture origin is bottom-left, an <img>'s is top-left. Without this every
  // photo renders vertically mirrored.
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);

  const makeTexture = () => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return { tex: t, w: 1, h: 1, photo: null };
  };
  const slots = [makeTexture(), makeTexture()];

  const loadInto = (slot, photoId) => new Promise((resolve) => {
    if (slot.photo === photoId) { resolve(); return; }
    slot.photo = photoId;
    const img = new Image();
    img.onload = () => {
      gl.bindTexture(gl.TEXTURE_2D, slot.tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      slot.w = img.naturalWidth; slot.h = img.naturalHeight;
      resolve();
    };
    img.onerror = () => { slot.photo = null; resolve(); };
    img.src = `/thumb/${photoId}?size=detail`;
  });

  // The shader samples `0.5 + (uv - 0.5) * fit / scale`.
  //   fit < 1 shrinks the sampled range -> COVER   (crops the overflowing axis)
  //   fit > 1 grows it beyond [0,1]     -> CONTAIN (letterboxes; outside is backdrop)
  // The sharp image uses CONTAIN so nothing is ever cropped; the blurred backdrop
  // uses COVER so it always fills the frame.
  const containFitOf = (slot) => {
    const canvasAspect = canvas.width / canvas.height;
    const imageAspect = slot.w / slot.h;
    return imageAspect > canvasAspect
      ? [1.0, imageAspect / canvasAspect]
      : [canvasAspect / imageAspect, 1.0];
  };
  const coverFitOf = (slot) => {
    const canvasAspect = canvas.width / canvas.height;
    const imageAspect = slot.w / slot.h;
    return imageAspect > canvasAspect
      ? [canvasAspect / imageAspect, 1.0]
      : [1.0, imageAspect / canvasAspect];
  };

  const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

  // The camera is NOT gated on `prefers-reduced-motion`. That check froze every shot —
  // no zoom, no drift, no blur — turning the whole film into a slideshow on any machine
  // with macOS "Reduce Motion" on, a global setting people enable for window animations
  // and never connect to this page. A film is not incidental motion in a UI: it is the
  // entire feature, and it plays only after a deliberate Play press. Opting out is
  // closing the film.
  //
  // `overrun` continues the move PAST a shot's end, in the same direction, so the
  // outgoing layer of a dissolve keeps moving instead of freezing into a still.
  const moveAt = (shot, t, overrun = 0) => {
    const e = (shot.move.ease === 'linear' ? t : easeInOut(t)) + overrun;
    const span = (from, to) => from + (to - from) * e;
    const speed = Math.abs(shot.move.scale_to - shot.move.scale_from) / shot.seconds;
    return {
      scale: span(shot.move.scale_from, shot.move.scale_to),
      x: span(shot.move.x_from, shot.move.x_to),
      y: span(shot.move.y_from, shot.move.y_to),
      blur: Math.min(speed * 0.35, 0.03),
    };
  };

  // How long a beat takes to fade in and out. Long enough that a hand-over reads as
  // one effect becoming another; short enough to still feel like a change happened.
  const BEAT_RAMP = 0.5;
  const NO_BEAT = { effect: 'plain', params: [0, 0, 0, 0], weight: 0 };
  const idOf = (beat) => fxShot[beat.effect] ?? 0;

  // The (at most two) beats playing at `into` seconds through `shot`, strongest first.
  // Weights ramp at each edge — EXCEPT where a beat runs to the shot's own boundary,
  // which is a bed rather than an entrance and should simply be on.
  const activeBeats = (shot, into) => {
    const dur = Math.max(shot.seconds, 0.001);
    const live = [];
    for (const beat of shot.beats) {
      const from = beat.from * dur, to = beat.to * dur;
      // EASED, not linear. A weight that ramps in a straight line still announces its
      // start and stop; eased, one effect simply becomes another.
      const rise = beat.from <= 0.001 ? 1 : easeInOut(clamp01((into - from) / BEAT_RAMP));
      const fall = beat.to >= 0.999 ? 1 : easeInOut(clamp01((to - into) / BEAT_RAMP));
      const weight = Math.min(rise, fall);
      if (weight > 0.002) { live.push({ ...beat, weight }); }
    }
    live.sort((a, b) => b.weight - a.weight);
    return [live[0] || NO_BEAT, live[1] || NO_BEAT];
  };
  const clamp01 = (value) => Math.min(Math.max(value, 0), 1);

  // MATCH THE CUT. The outgoing shot ends its push at, say, 1.12 while the incoming
  // one begins at 1.00 — and because a dissolve shows both at once, the eye reads that
  // difference as the picture JUMPING as the blend crosses over.
  //
  // So the incoming shot does not start on its own trajectory: it starts exactly where
  // the outgoing shot currently is, and eases onto its own over the length of the
  // transition. At tTrans = 0 the two transforms are identical, so there is nothing to
  // jump; by tTrans = 1 the new shot owns its move completely. A hard `cut` has no
  // transition time and is therefore left alone — a cut is SUPPOSED to change
  // everything at once.
  const matchInto = (incoming, outgoing, shot, tTrans) => {
    if (!(shot.transition_seconds > 0) || tTrans >= 1) return incoming;
    const k = easeInOut(tTrans);
    const towards = (from, to) => from + (to - from) * k;
    return {
      scale: towards(outgoing.scale, incoming.scale),
      x: towards(outgoing.x, incoming.x),
      y: towards(outgoing.y, incoming.y),
      blur: towards(outgoing.blur, incoming.blur),
    };
  };

  let edit = null, index = 0, playing = false, front = 0, mixed = false;
  // Nothing starts before the first photograph is in its texture, or has been given up
  // on: the draw loop starts there, so a film started earlier is sound over a canvas
  // nothing paints.
  let ready = false;
  // The film could not load at all: Play stays dead, and saying Loading… would be a lie.
  let unplayable = false;
  const playText = playLabel.textContent;
  let fxShot = {}, fxTransition = {};
  // Only used before the first Play, and for the rare film with no audio at all: the
  // soundtrack is the clock everywhere else.
  let silentStart = 0;

  const resize = () => {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.floor(innerWidth * dpr);
    canvas.height = Math.floor(innerHeight * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
  };
  addEventListener('resize', resize);

  // ONE clock for picture, score and voice, and it only ever goes FORWARD.
  //
  // The narration is the spine, so while it speaks the film is pinned to its
  // `currentTime`. But it ends before the film does, and the score is its own length —
  // reading `music.currentTime` straight would hand the film a second, disagreeing
  // clock. So the wall clock carries the film, and the narration corrects it while it
  // is the authority.
  let filmTime = 0;
  const clock = () => {
    // PAUSED HOLDS ITS PLACE. Returning 0 here sent the picture back to the first shot
    // the instant anything was paused — including the moment you stop to write a note,
    // which is precisely when you need to be looking at the frame you stopped on.
    if (!playing) return filmTime;
    const wall = (performance.now() - silentStart) / 1000;
    const spine = narration && !narration.paused && !narration.ended
      && narration.currentTime > 0 ? narration.currentTime : wall;
    // Clamped to the film's own length. The wall clock never stops, so without this
    // `filmTime` ran away past the end while the picture sat on its last frame — and
    // then every resume looked like a restart, because by then it genuinely was past
    // the end.
    filmTime = Math.min(Math.max(filmTime, spine), edit.seconds);
    if (filmTime >= edit.seconds) { atEnd(); }
    return filmTime;
  };

  // Which shot owns time `t`. A scan, not a counter: it must be right after a SEEK,
  // not merely after a tick, and a film is tens of shots rather than thousands.
  const shotAt = (t) => {
    for (let i = edit.shots.length - 1; i >= 0; i--) {
      if (t >= edit.shots[i].at) return i;
    }
    return 0;
  };

  let reportedDrawError = false;
  const draw = () => {
    requestAnimationFrame(draw);
    if (!edit || !edit.shots.length) return;
    try { render(); } catch (error) {
      // Logged ONCE. A throw in here used to mean an unrecoverable black screen with
      // nothing in the console but thousands of identical lines.
      if (!reportedDrawError) { reportedDrawError = true; console.error('[feed] draw', error); }
    }
  };

  const render = () => {

    const t = clock();
    floatNotes(t);
    const wanted = shotAt(t);
    if (wanted !== index) { index = wanted; present(); }

    const shot = edit.shots[index];
    const prevShot = edit.shots[Math.max(index - 1, 0)];
    const into = Math.max(t - shot.at, 0);
    const tShot = Math.min(into / Math.max(shot.seconds, 0.001), 1);
    const tTrans = shot.transition_seconds > 0
      ? Math.min(into / shot.transition_seconds, 1) : 1;

    // The outgoing shot KEEPS MOVING through the dissolve. Freezing it at its end was
    // what made every cross-fade read cheap — a still fading under a moving picture.
    const overrun = shot.transition_seconds > 0
      ? tTrans * (shot.transition_seconds / Math.max(prevShot.seconds, 0.001)) : 0;
    const a = moveAt(prevShot, 1, overrun);
    const b = matchInto(moveAt(shot, tShot), a, shot, tTrans);
    const back = slots[1 - front], fore = slots[front];

    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, back.tex);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, fore.tex);
    gl.uniform1i(uni.prev, 0); gl.uniform1i(uni.next, 1);
    gl.uniform4f(uni.prevXf, a.scale, a.x, a.y, a.blur);
    gl.uniform4f(uni.nextXf, b.scale, b.x, b.y, b.blur);
    gl.uniform2fv(uni.prevFit, containFitOf(back));
    gl.uniform2fv(uni.nextFit, containFitOf(fore));
    gl.uniform2fv(uni.prevFill, coverFitOf(back));
    gl.uniform2fv(uni.nextFill, coverFitOf(fore));
    // Which effects are live on each layer RIGHT NOW, and how strongly.
    const behind = activeBeats(prevShot, Math.max(t - prevShot.at, 0));
    const front_ = activeBeats(shot, into);
    gl.uniform4fv(uni.prevPA, behind[0].params); gl.uniform4fv(uni.prevPB, behind[1].params);
    gl.uniform4fv(uni.nextPA, front_[0].params); gl.uniform4fv(uni.nextPB, front_[1].params);
    gl.uniform2i(uni.prevFx, idOf(behind[0]), idOf(behind[1]));
    gl.uniform2i(uni.nextFx, idOf(front_[0]), idOf(front_[1]));
    gl.uniform2f(uni.prevW, behind[0].weight, behind[1].weight);
    gl.uniform2f(uni.nextW, front_[0].weight, front_[1].weight);
    // The outgoing layer's effect runs past its own end too, for the same reason its
    // camera move does.
    gl.uniform2f(uni.prevClock,
      Math.max(t - prevShot.at, 0), Math.max(prevShot.seconds, 0.001));
    gl.uniform2f(uni.nextClock, into, Math.max(shot.seconds, 0.001));
    gl.uniform1f(uni.mix, tTrans);
    gl.uniform1f(uni.time, t);
    // A radial smear on BOTH layers, hardest mid-cut and gone by its end. Three
    // transitions want one and they differ in how hard, not in kind — optically a whip
    // and a zoom blur are the same thing, which is why uWhip joins the camera's own
    // motion blur on one radial sample inside layer().
    //
    // Zero at BOTH ends by construction: sin(0) and sin(PI) are 0, so the smear cannot
    // outlive the join.
    gl.uniform1f(uni.whip,
      (SMEAR[shot.transition] || 0) * Math.sin(Math.PI * tTrans));
    // Direction of travel for a slide, in screen space; zero disables the shift for
    // every other transition, so one uniform serves both.
    gl.uniform2f(uni.slide,
      shot.transition === 'slide' ? 1.0 : 0.0,
      shot.transition === 'slide_up' ? -1.0 : 0.0);
    // Which way an iris travels is not a knob anybody sets: it follows the INCOMING
    // shot's own camera, so the circle and the push are never going opposite ways. A
    // shot that pushes in opens out of the iris; one that pulls back closes into it.
    // (CLAUDE.md — a choice that follows from context is made automatically and hidden.)
    const move = shot.move || {};
    const pushingIn = (move.scale_to ?? 1) >= (move.scale_from ?? 1);
    gl.uniform2f(uni.axis,
      shot.transition === 'iris' ? (pushingIn ? 1.0 : -1.0)
      : shot.transition === 'page_turn' ? -1.0 : 0.0,
      0.0);
    gl.uniform1f(uni.vignette, 0.35);
    gl.uniform1f(uni.grain, 0.025);
    gl.uniform1i(uni.mode, fxTransition[shot.transition] ?? 0);
    const grade = edit.grades[shot.grade] || edit.grades.neutral;
    gl.uniform3fv(uni.lift, grade.lift);
    gl.uniform3fv(uni.gamma, grade.gamma);
    gl.uniform3fv(uni.gain, grade.gain);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };

  // Put shot `index` in the front slot with its predecessor behind it, then preload the
  // one after. Driven by the clock rather than by a timer, so a SEEK lands on the right
  // pair instead of replaying every shot in between.
  //
  // The front slot is chosen BY CONTENT, never by flipping. Flipping made this function
  // non-idempotent: calling it without an index change — which is exactly what resuming
  // from a pause does — swapped the two slots, so the front one ended up holding the
  // PREVIOUS shot's photograph and the film jumped back a picture on every resume.
  const present = () => {
    const shots = edit.shots;
    const current = shots[index].photo_id;
    const previous = shots[Math.max(index - 1, 0)].photo_id;
    let target = slots.findIndex((slot) => slot.photo === current);
    if (target === -1) {
      // Nothing holds it yet: take whichever slot is NOT holding the outgoing photo,
      // so the dissolve still has something to come from.
      const holding = slots.findIndex((slot) => slot.photo === previous);
      target = holding === -1 ? front : 1 - holding;
    }
    front = target;
    loadInto(slots[front], current);
    loadInto(slots[1 - front], previous);
    showLine(shots[index].line);
    const upcoming = shots[index + 1];
    if (upcoming) { new Image().src = `/thumb/${upcoming.photo_id}?size=detail`; }
  };

  const showLine = (line) => {
    caption.textContent = line || '';
    caption.classList.remove('is-in');
    void caption.offsetWidth;
    caption.classList.add('is-in');
  };

  // The mix is built ONCE and kept. `createMediaElementSource` may be called only once
  // per element — a second call throws and silences the film — so the export taps this
  // graph through `mix.tap` rather than building its own (§11.2).
  let mix = null;
  const buildMix = () => {
    if (mixed || !music) return;
    mixed = true;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;   // no WebAudio: both tracks still play, just unducked
    const context = new Ctx();
    // Everything meets at `bus`, so the speakers and the recorder are two taps on one
    // mix rather than two mixes that could drift apart.
    const bus = context.createGain();
    bus.connect(context.destination);
    const gain = context.createGain();
    gain.gain.value = MUSIC_LEVEL;
    context.createMediaElementSource(music).connect(gain).connect(bus);
    mix = { context, bus, tap: () => {
      const destination = context.createMediaStreamDestination();
      bus.connect(destination);
      return destination;
    } };
    if (!narration) return;
    context.createMediaElementSource(narration).connect(bus);
    const duck = (to) => gain.gain.linearRampToValueAtTime(to, context.currentTime + DUCK_SECONDS);
    narration.addEventListener('play', () => duck(MUSIC_DUCKED));
    narration.addEventListener('pause', () => duck(MUSIC_LEVEL));
    narration.addEventListener('ended', () => duck(MUSIC_LEVEL));
  };

  // Space RESUMES; it does not start over. Pausing leaves `filmTime` and both audio
  // elements exactly where they stopped, so playing again simply continues — the only
  // rewind is when the film has already reached its end, where pressing play plainly
  // means "again".
  const play = () => {
    playing = true; buildMix();
    showPanel(false);
    start.classList.add('is-hidden');
    body.classList.add('is-playing');
    // `is-paused` is deliberately NOT cleared here. The overlay fades out over 0.6s,
    // and that class is what keeps the title and description `display: none` — drop it
    // now and they reappear for the length of the fade, so resuming flashes the title
    // card back over the film. It is cleared in `atEnd()`, which is the one moment the
    // title card genuinely should return.
    if (filmTime >= edit.seconds - 0.05) {
      filmTime = 0;               // the one place the clock is allowed to go back
      forgetShownNotes();
      for (const track of [music, narration]) {
        if (!track) continue;
        try { track.currentTime = 0; } catch { /* not seekable yet */ }
      }
    }
    // Anchor the wall clock to wherever we are resuming from, so a film with no audio
    // at all continues rather than jumping.
    silentStart = performance.now() - filmTime * 1000;
    for (const track of [music, narration]) {
      if (track) { track.play().catch(() => {}); }
    }
    index = shotAt(filmTime);
    present();
  };
  // Stop playing and show the overlay. Shared by a pause and by reaching the end; the
  // difference between those two is `is-paused`, set by the caller.
  const halt = () => {
    playing = false;
    start.classList.remove('is-hidden');
    body.classList.remove('is-playing');
    if (music) music.pause();
    if (narration) narration.pause();
  };

  // BACK TO THE START — what the page looked like before anything was pressed: the
  // title card, both tracks rewound, no notes already shown, the first photograph on
  // screen. Reached by Esc, which is the one key that means "undo whatever this is".
  const toStart = () => {
    halt();
    body.classList.remove('is-paused');
    filmTime = 0;
    silentStart = performance.now();
    forgetShownNotes();
    for (const track of [music, narration]) {
      if (!track) continue;
      try { track.currentTime = 0; } catch { /* not seekable yet */ }
    }
    index = 0;
    present();
  };

  // Reaching the end stops the film rather than leaving music over a frozen frame, and
  // returns the title card: `play()` there plainly means "again".
  const atEnd = () => {
    if (!playing) return;
    halt();
    body.classList.remove('is-paused');
  };

  // A PAUSE keeps the frame you stopped on. It shows a resume control over an almost
  // clear scrim rather than the title card — you paused to look at that picture, and
  // usually to write something about it.
  const pause = () => {
    halt();
    body.classList.add('is-paused');
    showPanel(true);
  };
  const toggle = () => {
    if (!ready) { if (!unplayable) { playLabel.textContent = 'Loading…'; } return; }
    if (playing) { pause(); } else { play(); }
  };

  // Stepping SEEKS the soundtrack, because the soundtrack is the clock. Moving the
  // picture alone would desync it from the voice on the very first press.
  const seekTo = (seconds) => {
    const at = Math.max(0, Math.min(seconds, edit.seconds));
    for (const track of [music, narration]) {
      if (!track) continue;
      try { track.currentTime = at; } catch { /* not seekable yet */ }
    }
    silentStart = performance.now() - at * 1000;
    filmTime = at;                // a seek is the other one
    forgetShownNotes();
    index = shotAt(at);
    present();
  };

  // --- the viewer's notes (design §11.3) ---------------------------------------------
  //
  // Notes live in the DOM and never touch the canvas. That is not decoration: the
  // export records `canvas.captureStream()`, so anything drawn outside the canvas is
  // absent from a downloaded film by construction. These are working notes for the
  // person MAKING the memory, not something to send to whoever receives it.
  // --- the viewer's notes (design §11.3) ---------------------------------------------
  //
  // The PANEL lives in `feed-notes.js`, which works with or without a film — the
  // Prepare screen has no player and is exactly where notes are worth writing. What
  // belongs here is the half that needs a clock: floating each note at its own second,
  // and telling the panel where the film is standing.
  //
  // Bubbles are DOM and never touch the canvas. That is not decoration: the export
  // records `canvas.captureStream()`, so anything drawn outside the canvas is absent
  // from a downloaded film by construction. These are working notes for the person
  // MAKING the memory, not for whoever receives it.
  const bubbleHost = document.getElementById('bubbles');
  // Seconds a note spends crossing the whole column, bottom edge to top edge. Half of
  // that is how long before its own timestamp it enters from below.
  const BUBBLE_TRAVEL = 8;
  const BUBBLE_GAP = 12;               // px kept between two bubbles, minimum
  const live = new Map();              // note id -> its element, while it is on screen

  // Notes are POSITIONED FROM THE FILM CLOCK, every frame — not animated by CSS.
  //
  // Three things follow from that and none of them are possible with keyframes: a
  // bubble is exactly at the centre of the screen at exactly its own timestamp, the
  // column is laid out so two never overlap, and a pause freezes them where they are
  // because the clock they are read from is frozen too.
  const floatNotes = (t) => {
    if (!bubbleHost || !window.feedNotes) return;
    const height = bubbleHost.clientHeight || innerHeight;
    const speed = height / BUBBLE_TRAVEL;        // px per second
    const centre = height / 2;
    const halfTravel = centre / speed;           // seconds from the bottom to the centre

    for (const note of window.feedNotes.all()) {
      const since = t - note.at;                 // 0 exactly as it crosses the centre
      const onScreen = since >= -halfTravel - 1 && since <= halfTravel + 1;
      if (onScreen && !live.has(note.id)) {
        // Created only while running, so the title card — where the clock rests at 0
        // for ever — never shows one. A pause keeps whatever is already up.
        if (!playing) continue;
        const bubble = document.createElement('p');
        bubble.className = 'feed-bubble';
        bubble.textContent = note.text;
        bubbleHost.append(bubble);
        live.set(note.id, bubble);
      } else if (!onScreen && live.has(note.id)) {
        live.get(note.id).remove();
        live.delete(note.id);
      }
    }

    // Where each one WANTS to be: purely its own time. Then anything too close to the
    // bubble above it is pushed down, which keeps them apart without changing that
    // they are all travelling at one speed.
    const placed = [];
    for (const note of window.feedNotes.all()) {
      const bubble = live.get(note.id);
      if (bubble) { placed.push({ bubble, y: centre - (t - note.at) * speed }); }
    }
    placed.sort((a, b) => a.y - b.y);
    for (let i = 1; i < placed.length; i++) {
      const floor = placed[i - 1].y + placed[i - 1].bubble.offsetHeight + BUBBLE_GAP;
      if (placed[i].y < floor) { placed[i].y = floor; }
    }
    for (const { bubble, y } of placed) {
      bubble.style.transform = `translateY(${Math.round(y)}px)`;
      // Fade at both edges so nothing pops in or out at the frame boundary.
      const edge = Math.min(y + bubble.offsetHeight, height - y) / (height * 0.18);
      bubble.style.opacity = String(Math.max(0, Math.min(1, edge)));
    }
  };

  // A seek or a replay must let the notes come round again. Cleared on those EVENTS
  // rather than by watching the clock: a time-based test is true on every frame the
  // film happens to be sitting near it, which is not an event at all.
  const forgetShownNotes = () => {
    for (const bubble of live.values()) { bubble.remove(); }
    live.clear();
  };

  // Opening the panel hands it the moment the film is standing at, so the stamp shown
  // beside the box is the one the note is saved with.
  //
  // `filmTime`, never `clock()`: `play()` calls this before it re-anchors `silentStart`,
  // so the clock would count the time spent idle as film time.
  const showPanel = (open) => {
    if (window.feedNotes) { window.feedNotes.open(open, filmTime); }
  };

  // --- export (design §11.2) -------------------------------------------------------
  //
  // The file is recorded from THIS canvas and THIS audio bus, so the export is
  // literally the film you just watched — no second renderer to drift from the first,
  // and no ffmpeg. It is the reason the effect library could grow to eight shaders
  // without an export rewrite behind it.
  //
  // It records in REAL TIME. `MediaRecorder` muxes a live audio track, and there is no
  // way to feed it a faster-than-realtime one without shipping a muxer, so a 50-second
  // film takes 50 seconds to write. That is stated on the button rather than hidden.
  const EXPORT_FPS = 30;
  // Preferred first. Chrome ≥ 130 and Safari can encode H.264 into MP4; older Chrome
  // and Firefox only do WebM, and a WebM the user can actually play beats an MP4 the
  // browser refuses to make.
  const EXPORT_TYPES = [
    'video/mp4;codecs="avc1.42E01E,mp4a.40.2"',
    'video/mp4',
    'video/webm;codecs=vp9,opus',
    'video/webm',
  ];
  const exportType = () => (window.MediaRecorder
    ? EXPORT_TYPES.find((type) => MediaRecorder.isTypeSupported(type))
    : undefined);

  const download = document.getElementById('download');
  const downloadLabel = download && download.querySelector('.feed-download-text');
  let recorder = null;

  const setDownloadLabel = (text) => {
    if (downloadLabel) { downloadLabel.textContent = text; }
  };

  const saveFile = (blob, type) => {
    const extension = type.startsWith('video/mp4') ? 'mp4' : 'webm';
    const name = `${(document.title.split(' — ')[0] || 'memory')}.${extension}`
      .replace(/[/\\?%*:|"<>]/g, '-');
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    link.click();
    // Revoked on the next turn of the loop: revoking synchronously races the click on
    // some browsers and produces a zero-byte file.
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  };

  const stopExport = () => {
    if (recorder && recorder.state !== 'inactive') { recorder.stop(); }
  };

  // NOTHING IS RECORDED UNTIL THE FILM CAN ACTUALLY PLAY.
  //
  // The export records in real time, so whatever the film does while the recorder runs
  // is what lands in the file. On a COLD page neither WAV is buffered (`preload` is a
  // hint, not a promise) and not one photograph is in the browser's cache — the
  // textures are fetched on demand as each shot comes up. Starting the recorder and
  // calling `play()` in the same tick therefore recorded the buffering: the first
  // download stalled, and the second one — after a reload had warmed the HTTP cache —
  // was smooth. That is the whole bug.
  //
  // Both waits are bounded: a browser that never fires `canplaythrough` (it happens)
  // must not hang the button, and a missing thumbnail must not block a film.
  const READY_TIMEOUT_MS = 20000;

  const audioReady = (track) => new Promise((resolve) => {
    // 4 is HAVE_ENOUGH_DATA — "can play to the end without stopping to buffer", which
    // is precisely the condition a real-time recording needs.
    if (!track || track.readyState >= 4) { resolve(); return; }
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      track.removeEventListener('canplaythrough', finish);
      track.removeEventListener('error', finish);
      resolve();
    };
    const timer = setTimeout(finish, READY_TIMEOUT_MS);
    track.addEventListener('canplaythrough', finish);
    track.addEventListener('error', finish);
    // Only when nothing has been fetched at all: `load()` on a partly-buffered element
    // throws the buffer away and starts again.
    if (track.readyState === 0) { try { track.load(); } catch { finish(); } }
  });

  const picturesReady = () => Promise.all(
    [...new Set(edit.shots.map((shot) => shot.photo_id))].map((id) => new Promise(
      (resolve) => {
        const img = new Image();
        img.onload = img.onerror = () => resolve();
        img.src = `/thumb/${id}?size=detail`;
      })));

  let preparing = false;
  // Set while cancelling, so `onstop` knows this recording is being thrown away.
  let discard = false;

  const cancelExport = () => {
    discard = true;
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop();                 // `onstop` clears the button and drops the chunks
      return;
    }
    recorder = null;                   // never started: it was still warming up
    preparing = false;
    body.classList.remove('is-exporting');
    setDownloadLabel('↓ Download');
  };

  const startExport = async () => {
    if (!ready) { if (!unplayable) { playLabel.textContent = 'Loading…'; } return; }
    // A click WHILE IT WARMS UP means "never mind" — the recorder exists but has not
    // started, so `stopExport` would have nothing to stop and the film would begin
    // recording a moment later anyway.
    if (preparing) {
      recorder = null;
      body.classList.remove('is-exporting');
      setDownloadLabel('↓ Download');
      return;
    }
    if (recorder) { stopExport(); return; }
    const type = exportType();
    if (!type) {
      setDownloadLabel('This browser cannot record video');
      return;
    }
    buildMix();
    const stream = canvas.captureStream(EXPORT_FPS);
    // No WebAudio at all means no way to mux the sound; a silent film is not worth
    // downloading, so say so rather than handing over a mute file.
    if (!mix) {
      setDownloadLabel('Sound is unavailable — cannot export');
      return;
    }
    for (const track of mix.tap().stream.getAudioTracks()) { stream.addTrack(track); }

    const chunks = [];
    discard = false;
    recorder = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 8e6 });
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size) { chunks.push(event.data); }
    };
    recorder.onstop = () => {
      const keep = !discard;
      recorder = null;
      discard = false;
      body.classList.remove('is-exporting');
      setDownloadLabel('↓ Download');
      // A CANCELLED recording produces no file. Half a film in the downloads folder is
      // not a smaller version of the thing that was asked for; it is litter.
      if (keep && chunks.length) { saveFile(new Blob(chunks, { type }), type); }
    };

    body.classList.add('is-exporting');
    // The wait is VISIBLE: a button that says nothing for ten seconds is
    // indistinguishable from a broken one.
    preparing = true;
    setDownloadLabel('Preparing…');
    try {
      await Promise.all([audioReady(music), audioReady(narration), picturesReady()]);
    } finally {
      preparing = false;
    }
    if (!recorder) { return; }          // stopped while it was warming up
    recorder.start();
    play();
    // Stop on the film's own last frame. `ended` is not enough: neither the score nor
    // the narration is exactly the length of the picture.
    const finish = () => {
      if (!recorder) return;
      if (clock() >= edit.seconds) { pause(); stopExport(); return; }
      setDownloadLabel(
        `Recording ${Math.round(100 * clock() / Math.max(edit.seconds, 0.001))}% — click to stop`);
      requestAnimationFrame(finish);
    };
    requestAnimationFrame(finish);
  };

  if (download) { download.addEventListener('click', startExport); }

  // The panel's timestamps jump the film; it has no clock of its own.
  if (window.feedNotes) { window.feedNotes.seek = (seconds) => seekTo(seconds); }

  // ESC IS LAYERED. It has always meant "up one level" (§13.1 rule 5), and that is
  // still what it does — but only once there is nothing inside the film to undo.
  // Recording? Cancel it. Playing or paused? Back to the title card. Only then close.
  // Pressing it mid-download and landing on the Organize grid, with the recording still
  // running behind the page, was the behaviour this replaces.
  window.feedFilm = {
    escape() {
      if (recorder || preparing) { cancelExport(); toStart(); return true; }
      if (playing || filmTime > 0) { toStart(); return true; }
      return false;                    // nothing to undo — let the page close
    },
  };

  start.addEventListener('click', toggle);
  document.addEventListener('keydown', (e) => {
    if (!ready) return;
    // Space is play/pause EXCEPT while writing a note, where it is a space.
    const typing = e.target instanceof HTMLElement
      && (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT');
    if (typing) return;
    if (e.key === ' ') { e.preventDefault(); toggle(); }
    if (e.key === 'ArrowRight') {
      const upcoming = edit.shots[index + 1];
      seekTo(upcoming ? upcoming.at : edit.seconds);
    }
    if (e.key === 'ArrowLeft') {
      seekTo(edit.shots[Math.max(index - 1, 0)].at);
    }
  });

  // A timeline PREPARED BEFORE the conductor existed has no `at`, no `params` and no
  // effect table — and `gl.uniform4fv(loc, undefined)` throws, on the first frame, for
  // ever, which is a black screen and no error the user can see. Every film already in
  // the database is one of those, so the player repairs the shape it is given instead
  // of assuming the newest one: absolute times are re-derived by accumulating the
  // durations, and anything missing falls back to a plain shot.
  const normalise = (timeline) => {
    let cursor = 0;
    for (const shot of timeline.shots || []) {
      if (typeof shot.at !== 'number') { shot.at = cursor; }
      cursor = shot.at + (shot.seconds || 0);
      if (!Array.isArray(shot.params) || shot.params.length !== 4) {
        shot.params = [0, 0, 0, 0];
      }
      if (typeof shot.effect !== 'string') { shot.effect = 'plain'; }
      // A film prepared before effects ran over TIME carries one treatment; it becomes
      // a single beat covering the whole shot, so there is one code path here.
      if (!Array.isArray(shot.beats) || !shot.beats.length) {
        shot.beats = [{ effect: shot.effect, params: shot.params, from: 0, to: 1 }];
      }
    }
    if (typeof timeline.seconds !== 'number') { timeline.seconds = cursor; }
    return timeline;
  };

  (async () => {
    resize();
    try {
      edit = normalise(await (await fetch(`/feed/${key}/timeline.json?lang=${lang}`)).json());
    } catch { edit = null; }
    if (!edit || !edit.shots || !edit.shots.length) {
      unplayable = true;
      playLabel.textContent = playText;
      return;
    }
    fxShot = (edit.effects && edit.effects.shot) || {};
    fxTransition = (edit.effects && edit.effects.transition) || {};
    const bounded = (waiting) => Promise.race([
      waiting, new Promise((resolve) => setTimeout(resolve, READY_TIMEOUT_MS))]);
    await bounded(loadInto(slots[front], edit.shots[0].photo_id));
    await bounded(loadInto(slots[1 - front], edit.shots[0].photo_id));
    ready = true;
    playLabel.textContent = playText;
    // The editor is hidden during PLAYBACK and shown whenever the film is not running
    // — the title card included. The title card is not playback: it is where you read
    // what you already said before deciding to watch or to prepare again.
    showPanel(true);
    requestAnimationFrame(draw);
  })();
})();
