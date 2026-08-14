/* mlx-minimax-music3 — site behaviour.

   Three independent pieces, each guarded so a failure in one cannot take the
   page down with it:

     1. the hero lattice, drawn from the model's real frame shape
     2. the residency meter, stepped by which pipeline stage is on screen
     3. the handoff readout, computed from the documented tensor shape

   Every number here comes from docs/architecture.md and docs/weights.md.
   Nothing is invented to make a chart look better, and the release number is
   fetched rather than written down, so the page cannot go stale.
*/
(() => {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');

  /* ══ model constants ═══════════════════════════════════════════════════
     docs/architecture.md: 25 frames per second, eight RVQ codebooks per
     frame, fused hidden states shaped [1, frames, 8 * 4096], 9,000 frames
     maximum. docs/weights.md: published component sizes in GiB. */
  const FPS = 25;
  const CODEBOOKS = 8;
  const HIDDEN = CODEBOOKS * 4096;   // 32,768 per frame
  const BF16 = 2;                    // bytes per element
  const CHECKPOINT_GIB = 26.56;      // all 25 files of the componentized layout

  const PHASES = [
    { resident: [], gib: 0, kind: 'code' },
    { resident: ['language_model', 'rvq_depth_decoder'], gib: 17.19, kind: 'code' },
    { resident: ['condition_encoder', 'transformer'], gib: 9.15, kind: 'flow' },
    { resident: ['vocoder'], gib: 0.2, kind: 'flow' },
    { resident: [], gib: 0, kind: 'flow' },
  ];

  /* ══ helpers ═══════════════════════════════════════════════════════════ */

  /* A small deterministic generator. The lattice should look uneven in a way
     that is the same on every load, so a screenshot taken today matches one
     taken tomorrow. */
  const rng = (seed) => () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  const clock = (frames) => {
    const total = Math.round(frames / FPS);
    return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
  };

  const bytes = (n) => {
    const MiB = n / 1048576;
    return MiB >= 1024 ? `${(MiB / 1024).toFixed(2)} GiB` : `${MiB.toFixed(1)} MiB`;
  };

  /* ══ 1. the hero lattice ═══════════════════════════════════════════════
     Eight rows, one column per frame, resolving left to right at the rate
     the model actually emits them. Row 0 is the semantic codebook and is
     drawn in the discrete accent; rows 1-7 are residuals and fade with
     depth, because that is what a residual is. */
  const lattice = () => {
    const svg = document.getElementById('lattice');
    if (!svg) return;

    const ROWS = CODEBOOKS;
    const CELL = 6;
    const GAP = 2;
    const TRAIL = 14;      // frames shown as queued for acoustic synthesis
    const LATENT = 0.1;    // unresolved cells: a visible field, not an empty box

    let cells = [];
    let cols = 0;
    let raf = 0;
    let head = 0;
    let last = 0;

    const build = () => {
      const w = svg.clientWidth || svg.getBoundingClientRect().width;
      const h = svg.clientHeight || svg.getBoundingClientRect().height;
      if (!w || !h) return false;

      cols = Math.max(1, Math.floor((w + GAP) / (CELL + GAP)));
      const rowH = (h - GAP * (ROWS - 1)) / ROWS;

      svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      const frag = document.createDocumentFragment();
      const rand = rng(0x6d75);
      cells = [];

      for (let r = 0; r < ROWS; r += 1) {
        for (let c = 0; c < cols; c += 1) {
          const rect = document.createElementNS(SVG_NS, 'rect');
          rect.setAttribute('x', (c * (CELL + GAP)).toFixed(2));
          rect.setAttribute('y', (r * (rowH + GAP)).toFixed(2));
          rect.setAttribute('width', CELL);
          rect.setAttribute('height', Math.max(2, rowH).toFixed(2));
          rect.setAttribute('rx', '1');
          rect.setAttribute('class', 'lattice__cell');

          /* Depth sets the ceiling: the semantic codebook is the loudest
             row, and each residual below it corrects a little less. */
          const depth = r === 0 ? 1 : 0.62 - (r - 1) * 0.068;
          const jitter = 0.72 + rand() * 0.28;

          cells.push({ el: rect, r, c, peak: depth * jitter, lag: rand() * 6 });
          rect.setAttribute('fill', r === 0 ? 'var(--code)' : 'var(--bone)');
          rect.style.opacity = String(LATENT);
          frag.appendChild(rect);
        }
      }

      svg.appendChild(frag);
      return true;
    };

    const FADE = 45;   // columns of dissolve before the sweep begins again

    const paint = () => {
      /* Once the sweep has run off the end, the field dissolves back to
         latent instead of snapping, so the loop reads as one continuous
         take rather than a hard cut. */
      const over = head - (cols + TRAIL);
      const fade = over > 0 ? Math.max(0, 1 - over / FADE) : 1;

      for (const cell of cells) {
        const since = head - cell.c - cell.lag;
        if (since < 0) {
          cell.el.style.opacity = String(LATENT);
          continue;
        }
        /* Resolved. The most recent frames are the ones queued for the
           acoustic phase, so they carry the continuous accent. */
        const queued = since < TRAIL;
        cell.el.setAttribute('fill', queued ? 'var(--flow)' : (cell.r === 0 ? 'var(--code)' : 'var(--bone)'));
        const lit = queued ? Math.min(1, cell.peak + 0.2) : cell.peak * 0.85;
        cell.el.style.opacity = String(LATENT + (lit - LATENT) * fade);
      }
    };

    /* Reduced motion gets the finished field, held: every frame resolved,
       nothing queued, nothing moving. */
    const still = () => {
      head = cols + TRAIL;
      paint();
    };

    const tick = (now) => {
      if (!last) last = now;
      head += ((now - last) / 1000) * FPS;
      last = now;
      if (head > cols + TRAIL + FADE) { head = 0; }
      paint();
      raf = requestAnimationFrame(tick);
    };

    const start = () => {
      if (raf || reduced.matches) return;
      last = 0;
      raf = requestAnimationFrame(tick);
    };
    const stop = () => {
      if (!raf) return;
      cancelAnimationFrame(raf);
      raf = 0;
    };

    const reset = () => {
      if (!build()) return;
      if (reduced.matches) still();
    };

    reset();

    /* Only animate while the artwork is actually on screen. */
    if ('IntersectionObserver' in window) {
      new IntersectionObserver((entries) => {
        for (const e of entries) (e.isIntersecting ? start : stop)();
      }, { threshold: 0 }).observe(svg);
    } else {
      start();
    }

    let resizeTimer = 0;
    addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => { stop(); head = 0; reset(); start(); }, 180);
    }, { passive: true });

    reduced.addEventListener('change', () => {
      stop();
      if (reduced.matches) still(); else start();
    });
  };

  /* ══ 2. the residency meter ════════════════════════════════════════════
     The stage that owns the viewport owns the meter. Residency is a step
     function -- weights are held or they are not -- so the meter changes
     state per stage rather than sliding continuously with the scrollbar. */
  const residency = () => {
    const meter = document.getElementById('meter');
    const stages = Array.from(document.querySelectorAll('.stage'));
    if (!meter || !stages.length) return;

    const value = document.getElementById('meterValue');
    const fill = document.getElementById('gaugeFill');
    const comps = new Map(
      Array.from(document.querySelectorAll('.comp'))
        .map((el) => [el.dataset.comp, el]),
    );

    let current = -1;
    let tween = 0;

    const show = (i) => {
      if (i === current) return;
      const phase = PHASES[i];
      if (!phase) return;
      current = i;

      meter.dataset.phase = String(i);
      meter.classList.toggle('is-code', phase.kind === 'code' && phase.gib > 0);
      meter.classList.toggle('is-flow', phase.kind === 'flow' && phase.gib > 0);

      for (const [name, el] of comps) {
        el.classList.toggle('is-on', phase.resident.includes(name));
      }

      fill.style.transform = `scaleX(${(phase.gib / CHECKPOINT_GIB).toFixed(4)})`;

      /* Tween only the readout digits, over the same duration the bar uses,
         so the two do not disagree about how long the change took. */
      cancelAnimationFrame(tween);
      const from = parseFloat(value.textContent) || 0;
      const to = phase.gib;
      if (reduced.matches || from === to) {
        value.textContent = to.toFixed(2);
        return;
      }
      const t0 = performance.now();
      const step = (now) => {
        const p = Math.min(1, (now - t0) / 600);
        const eased = 1 - Math.pow(1 - p, 3);
        value.textContent = (from + (to - from) * eased).toFixed(2);
        if (p < 1) tween = requestAnimationFrame(step);
      };
      tween = requestAnimationFrame(step);
    };

    stages.forEach((s, i) => { if (!s.dataset.phase) s.dataset.phase = String(i); });

    if (!('IntersectionObserver' in window)) {
      stages.forEach((s) => s.classList.add('is-on'));
      show(1);
      return;
    }

    /* A band across the middle of the viewport decides which stage is
       current, so the meter changes when a stage is being read rather than
       when it first peeks over the edge. */
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        e.target.classList.toggle('is-on', e.isIntersecting);
      }
      const on = stages.filter((s) => s.classList.contains('is-on'));
      if (on.length) show(Number(on[0].dataset.phase));
    }, { rootMargin: '-45% 0px -45% 0px', threshold: 0 });

    stages.forEach((s) => io.observe(s));
    show(0);
  };

  /* ══ 3. the handoff readout ════════════════════════════════════════════ */
  const handoff = () => {
    const input = document.getElementById('frames');
    if (!input) return;

    const out = {
      frames: document.getElementById('framesOut'),
      duration: document.getElementById('outDuration'),
      codes: document.getElementById('outCodes'),
      shape: document.getElementById('outShape'),
      bytes: document.getElementById('outBytes'),
    };

    const update = () => {
      const f = Number(input.value);
      const t = clock(f);
      out.frames.textContent = `${f.toLocaleString('en-US')} · ${t}`;
      out.duration.textContent = t;
      out.codes.textContent = (f * CODEBOOKS).toLocaleString('en-US');
      /* Left unseparated on purpose: a tensor shape is already comma
         delimited, and thousands separators inside it read as more axes. */
      out.shape.textContent = `[1, ${f}, ${HIDDEN}]`;
      out.bytes.textContent = bytes(f * HIDDEN * BF16);
    };

    input.addEventListener('input', update);
    update();
  };

  /* ══ 4. the published release ══════════════════════════════════════════
     Read from PyPI at view time. Nothing about the version is written into
     the page, so it cannot drift from what is actually installable. If the
     package is not published yet, or the request fails, the badge simply
     stays hidden and the page still reads correctly. */
  const release = () => {
    const el = document.getElementById('release');
    if (!el || !('fetch' in window)) return;

    fetch('https://pypi.org/pypi/mlx-minimax-music3/json', {
      headers: { Accept: 'application/json' },
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d) => {
        const v = d && d.info && d.info.version;
        if (!v) return;
        el.textContent = `v${v}`;
        el.hidden = false;
      })
      .catch(() => { /* not published yet; the chip stays as it is */ });
  };

  const boot = () => {
    for (const fn of [lattice, residency, handoff, release]) {
      try { fn(); } catch (e) { console.warn(`${fn.name} failed`, e); }
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot, { once: true });
  } else {
    boot();
  }
})();
