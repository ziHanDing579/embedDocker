// chart.js
// A dependency-free SVG scatter plot confined to the first quadrant [0,1] x [0,1].
// X = cosine similarity to the X-axis sentence, Y = cosine similarity to the
// Y-axis sentence. Points are keyed by id so that changing an axis animates them.

const NS = "http://www.w3.org/2000/svg";
const TICKS = [0, 0.25, 0.5, 0.75, 1];

function el(name, attrs = {}) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

function truncate(s, n) {
  return s && s.length > n ? s.slice(0, n - 1) + "…" : s;
}

export class Plot {
  constructor(svg) {
    this.svg = svg;
    this.W = 760;
    this.H = 560;
    this.pad = { t: 28, r: 28, b: 66, l: 70 };
    this.nodes = new Map(); // id -> <g> for a point

    this.svg.setAttribute("viewBox", `0 0 ${this.W} ${this.H}`);
    this.svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    this.svg.setAttribute("role", "img");
    this.svg.setAttribute("aria-label", "Scatter plot of sentence similarities");

    this._buildStatic();
  }

  get plotW() {
    return this.W - this.pad.l - this.pad.r;
  }
  get plotH() {
    return this.H - this.pad.t - this.pad.b;
  }
  px(x) {
    return this.pad.l + x * this.plotW;
  }
  py(y) {
    return this.pad.t + (1 - y) * this.plotH;
  }

  _buildStatic() {
    const { pad } = this;
    const x0 = this.px(0);
    const y0 = this.py(0);
    const x1 = this.px(1);
    const y1 = this.py(1);

    this.svg.appendChild(
      el("rect", {
        x: pad.l,
        y: pad.t,
        width: this.plotW,
        height: this.plotH,
        rx: 6,
        class: "plate-bg",
      })
    );

    const grid = el("g", { class: "grid" });
    for (const t of TICKS) {
      const gx = this.px(t);
      const gy = this.py(t);
      grid.appendChild(el("line", { x1: gx, y1: y0, x2: gx, y2: y1, class: "gridline" }));
      grid.appendChild(el("line", { x1: x0, y1: gy, x2: x1, y2: gy, class: "gridline" }));

      const tx = el("text", { x: gx, y: y0 + 20, class: "tick" });
      tx.textContent = t.toFixed(2);
      grid.appendChild(tx);

      if (t !== 0) {
        const ty = el("text", { x: x0 - 12, y: gy + 4, class: "tick tick-y" });
        ty.textContent = t.toFixed(2);
        grid.appendChild(ty);
      }
    }
    this.svg.appendChild(grid);

    // Axes (the L in the corner), tinted to match each axis colour.
    this.svg.appendChild(el("line", { x1: x0, y1: y0, x2: x1, y2: y0, class: "axis axis-x-line" }));
    this.svg.appendChild(el("line", { x1: x0, y1: y0, x2: x0, y2: y1, class: "axis axis-y-line" }));

    // Axis titles show the chosen sentence (truncated).
    this.xTitle = el("text", { x: (x0 + x1) / 2, y: this.H - 14, class: "axis-title axis-title-x" });
    this.svg.appendChild(this.xTitle);

    const ymid = (y0 + y1) / 2;
    this.yTitle = el("text", {
      x: 20,
      y: ymid,
      class: "axis-title axis-title-y",
      transform: `rotate(-90 20 ${ymid})`,
    });
    this.svg.appendChild(this.yTitle);

    this.setAxisTitles(null, null);

    this.pointsLayer = el("g", { class: "points" });
    this.svg.appendChild(this.pointsLayer);

    this.empty = el("text", { x: this.px(0.5), y: this.py(0.5), class: "empty-note" });
    this.empty.textContent = "Assign one sentence to X and one to Y";
    this.svg.appendChild(this.empty);
  }

  /** Update the axis titles with the chosen sentences (or a placeholder). */
  setAxisTitles(xText, yText) {
    this.xTitle.textContent = xText ? `X · “${truncate(xText, 42)}”` : "X · choose a sentence";
    this.yTitle.textContent = yText ? `Y · “${truncate(yText, 42)}”` : "Y · choose a sentence";
  }

  /**
   * Render the given points.
   * @param {{id:string,text:string,x:number,y:number,isX:boolean,isY:boolean}[]} points
   */
  render(points) {
    this.empty.style.display = points.length ? "none" : "";

    const seen = new Set();

    for (const p of points) {
      seen.add(p.id);
      let g = this.nodes.get(p.id);
      if (!g) {
        g = el("g", { class: "pt", tabindex: "0" });
        g.appendChild(el("circle", { r: 12, class: "pt-halo" }));
        g.appendChild(el("circle", { r: 5.5, class: "pt-dot" }));
        g.appendChild(el("title"));
        this.pointsLayer.appendChild(g);
        this.nodes.set(p.id, g);
      }

      g.setAttribute("transform", `translate(${this.px(p.x)} ${this.py(p.y)})`);
      g.classList.toggle("is-x", !!p.isX);
      g.classList.toggle("is-y", !!p.isY);
      g.querySelector("title").textContent =
        `${p.text}\n( x ${p.x.toFixed(3)},  y ${p.y.toFixed(3)} )`;
      g.setAttribute("aria-label", `${p.text}. x ${p.x.toFixed(2)}, y ${p.y.toFixed(2)}`);
    }

    for (const [id, g] of this.nodes) {
      if (!seen.has(id)) {
        g.remove();
        this.nodes.delete(id);
      }
    }
  }
}