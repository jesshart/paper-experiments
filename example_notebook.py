# /// script
# dependencies = [
#     "anywidget==0.10.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.4",
#     "traitlets==5.14.3",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # False Positive Simulation

    Simulate voxel-wise correlation tests under the **complete null hypothesis**
    (pure noise, no real signal) and compare uncorrected vs. Bonferroni-corrected
    significance counts.

    Move the sliders below to see how the false-positive rate and Bonferroni
    threshold change with the experiment shape.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The gist, in plain English

    Imagine **20,000 people** each flip a coin 200 times, and you look for
    someone whose flips match a pattern you picked in advance. Even though
    every coin is fair, with that many people you'll find **about 1,000**
    whose flips "match" by pure luck.

    That's what happens here — except the "people" are **voxels** (tiny 3D
    pixels of a brain scan) and the "pattern" is a fake task. No voxel has
    any real signal. But if you accept anything with "p < 0.05" as a
    finding, you'll announce ~1,000 discoveries that are entirely noise.

    The **Bonferroni correction** raises the bar: a finding only counts if
    its p-value beats `0.05 / 20,000`. Apply it and the false alarms almost
    entirely disappear.
    """)
    return


@app.cell(hide_code=True)
def _():
    import math
    from dataclasses import dataclass

    import anywidget
    import marimo as mo
    import numpy as np
    import traitlets

    return anywidget, dataclass, math, mo, np, traitlets


@app.cell(hide_code=True)
def _(math, np):
    def normal_cdf(x: np.ndarray) -> np.ndarray:
        """Approximate the standard normal CDF elementwise."""
        erf_vec = np.vectorize(math.erf)
        return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))

    return (normal_cdf,)


@app.cell(hide_code=True)
def _(dataclass, normal_cdf, np):
    @dataclass
    class ExperimentResult:
        """Container for the simulation outputs."""

        n_voxels: int
        alpha: float
        n_uncorrected_significant: int
        bonferroni_threshold: float
        n_bonferroni_significant: int
        p_values: np.ndarray
        correlations: np.ndarray


    def simulate_false_positives(
        n_timepoints: int,
        n_voxels: int,
        alpha: float,
        seed: int,
    ) -> ExperimentResult:
        """Run voxel-wise tests on pure noise and count false positives."""
        rng = np.random.default_rng(seed)

        # Fake experimental regressor (e.g. "emotional vs non-emotional" blocks).
        task = rng.integers(0, 2, size=n_timepoints).astype(float)
        task = (task - task.mean()) / task.std(ddof=1)

        # Pure noise: no real signal in any voxel.
        noise = rng.normal(size=(n_timepoints, n_voxels))
        noise = (noise - noise.mean(axis=0)) / noise.std(axis=0, ddof=1)

        # Pearson correlation between each voxel and the fake task.
        correlations = (task[:, None] * noise).sum(axis=0) / (n_timepoints - 1)

        df = n_timepoints - 2
        t_stats = correlations * np.sqrt(
            df / np.maximum(1e-12, 1.0 - correlations**2)
        )
        p_values = 2.0 * (1.0 - normal_cdf(np.abs(t_stats)))

        n_uncorrected = int((p_values < alpha).sum())
        bonferroni_threshold = alpha / n_voxels
        n_bonferroni = int((p_values < bonferroni_threshold).sum())

        return ExperimentResult(
            n_voxels=n_voxels,
            alpha=alpha,
            n_uncorrected_significant=n_uncorrected,
            bonferroni_threshold=bonferroni_threshold,
            n_bonferroni_significant=n_bonferroni,
            p_values=p_values,
            correlations=correlations,
        )

    return (simulate_false_positives,)


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class FalsePositiveHunter(anywidget.AnyWidget):
        """Dual-view interactive hunter for the false-positive simulation.

        Drag the red line on the histogram to move the threshold; the voxel
        grid on the left repaints in real time. Buttons snap to the naive
        (alpha) or Bonferroni-corrected cutoff.
        """

        _esm = r"""
        function svgEl(tag, attrs) {
            const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
            for (const k in attrs) e.setAttribute(k, attrs[k]);
            return e;
        }

        function render({ model, el }) {
            const wrapper = document.createElement("div");
            wrapper.style.cssText = `
                font-family: sans-serif;
                color: #222;
                background: #fafaf7;
                border: 1px solid #d8d8d0;
                border-radius: 8px;
                padding: 14px;
                display: flex;
                flex-direction: column;
                gap: 10px;
            `;

            const row = document.createElement("div");
            row.style.cssText = "display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap;";

            // --- Grid (canvas) ---
            const gridWrap = document.createElement("div");
            gridWrap.style.cssText = "display: flex; flex-direction: column; align-items: center; gap: 4px;";
            const gridTitle = document.createElement("div");
            gridTitle.style.cssText = "font-size: 12px; color: #333; font-weight: 500;";
            gridTitle.textContent = "Voxel grid — red = 'significant'";
            const CANVAS_SIZE = 320;
            const canvas = document.createElement("canvas");
            canvas.width = CANVAS_SIZE;
            canvas.height = CANVAS_SIZE;
            canvas.style.cssText = "border: 1px solid #ccc; border-radius: 4px; background: #f6f6f0; image-rendering: pixelated;";
            gridWrap.appendChild(gridTitle);
            gridWrap.appendChild(canvas);

            // --- Histogram (SVG) ---
            const HIST_W = 460, HIST_H = 340;
            const M = { l: 50, r: 15, t: 30, b: 40 };
            const PW = HIST_W - M.l - M.r;
            const PH = HIST_H - M.t - M.b;

            const svg = svgEl("svg", { width: HIST_W, height: HIST_H, viewBox: `0 0 ${HIST_W} ${HIST_H}` });
            svg.style.cssText = "border: 1px solid #ccc; border-radius: 4px; background: #fff; user-select: none; cursor: ew-resize;";

            const pToX = p => M.l + p * PW;
            const xToP = x => (x - M.l) / PW;

            const histTitle = svgEl("text", { x: HIST_W / 2, y: 18, "text-anchor": "middle", "font-size": 12, fill: "#333", "font-weight": 500 });
            histTitle.textContent = "P-value distribution — drag the red line";
            svg.appendChild(histTitle);

            svg.appendChild(svgEl("line", { x1: M.l, y1: M.t + PH, x2: M.l + PW, y2: M.t + PH, stroke: "#333" }));
            svg.appendChild(svgEl("line", { x1: M.l, y1: M.t, x2: M.l, y2: M.t + PH, stroke: "#333" }));

            for (let i = 0; i <= 10; i++) {
                const p = i / 10;
                const x = pToX(p);
                svg.appendChild(svgEl("line", { x1: x, y1: M.t + PH, x2: x, y2: M.t + PH + 4, stroke: "#333" }));
                const t = svgEl("text", { x, y: M.t + PH + 16, "text-anchor": "middle", "font-size": 10, fill: "#333" });
                t.textContent = p.toFixed(1);
                svg.appendChild(t);
            }
            const xlab = svgEl("text", { x: M.l + PW / 2, y: HIST_H - 8, "text-anchor": "middle", "font-size": 11, fill: "#333" });
            xlab.textContent = "p-value";
            svg.appendChild(xlab);
            const ylab = svgEl("text", {
                x: 14, y: M.t + PH / 2, "text-anchor": "middle", "font-size": 11, fill: "#333",
                transform: `rotate(-90 14 ${M.t + PH / 2})`,
            });
            ylab.textContent = "voxels";
            svg.appendChild(ylab);

            const barsGroup = svgEl("g", {});
            svg.appendChild(barsGroup);

            // Reference lines (no labels here — legend handles that)
            const alphaLine = svgEl("line", { y1: M.t, y2: M.t + PH, stroke: "#888", "stroke-dasharray": "3 3", "stroke-width": 1 });
            svg.appendChild(alphaLine);
            const bonfLine = svgEl("line", { y1: M.t, y2: M.t + PH, stroke: "#1a6b3a", "stroke-dasharray": "3 3", "stroke-width": 1 });
            svg.appendChild(bonfLine);

            // Legend in the top-right of the plot
            const legendGroup = svgEl("g", {});
            svg.appendChild(legendGroup);

            // Threshold handle (drawn last so it's on top)
            const thresholdLine = svgEl("line", { y1: M.t, y2: M.t + PH, stroke: "#c44", "stroke-width": 2, cursor: "ew-resize" });
            svg.appendChild(thresholdLine);
            const thresholdHandle = svgEl("rect", {
                y: M.t - 8, width: 14, height: 14, fill: "#c44",
                cursor: "ew-resize", rx: 3, stroke: "#fff", "stroke-width": 1.5,
            });
            svg.appendChild(thresholdHandle);

            row.appendChild(gridWrap);
            row.appendChild(svg);

            const readout = document.createElement("div");
            readout.style.cssText = "font-family: ui-monospace, monospace; font-size: 13px; line-height: 1.6; padding: 10px 12px; background: #fff; color: #222; border: 1px solid #e0e0d8; border-radius: 4px;";

            const buttons = document.createElement("div");
            buttons.style.cssText = "display: flex; gap: 8px; flex-wrap: wrap;";

            function makeBtn(label, bg, onClick) {
                const b = document.createElement("button");
                b.textContent = label;
                b.style.cssText = `padding: 6px 12px; border: 1px solid #888; background: ${bg}; color: #222; border-radius: 4px; cursor: pointer; font-size: 12px; font-family: sans-serif; font-weight: 500;`;
                b.onclick = onClick;
                return b;
            }
            buttons.appendChild(makeBtn("Naive: p < alpha", "#fff", () => {
                model.set("threshold", model.get("alpha"));
                model.save_changes();
            }));
            buttons.appendChild(makeBtn("Apply Bonferroni", "#d8f0d8", () => {
                model.set("threshold", model.get("bonferroni_threshold"));
                model.save_changes();
            }));

            wrapper.appendChild(row);
            wrapper.appendChild(readout);
            wrapper.appendChild(buttons);
            el.appendChild(wrapper);

            const c2d = canvas.getContext("2d");
            let binsCache = null;
            let binsMax = 1;

            function computeBins() {
                const pvals = model.get("p_values");
                const nBins = 50;
                const bins = new Array(nBins).fill(0);
                for (let i = 0; i < pvals.length; i++) {
                    let bi = Math.floor(pvals[i] * nBins);
                    if (bi >= nBins) bi = nBins - 1;
                    if (bi < 0) bi = 0;
                    bins[bi]++;
                }
                binsCache = bins;
                binsMax = 0;
                for (const v of bins) if (v > binsMax) binsMax = v;
                if (binsMax === 0) binsMax = 1;
            }

            function drawGrid() {
                const pvals = model.get("p_values");
                const n = pvals.length;
                const threshold = model.get("threshold");
                const side = Math.ceil(Math.sqrt(n));
                const cell = CANVAS_SIZE / side;
                c2d.fillStyle = "#f0f0e8";
                c2d.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
                c2d.fillStyle = "#c44";
                const draw = Math.max(1, Math.ceil(cell));
                for (let i = 0; i < n; i++) {
                    if (pvals[i] < threshold) {
                        const r = Math.floor(i / side);
                        const col = i % side;
                        c2d.fillRect(Math.floor(col * cell), Math.floor(r * cell), draw, draw);
                    }
                }
            }

            function drawLegend() {
                while (legendGroup.firstChild) legendGroup.removeChild(legendGroup.firstChild);
                const alpha = model.get("alpha");
                const bonf = model.get("bonferroni_threshold");
                const lw = 150;
                const lh = 38;
                const lx = M.l + PW - lw - 6;
                const ly = M.t + 6;
                legendGroup.appendChild(svgEl("rect", {
                    x: lx, y: ly, width: lw, height: lh,
                    fill: "#fff", stroke: "#ccc", "stroke-width": 1, rx: 3, opacity: 0.96,
                }));
                // alpha
                legendGroup.appendChild(svgEl("line", {
                    x1: lx + 8, y1: ly + 12, x2: lx + 26, y2: ly + 12,
                    stroke: "#888", "stroke-dasharray": "3 2", "stroke-width": 1.5,
                }));
                const aT = svgEl("text", { x: lx + 32, y: ly + 15, "font-size": 11, fill: "#444" });
                aT.textContent = `alpha = ${alpha.toFixed(3)}`;
                legendGroup.appendChild(aT);
                // bonferroni
                legendGroup.appendChild(svgEl("line", {
                    x1: lx + 8, y1: ly + 28, x2: lx + 26, y2: ly + 28,
                    stroke: "#1a6b3a", "stroke-dasharray": "3 2", "stroke-width": 1.5,
                }));
                const bT = svgEl("text", { x: lx + 32, y: ly + 31, "font-size": 11, fill: "#1a6b3a" });
                bT.textContent = `Bonf = ${bonf.toExponential(1)}`;
                legendGroup.appendChild(bT);
            }

            function drawHist() {
                while (barsGroup.firstChild) barsGroup.removeChild(barsGroup.firstChild);
                if (!binsCache) computeBins();
                const bins = binsCache;
                const nBins = bins.length;
                const threshold = model.get("threshold");
                const binW = PW / nBins;
                for (let i = 0; i < nBins; i++) {
                    const h = (bins[i] / binsMax) * PH;
                    const x = M.l + i * binW;
                    const y = M.t + PH - h;
                    const binRight = (i + 1) / nBins;
                    const flagged = binRight <= threshold + 1e-9;
                    barsGroup.appendChild(svgEl("rect", {
                        x, y,
                        width: Math.max(0.5, binW - 0.5),
                        height: h,
                        fill: flagged ? "#c44" : "#aaa",
                        opacity: flagged ? 0.9 : 0.55,
                    }));
                }

                const alpha = model.get("alpha");
                const bonf = model.get("bonferroni_threshold");

                const aX = pToX(Math.max(0, Math.min(1, alpha)));
                alphaLine.setAttribute("x1", aX);
                alphaLine.setAttribute("x2", aX);

                const bX = pToX(Math.max(0, Math.min(1, bonf)));
                bonfLine.setAttribute("x1", bX);
                bonfLine.setAttribute("x2", bX);

                const tX = pToX(Math.max(0, Math.min(1, threshold)));
                thresholdLine.setAttribute("x1", tX);
                thresholdLine.setAttribute("x2", tX);
                thresholdHandle.setAttribute("x", tX - 7);

                drawLegend();
            }

            function updateReadout() {
                const pvals = model.get("p_values");
                const n = pvals.length;
                const threshold = model.get("threshold");
                let count = 0;
                for (let i = 0; i < n; i++) if (pvals[i] < threshold) count++;
                const alpha = model.get("alpha");
                const bonf = model.get("bonferroni_threshold");
                const pct = (100 * count / n).toFixed(2);
                const expected = Math.round(alpha * n);

                let verdict;
                if (Math.abs(threshold - alpha) < alpha * 0.01) {
                    verdict = `<b style="color:#c44">Naive threshold.</b> Pure noise produces ${count.toLocaleString()} 'discoveries' — close to alpha*N = ${expected.toLocaleString()}.`;
                } else if (Math.abs(threshold - bonf) < Math.max(bonf * 0.01, 1e-12)) {
                    verdict = `<b style="color:#1a6b3a">Bonferroni threshold.</b> Only voxels with p &lt; 0.05/N pass. ${count.toLocaleString()} 'discoveries'.`;
                } else {
                    verdict = `<span style="color:#555">Custom threshold.</span> ${count.toLocaleString()} 'discoveries' from pure noise.`;
                }

                const thrStr = threshold < 1e-3 ? threshold.toExponential(2) : threshold.toFixed(4);
                readout.innerHTML =
                    `<span style="color:#555">Threshold:</span> <b>p &lt; ${thrStr}</b> &nbsp;·&nbsp; ` +
                    `<span style="color:#555">Flagged:</span> <b>${count.toLocaleString()}</b> of ${n.toLocaleString()} (${pct}%)<br>` +
                    verdict;
            }

            function redraw() { drawGrid(); drawHist(); updateReadout(); }

            let dragging = false;
            function ptX(evt) {
                const rect = svg.getBoundingClientRect();
                return (evt.clientX - rect.left) * (HIST_W / rect.width);
            }
            function setFromX(x) {
                let p = xToP(x);
                p = Math.max(1e-8, Math.min(1.0, p));
                model.set("threshold", p);
                model.save_changes();
            }
            svg.addEventListener("mousedown", (e) => { dragging = true; setFromX(ptX(e)); });
            window.addEventListener("mousemove", (e) => { if (dragging) setFromX(ptX(e)); });
            window.addEventListener("mouseup", () => { dragging = false; });

            model.on("change:threshold", redraw);
            model.on("change:p_values", () => { binsCache = null; redraw(); });
            model.on("change:alpha", redraw);
            model.on("change:bonferroni_threshold", redraw);

            redraw();
        }

        export default { render };
        """

        p_values = traitlets.List(trait=traitlets.Float()).tag(sync=True)
        alpha = traitlets.Float(0.05).tag(sync=True)
        bonferroni_threshold = traitlets.Float(2.5e-6).tag(sync=True)
        threshold = traitlets.Float(0.05).tag(sync=True)

    return (FalsePositiveHunter,)


@app.cell
def _(mo):
    n_timepoints = mo.ui.slider(
        start=20, stop=1000, step=10, value=200, label="n_timepoints"
    )
    n_voxels = mo.ui.slider(
        start=100, stop=100_000, step=100, value=20_000, label="n_voxels"
    )
    alpha = mo.ui.slider(
        start=0.001, stop=0.2, step=0.001, value=0.05, label="alpha"
    )
    seed = mo.ui.number(start=0, stop=10_000, step=1, value=42, label="seed")

    mo.vstack([n_timepoints, n_voxels, alpha, seed])
    return alpha, n_timepoints, n_voxels, seed


@app.cell(hide_code=True)
def _(alpha, n_timepoints, n_voxels, seed, simulate_false_positives):
    result = simulate_false_positives(
        n_timepoints=n_timepoints.value,
        n_voxels=n_voxels.value,
        alpha=alpha.value,
        seed=int(seed.value),
    )
    return (result,)


@app.cell(hide_code=True)
def _(FalsePositiveHunter, mo, result):
    hunter = mo.ui.anywidget(
        FalsePositiveHunter(
            p_values=result.p_values.tolist(),
            alpha=float(result.alpha),
            bonferroni_threshold=float(result.bonferroni_threshold),
            threshold=float(result.alpha),
        )
    )
    hunter
    return


if __name__ == "__main__":
    app.run()
