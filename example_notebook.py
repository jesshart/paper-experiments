# /// script
# dependencies = [
#     "anywidget==0.10.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.4",
#     "scikit-learn==1.8.0",
#     "traitlets==5.14.3",
#     "vega-datasets==0.9.0",
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
    from vega_datasets import data

    return anywidget, data, dataclass, math, mo, np, traitlets


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Does this happen with real data? Yes.

    We just showed pure noise producing false positives. The math doesn't
    care whether the "noise" is synthetic — any time you run many tests
    with no real effect, ~alpha of them will flag as "significant".

    Below we pull the classic **cars dataset** from `vega-datasets` and
    p-hack it: split the cars into two groups by coin flip (no real
    difference between groups), then run a t-test on every numeric
    feature. Repeat many times. By design every test is null — yet ~5%
    of them will come up "significant".
    """)
    return


@app.cell
def _(mo):
    n_splits = mo.ui.slider(
        start=50, stop=2000, step=50, value=500, label="random splits"
    )
    n_splits
    return (n_splits,)


@app.cell(hide_code=True)
def _(data, n_splits, normal_cdf, np):
    def _p_hack_dataset(df, n_splits, seed=1):
        """Coin-flip cars into two groups; Welch t-test each numeric feature.

        Returns p-values, raw effect sizes (mean_A - mean_B), and the feature
        tested for every (split, feature) pair — flattened in the same order.
        """
        rng = np.random.default_rng(seed)
        numeric = df.select_dtypes(include="number")
        X = numeric.to_numpy(dtype=float)
        X = X[~np.isnan(X).any(axis=1)]
        n = X.shape[0]
        feat_names = numeric.columns.tolist()

        p_chunks, e_chunks = [], []
        for _ in range(n_splits):
            group = rng.integers(0, 2, size=n).astype(bool)
            if not group.any() or group.all():
                continue
            a = X[group]
            b = X[~group]
            mean_diff = a.mean(axis=0) - b.mean(axis=0)
            se = np.sqrt(
                a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b)
            )
            se = np.maximum(se, 1e-12)
            t = mean_diff / se
            p_chunks.append(2.0 * (1.0 - normal_cdf(np.abs(t))))
            e_chunks.append(mean_diff)

        return {
            "p_values": np.concatenate(p_chunks),
            "effect_sizes": np.concatenate(e_chunks),
            "features_per_test": feat_names * len(p_chunks),
            "feature_names": feat_names,
        }


    cars_df = data.cars()
    cars_phack = _p_hack_dataset(cars_df, n_splits.value)
    cars_p_values = cars_phack["p_values"]
    cars_features = cars_phack["feature_names"]
    cars_total_tests = len(cars_p_values)
    cars_alpha = 0.05
    cars_bonf = cars_alpha / cars_total_tests
    return (
        cars_alpha,
        cars_bonf,
        cars_df,
        cars_features,
        cars_p_values,
        cars_phack,
        cars_total_tests,
    )


@app.cell(hide_code=True)
def _(cars_df, cars_features, cars_total_tests, mo, n_splits):
    mo.md(f"""
    **Dataset:** `cars.json` — {len(cars_df):,} cars ·
    features tested: {", ".join(f"`{c}`" for c in cars_features)}

    **Tests run:** {n_splits.value:,} random splits × {len(cars_features)} features
    = **{cars_total_tests:,}** t-tests, all null by construction.
    """)
    return


@app.cell(hide_code=True)
def _(FalsePositiveHunter, cars_alpha, cars_bonf, cars_p_values, mo):
    cars_hunter = mo.ui.anywidget(
        FalsePositiveHunter(
            p_values=cars_p_values.tolist(),
            alpha=float(cars_alpha),
            bonferroni_threshold=float(cars_bonf),
            threshold=float(cars_alpha),
        )
    )
    cars_hunter
    return (cars_hunter,)


@app.cell(hide_code=True)
def _(
    cars_bonf,
    cars_df,
    cars_features,
    cars_hunter,
    cars_phack,
    cars_total_tests,
    mo,
    np,
):
    _threshold = cars_hunter.value["threshold"]
    _p_vals = cars_phack["p_values"]
    _effects = cars_phack["effect_sizes"]
    _feats = cars_phack["features_per_test"]

    _UNITS = {
        "Miles_per_Gallon": "mpg",
        "Cylinders": "cylinders",
        "Displacement": "cc displacement",
        "Horsepower": "hp",
        "Weight_in_lbs": "lbs",
        "Acceleration": "sec 0-60",
    }

    _passing = np.where(_p_vals < _threshold)[0]
    _n_passing = int(len(_passing))

    _items = []
    if _n_passing > 0:
        _top = _passing[np.argsort(_p_vals[_passing])[:5]]
        for _i in _top:
            _feat = _feats[int(_i)]
            _eff = float(_effects[int(_i)])
            _p = float(_p_vals[int(_i)])
            _dir = "higher" if _eff > 0 else "lower"
            _unit = _UNITS.get(_feat, _feat)
            _p_str = f"{_p:.2e}" if _p < 1e-3 else f"{_p:.4f}"
            _items.append(
                "<li style='margin-bottom:6px;'>"
                f"Cohort A exhibited <b>{abs(_eff):.2f} {_unit}</b> {_dir} than Cohort B "
                f"(<i>p</i> = {_p_str}).</li>"
            )

    _show_stamp = _n_passing > 0 and _threshold > cars_bonf + 1e-10

    _stamp = (
        (
            """
    <div style="
        position: absolute;
        top: 80px;
        right: 50px;
        transform: rotate(-14deg);
        font-family: 'Arial Black', Impact, sans-serif;
        font-size: 108px;
        font-weight: 900;
        color: rgba(190, 30, 30, 0.80);
        border: 10px double rgba(190, 30, 30, 0.80);
        padding: 4px 24px;
        letter-spacing: 8px;
        pointer-events: none;
        user-select: none;
        line-height: 1;
        z-index: 2;
    ">FALSE</div>
    """
        )
        if _show_stamp
        else ""
    )

    _results_list = (
        "<ul style='padding-left: 22px; margin: 0 0 14px 0;'>"
        + "".join(_items)
        + "</ul>"
        if _items
        else "<p style='color:#888; font-style:italic; margin: 0 0 14px 0;'>No findings at this threshold &mdash; there is no paper to publish.</p>"
    )

    _closing = (
        (
            "Our analysis reveals consistent and measurable differences between cohorts "
            "across multiple vehicle dimensions. We recommend manufacturer review and "
            "further investigation in light of these findings."
        )
        if _n_passing > 0
        else (
            "No effects were detected. The null hypothesis could not be rejected."
        )
    )

    _threshold_str = (
        f"{_threshold:.2e}" if _threshold < 1e-3 else f"{_threshold:.4f}"
    )

    mo.md(f"""
    <div style="
        position: relative;
        max-width: 800px;
        margin: 20px auto;
        padding: 44px 56px 40px 56px;
        background: #fefef8;
        border: 1px solid #cac8c0;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        font-family: Georgia, 'Times New Roman', serif;
        color: #222;
        line-height: 1.55;
        overflow: hidden;
    ">
        {_stamp}
        <div style="text-align: center; border-bottom: 1px solid #888; padding-bottom: 14px; margin-bottom: 18px;">
            <div style="font-size: 10px; color: #666; letter-spacing: 2px; text-transform: uppercase;">
                Journal of Automotive Cohort Analysis &middot; Vol. 47 &middot; No. 3 &middot; pp. 412-418
            </div>
            <div style="font-size: 22px; margin: 10px 0 6px 0; line-height: 1.25; color: #111; font-weight: 700;">
                Systematic Variation Across Randomly Partitioned Automotive Cohorts
            </div>
            <div style="font-size: 15px; font-weight: 400; font-style: italic; color: #444; margin-bottom: 8px;">
                A Multi-Feature Analysis of {len(cars_df)} Vehicles
            </div>
            <div style="font-style: italic; color: #444; font-size: 13px;">
                M. Anon<sup>1</sup>, J. Hartman<sup>1</sup>, A. Claude<sup>2</sup>
            </div>
            <div style="font-size: 10px; color: #666; margin-top: 4px; letter-spacing: 0.5px;">
                <sup>1</sup>Institute for Premature Publication &middot; <sup>2</sup>Dept. of Exuberant Inference
            </div>
        </div>

        <div style="font-size: 13px;">
            <div style="font-variant: small-caps; letter-spacing: 1.5px; font-size: 13px; margin: 0 0 6px 0; color: #333; font-weight: 700;">Abstract</div>
            <p style="text-align: justify; margin: 0 0 14px 0;">
                We analyzed <b>{len(cars_df)}</b> automobiles partitioned into two cohorts
                and conducted a systematic comparison across {len(cars_features)} mechanical
                and efficiency attributes. At a significance threshold of
                <i>p</i> &lt; {_threshold_str}, we identified
                <b>{_n_passing:,} statistically significant differences</b>.
                These results have implications for manufacturing specification,
                regulatory policy, and consumer guidance.
            </p>

            <div style="font-variant: small-caps; letter-spacing: 1.5px; font-size: 13px; margin: 0 0 6px 0; color: #333; font-weight: 700;">Selected Results</div>
            {_results_list}

            <div style="font-variant: small-caps; letter-spacing: 1.5px; font-size: 13px; margin: 0 0 6px 0; color: #333; font-weight: 700;">Conclusion</div>
            <p style="text-align: justify; margin: 0 0 14px 0;">
                {_closing}
            </p>

            <div style="font-size: 10px; color: #888; border-top: 1px dashed #bbb; padding-top: 10px; text-align: center; letter-spacing: 0.5px;">
                Manuscript generated from {cars_total_tests:,} null t-tests on vega-datasets/cars.json.
                Cohort assignment: uniformly random coin flip per vehicle.
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # The same trap, in modern clothes

    Everything above was classical statistics dressed for fMRI and for cars.
    The same failure mode shows up across modern ML interpretability tools.
    Below: three case studies where confident-looking explanations turn out
    to be artifacts of high-dimensional noise, not of anything the model
    actually learned or the data actually contains.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Feature Attribution

    Saliency maps, Grad-CAM, SHAP, Integrated Gradients — the promise is:
    "here are the parts of the input the model cared about." Adebayo et al.
    (2018) showed that many popular attribution methods produce **nearly
    identical heatmaps even when you randomize the model's weights**. That
    means the pretty overlay is mostly a projection of the input, not
    evidence of what the model learned.

    Below: we train a tiny linear classifier on synthetic 16×16 shapes
    (circles vs squares), then compute input-times-gradient saliency for
    both the trained weights and a fresh random model. The "hot" regions
    line up on the shape in both cases — because `|W·x|` is near zero
    wherever `x` is near zero, regardless of what `W` is.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    from sklearn.linear_model import LogisticRegression


    def _make_shape_dataset(n_per_class=60, size=16, noise=0.12, seed=0):
        """Generate 16x16 grayscale shape images (0=disk, 1=square) with
        position jitter and Gaussian noise. Returns flat (n, size*size) array."""
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:size, 0:size]
        imgs, labels = [], []
        for lbl in (0, 1):
            for _ in range(n_per_class):
                img = rng.normal(0.0, noise, size=(size, size))
                cy = size // 2 + int(rng.integers(-2, 3))
                cx = size // 2 + int(rng.integers(-2, 3))
                r = 3
                if lbl == 0:
                    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
                else:
                    mask = (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)
                img = img + 0.9 * mask
                imgs.append(img.ravel())
                labels.append(lbl)
        X = np.array(imgs, dtype=float)
        y = np.array(labels, dtype=int)
        perm = rng.permutation(len(X))
        return X[perm], y[perm]


    _attr_X, _attr_y = _make_shape_dataset(n_per_class=60, seed=0)
    _attr_clf = LogisticRegression(max_iter=1000, C=10.0).fit(_attr_X, _attr_y)
    attr_W = _attr_clf.coef_[0].astype(float)
    attr_b = float(_attr_clf.intercept_[0])
    attr_train_acc = float(_attr_clf.score(_attr_X, _attr_y))

    attr_X_test, attr_y_test = _make_shape_dataset(n_per_class=10, seed=7)
    attr_label_names = ["circle", "square"]
    return attr_W, attr_X_test, attr_label_names, attr_y_test


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class AttributionCompare(anywidget.AnyWidget):
        """Side-by-side saliency comparison: trained vs random model weights.

        The demo shows input-times-gradient saliency |W * x| for both a trained
        linear classifier and a fresh model with random weights of matching scale.
        Both maps light up the shape region — because |W * x| is near zero
        wherever x is near zero, regardless of W. The fingerprint of the input
        dominates the fingerprint of the model.
        """

        _esm = r"""
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
                gap: 12px;
            `;

            const row = document.createElement("div");
            row.style.cssText = "display: flex; gap: 14px; justify-content: center; flex-wrap: wrap;";

            const DISPLAY = 180;

            function makePanel(title) {
                const p = document.createElement("div");
                p.style.cssText = "display: flex; flex-direction: column; align-items: center; gap: 4px;";
                const t = document.createElement("div");
                t.style.cssText = "font-size: 12px; color: #333; font-weight: 500;";
                t.textContent = title;
                const c = document.createElement("canvas");
                c.width = DISPLAY;
                c.height = DISPLAY;
                c.style.cssText = "border: 1px solid #aaa; image-rendering: pixelated; background: #000;";
                p.appendChild(t);
                p.appendChild(c);
                return { panel: p, canvas: c, title: t };
            }

            const pInput = makePanel("Input image");
            const pTrained = makePanel("Saliency — trained model");
            const pRandom = makePanel("Saliency — random model");
            row.appendChild(pInput.panel);
            row.appendChild(pTrained.panel);
            row.appendChild(pRandom.panel);

            const controls = document.createElement("div");
            controls.style.cssText = "display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: center;";

            function makeBtn(label, onClick) {
                const b = document.createElement("button");
                b.textContent = label;
                b.style.cssText = "padding: 6px 12px; border: 1px solid #888; background: #fff; color: #222; border-radius: 4px; cursor: pointer; font-size: 12px; font-family: sans-serif;";
                b.onclick = onClick;
                return b;
            }

            const idxLabel = document.createElement("span");
            idxLabel.style.cssText = "font-family: ui-monospace, monospace; font-size: 12px; color: #333; min-width: 110px; text-align: center;";

            controls.appendChild(makeBtn("Prev", () => {
                const n = model.get("n_test");
                let s = model.get("selected");
                s = (s - 1 + n) % n;
                model.set("selected", s);
                model.save_changes();
            }));
            controls.appendChild(idxLabel);
            controls.appendChild(makeBtn("Next", () => {
                const n = model.get("n_test");
                let s = model.get("selected");
                s = (s + 1) % n;
                model.set("selected", s);
                model.save_changes();
            }));
            controls.appendChild(makeBtn("Re-roll random model", () => {
                model.set("random_seed", model.get("random_seed") + 1);
                model.save_changes();
            }));

            const readout = document.createElement("div");
            readout.style.cssText = "font-family: ui-monospace, monospace; font-size: 13px; line-height: 1.55; padding: 10px 12px; background: #fff; color: #222; border: 1px solid #e0e0d8; border-radius: 4px;";

            wrapper.appendChild(row);
            wrapper.appendChild(controls);
            wrapper.appendChild(readout);
            el.appendChild(wrapper);

            function mulberry32(seed) {
                return function() {
                    seed = (seed + 0x6D2B79F5) | 0;
                    let t = seed;
                    t = Math.imul(t ^ (t >>> 15), t | 1);
                    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
                    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
                };
            }
            function randn(rng) {
                const u = 1 - rng();
                const v = rng();
                return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
            }
            function stddev(arr) {
                let m = 0;
                for (const v of arr) m += v;
                m /= arr.length;
                let s = 0;
                for (const v of arr) s += (v - m) * (v - m);
                return Math.sqrt(s / arr.length);
            }
            function pearson(a, b) {
                const n = a.length;
                let sa=0, sb=0, saa=0, sbb=0, sab=0;
                for (let i = 0; i < n; i++) {
                    sa += a[i]; sb += b[i];
                    saa += a[i] * a[i]; sbb += b[i] * b[i];
                    sab += a[i] * b[i];
                }
                const ma = sa / n, mb = sb / n;
                const cov = sab / n - ma * mb;
                const va = saa / n - ma * ma, vb = sbb / n - mb * mb;
                return cov / (Math.sqrt(va * vb) + 1e-12);
            }

            function grayscale(t) {
                t = Math.max(0, Math.min(1, t));
                const v = Math.floor(t * 255);
                return "rgb(" + v + "," + v + "," + v + ")";
            }
            function hot(t) {
                t = Math.max(0, Math.min(1, t));
                let r, g, b;
                if (t < 0.33) { r = Math.floor(t / 0.33 * 255); g = 0; b = 0; }
                else if (t < 0.67) { r = 255; g = Math.floor((t - 0.33) / 0.34 * 255); b = 0; }
                else { r = 255; g = 255; b = Math.floor((t - 0.67) / 0.33 * 255); }
                return "rgb(" + r + "," + g + "," + b + ")";
            }

            function drawImage(canvas, arr, colormap, clipPct) {
                const ctx = canvas.getContext("2d");
                const size = Math.round(Math.sqrt(arr.length));
                // robust normalization: clip top 2%
                const sorted = arr.slice().sort((a, b) => a - b);
                const mn = sorted[Math.floor(sorted.length * (clipPct || 0))];
                const mx = sorted[Math.floor(sorted.length * (1 - (clipPct || 0))) - 1];
                const range = Math.max(mx - mn, 1e-12);
                const px = canvas.width / size;
                for (let i = 0; i < size; i++) {
                    for (let j = 0; j < size; j++) {
                        const val = (arr[i * size + j] - mn) / range;
                        ctx.fillStyle = colormap(val);
                        ctx.fillRect(Math.floor(j * px), Math.floor(i * px), Math.ceil(px), Math.ceil(px));
                    }
                }
            }

            function refresh() {
                const images = model.get("images");
                const labels = model.get("labels");
                const Wt = model.get("W_trained");
                const size = model.get("image_size");
                const n = model.get("n_test");
                const names = model.get("label_names");
                const sel = model.get("selected");
                const seed = model.get("random_seed");

                const pixels = size * size;
                const img = images.slice(sel * pixels, (sel + 1) * pixels);

                const wStd = stddev(Wt);
                const rng = mulberry32(seed * 1337 + 42);
                const Wr = new Array(pixels);
                for (let i = 0; i < pixels; i++) Wr[i] = randn(rng) * wStd;

                const trainedSal = new Array(pixels);
                const randomSal = new Array(pixels);
                for (let i = 0; i < pixels; i++) {
                    trainedSal[i] = Math.abs(Wt[i] * img[i]);
                    randomSal[i] = Math.abs(Wr[i] * img[i]);
                }

                drawImage(pInput.canvas, img, grayscale, 0.02);
                drawImage(pTrained.canvas, trainedSal, hot, 0.02);
                drawImage(pRandom.canvas, randomSal, hot, 0.02);

                pInput.title.textContent = "Input: " + names[labels[sel]];
                idxLabel.textContent = "Image " + (sel + 1) + " of " + n;

                const r = pearson(trainedSal, randomSal);
                const rColor = r > 0.5 ? "#c44" : (r > 0.2 ? "#b80" : "#555");
                readout.innerHTML =
                    "<b>Pearson r between trained and random saliency:</b> " +
                    "<span style='color:" + rColor + "; font-weight:700;'>" + r.toFixed(3) + "</span><br>" +
                    "<span style='color:#555;'>Both maps emphasize the shape region because " +
                    "saliency = |W &middot; x| is near zero wherever x is near zero &mdash; " +
                    "regardless of whether W came from training or from random noise.</span>";
            }

            model.on("change:selected", refresh);
            model.on("change:random_seed", refresh);
            refresh();
        }

        export default { render };
        """

        images = traitlets.List(trait=traitlets.Float()).tag(sync=True)
        labels = traitlets.List(trait=traitlets.Int()).tag(sync=True)
        W_trained = traitlets.List(trait=traitlets.Float()).tag(sync=True)
        image_size = traitlets.Int(16).tag(sync=True)
        n_test = traitlets.Int(0).tag(sync=True)
        label_names = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
        selected = traitlets.Int(0).tag(sync=True)
        random_seed = traitlets.Int(0).tag(sync=True)

    return (AttributionCompare,)


@app.cell(hide_code=True)
def _(
    AttributionCompare,
    attr_W,
    attr_X_test,
    attr_label_names,
    attr_y_test,
    mo,
):
    attr_widget = mo.ui.anywidget(
        AttributionCompare(
            images=attr_X_test.flatten().astype(float).tolist(),
            labels=attr_y_test.astype(int).tolist(),
            W_trained=attr_W.tolist(),
            image_size=16,
            n_test=int(len(attr_X_test)),
            label_names=attr_label_names,
        )
    )
    attr_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Probing

    Probing classifiers are the "did the model encode X?" tool: train a
    linear probe on a model's internal activations to decode some property,
    and take success to mean the model "knows about" that property.

    Hewitt & Liang (2019) showed the trap: **a probe can succeed even when
    the model is not using the decoded information for its output**. A
    high-dimensional projection preserves almost everything fed into it — a
    probe can recover a feature from the activations even if the downstream
    readout completely ignores it.

    Below: synthetic features flow through a fixed random hidden layer. A
    linear readout is trained on `y = sin(1.5·x1) + noise` — only `x1`
    matters. For every feature we measure:

    - **Probe R²** — can a linear probe decode `xⱼ` from the hidden activations?
    - **Ablation impact** — when we replace `xⱼ` with its mean, how much
      does the model's output change?

    The probe says "yes, it's all in there!" The ablation says "no, only
    `x1` is actually used." Both are true at the same time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    probing_n_nuisance = mo.ui.slider(
        start=2, stop=20, step=1, value=5, label="nuisance features"
    )
    probing_n_nuisance
    return (probing_n_nuisance,)


@app.cell(hide_code=True)
def _(np, probing_n_nuisance):
    from sklearn.linear_model import Ridge


    def _probing_experiment(n_samples=500, n_nuisance=5, hidden=32, seed=0):
        """Fixed random hidden layer + trained linear readout. Only x1 drives y.

        For each feature xj, computes: (a) the R² of a linear probe that
        decodes xj from the hidden activations, and (b) the RMS change in
        model output when xj is replaced by its mean.
        """
        rng = np.random.default_rng(seed)
        k = 1 + n_nuisance
        X = rng.normal(0, 1, size=(n_samples, k))
        y = np.sin(X[:, 0] * 1.5) + 0.1 * rng.normal(size=n_samples)

        W0 = rng.normal(0, 1 / np.sqrt(k), size=(k, hidden))
        b0 = rng.normal(0, 0.1, size=hidden)
        H = np.tanh(X @ W0 + b0)

        readout = Ridge(alpha=0.1).fit(H, y)
        y_pred = readout.predict(H)

        probe_r2 = []
        for j in range(k):
            p = Ridge(alpha=0.01).fit(H, X[:, j])
            probe_r2.append(float(max(0.0, p.score(H, X[:, j]))))

        abl_deltas = []
        for j in range(k):
            X_abl = X.copy()
            X_abl[:, j] = X[:, j].mean()
            H_abl = np.tanh(X_abl @ W0 + b0)
            y_abl = readout.predict(H_abl)
            abl_deltas.append(float(np.sqrt(np.mean((y_abl - y_pred) ** 2))))

        abl_arr = np.array(abl_deltas)
        abl_norm = (abl_arr / max(float(abl_arr.max()), 1e-12)).tolist()

        return {
            "feature_names": [f"x{j + 1}" for j in range(k)],
            "probe_r2": probe_r2,
            "ablation_impact": abl_norm,
            "ablation_raw": abl_deltas,
        }


    probing_result = _probing_experiment(
        n_nuisance=probing_n_nuisance.value, seed=1
    )
    return (probing_result,)


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class ProbingCompare(anywidget.AnyWidget):
        """Grouped bar chart: per-feature probe R² vs ablation impact.

        Visualizes the Hewitt-Liang insight: probes can recover features the
        model does not actually use for its output.
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

            const W = 640, H = 340;
            const M = { l: 50, r: 15, t: 50, b: 42 };
            const PW = W - M.l - M.r;
            const PH = H - M.t - M.b;

            const svg = svgEl("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H });
            svg.style.cssText = "background: #fff; border: 1px solid #ccc; border-radius: 4px; user-select: none;";

            const title = svgEl("text", {
                x: W / 2, y: 20, "text-anchor": "middle",
                "font-size": 13, "font-weight": 600, fill: "#333",
            });
            title.textContent = "Probe-decodability vs ablation-impact, per feature";
            svg.appendChild(title);

            const legend = svgEl("g", {});
            legend.appendChild(svgEl("rect", { x: 70, y: 30, width: 14, height: 10, fill: "#3a77bf" }));
            const l1 = svgEl("text", { x: 90, y: 39, "font-size": 11, fill: "#333" });
            l1.textContent = "Probe R² (linear decode from hidden)";
            legend.appendChild(l1);
            legend.appendChild(svgEl("rect", { x: 360, y: 30, width: 14, height: 10, fill: "#c44" }));
            const l2 = svgEl("text", { x: 380, y: 39, "font-size": 11, fill: "#333" });
            l2.textContent = "Ablation impact on output (normalized)";
            legend.appendChild(l2);
            svg.appendChild(legend);

            svg.appendChild(svgEl("line", { x1: M.l, y1: M.t, x2: M.l, y2: M.t + PH, stroke: "#333" }));
            for (let i = 0; i <= 4; i++) {
                const yv = i / 4;
                const py = M.t + PH - yv * PH;
                svg.appendChild(svgEl("line", { x1: M.l - 4, y1: py, x2: M.l, y2: py, stroke: "#333" }));
                const tk = svgEl("text", {
                    x: M.l - 6, y: py + 3, "font-size": 10, fill: "#555", "text-anchor": "end",
                });
                tk.textContent = yv.toFixed(2);
                svg.appendChild(tk);
            }
            const ylab = svgEl("text", {
                x: 14, y: M.t + PH / 2, "text-anchor": "middle", "font-size": 11, fill: "#333",
                transform: "rotate(-90 14 " + (M.t + PH / 2) + ")",
            });
            ylab.textContent = "score";
            svg.appendChild(ylab);

            svg.appendChild(svgEl("line", { x1: M.l, y1: M.t + PH, x2: M.l + PW, y2: M.t + PH, stroke: "#333" }));

            const barsGroup = svgEl("g", {});
            svg.appendChild(barsGroup);

            const readout = document.createElement("div");
            readout.style.cssText = "font-family: ui-monospace, monospace; font-size: 13px; line-height: 1.55; padding: 10px 12px; background: #fff; color: #222; border: 1px solid #e0e0d8; border-radius: 4px;";

            wrapper.appendChild(svg);
            wrapper.appendChild(readout);
            el.appendChild(wrapper);

            function redraw() {
                while (barsGroup.firstChild) barsGroup.removeChild(barsGroup.firstChild);
                const features = model.get("feature_names");
                const probes = model.get("probe_r2");
                const ablations = model.get("ablation_impact");
                const k = features.length;
                if (k === 0) return;
                const slot = PW / k;
                const barW = Math.min(24, slot * 0.35);

                for (let i = 0; i < k; i++) {
                    const cx = M.l + i * slot + slot / 2;
                    const probeH = Math.max(0, Math.min(1, probes[i])) * PH;
                    barsGroup.appendChild(svgEl("rect", {
                        x: cx - barW - 2, y: M.t + PH - probeH,
                        width: barW, height: probeH,
                        fill: "#3a77bf", opacity: 0.85,
                    }));
                    const ablH = Math.max(0, Math.min(1, ablations[i])) * PH;
                    barsGroup.appendChild(svgEl("rect", {
                        x: cx + 2, y: M.t + PH - ablH,
                        width: barW, height: ablH,
                        fill: "#c44", opacity: 0.85,
                    }));
                    const lbl = svgEl("text", {
                        x: cx, y: M.t + PH + 16, "text-anchor": "middle",
                        "font-size": 11, fill: "#333",
                        "font-weight": i === 0 ? 700 : 400,
                    });
                    lbl.textContent = features[i];
                    barsGroup.appendChild(lbl);
                }

                const nProbeHigh = probes.filter(function(r) { return r > 0.5; }).length;
                const nAblHigh = ablations.filter(function(a) { return a > 0.1; }).length;
                readout.innerHTML =
                    "Probes can decode <b style='color:#3a77bf'>" + nProbeHigh + " of " + k +
                    " features</b> from the hidden activations. " +
                    "Ablating features changes the model's output for only " +
                    "<b style='color:#c44'>" + nAblHigh + " of " + k + "</b>.<br>" +
                    "<span style='color:#555'>The representation <i>contains</i> the nuisance features; " +
                    "the output does not <i>use</i> them. A successful probe is not evidence " +
                    "that the model relies on the decoded feature.</span>";
            }

            model.on("change:probe_r2", redraw);
            model.on("change:ablation_impact", redraw);
            model.on("change:feature_names", redraw);
            redraw();
        }

        export default { render };
        """

        feature_names = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
        probe_r2 = traitlets.List(trait=traitlets.Float()).tag(sync=True)
        ablation_impact = traitlets.List(trait=traitlets.Float()).tag(sync=True)

    return (ProbingCompare,)


@app.cell(hide_code=True)
def _(ProbingCompare, mo, probing_result):
    probing_widget = mo.ui.anywidget(
        ProbingCompare(
            feature_names=probing_result["feature_names"],
            probe_r2=probing_result["probe_r2"],
            ablation_impact=probing_result["ablation_impact"],
        )
    )
    probing_widget
    return


if __name__ == "__main__":
    app.run()
