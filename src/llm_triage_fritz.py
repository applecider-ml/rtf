"""
LLM Triage on Fritz sources using Groq/Llama with full AppleCiDEr physics.

Takes the anomaly-scored Fritz sources, merges with physics features from
the Rust fitter, and sends the top N per group to Groq for reasoning.

Usage:
    python llm_triage_fritz.py \
        --scores-csv anomaly_scores.csv \
        --features fritz_features.parquet \
        --manifest fritz_manifest.csv \
        --fritz-dir ../fritz_sources \
        --groq-key gsk_XXXXX \
        --top-n 30 \
        --min-detections 10 \
        --output-json llm_triage_results.json \
        --output-report triage_report.md

    Dry run (no API calls):
        --dry-run
"""

import argparse
import json
import math
import os
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2

# Parametric model ID → name
MODEL_NAMES = {
    0: "Bazin",
    1: "Villar",
    2: "MetzgerKN",
    3: "Tde",
    4: "Arnett",
    5: "Magnetar",
    6: "ShockCooling",
    7: "Afterglow",
}

# Per-group system prompts
SYSTEM_PROMPTS = {
    "SLSNe": """You are an expert observational astrophysicist specializing in Superluminous Supernovae (SLSNe).
An autoencoder trained on 1.74 million ZTF alerts has flagged this object as anomalous, meaning its light curve shape
deviates significantly from the learned baseline of "normal" transients.

Focus your analysis on:
- Is the peak luminosity consistent with SLSNe (peak mag brighter than -21)?
- Is the evolution slow (rise > 20-30 days)?
- Are the colors consistent with hydrogen-poor (SLSN-I) or hydrogen-rich (SLSN-II)?
- Does the parametric fitting favor Magnetar or Arnett models?
- Could this be a misclassified normal SN Ia or SN IIn?

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{"verdict": "interesting|known_type|artifact", "confidence": 0.0-1.0, "reasoning": "2-3 sentences", "suggested_class": "your best guess", "followup": "recommended observations"}""",
    "Fast_Transients": """You are an expert observational astrophysicist specializing in fast transients (FBOTs, afterglows, rapid novae).
An autoencoder trained on 1.74 million ZTF alerts has flagged this object as anomalous.

Focus your analysis on:
- Is the rise time very fast (< 5 days)? This is the hallmark of FBOTs.
- Is the peak color blue (g-r < 0)? Blue + fast = possible FBOT.
- Does the decay follow a power law (afterglow) or exponential (nova)?
- Does the parametric fitting favor Afterglow or ShockCooling models?
- Could this be a cataclysmic variable or flare star masquerading as a fast transient?

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{"verdict": "interesting|known_type|artifact", "confidence": 0.0-1.0, "reasoning": "2-3 sentences", "suggested_class": "your best guess", "followup": "recommended observations"}""",
    "LIONS": """You are an expert observational astrophysicist reviewing live interesting transient candidates from the LIONS program.
An autoencoder trained on 1.74 million ZTF alerts has flagged this object as anomalous.

This is a mixed-science group. Evaluate broadly:
- Does the light curve morphology match any known transient class?
- Are there unusual features (double-peaked, extremely fast/slow, color reversals)?
- Does the parametric fitting strongly prefer one model over others (low delta chi2 = ambiguous)?
- Is there evidence of interaction (CSM, late-time rebrightening)?
- Could this be an instrumental artifact or data quality issue?

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{"verdict": "interesting|known_type|artifact", "confidence": 0.0-1.0, "reasoning": "2-3 sentences", "suggested_class": "your best guess", "followup": "recommended observations"}""",
}

# Default system prompt for unknown groups
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPTS["LIONS"]

# Band colors for light curve plots
BAND_COLORS = {0: "#2ecc71", 1: "#e74c3c", 2: "#f39c12"}  # g=green, r=red, i=orange
BAND_LABELS = {0: "g", 1: "r", 2: "i"}


def plot_lightcurve(fritz_npz_path, output_png_path, obj_id, classification=""):
    """Plot a multi-band light curve from a Fritz NPZ file.

    Returns True if plot was generated, False otherwise.
    """
    try:
        data = np.load(fritz_npz_path, allow_pickle=True)
        x = data["x"]  # (L, 37)
    except Exception:
        return False

    L = x.shape[0]
    if L < 5:
        return False

    # Extract columns
    dt = np.expm1(x[:, 0])  # days since first detection
    logflux = x[:, 2]  # -0.4 * mag → mag = logflux / -0.4
    logflux_err = x[:, 3]  # 0.4 * magerr → magerr = logflux_err / 0.4
    band_id = x[:, 4:7].argmax(axis=1)  # 0=g, 1=r, 2=i

    # Convert to magnitude
    mag = logflux / -0.4
    magerr = logflux_err / 0.4

    # Filter out non-finite values
    valid = np.isfinite(mag) & np.isfinite(magerr) & (magerr > 0) & (magerr < 5)
    dt = dt[valid]
    mag = mag[valid]
    magerr = magerr[valid]
    band_id = band_id[valid]

    if len(dt) < 3:
        return False

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    for bid in sorted(set(band_id)):
        mask = band_id == bid
        ax.errorbar(
            dt[mask],
            mag[mask],
            yerr=magerr[mask],
            fmt="o",
            ms=3.5,
            lw=0.8,
            capsize=1.5,
            color=BAND_COLORS.get(bid, "gray"),
            label=BAND_LABELS.get(bid, f"band{bid}"),
            alpha=0.8,
            zorder=3,
        )

    ax.invert_yaxis()  # Astronomical convention: brighter = lower mag
    ax.set_xlabel("Days since first detection", fontsize=9)
    ax.set_ylabel("Magnitude", fontsize=9)
    title = obj_id
    if classification and str(classification) != "nan":
        title += f"  [{classification}]"
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_png_path, bbox_inches="tight", facecolor="#fafafa")
    plt.close(fig)
    return True


def _safe_get(series, col, fmt=".2f", default="N/A"):
    """Safely extract a value from a pandas Series."""
    val = series.get(col, None)
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return default
    return f"{val:{fmt}}"


def build_physics_prompt(obj_row, feature_row):
    """Build a physics-rich prompt for a single source.

    Mirrors _build_physics_prompt() from llm_triage.py but adapted for
    Fritz sources with RTF anomaly context.
    """
    obj_id = obj_row["obj_id"]
    group = obj_row["group_name"]
    anomaly_score = obj_row["anomaly_score"]
    recon_error = obj_row["recon_error"]
    if_score = obj_row["if_score"]
    classification = obj_row.get("classification", "")
    n_det = obj_row.get("n_detections", 0)

    # Best parametric model
    best_model_val = feature_row.get("param_best_model", None)
    if best_model_val is not None and not (
        isinstance(best_model_val, float) and math.isnan(best_model_val)
    ):
        best_model_name = MODEL_NAMES.get(int(best_model_val), f"Model_{int(best_model_val)}")
    else:
        best_model_name = "N/A"

    # Per-model chi2 breakdown
    chi2_lines = []
    for model_name in MODEL_NAMES.values():
        chi2_val = feature_row.get(f"param_chi2_{model_name}", None)
        if chi2_val is not None and isinstance(chi2_val, (int, float)) and math.isfinite(chi2_val):
            chi2_lines.append(f"    {model_name}: chi2={chi2_val:.2f}")
    chi2_breakdown = "\n".join(chi2_lines) if chi2_lines else "    No parametric fits available"

    prompt = f"""Review this ZTF transient flagged by the anomaly detection system:

Object ID: {obj_id}
SkyPortal Group: {group}
Existing Classification: {classification if classification else "None"}
Number of Detections: {n_det}

RTF Autoencoder Analysis:
  Anomaly Score: {anomaly_score:.4f}
  Reconstruction Error: {recon_error:.6f} (training population mean: 0.001251)
  Reconstruction Error Ratio: {recon_error / 0.001251:.0f}x above baseline
  Isolation Forest Score: {if_score:.4f}

Light Curve Fitting (GP nonparametric):
  Rise time: {_safe_get(feature_row, "np_rise_time", ".1f")} days
  Decay time: {_safe_get(feature_row, "np_decay_time", ".1f")} days
  Rise rate: {_safe_get(feature_row, "np_rise_rate", ".3f")} mag/day
  Decay rate: {_safe_get(feature_row, "np_decay_rate", ".3f")} mag/day
  FWHM: {_safe_get(feature_row, "np_fwhm", ".1f")} days
  Peak magnitude: {_safe_get(feature_row, "np_peak_mag", ".2f")}
  Peak color (g-r): {_safe_get(feature_row, "np_peak_color_g_r", ".3f")}
  Color at +30d (g-r): {_safe_get(feature_row, "np_color_g_r_30d", ".3f")}
  Color evolution rate: {_safe_get(feature_row, "np_color_evolution_rate", ".4f")} mag/day
  Von Neumann ratio: {_safe_get(feature_row, "np_von_neumann_ratio", ".3f")}
  GP chi2: {_safe_get(feature_row, "np_chi2", ".2f")}
  Decay power law index: {_safe_get(feature_row, "np_decay_power_law_index", ".2f")}

Parametric Model Fitting:
  Best-fit model: {best_model_name}
  PSO chi2: {_safe_get(feature_row, "param_pso_chi2", ".3f")}
  Delta chi2 (best vs 2nd): {_safe_get(feature_row, "param_delta_chi2", ".3f")}
  Per-model chi2 breakdown:
{chi2_breakdown}

Thermal (Blackbody) Fitting:
  Peak temperature: {_safe_get(feature_row, "thermal_log_temp_peak", ".2f")} log K
  Cooling rate: {_safe_get(feature_row, "thermal_cooling_rate", ".4f")}
  Thermal chi2: {_safe_get(feature_row, "thermal_chi2", ".2f")}"""

    return prompt


def parse_llm_response(text):
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to find JSON in code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        # Try to find raw JSON
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "verdict": "error",
            "confidence": 0.0,
            "reasoning": f"Failed to parse LLM response: {text[:200]}",
            "suggested_class": "unknown",
            "followup": "N/A",
        }


def call_groq(prompt, system_prompt, groq_key, retries=3):
    """Call Groq REST API with retry logic."""
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": 500,
    }

    for attempt in range(retries):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 2)
                print(f"      Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"      API error (attempt {attempt + 1}): {e}")
            time.sleep(2**attempt)

    return '{"verdict": "error", "confidence": 0, "reasoning": "API call failed after retries", "suggested_class": "unknown", "followup": "retry"}'


def generate_report(results, output_path, plots_dir=None):
    """Generate a human-readable markdown triage report with embedded light curves."""
    lines = [
        "# RTF Anomaly Triage Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "**Model:** RTF Autoencoder (128-dim, trained on 1.74M ZTF alerts)",
        f"**LLM:** {GROQ_MODEL}",
        f"**Sources reviewed:** {len(results)}",
        "",
    ]

    # Group results
    groups = {}
    for r in results:
        g = r["group_name"]
        if g not in groups:
            groups[g] = []
        groups[g].append(r)

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Group | Reviewed | Interesting | Known Type | Artifact/Error |")
    lines.append("|---|---|---|---|---|")
    for group, items in sorted(groups.items()):
        n_int = sum(1 for r in items if r.get("verdict") == "interesting")
        n_known = sum(1 for r in items if r.get("verdict") == "known_type")
        n_art = sum(1 for r in items if r.get("verdict") in ("artifact", "error"))
        lines.append(f"| {group} | {len(items)} | {n_int} | {n_known} | {n_art} |")
    lines.append("")

    # Per-group details
    for group, items in sorted(groups.items()):
        lines.append("---")
        lines.append("")
        lines.append(f"## {group}")
        lines.append("")

        for rank, r in enumerate(items, 1):
            verdict = r.get("verdict", "unknown")
            confidence = r.get("confidence", 0)
            reasoning = r.get("reasoning", "N/A")
            suggested = r.get("suggested_class", "unknown")
            followup = r.get("followup", "N/A")

            emoji = {"interesting": "⭐", "known_type": "📋", "artifact": "🗑️"}.get(verdict, "❓")

            lines.append(f"### Rank {rank}: {r['obj_id']} (Score: {r['anomaly_score']:.4f})")
            lines.append("")

            # Embed light curve plot if available
            if plots_dir:
                png_name = f"{r['obj_id']}.png"
                png_path = os.path.join(plots_dir, png_name)
                if os.path.exists(png_path):
                    lines.append(f"![{r['obj_id']}]({plots_dir}/{png_name})")
                    lines.append("")

            lines.append(
                f"**LLM Verdict:** {emoji} {verdict.replace('_', ' ').title()} (confidence: {confidence:.2f})"
            )
            lines.append(f"**Suggested Type:** {suggested}")
            lines.append(f"**Reasoning:** {reasoning}")
            lines.append(f"**Follow-up:** {followup}")
            lines.append("")

            # Key physics features
            lines.append("| Feature | Value |")
            lines.append("|---|---|")
            lines.append(
                f"| Recon Error | {r['recon_error']:.4f} ({r['recon_error'] / 0.001251:.0f}x baseline) |"
            )
            lines.append(f"| IF Score | {r['if_score']:.4f} |")
            lines.append(f"| Detections | {r.get('n_detections', 'N/A')} |")
            lines.append(f"| Classification | {r.get('classification', 'None') or 'None'} |")

            # Add physics features if available
            for feat_key, feat_name in [
                ("np_rise_time", "Rise Time (days)"),
                ("np_decay_time", "Decay Time (days)"),
                ("np_peak_mag", "Peak Magnitude"),
                ("np_peak_color_g_r", "Peak Color (g-r)"),
                ("np_fwhm", "FWHM (days)"),
                ("np_von_neumann_ratio", "Von Neumann Ratio"),
                ("param_best_model_name", "Best Parametric Model"),
            ]:
                val = r.get(feat_key, None)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    if isinstance(val, float):
                        lines.append(f"| {feat_name} | {val:.3f} |")
                    else:
                        lines.append(f"| {feat_name} | {val} |")

            lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="LLM triage on Fritz sources with AppleCiDEr physics"
    )
    parser.add_argument("--scores-csv", required=True, help="anomaly_scores.csv")
    parser.add_argument("--features", required=True, help="fritz_features.parquet")
    parser.add_argument("--manifest", required=True, help="fritz_manifest.csv")
    parser.add_argument("--fritz-dir", required=True, help="Fritz sources dir (with NPZ files)")
    parser.add_argument("--groq-key", required=True, help="Groq API key")
    parser.add_argument("--top-n", type=int, default=30, help="Top N per group for LLM")
    parser.add_argument(
        "--min-detections", type=int, default=10, help="Minimum detections to include"
    )
    parser.add_argument("--output-json", default="llm_triage_results.json")
    parser.add_argument("--output-report", default="triage_report.md")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show candidates without calling API"
    )
    args = parser.parse_args()

    print("=== LLM Triage with AppleCiDEr Physics ===")
    print(f"Model: {GROQ_MODEL}")
    print(f"Top N per group: {args.top_n}")
    print(f"Min detections: {args.min_detections}")

    # ── Load data ──
    print("\n[1/4] Loading data...")
    scores_df = pd.read_csv(args.scores_csv)
    features_df = pd.read_parquet(args.features)
    manifest_df = pd.read_csv(args.manifest)
    print(f"  Scores: {len(scores_df)} sources")
    print(f"  Features: {features_df.shape}")
    print(f"  Manifest: {len(manifest_df)} sources")

    # ── Filter by min detections ──
    pre_filter = len(scores_df)
    scores_df = scores_df[scores_df["n_detections"] >= args.min_detections]
    print(
        f"  After min-detections filter (>={args.min_detections}): {len(scores_df)} / {pre_filter}"
    )

    # ── Select candidates ──
    print(f"\n[2/5] Selecting top {args.top_n} per group...")
    groups = scores_df["group_name"].unique()
    candidates = []

    for group in sorted(groups):
        group_df = scores_df[scores_df["group_name"] == group].sort_values(
            "anomaly_score", ascending=False
        )
        top = group_df.head(args.top_n)
        candidates.append(top)
        print(
            f"  {group}: {len(top)} candidates (score range: "
            f"{top['anomaly_score'].iloc[-1]:.4f} - {top['anomaly_score'].iloc[0]:.4f})"
        )

    candidates_df = pd.concat(candidates, ignore_index=True)
    total = len(candidates_df)
    print(f"  Total candidates for LLM: {total}")

    # ── Generate light curve plots ──
    report_dir = os.path.dirname(os.path.abspath(args.output_report))
    plots_dir = os.path.join(report_dir, "lightcurves")
    os.makedirs(plots_dir, exist_ok=True)
    print("\n[3/5] Generating light curve plots...")
    n_plotted = 0
    for _, row in candidates_df.iterrows():
        obj_id = row["obj_id"]
        npz_path = os.path.join(args.fritz_dir, f"{obj_id}.npz")
        png_path = os.path.join(plots_dir, f"{obj_id}.png")
        classification = row.get("classification", "")
        if os.path.exists(npz_path) and plot_lightcurve(npz_path, png_path, obj_id, classification):
            n_plotted += 1
    print(f"  Plotted {n_plotted}/{total} light curves")

    if args.dry_run:
        print("\n=== DRY RUN — No API calls ===")
        for _, row in candidates_df.iterrows():
            print(
                f"  {row['group_name']:20s} | {row['obj_id']:20s} | "
                f"score={row['anomaly_score']:.4f} | recon={row['recon_error']:.4f}"
            )
        print(f"\nWould send {total} sources to Groq. Exiting.")
        return

    # ── Run LLM triage ──
    print(f"\n[4/5] Running LLM triage on {total} candidates...")
    results = []

    for idx, (_, row) in enumerate(candidates_df.iterrows()):
        obj_id = row["obj_id"]
        group = row["group_name"]

        # Get feature row
        if obj_id in features_df.index:
            feature_row = features_df.loc[obj_id]
        else:
            print(f"  [{idx + 1}/{total}] {obj_id}: no features, using empty")
            feature_row = pd.Series(dtype=float)

        # Build prompt
        prompt = build_physics_prompt(row, feature_row)
        system_prompt = SYSTEM_PROMPTS.get(group, DEFAULT_SYSTEM_PROMPT)

        # Call Groq
        print(f"  [{idx + 1}/{total}] {obj_id} ({group})...", end=" ", flush=True)
        response_text = call_groq(prompt, system_prompt, args.groq_key)
        parsed = parse_llm_response(response_text)
        print(f"{parsed.get('verdict', '?')} ({parsed.get('confidence', 0):.2f})")

        # Build result
        result = {
            "obj_id": obj_id,
            "group_name": group,
            "anomaly_score": float(row["anomaly_score"]),
            "recon_error": float(row["recon_error"]),
            "if_score": float(row["if_score"]),
            "n_detections": int(row.get("n_detections", 0)),
            "classification": row.get("classification", ""),
            "verdict": parsed.get("verdict", "error"),
            "confidence": float(parsed.get("confidence", 0)),
            "reasoning": parsed.get("reasoning", ""),
            "suggested_class": parsed.get("suggested_class", "unknown"),
            "followup": parsed.get("followup", ""),
        }

        # Attach key physics features for the report
        for feat_key in [
            "np_rise_time",
            "np_decay_time",
            "np_peak_mag",
            "np_peak_color_g_r",
            "np_fwhm",
            "np_von_neumann_ratio",
        ]:
            val = feature_row.get(feat_key, None)
            if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
                result[feat_key] = float(val)

        best_model = feature_row.get("param_best_model", None)
        if best_model is not None and not (
            isinstance(best_model, float) and math.isnan(best_model)
        ):
            result["param_best_model_name"] = MODEL_NAMES.get(int(best_model), "Unknown")

        results.append(result)

        # Rate limit: ~2s between calls
        time.sleep(2.0)

    # ── Save results ──
    print("\n[5/5] Saving results...")
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON: {args.output_json}")

    generate_report(results, args.output_report, plots_dir=plots_dir)
    print(f"  Report: {args.output_report}")
    print(f"  Plots: {plots_dir}/")

    # Print summary
    verdicts = {}
    for r in results:
        v = r["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1
    print("\n=== SUMMARY ===")
    for v, count in sorted(verdicts.items()):
        print(f"  {v}: {count}")
    print(f"  Total: {len(results)}")

    n_interesting = verdicts.get("interesting", 0)
    print(
        f"\n  LLM hit rate: {n_interesting}/{len(results)} = "
        f"{n_interesting / len(results) * 100:.0f}% interesting"
    )

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
