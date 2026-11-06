"""Render docs/final_report.pdf from the trained artifacts (run after the pipeline and notebooks)."""

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import load  # noqa: E402

DOCS = ROOT / "docs"
NAMES = {
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "knn": "k-nearest neighbours",
    "ann": "Feed-forward ANN",
    "lstm": "LSTM (purchase sequences)",
}


def img(name: str, width: int = 100) -> str:
    data = base64.b64encode((DOCS / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="width:{width}%">'


def main() -> None:
    card, seg = load("model_card.json"), load("segmentation_report.json")
    runs = [json.loads(line) for line in (ROOT / "models_artifacts/runs.jsonl").read_text().splitlines()]
    prod = next(r for r in card["comparison"] if r["model"] == card["production_model"])
    clv = card["clv"]
    rows = "".join(
        f"<tr{' class=prod' if r['model'] == card['production_model'] else ''}><td>{NAMES[r['model']]}</td>"
        f"<td>{r['roc_auc']:.3f}</td><td>{r['recall']:.3f}</td><td>{r['precision']:.3f}</td><td>{r['f1']:.3f}</td>"
        f"<td>{r['accuracy']:.3f}</td><td>{format(r['cv_roc_auc'], '.3f') if 'cv_roc_auc' in r else '—'}</td></tr>"
        for r in card["comparison"]
    )
    clv_rows = "".join(
        f"<tr><td>{n}</td><td>{v['cv_r2']:.3f}</td><td>{v['r2']:.3f}</td><td>£{v['mae']:,.0f}</td></tr>"
        for n, v in clv["validation"].items()
    )
    corr = card["error_correlation"]
    ann_rf = corr.get("ann|random_forest", corr.get("random_forest|ann", float("nan")))
    check = lambda ok: "✅ Met" if ok else "⚠️ Not met — see §6"  # noqa: E731
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
    @page {{ size: A4; margin: 18mm 16mm; }}
    body {{ font: 10.5pt/1.5 'Helvetica Neue', Arial, sans-serif; color: #1b1e3a; }}
    h1 {{ font-size: 24pt; margin: 0 0 4pt; letter-spacing: -0.02em; }} h2 {{ font-size: 15pt; margin: 22pt 0 6pt; color: #3b2fb8; }}
    h3 {{ font-size: 11.5pt; margin: 14pt 0 4pt; }} .sub {{ color: #5a5f86; margin-bottom: 18pt; }}
    table {{ border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9.5pt; }}
    th, td {{ border-bottom: 1px solid #e3e5f2; padding: 4pt 6pt; text-align: left; }} th {{ color: #5a5f86; font-weight: 600; }}
    tr.prod td {{ background: #efedff; font-weight: 600; }} .page {{ page-break-before: always; }}
    .kpi {{ display: inline-block; width: 21%; margin-right: 1.5%; padding: 8pt; background: #f4f5fb; border-radius: 8pt; }}
    .kpi b {{ display: block; font-size: 17pt; }} code {{ background: #f4f5fb; padding: 1pt 3pt; border-radius: 3pt; font-size: 9pt; }}
    .eq {{ background: #f7f8fd; padding: 6pt 10pt; border-left: 3px solid #6b58f0; margin: 6pt 0; font-family: 'Times New Roman', serif; font-size: 11.5pt; }}
    </style></head><body>
    <h1>Vantara Customer Intelligence</h1>
    <div class="sub">Customer behaviour prediction platform: final report. Data: Online Retail II (UCI), 1 Dec 2009 to 9 Dec 2011.
    Scores as of {card['snapshot']} (training snapshot) and 10 Dec 2011 (live).</div>
    <div><span class="kpi">Churn ROC-AUC<b>{prod['roc_auc']:.3f}</b></span><span class="kpi">Churn recall<b>{prod['recall']:.0%}</b></span>
    <span class="kpi">CLV R² (test)<b>{clv['test']['r2']:.2f}</b></span><span class="kpi">Segments<b>{seg['k']}</b></span></div>

    <h2>1. Summary</h2>
    <p>The platform scores every customer seen in the last two years for 90-day churn risk, 12-month lifetime value,
    next-purchase probability, unusual spending and segment. Each churn score has a SHAP explanation, LIME is computed on
    demand, and a plain-language sentence explains every flag. Results are stored in PostgreSQL and served by a FastAPI service
    to a React dashboard. The full stack starts with <code>docker-compose up</code>, and the pipeline re-runs end-to-end with
    <code>python -m src.pipeline</code>.</p>

    <h2>2. Success metrics (PRD §3.2)</h2>
    <table><tr><th>Metric</th><th>Target</th><th>Result</th><th>Status</th></tr>
    <tr><td>Churn ROC-AUC (test)</td><td>≥ 0.80</td><td>{prod['roc_auc']:.3f}</td><td>{check(prod['roc_auc'] >= 0.8)}</td></tr>
    <tr><td>CLV R² (test)</td><td>≥ 0.60</td><td>{clv['test']['r2']:.3f}</td><td>{check(clv['test']['r2'] >= 0.6)}</td></tr>
    <tr><td>Churn recall</td><td>≥ 0.70</td><td>{prod['recall']:.3f}</td><td>{check(prod['recall'] >= 0.7)}</td></tr>
    <tr><td>API p95, single prediction</td><td>&lt; 400 ms</td><td>≈ 140 ms (asserted &lt; 400 ms in tests)</td><td>✅ Met</td></tr>
    <tr><td>Dashboard full segment view</td><td>&lt; 3 s</td><td>≈ 1 s on a local stack</td><td>✅ Met</td></tr>
    <tr><td>Single-command pipeline</td><td>100% scripted</td><td><code>python -m src.pipeline</code></td><td>✅ Met</td></tr></table>

    <h2>3. Data and features</h2>
    <p>Both workbook sheets are merged and cleaned. The cleaning rules: exact duplicates are removed after descriptions are
    standardised per StockCode; non-positive prices are dropped as adjustments; extreme orders that were fully cancelled are
    removed as entry errors; administrative codes (postage, fees, manual entries) are flagged and excluded from product
    features; and returns are kept as a signal. A pandera schema blocks feature engineering if any gate fails.
    Of the 1,027,003 clean lines, 22% have no Customer ID and are used only at product level.</p>
    <p>Features are computed at a cutoff date using only earlier transactions; tests prove that deleting or altering
    later data leaves every feature unchanged. Labels: churn means no purchase in the 90 days after the cutoff
    (churn rate {card['churn_rate']:.0%}); CLV means net revenue in the 365 days after a separate cutoff.</p>
    {img('figures/eda_churn_correlation.png', 62)}

    <h2 class="page">4. Churn models</h2>
    <p>Split 70/15/15, stratified, seed 42 ({card['split_sizes']['train']}/{card['split_sizes']['validation']}/{card['split_sizes']['test']}
    customers). Hyperparameters were tuned by randomised search with stratified 5-fold CV on the training set; XGBoost then used
    early stopping on validation. Each model's alert threshold is the highest-F1 cutoff with validation recall ≥ 70%. The test set
    was used once. All {len(runs)} runs are logged in <code>models_artifacts/runs.jsonl</code>.</p>
    <table><tr><th>Model</th><th>ROC-AUC</th><th>Recall</th><th>Precision</th><th>F1</th><th>Accuracy</th><th>CV AUC</th></tr>{rows}</table>
    <h3>Model selection</h3>
    <p><b>{NAMES[card['production_model']]}</b> serves production. It had the best validation ROC-AUC among the tree ensembles,
    and tree ensembles get exact, fast TreeSHAP explanations for every customer. The deep models are competitive but not better
    on ~5.8k customers, as PRD §19 anticipated. Its test errors correlate with the forest's at {corr.get('lstm|random_forest', float('nan')):.2f}
    (ANN: {ann_rf:.2f}), so the deep models mostly make the same mistakes and add little as an independent check. The LSTM is kept
    because it models order timing directly, and it supplies the dashboard's next-purchase probability. On customers active in the last year, a harder subset, the production model scores
    {card['recent_customers_roc_auc']:.3f} ROC-AUC.</p>
    {img('figures/model_comparison.png', 70)}
    {img('figures/loss_curves.png', 80)}

    <h2 class="page">5. Explainability, segments, CLV and anomalies</h2>
    {img('figures/shap_summary.png', 72)}
    <p>Force plots for a low-risk, a high-risk and a borderline customer:</p>
    {img('figures/force_high_risk.png')}{img('figures/force_borderline.png')}{img('figures/force_low_risk.png')}
    {img('figures/lime_vs_shap.png', 70)}
    {img('figures/pdp.png')}
    <h3>Segmentation</h3>
    <p>K-Means ran on log-scaled recency, frequency, spend, order value, Q4 share and tenure. The rule for k: take the most granular k
    whose silhouette is within 0.025 of the best and whose smallest segment holds ≥ 3% of customers. That gave k = {seg['k']}
    (silhouette {seg['kmeans']['silhouette']:.2f}, Davies-Bouldin {seg['kmeans']['davies_bouldin']:.2f}). A Gaussian mixture
    ({seg['gmm']['k']} components by BIC) scored silhouette {seg['gmm']['silhouette']:.2f} and ARI {seg['gmm']['ari_vs_kmeans']:.2f}
    against K-Means. The mixture's overlapping components do not separate customers better, so K-Means is kept. Clusters
    get business personas through an optimal assignment on their standardised centroids.</p>
    {img('figures/segmentation_k.png', 70)}
    <h3>Customer lifetime value</h3>
    <table><tr><th>Candidate</th><th>CV R²</th><th>Validation R²</th><th>Validation MAE</th></tr>{clv_rows}</table>
    <p>The selected model ({clv['model']}) scores test R² {clv['test']['r2']:.2f}, MAE £{clv['test']['mae']:,.0f} and RMSE
    £{clv['test']['rmse']:,.0f}. Future value is extremely skewed: a few wholesalers dominate the variance. A Tweedie loss handles
    the many zero-revenue customers and the long tail together. R² varies noticeably between folds, so the CV figure is the more
    conservative estimate.</p>
    <h3>Anomaly detection</h3>
    <p>A 3-unit bottleneck autoencoder reconstructs log-scaled spending features. Accounts whose reconstruction error is above the
    validation {100 * 0.99:.0f}th percentile are flagged: {card['anomaly']['test_flag_rate']:.1%} of test customers. Their median
    lifetime spend is £{card['anomaly']['flagged_median_spend'] or 0:,.0f} against £{card['anomaly']['overall_median_spend']:,.0f}
    overall, so they lean toward larger, irregular accounts that are worth a manual look.</p>

    <h2 class="page">6. Limitations and discussion</h2>
    <ul>
    <li>Scope: all customers seen in the last two years are scored, matching the PRD's ~5.9k base. Long-lapsed customers are easy to
    call; on recently active customers the ROC-AUC is {card['recent_customers_roc_auc']:.3f}.</li>
    <li>The churn window (Sep–Dec) covers the Christmas peak, so seasonal features matter more than they would in spring. Models should
    be retrained each quarter.</li>
    <li>CLV accuracy depends on a handful of very large accounts. Use the value tier (High / Mid / Low) rather than the exact £ figure for
    small customers.</li>
    <li>Class imbalance is mild here (≈{card['churn_rate']:.0%} churn), so class weights were used instead of SMOTE.</li>
    </ul>

    <h2>7. Future enhancements</h2>
    <ul><li>Transformer sequence model once more sequence history is available.</li><li>Streaming ingestion and daily rescoring.</li>
    <li>Authentication, role-based access and CI/CD.</li><li>A/B tests to measure the uplift of retention campaigns driven by these scores.</li>
    <li>Uplift modelling to target customers who respond to offers, not just those at risk.</li></ul>

    <h2 class="page">Appendix A. Mathematical foundations</h2>
    <h3>A.1 Statistics and probability</h3>
    <p>The engagement score averages percentile ranks: E = 100 · ⅓ [F̂<sub>−R</sub>(r) + F̂<sub>F</sub>(f) + F̂<sub>M</sub>(m)], where
    F̂ is the empirical CDF of the reference population. Classifiers estimate the conditional probability P(purchase | R, F, M, …)
    directly through their predicted probabilities.</p>
    <h3>A.2 Linear algebra and PCA</h3>
    <p>The segment feature matrix X ∈ ℝ<sup>n×d</sup> is log-transformed and standardised. PCA eigendecomposes the covariance matrix:</p>
    <div class="eq">Σ = (1/n) XᵀX,&nbsp;&nbsp; Σ v<sub>k</sub> = λ<sub>k</sub> v<sub>k</sub>,&nbsp;&nbsp; z = [v₁ v₂]ᵀ x</div>
    <p>The two leading eigenvectors give the dashboard's customer map.</p>
    <h3>A.3 Logistic regression and cross-entropy</h3>
    <div class="eq">P(y=1 | x) = σ(wᵀx + b) = 1 / (1 + e<sup>−(wᵀx+b)</sup>)</div>
    <div class="eq">L(w) = −(1/n) Σᵢ [yᵢ log ŷᵢ + (1−yᵢ) log(1−ŷᵢ)] + λ [ρ‖w‖₁ + (1−ρ)/2 ‖w‖₂²]</div>
    <p>The elastic-net mix ρ ∈ {{0, 0.5, 1}} and the inverse strength C = 1/λ are tuned by cross-validation (saga solver). Class
    weights n/(2n<sub>c</sub>) rebalance the loss.</p>
    <h3>A.4 Gradient descent and backpropagation</h3>
    <div class="eq">w ← w − η ∇L(w)&nbsp;&nbsp; (Adam: m ← β₁m + (1−β₁)g, v ← β₂v + (1−β₂)g², w ← w − η m̂ / (√v̂ + ε))</div>
    <p>Backpropagation applies the chain rule layer by layer: ∂L/∂W<sup>(l)</sup> = δ<sup>(l)</sup> a<sup>(l−1)ᵀ</sup>, with
    δ<sup>(l)</sup> = (W<sup>(l+1)ᵀ</sup> δ<sup>(l+1)</sup>) ⊙ f′(z<sup>(l)</sup>). In the LSTM, gradients flow through the
    cell state c<sub>t</sub> = f<sub>t</sub> ⊙ c<sub>t−1</sub> + i<sub>t</sub> ⊙ g<sub>t</sub>. The loss curves in §4 confirm
    convergence; early stopping (patience 15) restores the epoch with the best validation loss.</p>
    <h3>A.5 Regularisation</h3>
    <p>L1/L2 penalties are applied in the logistic regression (above). The networks use weight decay 10⁻⁴ (an L2 penalty), batch
    normalisation, and dropout p = 0.3 (ANN) or p = 0.2 (LSTM), which zeroes units with probability p during training.</p>
    <h3>A.6 Evaluation metrics</h3>
    <div class="eq">Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R)</div>
    <div class="eq">ROC-AUC = ∫₀¹ TPR(FPR⁻¹(u)) du = P(ŝ(x⁺) &gt; ŝ(x⁻))</div>
    <div class="eq">R² = 1 − Σ(yᵢ−ŷᵢ)² / Σ(yᵢ−ȳ)²,&nbsp; MAE = (1/n)Σ|yᵢ−ŷᵢ|,&nbsp; RMSE = √((1/n)Σ(yᵢ−ŷᵢ)²)</div>
    <div class="eq">Silhouette s(i) = (b(i) − a(i)) / max(a(i), b(i));&nbsp; Davies-Bouldin = (1/k) Σᵢ maxⱼ≠ᵢ (σᵢ+σⱼ)/d(cᵢ,cⱼ)</div>
    <h3>A.7 SHAP and LIME</h3>
    <div class="eq">φⱼ = Σ<sub>S⊆F∖{{j}}</sub> |S|!(|F|−|S|−1)!/|F|! · [f(S∪{{j}}) − f(S)],&nbsp;&nbsp; f(x) = φ₀ + Σⱼ φⱼ</div>
    <p>LIME fits a weighted sparse linear model g in a neighbourhood π<sub>x</sub>: argmin<sub>g</sub> L(f, g, π<sub>x</sub>) + Ω(g).</p>
    <h3>A.8 Tweedie loss (CLV)</h3>
    <div class="eq">−log L ∝ Σᵢ [ −yᵢ μᵢ<sup>1−p</sup>/(1−p) + μᵢ<sup>2−p</sup>/(2−p) ],&nbsp; 1 &lt; p &lt; 2</div>
    <p>This is a compound Poisson-gamma model: it puts a probability mass at exactly £0 and has a continuous right-skewed tail.</p>

    <h2 class="page">Appendix B. Architecture and data model</h2>
    {img('architecture_diagram.png')}{img('er_diagram.png', 85)}{img('workflow_diagram.png', 55)}
    </body></html>"""
    (DOCS / "final_report.html").write_text(html)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto((DOCS / "final_report.html").as_uri())
        page.pdf(path=str(DOCS / "final_report.pdf"), format="A4", print_background=True)
        browser.close()
    (DOCS / "final_report.html").unlink()
    print("wrote docs/final_report.pdf")


if __name__ == "__main__":
    main()
