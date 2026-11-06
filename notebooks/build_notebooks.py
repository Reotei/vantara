"""Generate and execute the three analysis notebooks (run from the repo root after the pipeline)."""

import sys
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).parent
SETUP = """import sys, warnings, json
sys.path.insert(0, '..'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from src.utils import load_config, load
from src.features.build import build_features, build_labels, data_end, feature_columns
cfg = load_config(); FIG = '../docs/figures/'
import os; os.makedirs(FIG, exist_ok=True)
sns.set_theme(style='whitegrid', palette=['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6'])
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
tx = pd.read_parquet(cfg['paths']['interim'])"""

NOTEBOOKS = {
    "01_eda.ipynb": [
        ("md", "# Exploratory data analysis\nOnline Retail II, cleaned by `src.data.clean` and validated by `src.data.validate`."),
        ("code", SETUP),
        ("code", "print(tx.shape); tx.head()"),
        ("code", "summary = {'rows': len(tx), 'customers': tx.customer_id.nunique(), 'guest_line_share': tx.customer_id.isna().mean(),\n"
                 " 'return_line_share': tx.is_return.mean(), 'first': tx.invoice_date.min(), 'last': tx.invoice_date.max()}\nsummary"),
        ("md", "## Univariate: spend, frequency, recency per customer"),
        ("code", "snap = pd.read_parquet(f\"{cfg['paths']['processed']}/churn_snapshot.parquet\")\n"
                 "snap[['recency_days','frequency','monetary_total','avg_order_value','return_rate']].describe().T.round(2)"),
        ("code", "fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))\n"
                 "for a, col, log in zip(ax, ['recency_days', 'frequency', 'monetary_total'], [False, True, True]):\n"
                 "    sns.histplot(np.log10(snap[col].clip(lower=1)) if log else snap[col], ax=a, bins=40, color='#2a78d6')\n"
                 "    a.set_xlabel(('log10 ' if log else '') + col)\nplt.tight_layout(); plt.savefig(FIG + 'eda_distributions.png')"),
        ("md", "## Bivariate: features vs churn and future value"),
        ("code", "num = [c for c in feature_columns(snap.drop(columns=['churn','future_revenue'])) if not c.startswith(('cat_','country_'))]\n"
                 "corr = snap[num + ['churn']].corr(method='spearman')['churn'].drop('churn').sort_values()\n"
                 "ax = corr.plot.barh(figsize=(7, 7), color=np.where(corr > 0, '#e34948', '#1baf7a')); ax.set_title('Spearman correlation with churn')\n"
                 "plt.tight_layout(); plt.savefig(FIG + 'eda_churn_correlation.png'); corr.round(3)"),
        ("code", "clv = pd.read_parquet(f\"{cfg['paths']['processed']}/clv_snapshot.parquet\")\n"
                 "clv[num].corrwith(clv.future_revenue, method='spearman').sort_values(ascending=False).head(8).round(3)"),
        ("md", "## Multicollinearity (VIF)"),
        ("code", "from statsmodels.stats.outliers_influence import variance_inflation_factor\n"
                 "X = np.log1p(snap[num].fillna(snap[num].median()).clip(lower=0)); X = (X - X.mean()) / X.std()\n"
                 "vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=num).sort_values(ascending=False)\nvif.round(1)"),
        ("md", "High-VIF pairs (e.g. frequency vs. active months, revenue_last_90d vs. orders_last_90d) are kept: the production model is a "
               "tree ensemble, which is insensitive to collinearity, and the linear baseline is L1/L2 regularised."),
        ("md", "## Country and seasonality"),
        ("code", "rev = tx[~tx.is_admin & tx.customer_id.notna()]\n"
                 "rev.groupby('country').revenue.sum().sort_values(ascending=False).head(10).div(1e3).round(0).rename('revenue £k')"),
        ("code", "monthly = rev.groupby('month').revenue.sum()\nax = monthly.div(1e3).plot(figsize=(11, 3.5), marker='o', color='#2a78d6')\n"
                 "ax.set_ylabel('£k'); ax.set_title('Monthly net revenue (identified customers)'); plt.tight_layout(); plt.savefig(FIG + 'eda_monthly_revenue.png')"),
        ("md", "## Hypotheses for modelling\n1. Recency is the strongest single churn signal, but its effect depends on each customer's usual "
               "order gap (overdue ratio).\n2. Customers who bought in the same season last year are less likely to churn before Christmas.\n"
               "3. Return rate and markdown share add signal beyond RFM.\n4. Past spend is close to linear in future 12-month value; "
               "tree models will need help extrapolating whales (Tweedie loss or a linear model).\n"
               "5. Spending behaviour forms a handful of interpretable segments (wholesalers, seasonal gift buyers, lapsed big spenders)."),
    ],
    "02_feature_engineering.ipynb": [
        ("md", "# Feature engineering\nAll features come from `src.features.build.build_features(tx, cutoff, cfg)` using only data before `cutoff`."),
        ("code", SETUP),
        ("code", "cutoff = data_end(tx) - pd.Timedelta(days=90)\nf = build_features(tx, cutoff, cfg)\nprint(cutoff.date(), f.shape)\nf.describe().T.round(2)"),
        ("md", "## Leakage check\nRebuilding the table with every post-cutoff row removed must give identical features (also enforced in `tests/test_features.py`)."),
        ("code", "pd.testing.assert_frame_equal(f, build_features(tx[tx.invoice_date < cutoff], cutoff, cfg)); 'no leakage'"),
        ("md", "## Why each feature\n| Feature | Why it matters |\n|---|---|\n"
               "| recency_days | Directly precedes churn; strongest univariate predictor |\n"
               "| frequency, active_months_last_year | Habitual vs one-time buyers |\n"
               "| monetary_total, avg_order_value | Core input for churn weighting and CLV |\n"
               "| avg_basket_items / lines | Top-up vs bulk shopper |\n| order_trend | Acceleration or deceleration hidden by averages |\n"
               "| gap_mean, gap_std, overdue_ratio | Irregular customers; how late the next order is vs normal |\n"
               "| q4_share, month_concentration, orders_same_season_last_year | Seasonal / gift shoppers |\n"
               "| cat_* | Category affinity vector, feeds recommendations |\n| return_rate | Leading indicator of dissatisfaction |\n"
               "| markdown_share | Discount sensitivity: buys below the usual price |\n| product_popularity | Frequency-encoded SKU mainstreamness |\n"
               "| country_* | One-hot for the five largest markets |\n| engagement_score | Single 0-100 RFM score for dashboards |"),
        ("code", "labels = build_labels(tx, cutoff, 90, f.index)\nprint('churn rate', labels.churn.mean().round(3))\n"
                 "f.join(labels).groupby('churn')[['recency_days', 'frequency', 'overdue_ratio', 'engagement_score']].median().round(1)"),
    ],
    "03_model_experiments.ipynb": [
        ("md", "# Model experiments\nReads the run log and artifacts written by `python -m src.pipeline --stages train`."),
        ("code", SETUP + "\nimport shap\nfrom src.explainability.explain import ChurnExplainer, label\ncard = load('model_card.json')"),
        ("code", "runs = pd.DataFrame([json.loads(l) for l in open(cfg['paths']['artifacts'] + '/runs.jsonl')])\n"
                 "runs[runs.task == 'churn'][['model', 'cv_roc_auc', 'val_roc_auc', 'roc_auc', 'recall', 'precision', 'f1', 'accuracy', 'train_seconds']].round(3)"),
        ("code", "cmp = pd.DataFrame(card['comparison']).set_index('model')[['roc_auc', 'recall', 'precision', 'f1', 'accuracy']]\n"
                 "ax = cmp[['roc_auc', 'recall', 'f1']].sort_values('roc_auc').plot.barh(figsize=(8, 5)); ax.set_xlim(0.5, 1)\n"
                 "ax.set_title('Churn models on the held-out test set'); plt.tight_layout(); plt.savefig(FIG + 'model_comparison.png'); cmp.round(3)"),
        ("code", "fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))\nfor a, name in zip(ax, ['ann', 'lstm']):\n"
                 "    lc = pd.DataFrame(card['loss_curves'][name]).set_index('epoch'); lc.plot(ax=a); a.set_title(f'{name.upper()} loss'); a.set_ylabel('BCE')\n"
                 "plt.tight_layout(); plt.savefig(FIG + 'loss_curves.png')"),
        ("code", "pd.Series(card['error_correlation']).sort_values().round(2).to_frame('error correlation')"),
        ("md", "## SHAP"),
        ("code", "snap = pd.read_parquet(f\"{cfg['paths']['processed']}/churn_snapshot.parquet\")\ncols = card['features']\n"
                 "model = load('churn_model.joblib'); exp = ChurnExplainer(model, load('background.joblib'), cfg['seed'])\n"
                 "X = snap[cols].sample(800, random_state=0); sv = exp.shap_values(X)\n"
                 "shap.summary_plot(sv, X.rename(columns=label), show=False, max_display=12, plot_size=(9, 6)); plt.tight_layout(); plt.savefig(FIG + 'shap_summary.png')"),
        ("code", "p = model.predict_proba(X)[:, 1]\npicks = {'low risk': X.index[np.argmin(p)], 'high risk': X.index[np.argmax(p)], 'borderline': X.index[np.argmin(abs(p - card['threshold']))]}\n"
                 "for name, cid in picks.items():\n    i = X.index.get_loc(cid)\n"
                 "    shap.force_plot(exp.base_value, sv[i], X.iloc[i].round(2).rename(index=label), matplotlib=True, show=False, figsize=(16, 3), text_rotation=12)\n"
                 "    plt.title(f'Customer {cid} ({name}, p={p[i]:.2f})', y=1.5); plt.savefig(FIG + f\"force_{name.replace(' ', '_')}.png\", bbox_inches='tight'); plt.show()"),
        ("md", "## LIME vs SHAP for the borderline customer"),
        ("code", "row = snap.loc[picks['borderline']]\nlime_w = pd.DataFrame(exp.lime_weights(row, 3000)).set_index('feature')['weight']\n"
                 "shap_w = pd.Series(sv[X.index.get_loc(picks['borderline'])], index=cols)\n"
                 "both = pd.DataFrame({'LIME': lime_w, 'SHAP': shap_w.reindex(lime_w.index)}).rename(index=label)\n"
                 "ax = both.plot.barh(figsize=(8, 4.5)); ax.set_title('LIME vs SHAP for the borderline customer')\n"
                 "plt.tight_layout(); plt.savefig(FIG + 'lime_vs_shap.png'); both.round(3)"),
        ("md", "## Partial dependence"),
        ("code", "fig, ax = plt.subplots(1, len(card['pdp']), figsize=(13, 3.4))\nfor a, (f, pts) in zip(ax, card['pdp'].items()):\n"
                 "    d = pd.DataFrame(pts); a.plot(d.x, d.y, color='#1baf7a'); a.set_title(label(f)); a.set_ylabel('P(churn)')\n"
                 "plt.tight_layout(); plt.savefig(FIG + 'pdp.png')"),
        ("md", "## Segmentation diagnostics"),
        ("code", "seg = load('segmentation_report.json'); el = pd.DataFrame(seg['elbow']).set_index('k')\n"
                 "fig, ax = plt.subplots(1, 2, figsize=(10, 3.2)); el.inertia.plot(ax=ax[0], marker='o', title='Elbow (inertia)')\n"
                 "el.silhouette.plot(ax=ax[1], marker='o', title='Silhouette'); plt.tight_layout(); plt.savefig(FIG + 'segmentation_k.png')\n"
                 "{k: seg[k] for k in ['k', 'kmeans']} | {'gmm': {k: v for k, v in seg['gmm'].items() if k != 'bic'}}"),
        ("md", "## CLV"),
        ("code", "pd.DataFrame(card['clv']['validation']).T.round(3).assign(selected=lambda d: d.index == card['clv']['model'])"),
    ],
}


def build(execute: bool = True) -> None:
    """Write each notebook and (optionally) execute it in place."""
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        if execute:
            ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
        nbf.write(nb, HERE / name)
        print("wrote", name)


if __name__ == "__main__":
    build("--no-exec" not in sys.argv)
