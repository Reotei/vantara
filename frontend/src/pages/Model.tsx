import { Crown } from 'lucide-react'
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useModelCard, type ModelCard } from '@/api/client'
import { axis, ChartTip, grid } from '@/components/charts'
import { Card, CardTitle, PageHeader, Skeleton, StateMessage, Tip } from '@/components/ui'
import { cn, fmt } from '@/lib/utils'

const NAMES: Record<string, string> = {
  logistic_regression: 'Logistic regression', decision_tree: 'Decision tree', random_forest: 'Random forest', xgboost: 'XGBoost',
  lightgbm: 'LightGBM', knn: 'k-nearest neighbours', ann: 'Neural network (ANN)', lstm: 'Sequence model (LSTM)',
}

function Comparison({ card }: { card: ModelCard }) {
  const best = Math.max(...card.comparison.map((r) => r.roc_auc))
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-sm">
        <thead>
          <tr className="border-b border-line text-left text-ink-3">
            {['Model', 'ROC-AUC', 'Recall', 'Precision', 'F1', 'Accuracy', 'Alert threshold'].map((h) => <th key={h} className="px-3 py-2.5 font-medium">{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {card.comparison.map((r) => {
            const prod = r.model === card.production_model
            return (
              <tr key={r.model} className={cn('border-b border-line/60 last:border-0', prod && 'bg-iris/8')}>
                <td className="px-3 py-2.5">
                  <span className="flex items-center gap-2 font-medium">
                    {NAMES[r.model] ?? r.model}
                    {prod && <Tip content="Serves every score in this dashboard: best validation ROC-AUC among tree ensembles, with exact SHAP explanations."><span className="flex items-center gap-1 rounded-full bg-iris px-2 py-0.5 text-[11px] text-white"><Crown className="size-3" />In production</span></Tip>}
                    {r.family === 'deep' && <span className="rounded-full border border-line px-2 py-0.5 text-[11px] text-ink-3">Deep learning</span>}
                  </span>
                </td>
                <td className="px-3 py-2.5">
                  <div className="flex items-center gap-2">
                    <span className="w-12 font-semibold tabular">{r.roc_auc.toFixed(3)}</span>
                    <div className="h-1.5 w-24 rounded-full bg-[var(--grid)]"><div className="h-full rounded-full bg-iris" style={{ width: `${((r.roc_auc - 0.5) / (best - 0.5)) * 100}%` }} /></div>
                  </div>
                </td>
                {[r.recall, r.precision, r.f1, r.accuracy].map((v, i) => <td key={i} className="px-3 py-2.5 tabular text-ink-2">{v.toFixed(3)}</td>)}
                <td className="px-3 py-2.5 tabular text-ink-3">{fmt.pct(r.threshold)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function Confusion({ m }: { m: number[][] }) {
  const [[tn, fp], [fn, tp]] = m
  const cells = [
    { label: 'Stayed, predicted to stay', v: tn, good: true }, { label: 'Stayed, flagged as churn', v: fp, good: false },
    { label: 'Churned, missed', v: fn, good: false }, { label: 'Churned, caught', v: tp, good: true },
  ]
  const total = tn + fp + fn + tp
  return (
    <div className="grid grid-cols-2 gap-2">
      {cells.map((c) => (
        <div key={c.label} className="rounded-2xl p-4" style={{ background: `color-mix(in oklab, ${c.good ? 'var(--risk-low)' : 'var(--risk-high)'} ${8 + (c.v / total) * 40}%, transparent)` }}>
          <p className="font-display text-3xl font-semibold tabular">{fmt.num(c.v)}</p>
          <p className="mt-1 text-xs text-ink-2">{c.label}</p>
        </div>
      ))}
    </div>
  )
}

export default function Model() {
  const q = useModelCard()
  if (q.isError) return <StateMessage kind="error" title="Couldn't load model metadata" detail={q.error.message} onRetry={() => q.refetch()} />
  if (!q.data) return <div className="grid gap-4"><Skeleton className="h-80" /><Skeleton className="h-72" /></div>
  const card = q.data
  const prod = card.comparison.find((r) => r.model === card.production_model)!
  const seg = card.segmentation
  return (
    <>
      <PageHeader title="Model quality" description={`How far to trust the scores. All metrics are measured once on a held-out test set of ${fmt.num(card.split_sizes.test)} customers the models never saw during training or tuning.`} />
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[
          ['Churn ROC-AUC', prod.roc_auc.toFixed(3), `${card.recent_customers_roc_auc.toFixed(3)} on customers active in the last year`],
          ['Churners caught (recall)', fmt.pct(prod.recall), `At a ${fmt.pct(card.threshold)} alert threshold`],
          ['CLV accuracy (R²)', card.clv.test.r2.toFixed(2), `Mean absolute error ${fmt.gbp(card.clv.test.mae)} per customer`],
          ['Segment separation', seg.kmeans.silhouette.toFixed(2), `Silhouette score, ${seg.k} segments`],
        ].map(([k, v, h]) => (
          <Card key={k}><p className="text-sm text-ink-2">{k}</p><p className="mt-2 font-display text-4xl font-semibold tabular">{v}</p><p className="mt-1 text-xs text-ink-3">{h}</p></Card>
        ))}
      </div>

      <Card className="mt-4">
        <CardTitle title="Churn model comparison" hint="Six classical models and two deep-learning models, ranked by ROC-AUC then recall. Thresholds are tuned on validation data to catch at least 70% of churners." />
        <Comparison card={card} />
      </Card>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card>
          <CardTitle title="ROC curve" hint={`${NAMES[card.production_model]}, test set`} />
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={card.roc_curve} margin={{ left: -16, right: 8, top: 8 }}>
              <defs><linearGradient id="roc" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="var(--iris)" stopOpacity={0.4} /><stop offset="1" stopColor="var(--iris)" stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid stroke="var(--grid)" />
              <XAxis type="number" dataKey="fpr" domain={[0, 1]} tickFormatter={(v) => fmt.pct(v)} {...axis} />
              <YAxis type="number" domain={[0, 1]} tickFormatter={(v) => fmt.pct(v)} {...axis} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--ink-3)" strokeDasharray="4 4" />
              <Tooltip content={({ payload }) => {
                const p = payload?.[0]?.payload as { fpr: number; tpr: number } | undefined
                return p ? <ChartTip rows={[{ label: 'Churners caught', value: fmt.pct(p.tpr) }, { label: 'False alarms', value: fmt.pct(p.fpr) }]} /> : null
              }} />
              <Area type="monotone" dataKey="tpr" stroke="var(--iris)" strokeWidth={2} fill="url(#roc)" />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
        <Card>
          <CardTitle title="Confusion matrix" hint={`Test set at the ${fmt.pct(card.threshold)} threshold`} />
          <Confusion m={prod.confusion_matrix} />
        </Card>
        <Card>
          <CardTitle title="What the model looks at" hint="Mean absolute SHAP value, top 8 features" />
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={card.global_importance.slice(0, 8)} layout="vertical" margin={{ left: 0, right: 12 }} barCategoryGap={6}>
              <XAxis type="number" hide />
              <YAxis type="category" dataKey="label" width={170} {...axis} tick={{ fill: 'var(--ink-2)', fontSize: 12 }} />
              <Tooltip cursor={{ fill: 'var(--grid)' }} content={({ payload }) => {
                const p = payload?.[0]?.payload as { label: string; importance: number } | undefined
                return p ? <ChartTip title={p.label} rows={[{ label: 'Mean |SHAP|', value: p.importance.toFixed(3) }]} /> : null
              }} />
              <Bar dataKey="importance" fill="var(--iris)" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        {Object.entries(card.pdp).map(([f, pts]) => (
          <Card key={f}>
            <CardTitle title={card.feature_labels[f] ?? f} hint="Partial dependence: average churn probability as this feature changes" />
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={pts} margin={{ left: -16, right: 8, top: 8 }}>
                <CartesianGrid {...grid} />
                <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(v) => (Math.abs(v) < 1 ? v.toFixed(2) : fmt.num(v))} {...axis} />
                <YAxis tickFormatter={(v) => fmt.pct(v)} {...axis} domain={['auto', 'auto']} />
                <Tooltip content={({ payload }) => {
                  const p = payload?.[0]?.payload as { x: number; y: number } | undefined
                  return p ? <ChartTip rows={[{ label: card.feature_labels[f] ?? f, value: p.x.toLocaleString('en-GB', { maximumFractionDigits: 2 }) }, { label: 'Churn probability', value: fmt.pct(p.y, 1), color: 'var(--iris)' }]} /> : null
                }} />
                <Line dataKey="y" stroke="var(--aqua)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        ))}
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card>
          <CardTitle title="Lifetime value model" hint={`Predicts net revenue over the next ${card.clv.horizon_days} days`} />
          <dl className="grid grid-cols-3 gap-3 text-sm">
            {[['R²', card.clv.test.r2.toFixed(2)], ['MAE', fmt.gbp(card.clv.test.mae)], ['RMSE', fmt.gbp(card.clv.test.rmse)]].map(([k, v]) => <div key={k}><dt className="text-ink-3">{k}</dt><dd className="font-display text-xl font-semibold tabular">{v}</dd></div>)}
          </dl>
          <p className="mt-4 text-sm text-ink-3">Selected from {Object.keys(card.clv.validation).length} candidates by cross-validated R²: {card.clv.model.replace('_', ' ')}.</p>
        </Card>
        <Card>
          <CardTitle title="Anomaly detection" hint="Autoencoder on spending patterns" />
          <p className="font-display text-3xl font-semibold tabular">{fmt.pct(card.anomaly.test_flag_rate, 1)}</p>
          <p className="mt-1 text-sm text-ink-3">of test customers flagged for review. Their median lifetime spend is {fmt.gbp(card.anomaly.flagged_median_spend ?? 0)} vs {fmt.gbp(card.anomaly.overall_median_spend)} overall.</p>
        </Card>
        <Card>
          <CardTitle title="Segmentation check" hint="K-Means vs Gaussian mixture" />
          <dl className="grid grid-cols-2 gap-3 text-sm">
            {[['K-Means silhouette', seg.kmeans.silhouette.toFixed(2)], ['K-Means Davies-Bouldin', seg.kmeans.davies_bouldin.toFixed(2)],
              [`GMM (${seg.gmm.k} components) silhouette`, seg.gmm.silhouette.toFixed(2)], ['Agreement (ARI)', seg.gmm.ari_vs_kmeans.toFixed(2)]].map(([k, v]) => (
              <div key={k}><dt className="text-ink-3">{k}</dt><dd className="font-display text-xl font-semibold tabular">{v}</dd></div>
            ))}
          </dl>
        </Card>
      </div>
    </>
  )
}
