import { motion } from 'motion/react'
import { Lightbulb } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { useFilters, useSegments, type S } from '@/api/client'
import { axis, ChartTip, grid } from '@/components/charts'
import { Card, CardTitle, PageHeader, Skeleton, StateMessage } from '@/components/ui'
import { cn, fmt, riskBand, riskColor, segmentColor } from '@/lib/utils'

const METRICS = [
  { key: 'churn_rate', label: 'Churn risk', format: (v: number) => fmt.pct(v) },
  { key: 'avg_clv', label: 'Predicted CLV', format: fmt.gbp },
  { key: 'avg_spend', label: 'Lifetime spend', format: fmt.gbp },
  { key: 'avg_frequency', label: 'Orders', format: (v: number) => v.toFixed(1) },
  { key: 'avg_recency', label: 'Days since last order', format: fmt.num },
] as const

function SegmentCard({ s, total, i }: { s: S['Segment']; total: number; i: number }) {
  const band = riskBand(s.churn_rate)
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
      <Card className="h-full">
        <div className="flex items-start justify-between gap-2">
          <div>
            <p className="flex items-center gap-2 font-display text-lg font-semibold tracking-tight"><span className="size-2.5 rounded-full" style={{ background: segmentColor(s.segment) }} />{s.segment}</p>
            <p className="mt-0.5 text-sm text-ink-3">{fmt.num(s.size)} customers, {fmt.pct(s.size / total)} of base</p>
          </div>
          <span className="rounded-full px-2.5 py-0.5 text-xs font-medium tabular" style={{ color: riskColor[band], background: `color-mix(in oklab, ${riskColor[band]} 14%, transparent)` }}>
            Churn risk {fmt.pct(s.churn_rate)}
          </span>
        </div>
        <dl className="mt-5 grid grid-cols-3 gap-3 text-sm">
          {[['Avg. spend', fmt.gbpShort(s.avg_spend)], ['Avg. CLV', fmt.gbpShort(s.avg_clv)], ['At risk', fmt.gbpShort(s.revenue_at_risk)]].map(([k, v]) => (
            <div key={k}><dt className="text-ink-3">{k}</dt><dd className="mt-0.5 font-display text-lg font-semibold tabular">{v}</dd></div>
          ))}
        </dl>
        <p className="mt-5 flex gap-2 rounded-2xl bg-iris/8 p-3 text-sm text-ink-2"><Lightbulb className="mt-0.5 size-4 shrink-0 text-iris" />{s.action}</p>
      </Card>
    </motion.div>
  )
}

export default function Segments() {
  const q = useSegments()
  const navigate = useNavigate()
  const { search } = useFilters()
  const [focus, setFocus] = useState<string | null>(null)
  const [metric, setMetric] = useState<(typeof METRICS)[number]>(METRICS[0])
  if (q.isError) return <StateMessage kind="error" title="Couldn't load segments" detail={q.error.message} onRetry={() => q.refetch()} />
  const data = q.data
  const total = data?.segments.reduce((n, s) => n + s.size, 0) ?? 0
  const names = data?.segments.map((s) => s.segment) ?? []
  const sorted = [...(data?.segments ?? [])].sort((a, b) => b[metric.key] - a[metric.key])

  return (
    <>
      <PageHeader title="Segments" description={`${names.length || 'Several'} behavioural groups found by K-Means clustering on recency, frequency, spend, basket size, seasonality and returns — each with a recommended play.`} />
      {!data ? <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">{Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-60" />)}</div>
        : !data.segments.length ? <StateMessage kind="empty" title="No customers match these filters" />
        : (
          <>
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">{data.segments.map((s, i) => <SegmentCard key={s.segment} s={s} total={total} i={i} />)}</div>
            <div className="mt-4 grid gap-4 xl:grid-cols-5">
              <Card className="xl:col-span-3">
                <CardTitle title="Customer map" hint="Each dot is a customer, placed by the first two principal components. Click a segment to isolate it; click a dot to open the profile." />
                <div className="mb-3 flex flex-wrap gap-2">
                  {names.map((n) => (
                    <button key={n} onClick={() => setFocus(focus === n ? null : n)} aria-pressed={focus === n}
                      className={cn('flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition', focus && focus !== n ? 'border-line text-ink-3 opacity-60' : 'border-line text-ink-2')}>
                      <span className="size-2 rounded-full" style={{ background: segmentColor(n) }} />{n}
                    </button>
                  ))}
                </div>
                <ResponsiveContainer width="100%" height={380}>
                  <ScatterChart margin={{ left: -16, right: 8, top: 8 }}>
                    <CartesianGrid stroke="var(--grid)" />
                    <XAxis type="number" dataKey="x" name="PC1" {...axis} tickFormatter={(v) => v.toFixed(0)} />
                    <YAxis type="number" dataKey="y" name="PC2" {...axis} tickFormatter={(v) => v.toFixed(0)} />
                    <ZAxis range={[22, 22]} />
                    <Tooltip cursor={false} content={({ payload }) => {
                      const p = payload?.[0]?.payload as S['ScatterPoint'] | undefined
                      return p ? <ChartTip title={`Customer #${p.customer_id}`} rows={[{ label: 'Segment', value: p.segment, color: segmentColor(p.segment) }, { label: 'Churn risk', value: fmt.pct(p.churn_prob) }]} /> : null
                    }} />
                    {names.map((n) => (
                      <Scatter key={n} name={n} data={data.scatter.filter((p) => p.segment === n)} fill={segmentColor(n)}
                        fillOpacity={focus && focus !== n ? 0.06 : 0.75} stroke="var(--canvas)" strokeWidth={1} isAnimationActive={false}
                        onClick={(p) => navigate(`/customers/${(p as unknown as S['ScatterPoint']).customer_id}${search}`)} className="cursor-pointer" />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
              </Card>
              <Card className="xl:col-span-2">
                <CardTitle title="Compare segments" hint="Average per customer" />
                <div className="mb-4 flex flex-wrap gap-1.5" role="tablist">
                  {METRICS.map((m) => (
                    <button key={m.key} role="tab" aria-selected={metric.key === m.key} onClick={() => setMetric(m)}
                      className={cn('rounded-lg px-2.5 py-1 text-xs transition', metric.key === m.key ? 'bg-iris text-white' : 'text-ink-2 hover:bg-iris/10')}>{m.label}</button>
                  ))}
                </div>
                <ResponsiveContainer width="100%" height={360}>
                  <BarChart data={sorted} layout="vertical" margin={{ left: 8, right: 16 }} barCategoryGap={8}>
                    <CartesianGrid {...grid} horizontal={false} vertical />
                    <XAxis type="number" {...axis} tickFormatter={(v) => metric.format(v)} />
                    <YAxis type="category" dataKey="segment" width={150} {...axis} tick={{ fill: 'var(--ink-2)', fontSize: 12 }} />
                    <Tooltip cursor={{ fill: 'var(--grid)' }} content={({ payload }) => {
                      const s = payload?.[0]?.payload as S['Segment'] | undefined
                      return s ? <ChartTip title={s.segment} rows={[{ label: metric.label, value: metric.format(s[metric.key]), color: segmentColor(s.segment) }, { label: 'Customers', value: fmt.num(s.size) }]} /> : null
                    }} />
                    <Bar dataKey={metric.key} radius={[0, 4, 4, 0]} animationDuration={500}>
                      {sorted.map((s) => <Cell key={s.segment} fill={segmentColor(s.segment)} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>
          </>
        )}
    </>
  )
}
