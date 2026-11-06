import { ArrowUpRight, ShieldAlert, Sparkles, UserCheck, Wallet } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Area, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useCustomers, useFilters, useKpis, useSegments, useTrends } from '@/api/client'
import { axis, ChartTip, grid } from '@/components/charts'
import { Animated, Card, CardTitle, Delta, PageHeader, RiskBadge, Skeleton, StateMessage, Tip } from '@/components/ui'
import { fmt, riskBand, riskColor, riskFill, segmentColor, type RiskBand } from '@/lib/utils'

function KpiCard({ label, icon: Icon, value, previous, format, invert, hint, spark }: {
  label: string; icon: typeof Wallet; value: number; previous?: number | null; format: 'gbp' | 'num'; invert?: boolean; hint: string; spark?: number[]
}) {
  return (
    <Card className="flex items-center gap-4 py-4">
      <span className="grid size-10 shrink-0 place-items-center rounded-2xl bg-iris/10 text-iris"><Icon className="size-[18px]" /></span>
      <div className="min-w-0 flex-1">
        <Tip content={hint}><span className="cursor-help text-sm text-ink-2 underline decoration-line decoration-dotted underline-offset-4">{label}</span></Tip>
        <div className="mt-1"><Delta now={value} before={previous} invert={invert} format={format === 'gbp' ? fmt.gbpShort : fmt.num} /></div>
      </div>
      {spark && spark.length > 1 && (
        <svg viewBox="0 0 100 32" className="hidden h-8 w-20 sm:block" aria-hidden>
          <polyline fill="none" stroke="var(--aqua)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
            points={spark.map((v, i) => `${(i / (spark.length - 1)) * 100},${30 - ((v - Math.min(...spark)) / (Math.max(...spark) - Math.min(...spark) || 1)) * 28}`).join(' ')} />
        </svg>
      )}
      <Animated value={value} format={format} className="font-display text-[28px] font-semibold leading-none tracking-tight" />
    </Card>
  )
}

function Hero() {
  const kpis = useKpis()
  if (kpis.isError) return <Card className="lg:col-span-3"><StateMessage kind="error" title="Couldn't load KPIs" detail={kpis.error.message} onRetry={() => kpis.refetch()} /></Card>
  if (!kpis.data) return <Skeleton className="h-72 lg:col-span-3" />
  const k = kpis.data
  const bands: RiskBand[] = ['High', 'Medium', 'Low']
  return (
    <Card beam className="relative overflow-hidden lg:col-span-3">
      <div className="flex items-center gap-2 text-sm text-ink-2"><ShieldAlert className="size-4 text-high" />Revenue at risk over the next 12 months</div>
      <Animated value={k.revenue_at_risk.value} format="gbp" className="mt-4 block font-display text-[56px] font-semibold leading-none tracking-[-0.03em] [text-shadow:0_0_40px_color-mix(in_oklab,var(--iris)_45%,transparent)]" />
      <div className="mt-3"><Delta now={k.revenue_at_risk.value} before={k.revenue_at_risk.previous} invert format={fmt.gbpShort} /></div>
      <p className="mt-5 text-sm text-ink-3">Churn probability × predicted lifetime value, summed over {fmt.num(k.customers)} customers.</p>
      <div className="mt-4 flex h-3 overflow-hidden rounded-full" role="img" aria-label="Customers by risk band">
        {bands.map((b) => (
          <Tip key={b} content={`${fmt.num(k.risk_bands[b] ?? 0)} customers at ${b.toLowerCase()} risk`}>
            <span className="h-full border-r-2 border-[var(--canvas)] last:border-0 transition-[flex-grow] duration-700" style={{ flexGrow: k.risk_bands[b] ?? 0, background: riskColor[b] }} />
          </Tip>
        ))}
      </div>
      <div className="mt-2.5 flex gap-4 text-xs text-ink-2">
        {bands.map((b) => <span key={b} className="flex items-center gap-1.5"><span className="size-2 rounded-full" style={{ background: riskColor[b] }} />{b} <b className="tabular text-ink">{fmt.num(k.risk_bands[b] ?? 0)}</b></span>)}
      </div>
    </Card>
  )
}

function Kpis() {
  const { data } = useKpis()
  const trend = useTrends().data?.filter((p) => p.customers !== null).slice(-12)
  if (!data) return <div className="grid gap-4 lg:col-span-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-24" />)}</div>
  return (
    <div className="grid gap-4 lg:col-span-2">
      <KpiCard label="Active customers" icon={UserCheck} value={data.active_customers.value} previous={data.active_customers.previous} format="num"
        hint="Customers who ordered in the last 90 days. The line shows buying customers per month over the last year." spark={trend?.map((p) => p.customers ?? 0)} />
      <KpiCard label="At-risk customers" icon={ShieldAlert} value={data.at_risk_customers.value} previous={data.at_risk_customers.previous} format="num" invert
        hint={`Customers whose churn probability is at least ${fmt.pct(data.threshold)}: the alert threshold tuned to catch 70%+ of churners.`} />
      <KpiCard label="Average CLV" icon={Wallet} value={data.avg_clv.value} previous={data.avg_clv.previous} format="gbp" hint="Predicted net revenue per customer over the next 12 months." />
    </div>
  )
}

function RevenueTrend() {
  const trends = useTrends()
  const data = trends.data?.map((p) => ({ ...p, band: p.lower !== null && p.upper !== null ? [p.lower, p.upper] : null }))
  return (
    <Card className="lg:col-span-2">
      <CardTitle title="Monthly revenue" hint="Net revenue from identified customers, with a 3-month forecast and 80% confidence band" />
      {trends.isError ? <StateMessage kind="error" title="Couldn't load the revenue trend" onRetry={() => trends.refetch()} />
        : !data ? <Skeleton className="h-72" />
        : !data.length ? <StateMessage kind="empty" title="No revenue in this selection" detail="Widen the filters to see a trend." />
        : (
          <ResponsiveContainer width="100%" height={288}>
            <ComposedChart data={data} margin={{ left: 4, right: 8, top: 8 }}>
              <defs>
                <linearGradient id="rev" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="var(--iris)" stopOpacity={0.45} /><stop offset="1" stopColor="var(--iris)" stopOpacity={0} /></linearGradient>
              </defs>
              <CartesianGrid {...grid} />
              <XAxis dataKey="month" tickFormatter={fmt.month} {...axis} minTickGap={24} />
              <YAxis tickFormatter={(v) => fmt.gbpShort(v)} {...axis} width={64} />
              <Tooltip cursor={{ stroke: 'var(--ink-3)', strokeDasharray: '3 3' }} content={({ payload, label }) => {
                const p = payload?.[0]?.payload as (typeof data)[number] | undefined
                if (!p) return null
                return <ChartTip title={fmt.month(String(label))} rows={p.revenue !== null
                  ? [{ label: 'Revenue', value: fmt.gbp(p.revenue ?? 0), color: 'var(--iris)' }, { label: 'Buying customers', value: fmt.num(p.customers ?? 0) }]
                  : [{ label: 'Forecast', value: fmt.gbp(p.forecast ?? 0), color: 'var(--aqua)' }, { label: '80% range', value: `${fmt.gbpShort(p.lower ?? 0)} – ${fmt.gbpShort(p.upper ?? 0)}` }]} />
              }} />
              <Area dataKey="band" stroke="none" fill="var(--aqua)" fillOpacity={0.14} isAnimationActive={false} />
              <Area dataKey="revenue" stroke="var(--iris)" strokeWidth={2} fill="url(#rev)" dot={false} activeDot={{ r: 5, strokeWidth: 2, stroke: 'var(--canvas)' }} />
              <Line dataKey="forecast" stroke="var(--aqua)" strokeWidth={2} strokeDasharray="6 5" dot={false} activeDot={{ r: 5 }} />
            </ComposedChart>
          </ResponsiveContainer>
        )}
    </Card>
  )
}

function RiskDistribution() {
  const { data } = useKpis()
  return (
    <Card>
      <CardTitle title="Churn-risk distribution" hint="Customers by predicted probability of churning" />
      {!data ? <Skeleton className="h-72" /> : (
        <ResponsiveContainer width="100%" height={288}>
          <BarChart data={data.risk_distribution} margin={{ left: -8, right: 4, top: 8 }} barCategoryGap={3}>
            <CartesianGrid {...grid} />
            <XAxis dataKey="low" tickFormatter={(v) => `${Math.round(v * 100)}%`} {...axis} interval={1} />
            <YAxis {...axis} width={44} />
            <Tooltip cursor={{ fill: 'var(--grid)' }} content={({ payload }) => {
              const b = payload?.[0]?.payload as (typeof data.risk_distribution)[number] | undefined
              return b ? <ChartTip title={`Churn risk ${b.bucket}`} rows={[{ label: 'Customers', value: fmt.num(b.customers), color: riskFill(b.low + 0.05) }, { label: 'Revenue at risk', value: fmt.gbp(b.revenue_at_risk) }]} /> : null
            }} />
            <Bar dataKey="customers" radius={[4, 4, 0, 0]}>
              {data.risk_distribution.map((b) => <Cell key={b.bucket} fill={riskFill(b.low + 0.05)} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </Card>
  )
}

function TopAtRisk() {
  const { search } = useFilters()
  const q = useCustomers({ page_size: 6, sort_by: 'revenue_at_risk', order: 'desc', risk_band: ['High', 'Medium'] })
  return (
    <Card className="lg:col-span-2">
      <CardTitle title="Call these customers first" hint="Medium- and high-risk customers with the most revenue at stake"
        action={<Link to={`/churn${search}`} className="flex items-center gap-1 text-sm text-iris hover:underline">Full leaderboard<ArrowUpRight className="size-4" /></Link>} />
      {!q.data ? <Skeleton className="h-64" /> : !q.data.items.length ? <StateMessage kind="empty" title="No customers match these filters" /> : (
        <ul className="divide-y divide-line">
          {q.data.items.map((c) => (
            <li key={c.customer_id}>
              <Link to={`/customers/${c.customer_id}${search}`} className="group flex items-center gap-4 rounded-xl px-2 py-3 transition hover:bg-iris/6">
                <span className="grid size-10 place-items-center rounded-xl text-xs font-semibold tabular" style={{ background: `color-mix(in oklab, ${segmentColor(c.segment)} 20%, transparent)`, color: segmentColor(c.segment) }}>
                  {String(c.customer_id).slice(-3)}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="font-medium">Customer #{c.customer_id}<span className="ml-2 text-sm font-normal text-ink-3">{c.segment}, {c.country}</span></p>
                  <p className="truncate text-sm text-ink-3">Main driver: {c.top_driver}</p>
                </div>
                <RiskBadge band={riskBand(c.churn_prob)} probability={c.churn_prob} />
                <span className="w-24 text-right font-display text-lg font-semibold tabular">{fmt.gbpShort(c.revenue_at_risk)}</span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}

function SegmentRisk() {
  const { data } = useSegments()
  const max = Math.max(...(data?.segments.map((s) => s.revenue_at_risk) ?? [1]))
  return (
    <Card>
      <CardTitle title="Revenue at risk by segment" hint="Where retention budget has the most to save" />
      {!data ? <Skeleton className="h-64" /> : (
        <ul className="space-y-3.5">
          {data.segments.map((s) => (
            <li key={s.segment}>
              <div className="mb-1.5 flex justify-between text-sm"><span className="flex items-center gap-2"><span className="size-2 rounded-full" style={{ background: segmentColor(s.segment) }} />{s.segment}</span><span className="font-medium tabular">{fmt.gbpShort(s.revenue_at_risk)}</span></div>
              <div className="h-2 rounded-full bg-[var(--grid)]"><div className="h-full rounded-full transition-[width] duration-700" style={{ width: `${(s.revenue_at_risk / max) * 100}%`, background: segmentColor(s.segment) }} /></div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}

export default function Overview() {
  const { search } = useFilters()
  return (
    <>
      <PageHeader title="Customer health" description="Who is likely to stop buying in the next 90 days, and how much revenue depends on keeping them."
        action={<Link to={`/churn${search}`} className="inline-flex items-center gap-2 rounded-xl bg-iris/12 px-4 py-2 text-sm font-medium text-iris hover:bg-iris/20"><Sparkles className="size-4" />Prioritise outreach</Link>} />
      <div className="grid gap-4 lg:grid-cols-5">
        <Hero />
        <Kpis />
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-3"><RevenueTrend /><RiskDistribution /></div>
      <div className="mt-4 grid gap-4 lg:grid-cols-3"><TopAtRisk /><SegmentRisk /></div>
    </>
  )
}
