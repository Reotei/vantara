import { ArrowLeft, CalendarClock, Gift, MapPin, Package, ScanSearch, ShoppingBag, Sparkles, Split } from 'lucide-react'
import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useCustomer, useFilters, useLime, type S } from '@/api/client'
import { axis, ChartTip, Gauge, grid, ShapWaterfall } from '@/components/charts'
import { Animated, Button, Card, CardTitle, RiskBadge, Skeleton, StateMessage, Tip } from '@/components/ui'
import { cn, fmt, riskBand, riskColor, segmentColor } from '@/lib/utils'

function LimeComparison({ id }: { id: number }) {
  const q = useLime(id, true)
  if (q.isError) {
    const snapshot = q.error.message.includes('hosted snapshot')
    return <StateMessage kind={snapshot ? 'empty' : 'error'} title={snapshot ? 'LIME check not included here' : "Couldn't compute the LIME explanation"}
      detail={q.error.message} onRetry={snapshot ? undefined : () => q.refetch()} />
  }
  if (!q.data) return <div className="space-y-2 py-2"><p className="text-sm text-ink-3">Fitting a local surrogate model around this customer…</p><Skeleton className="h-56" /></div>
  const max = Math.max(...q.data.lime.map((w) => Math.abs(w.weight)))
  return (
    <div>
      <p className="mb-4 text-sm text-ink-2">
        LIME and SHAP agree on <b className="text-ink">{q.data.shared_top_features.length} of the top 5</b> drivers
        ({fmt.pct(q.data.agreement)} agreement). LIME explains with simple rules; SHAP attributes exact contributions.
      </p>
      <ul className="space-y-2.5">
        {q.data.lime.map((w) => (
          <li key={w.rule} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] items-center gap-3 text-sm">
            <span className={cn('truncate', q.data.shared_top_features.includes(w.feature) ? 'text-ink' : 'text-ink-3')} title={w.rule}>{w.rule.replace(w.feature, w.label)}</span>
            <div className="flex h-5 items-center">
              <div className="relative h-full w-full">
                <span className="absolute inset-y-0 left-1/2 w-px bg-line" />
                <span className="absolute inset-y-1 rounded" style={{
                  background: w.weight > 0 ? riskColor.High : riskColor.Low,
                  left: w.weight > 0 ? '50%' : `${50 - (Math.abs(w.weight) / max) * 50}%`, width: `${(Math.abs(w.weight) / max) * 50}%`,
                }} />
              </div>
            </div>
          </li>
        ))}
      </ul>
      <p className="mt-3 flex justify-between text-xs text-ink-3"><span>← lowers churn risk</span><span>raises churn risk →</span></p>
    </div>
  )
}

function Timeline({ orders }: { orders: S['Order'][] }) {
  const data = orders.map((o) => ({ ...o, signed: o.is_return ? -Math.abs(o.value) : o.value }))
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ left: 4, right: 8, top: 8 }}>
        <CartesianGrid {...grid} />
        <XAxis dataKey="date" tickFormatter={(d) => new Date(d).toLocaleDateString('en-GB', { month: 'short', year: '2-digit' })} {...axis} minTickGap={36} />
        <YAxis tickFormatter={(v) => fmt.gbpShort(v)} {...axis} width={60} />
        <ReferenceLine y={0} stroke="var(--line)" />
        <Tooltip cursor={{ fill: 'var(--grid)' }} content={({ payload }) => {
          const o = payload?.[0]?.payload as (typeof data)[number] | undefined
          return o ? <ChartTip title={fmt.date(o.date)} rows={[{ label: o.is_return ? 'Returned' : 'Order value', value: fmt.gbp(Math.abs(o.value)), color: o.is_return ? riskColor.High : 'var(--iris)' }, { label: 'Items', value: fmt.num(o.items) }, { label: 'Invoice', value: o.invoice }]} /> : null
        }} />
        <Bar dataKey="signed" radius={[3, 3, 0, 0]} maxBarSize={14}>
          {data.map((o) => <Cell key={o.invoice} fill={o.is_return ? riskColor.High : 'var(--iris)'} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function Stat({ icon: Icon, label, value }: { icon: typeof Gift; label: string; value: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="grid size-9 place-items-center rounded-xl bg-iris/10 text-iris"><Icon className="size-4" /></span>
      <div><p className="text-xs text-ink-3">{label}</p><p className="font-medium tabular">{value}</p></div>
    </div>
  )
}

export default function CustomerProfile() {
  const id = Number(useParams().id)
  const { search } = useFilters()
  const q = useCustomer(id)
  const [lime, setLime] = useState(false)
  const back = <Link to={`/churn${search}`} className="mb-4 inline-flex items-center gap-1.5 text-sm text-ink-3 hover:text-ink"><ArrowLeft className="size-4" />Churn risk leaderboard</Link>
  if (q.isError) return <>{back}<Card><StateMessage kind="error" title={`Customer #${id} is not available`} detail={q.error.message} /></Card></>
  if (!q.data) return <>{back}<Skeleton className="mb-4 h-24" /><div className="grid gap-4 lg:grid-cols-3"><Skeleton className="h-72" /><Skeleton className="h-72 lg:col-span-2" /></div></>
  const c = q.data
  const band = riskBand(c.churn_prob)
  const change = c.prev_churn_prob === null || c.prev_churn_prob === undefined ? null : c.churn_prob - c.prev_churn_prob

  return (
    <>
      {back}
      <header className="mb-6 flex flex-wrap items-center gap-x-4 gap-y-3">
        <span className="grid size-14 place-items-center rounded-2xl font-display text-lg font-semibold" style={{ background: `color-mix(in oklab, ${segmentColor(c.segment)} 22%, transparent)`, color: segmentColor(c.segment) }}>
          {String(c.customer_id).slice(-2)}
        </span>
        <div>
          <h1 className="font-display text-3xl font-semibold tracking-tight">Customer #{c.customer_id}</h1>
          <p className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-ink-2">
            <span className="flex items-center gap-1.5"><span className="size-2 rounded-full" style={{ background: segmentColor(c.segment) }} />{c.segment}</span>
            <span className="flex items-center gap-1.5"><MapPin className="size-3.5" />{c.country}</span>
            <span>{c.value_tier} value tier</span>
            {c.is_anomaly && <Tip content="The autoencoder could not reconstruct this spending pattern well — check for bulk resellers, data-entry errors or fraud."><span className="flex items-center gap-1.5 text-mid"><ScanSearch className="size-3.5" />Unusual spending pattern</span></Tip>}
          </p>
        </div>
        <div className="ml-auto"><RiskBadge band={band} size="md" /></div>
      </header>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="flex flex-col items-center justify-center text-center">
          <Gauge value={c.churn_prob} band={band} />
          <p className="-mt-1 text-sm text-ink-2">Chance of churning in the next 90 days</p>
          {change !== null && (
            <p className="mt-2 text-sm text-ink-3">
              <span className={cn('font-semibold tabular', change > 0 ? 'text-high' : 'text-low')}>{change > 0 ? '+' : ''}{(change * 100).toFixed(0)} pts</span> since last quarter
            </p>
          )}
        </Card>
        <Card className="grid gap-5 sm:grid-cols-2 lg:col-span-2">
          <div>
            <p className="text-sm text-ink-2">Predicted CLV, next 12 months</p>
            <Animated value={c.clv_pred} format="gbp" className="mt-2 block font-display text-4xl font-semibold tracking-tight" />
            <p className="mt-1 text-sm text-ink-3">Revenue at risk <b className="tabular text-ink">{fmt.gbp(c.revenue_at_risk)}</b></p>
          </div>
          <div>
            <p className="text-sm text-ink-2">Next-purchase probability, 90 days</p>
            <Animated value={c.next_purchase_prob} format="pct" className="mt-2 block font-display text-4xl font-semibold tracking-tight" />
            <div className="mt-3 h-2 rounded-full bg-[var(--grid)]"><div className="h-full rounded-full bg-[linear-gradient(90deg,var(--iris),var(--aqua))]" style={{ width: fmt.pct(c.next_purchase_prob) }} /></div>
          </div>
          <div className="grid grid-cols-2 gap-4 sm:col-span-2 sm:grid-cols-4">
            <Stat icon={ShoppingBag} label="Orders" value={fmt.num(c.frequency)} />
            <Stat icon={Package} label="Lifetime spend" value={fmt.gbp(c.monetary_total)} />
            <Stat icon={CalendarClock} label="Last order" value={c.recency_days === 0 ? 'Today' : `${fmt.num(c.recency_days)} days ago`} />
            <Stat icon={Gift} label="Customer since" value={fmt.date(c.first_purchase)} />
          </div>
        </Card>
      </div>

      <Card className="mt-4">
        <CardTitle title="Why this customer is flagged" hint="Top drivers of the churn score. Red bars push risk up, green bars pull it down."
          action={<Button variant="ghost" onClick={() => setLime((v) => !v)} aria-pressed={lime}><Split className="size-4" />{lime ? 'Hide LIME check' : 'Cross-check with LIME'}</Button>} />
        <p className="mb-5 max-w-3xl rounded-2xl border border-line bg-panel p-4 text-[15px] leading-relaxed">
          <Sparkles className="mr-2 inline size-4 text-iris" />{c.explanation}
        </p>
        <div className={cn('grid gap-6', lime && 'xl:grid-cols-2')}>
          <div>
            <ShapWaterfall base={c.base_value} link={c.shap_link} prediction={c.churn_prob} contributions={c.contributions} />
          </div>
          {lime && <div><p className="mb-2 text-sm text-ink-3">LIME: local rule-based explanation for the same prediction</p><LimeComparison id={id} /></div>}
        </div>
      </Card>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardTitle title="Purchase timeline" hint={`${fmt.num(c.timeline.length)} invoices; returns shown below the line`} />
          {c.timeline.length ? <Timeline orders={c.timeline} /> : <StateMessage kind="empty" title="No orders on record" />}
        </Card>
        <Card>
          <CardTitle title="Recommended next products" hint="Popular items in this customer's favourite categories they haven't bought yet" />
          {c.recommendations.length ? (
            <ul className="space-y-2">
              {c.recommendations.map((p) => (
                <li key={p.stock_code} className="flex items-center justify-between gap-3 rounded-xl border border-line px-3 py-2.5 text-sm">
                  <div className="min-w-0"><p className="truncate font-medium">{p.description}</p><p className="text-xs text-ink-3">{p.category}, SKU {p.stock_code}</p></div>
                  <span className="tabular text-ink-2">{fmt.price(p.price)}</span>
                </li>
              ))}
            </ul>
          ) : <StateMessage kind="empty" title="No recommendations yet" detail="This customer has already bought every current best-seller in their categories." />}
          <div className="mt-5">
            <p className="mb-2 text-sm text-ink-3">Category mix</p>
            <div className="flex h-2.5 overflow-hidden rounded-full">
              {Object.entries(c.category_mix).map(([k, v], i) => (
                <Tip key={k} content={`${k}: ${fmt.pct(v)} of spend`}><span className="h-full border-r-2 border-[var(--canvas)] last:border-0" style={{ width: `${v * 100}%`, background: `var(--series-${(i % 7) + 1})` }} /></Tip>
              ))}
            </div>
            <p className="mt-2 text-xs text-ink-3">{Object.entries(c.category_mix).slice(0, 3).map(([k, v]) => `${k} ${fmt.pct(v)}`).join(', ')}</p>
          </div>
        </Card>
      </div>
      <p className="mt-6 rounded-2xl border border-line p-4 text-sm text-ink-2"><b className="text-ink">Recommended play for {c.segment}:</b> {c.segment_action}</p>
    </>
  )
}
