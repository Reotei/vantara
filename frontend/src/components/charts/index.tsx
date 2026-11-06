import { motion } from 'motion/react'
import type { ReactNode } from 'react'
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { S } from '@/api/client'
import { riskColor, type RiskBand } from '@/lib/utils'

export const axis = { stroke: 'var(--ink-3)', fontSize: 12, tickLine: false, axisLine: false } as const
export const grid = { stroke: 'var(--grid)', strokeDasharray: '0', vertical: false } as const

export function ChartTip({ title, rows }: { title?: ReactNode; rows: { label: ReactNode; value: ReactNode; color?: string }[] }) {
  return (
    <div className="min-w-44 rounded-xl border border-line bg-panel-strong/95 px-3 py-2.5 text-xs shadow-2xl backdrop-blur">
      {title && <p className="mb-1.5 font-medium text-ink">{title}</p>}
      {rows.map((r, i) => (
        <p key={i} className="flex items-center justify-between gap-4 py-0.5 text-ink-2">
          <span className="flex items-center gap-1.5">
            {r.color && <span className="size-2 rounded-full" style={{ background: r.color }} />}
            {r.label}
          </span>
          <span className="font-semibold text-ink tabular">{r.value}</span>
        </p>
      ))}
    </div>
  )
}

/** Semicircular churn gauge with the green → amber → red risk scale and an animated needle. */
export function Gauge({ value, band }: { value: number; band: RiskBand }) {
  const r = 80, cx = 100, cy = 96
  const point = (p: number) => [cx - r * Math.cos(Math.PI * p), cy - r * Math.sin(Math.PI * p)]
  const arc = (a: number, b: number) => {
    const [x1, y1] = point(a), [x2, y2] = point(b)
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`
  }
  const [nx, ny] = point(value)
  return (
    <svg viewBox="0 0 200 116" className="w-full max-w-[260px]" role="img" aria-label={`Churn risk ${Math.round(value * 100)}%`}>
      <path d={arc(0, 1)} stroke="var(--grid)" strokeWidth="14" fill="none" strokeLinecap="round" />
      {[[0, 0.4, 'Low'], [0.4, 0.7, 'Medium'], [0.7, 1, 'High']].map(([a, b, k]) => (
        <path key={k as string} d={arc((a as number) + 0.006, (b as number) - 0.006)} stroke={riskColor[k as RiskBand]} strokeOpacity={0.28} strokeWidth="14" fill="none" />
      ))}
      <motion.path d={arc(0, Math.max(value, 0.001))} stroke={riskColor[band]} strokeWidth="14" fill="none" strokeLinecap="round"
        initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1] }} />
      <motion.circle r="7" fill="var(--panel-strong)" stroke={riskColor[band]} strokeWidth="3"
        initial={{ cx: point(0)[0], cy: point(0)[1] }} animate={{ cx: nx, cy: ny }} transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1] }} />
      <text x={cx} y={cy - 6} textAnchor="middle" className="fill-ink font-display" fontSize="38" fontWeight="600">{Math.round(value * 100)}%</text>
    </svg>
  )
}

/** SHAP waterfall: each driver moves the churn probability from the average customer toward this customer's score. */
export function ShapWaterfall({ base, link, prediction, contributions }: {
  base: number; link: string; prediction: number; contributions: S['Contribution'][]
}) {
  const toProb = (z: number) => (link === 'logit' ? 1 / (1 + Math.exp(-z)) : z)
  let acc = base
  const step = (c: S['Contribution'], from: number, to: number) =>
    ({ ...c, range: [Math.min(from, to), Math.max(from, to)] as [number, number], from, to, shap: to - from })
  const rows = contributions.slice(0, 7).map((c) => {
    const start = acc
    acc += c.shap
    return step(c, toProb(start), toProb(acc))
  })
  rows.push(step({ feature: 'other', label: 'All other factors', value: null, shap: 0 }, toProb(acc), prediction))
  const fmtVal = (c: S['Contribution']) =>
    c.value === null ? '—' : Math.abs(c.value) < 1 && c.value !== 0 ? `${(c.value * 100).toFixed(0)}%` : c.value.toLocaleString('en-GB', { maximumFractionDigits: 1 })
  return (
    <ResponsiveContainer width="100%" height={rows.length * 38 + 50}>
      <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 24, top: 22, bottom: 4 }} barCategoryGap={8}>
        <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} {...axis} />
        <YAxis type="category" dataKey="label" width={190} {...axis} tick={{ fill: 'var(--ink-2)', fontSize: 12 }} />
        <ReferenceLine x={toProb(base)} stroke="var(--ink-3)" strokeDasharray="3 3" label={{ value: 'Average customer', position: 'top', fill: 'var(--ink-3)', fontSize: 11 }} />
        <ReferenceLine x={prediction} stroke={riskColor[prediction >= 0.7 ? 'High' : prediction >= 0.4 ? 'Medium' : 'Low']} label={{ value: 'This customer', position: 'top', fill: 'var(--ink-2)', fontSize: 11 }} />
        <Tooltip cursor={{ fill: 'var(--grid)' }} content={({ payload }) => {
          const r = payload?.[0]?.payload as (typeof rows)[number] | undefined
          return r ? <ChartTip title={r.label} rows={[
            ...(r.feature === 'other' ? [] : [{ label: 'This customer', value: fmtVal(r) }]),
            { label: 'Effect on risk', value: `${r.to >= r.from ? '+' : ''}${((r.to - r.from) * 100).toFixed(1)} pts`, color: r.shap > 0 ? riskColor.High : riskColor.Low },
          ]} /> : null
        }} />
        <Bar dataKey="range" radius={4} isAnimationActive animationDuration={700}>
          {rows.map((r) => <Cell key={r.feature} fill={r.shap > 0 ? riskColor.High : riskColor.Low} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

