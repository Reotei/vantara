import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs))

const gbp = new Intl.NumberFormat('en-GB', { style: 'currency', currency: 'GBP', maximumFractionDigits: 0 })
const gbpCompact = new Intl.NumberFormat('en-GB', { style: 'currency', currency: 'GBP', notation: 'compact', maximumFractionDigits: 1 })
const num = new Intl.NumberFormat('en-GB', { maximumFractionDigits: 0 })

export const fmt = {
  gbp: (v: number) => gbp.format(v),
  gbpShort: (v: number) => (Math.abs(v) >= 10_000 ? gbpCompact.format(v) : gbp.format(v)),
  num: (v: number) => num.format(v),
  price: (v: number) => v.toLocaleString('en-GB', { style: 'currency', currency: 'GBP' }),
  pct: (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`,
  date: (v: string) => new Date(v).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }),
  month: (v: string) => new Date(`${v}-01`).toLocaleDateString('en-GB', { month: 'short', year: '2-digit' }),
}

export type RiskBand = 'Low' | 'Medium' | 'High'
export const riskColor: Record<RiskBand, string> = { Low: 'var(--risk-low)', Medium: 'var(--risk-mid)', High: 'var(--risk-high)' }
export const riskBand = (p: number, bands = { medium: 0.4, high: 0.7 }): RiskBand =>
  p >= bands.high ? 'High' : p >= bands.medium ? 'Medium' : 'Low'
export const riskFill = (p: number) => riskColor[riskBand(p)]

// Colour follows the segment, never its rank. Blue, violet, aqua and magenta go first: they never read as a risk colour.
const SEGMENT_SLOT: Record<string, number> = {
  Champions: 1, 'Loyal Regulars': 7, 'New & Promising': 3, 'Seasonal Gift Shoppers': 5,
  'Big-Ticket Wholesalers': 2, 'Fading Occasionals': 6, 'Lapsed High-Value': 4,
}
export const segmentColor = (segment: string) => `var(--series-${SEGMENT_SLOT[segment] ?? 1})`

export function downloadBlob(content: BlobPart, name: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }))
  Object.assign(document.createElement('a'), { href: url, download: name }).click()
  URL.revokeObjectURL(url)
}
