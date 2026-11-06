import NumberFlow, { type Format } from '@number-flow/react'
import * as Popover from '@radix-ui/react-popover'
import * as RTooltip from '@radix-ui/react-tooltip'
import { Check, ChevronDown, CircleAlert, Inbox, RotateCw } from 'lucide-react'
import type { ComponentProps, ReactNode } from 'react'
import { cn, riskColor, type RiskBand } from '@/lib/utils'

export function Card({ className, children, beam, ...props }: ComponentProps<'section'> & { beam?: boolean }) {
  return (
    <section
      {...props}
      onMouseMove={(e) => {
        const r = e.currentTarget.getBoundingClientRect()
        e.currentTarget.style.setProperty('--mx', `${e.clientX - r.left}px`)
        e.currentTarget.style.setProperty('--my', `${e.clientY - r.top}px`)
      }}
      className={cn('glass spotlight rounded-3xl p-5', className)}
    >
      {beam && <span aria-hidden className="beam" />}
      {children}
    </section>
  )
}

export function CardTitle({ title, hint, action }: { title: string; hint?: string; action?: ReactNode }) {
  return (
    <header className="mb-4 flex items-start justify-between gap-3">
      <div>
        <h2 className="font-display text-[17px] font-semibold tracking-tight">{title}</h2>
        {hint && <p className="mt-0.5 text-[13px] text-ink-3">{hint}</p>}
      </div>
      {action}
    </header>
  )
}

export function PageHeader({ title, description, action }: { title: string; description: string; action?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="font-display text-3xl font-semibold tracking-[-0.02em] sm:text-[34px]">{title}</h1>
        <p className="mt-1 max-w-2xl text-[15px] text-ink-2">{description}</p>
      </div>
      {action}
    </div>
  )
}

export const Skeleton = ({ className }: { className?: string }) => <div className={cn('shimmer rounded-xl', className)} />

export function Animated({ value, format, className }: { value: number; format: 'gbp' | 'num' | 'pct'; className?: string }) {
  const options: Record<string, Format> = {
    gbp: { style: 'currency', currency: 'GBP', notation: value >= 100_000 ? 'compact' : 'standard', maximumFractionDigits: value >= 100_000 ? 1 : 0 },
    num: { maximumFractionDigits: 0 },
    pct: { style: 'percent', maximumFractionDigits: 0 },
  }
  return <NumberFlow value={value} format={options[format]} locales="en-GB" className={cn('tabular', className)} />
}

export function RiskBadge({ band, probability, size = 'sm' }: { band: RiskBand; probability?: number; size?: 'sm' | 'md' }) {
  return (
    <span
      className={cn('inline-flex items-center gap-1.5 rounded-full font-medium tabular whitespace-nowrap', size === 'sm' ? 'px-2.5 py-0.5 text-xs' : 'px-3 py-1 text-sm')}
      style={{ color: riskColor[band], background: `color-mix(in oklab, ${riskColor[band]} 14%, transparent)` }}
    >
      <span className="size-1.5 rounded-full" style={{ background: riskColor[band] }} />
      {probability === undefined ? `${band} risk` : `Churn risk ${Math.round(probability * 100)}%`}
    </span>
  )
}

export function Button({ className, variant = 'primary', ...props }: ComponentProps<'button'> & { variant?: 'primary' | 'ghost' }) {
  return (
    <button
      {...props}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-medium transition active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50',
        variant === 'primary'
          ? 'bg-[linear-gradient(120deg,var(--iris),color-mix(in_oklab,var(--iris)_55%,var(--aqua)))] text-white shadow-[0_8px_24px_-10px_var(--iris)] hover:brightness-110'
          : 'border border-line text-ink-2 hover:bg-panel hover:text-ink',
        className,
      )}
    />
  )
}

export function Tip({ content, children }: { content: ReactNode; children: ReactNode }) {
  return (
    <RTooltip.Root delayDuration={150}>
      <RTooltip.Trigger asChild>{children}</RTooltip.Trigger>
      <RTooltip.Portal>
        <RTooltip.Content sideOffset={6} className="z-50 max-w-xs rounded-lg border border-line bg-panel-strong px-3 py-2 text-xs text-ink-2 shadow-xl">
          {content}
        </RTooltip.Content>
      </RTooltip.Portal>
    </RTooltip.Root>
  )
}

export function StateMessage({ kind, title, detail, onRetry, className }: {
  kind: 'empty' | 'error'; title: string; detail?: string; onRetry?: () => void; className?: string
}) {
  const Icon = kind === 'error' ? CircleAlert : Inbox
  return (
    <div role={kind === 'error' ? 'alert' : 'status'} className={cn('flex flex-col items-center justify-center gap-2 py-10 text-center', className)}>
      <Icon className={cn('size-6', kind === 'error' ? 'text-high' : 'text-ink-3')} />
      <p className="font-medium">{title}</p>
      {detail && <p className="max-w-sm text-sm text-ink-3">{detail}</p>}
      {onRetry && <Button variant="ghost" onClick={onRetry} className="mt-2"><RotateCw className="size-4" />Try again</Button>}
    </div>
  )
}

export function MultiSelect({ label, options, value, onChange, render }: {
  label: string; options: string[]; value: string[]; onChange: (v: string[]) => void; render?: (o: string) => ReactNode
}) {
  const toggle = (o: string) => onChange(value.includes(o) ? value.filter((v) => v !== o) : [...value, o])
  return (
    <Popover.Root>
      <Popover.Trigger className={cn('inline-flex h-9 items-center gap-2 rounded-xl border px-3 text-sm transition hover:text-ink', value.length ? 'border-iris/50 bg-iris/10 text-ink' : 'border-line text-ink-2')}>
        {label}
        {value.length > 0 && <span className="rounded-md bg-iris px-1.5 text-[11px] font-semibold text-white">{value.length}</span>}
        <ChevronDown className="size-3.5 opacity-60" />
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content align="start" sideOffset={6} className="z-50 max-h-80 w-60 overflow-auto rounded-2xl border border-line bg-panel-strong p-1.5 shadow-2xl">
          {options.map((o) => (
            <button key={o} onClick={() => toggle(o)} className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm hover:bg-iris/10">
              <span className={cn('grid size-4 place-items-center rounded border', value.includes(o) ? 'border-iris bg-iris text-white' : 'border-line')}>
                {value.includes(o) && <Check className="size-3" />}
              </span>
              {render ? render(o) : o}
            </button>
          ))}
          {value.length > 0 && (
            <button onClick={() => onChange([])} className="mt-1 w-full rounded-lg px-2.5 py-2 text-left text-sm text-ink-3 hover:text-ink">Clear {label.toLowerCase()}</button>
          )}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  )
}

export function Delta({ now, before, invert, format }: { now: number; before?: number | null; invert?: boolean; format: (v: number) => string }) {
  if (before === undefined || before === null || before === 0) return <span className="text-xs text-ink-3">No prior period</span>
  const change = (now - before) / Math.abs(before)
  const good = invert ? change <= 0 : change >= 0
  return (
    <span className="text-xs text-ink-3">
      <span className={cn('mr-1 rounded-md px-1.5 py-0.5 font-semibold tabular', good ? 'bg-low/12 text-low' : 'bg-high/12 text-high')}>
        {change >= 0 ? '+' : ''}{(change * 100).toFixed(1)}%
      </span>
      vs {format(before)} last quarter
    </span>
  )
}
