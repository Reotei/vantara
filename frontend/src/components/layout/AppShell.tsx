import * as Dialog from '@radix-ui/react-dialog'
import { Command } from 'cmdk'
import { AnimatePresence, motion } from 'motion/react'
import {
  Activity, BrainCircuit, CalendarRange, FileUp, LayoutDashboard, Moon, PanelLeftClose, PanelLeftOpen, Search, Sun, Users, X,
} from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import { STATIC, useCustomers, useFilterOptions, useFilters, useHealth } from '@/api/client'
import { MultiSelect, RiskBadge } from '@/components/ui'
import { cn, fmt, riskBand, segmentColor } from '@/lib/utils'

const NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/churn', label: 'Churn risk', icon: Activity },
  { to: '/segments', label: 'Segments', icon: Users },
  { to: '/batch', label: 'Batch scoring', icon: FileUp },
  { to: '/model', label: 'Model quality', icon: BrainCircuit },
]

function Logo({ compact }: { compact: boolean }) {
  return (
    <div className="flex items-center gap-2.5 px-1">
      <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" className="size-8" />
      {!compact && (
        <div className="leading-tight">
          <p className="font-display text-[17px] font-semibold tracking-tight">Vantara</p>
          <p className="text-[11px] text-ink-3">Customer Intelligence</p>
        </div>
      )}
    </div>
  )
}

function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  const { search } = useFilters()
  const health = useHealth()
  const options = useFilterOptions()
  const ok = health.data?.status === 'ok'
  return (
    <aside className={cn('glass sticky top-3 z-30 m-3 hidden h-[calc(100vh-1.5rem)] shrink-0 flex-col rounded-3xl p-3 transition-[width] duration-300 md:flex', collapsed ? 'w-[76px]' : 'w-60')}>
      <div className="flex items-center justify-between py-2">
        <Logo compact={collapsed} />
      </div>
      <nav className="mt-6 flex flex-1 flex-col gap-1" aria-label="Main">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={`${to}${search}`} end={to === '/'} title={collapsed ? label : undefined}
            className={({ isActive }) => cn('group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition', isActive ? 'text-ink' : 'text-ink-2 hover:bg-panel hover:text-ink')}>
            {({ isActive }) => (
              <>
                {isActive && <motion.span layoutId="nav-active" className="absolute inset-0 rounded-xl border border-iris/30 bg-iris/12" transition={{ type: 'spring', stiffness: 400, damping: 34 }} />}
                <Icon className={cn('relative size-[18px] shrink-0', isActive && 'text-iris')} />
                {!collapsed && <span className="relative">{label}</span>}
              </>
            )}
          </NavLink>
        ))}
      </nav>
      <div className={cn('mb-2 rounded-2xl border border-line p-3 text-xs text-ink-3', collapsed && 'hidden')}>
        <p className="flex items-center gap-2 text-ink-2">
          <span className={cn('size-2 rounded-full', STATIC ? 'bg-iris' : ok ? 'bg-low shadow-[0_0_10px_var(--risk-low)]' : 'bg-mid')} />
          {STATIC ? 'Read-only snapshot' : health.isLoading ? 'Checking service…' : ok ? 'Scoring service online' : 'Service degraded'}
        </p>
        {options.data && <p className="mt-1.5">Scores as of {fmt.date(options.data.scored_at)}</p>}
      </div>
      <button onClick={onToggle} className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-ink-3 hover:text-ink" aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
        {collapsed ? <PanelLeftOpen className="size-[18px]" /> : <><PanelLeftClose className="size-[18px]" />Collapse</>}
      </button>
    </aside>
  )
}

function CustomerSearch() {
  const [open, setOpen] = useState(false)
  const [term, setTerm] = useState('')
  const navigate = useNavigate()
  const { search } = useFilters()
  const results = useCustomers({ search: term.replace(/\D/g, ''), page_size: 8, sort_by: 'revenue_at_risk' })
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setOpen((o) => !o) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger className="flex h-9 w-full items-center gap-2 rounded-xl border border-line px-3 text-sm text-ink-3 transition hover:text-ink-2 sm:w-64">
        <Search className="size-4" /> Find a customer ID
        <kbd className="ml-auto hidden rounded-md border border-line px-1.5 text-[11px] sm:block">Ctrl K</kbd>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm" />
        <Dialog.Content className="fixed left-1/2 top-[18vh] z-50 w-[min(560px,92vw)] -translate-x-1/2 overflow-hidden rounded-2xl border border-line bg-panel-strong shadow-2xl">
          <Dialog.Title className="sr-only">Find a customer</Dialog.Title>
          <Command shouldFilter={false} label="Find a customer">
            <div className="flex items-center gap-2 border-b border-line px-4">
              <Search className="size-4 text-ink-3" />
              <Command.Input value={term} onValueChange={setTerm} placeholder="Type a customer ID, e.g. 14646" inputMode="numeric"
                className="h-12 flex-1 bg-transparent text-sm outline-none placeholder:text-ink-3" />
            </div>
            <Command.List className="max-h-80 overflow-auto p-2">
              <Command.Empty className="px-3 py-6 text-center text-sm text-ink-3">No scored customer matches that ID.</Command.Empty>
              {results.data?.items.map((c) => (
                <Command.Item key={c.customer_id} value={String(c.customer_id)}
                  onSelect={() => { setOpen(false); setTerm(''); navigate(`/customers/${c.customer_id}${search}`) }}
                  className="flex cursor-pointer items-center gap-3 rounded-xl px-3 py-2.5 text-sm data-[selected=true]:bg-iris/12">
                  <span className="font-medium tabular">#{c.customer_id}</span>
                  <span className="flex items-center gap-1.5 text-ink-3"><span className="size-2 rounded-full" style={{ background: segmentColor(c.segment) }} />{c.segment}</span>
                  <span className="ml-auto"><RiskBadge band={riskBand(c.churn_prob)} probability={c.churn_prob} /></span>
                </Command.Item>
              ))}
            </Command.List>
          </Command>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

function FilterBar() {
  const { filters, set, clear, active } = useFilters()
  const options = useFilterOptions().data
  const dateInput = 'h-9 rounded-xl border border-line bg-transparent px-2 text-sm text-ink-2 [color-scheme:inherit]'
  return (
    <div className="flex flex-wrap items-center gap-2" aria-label="Filters">
      <MultiSelect label="Segment" options={options?.segments ?? []} value={filters.segment ?? []} onChange={(v) => set('segment', v)}
        render={(o) => <span className="flex items-center gap-2"><span className="size-2 rounded-full" style={{ background: segmentColor(o) }} />{o}</span>} />
      <MultiSelect label="Country" options={options?.countries ?? []} value={filters.country ?? []} onChange={(v) => set('country', v)} />
      <MultiSelect label="Value tier" options={options?.tiers ?? []} value={filters.tier ?? []} onChange={(v) => set('tier', v)} />
      <label className="flex items-center gap-1.5 text-sm text-ink-3">
        <CalendarRange className="size-4" /><span className="sr-only">Last purchase from</span>
        <input type="date" className={dateInput} min={options?.min_date} max={options?.max_date} value={filters.date_from ?? ''} onChange={(e) => set('date_from', e.target.value)} />
        <span aria-hidden>to</span><span className="sr-only">Last purchase to</span>
        <input type="date" className={dateInput} min={options?.min_date} max={options?.max_date} value={filters.date_to ?? ''} onChange={(e) => set('date_to', e.target.value)} />
      </label>
      {active > 0 && <button onClick={clear} className="flex h-9 items-center gap-1 rounded-xl px-2 text-sm text-ink-3 hover:text-ink"><X className="size-3.5" />Clear all</button>}
    </div>
  )
}

function ThemeToggle() {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme ?? 'dark')
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('vantara-theme', theme) } catch { /* storage unavailable */ }
  }, [theme])
  const dark = theme === 'dark'
  return (
    <button onClick={() => setTheme(dark ? 'light' : 'dark')} aria-label={`Switch to ${dark ? 'light' : 'dark'} mode`}
      className="grid size-9 place-items-center rounded-xl border border-line text-ink-2 transition hover:text-ink">
      <AnimatePresence mode="wait" initial={false}>
        <motion.span key={theme} initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }} transition={{ duration: 0.2 }}>
          {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
        </motion.span>
      </AnimatePresence>
    </button>
  )
}

function MobileNav() {
  const { search } = useFilters()
  return (
    <nav className="glass fixed inset-x-3 bottom-3 z-40 flex justify-around rounded-2xl p-1.5 md:hidden" aria-label="Main">
      {NAV.map(({ to, label, icon: Icon }) => (
        <NavLink key={to} to={`${to}${search}`} end={to === '/'} aria-label={label}
          className={({ isActive }) => cn('grid size-11 place-items-center rounded-xl', isActive ? 'bg-iris/15 text-iris' : 'text-ink-3')}>
          <Icon className="size-5" />
        </NavLink>
      ))}
    </nav>
  )
}

export function AppShell({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(() => window.innerWidth < 1200)
  const location = useLocation()
  return (
    <div className="flex min-h-screen">
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />
      <div className="min-w-0 flex-1 pb-24 md:pb-6">
        <header className="sticky top-0 z-20 px-4 pt-3 sm:px-6">
          <div className="glass flex flex-wrap items-center gap-3 rounded-2xl px-3 py-2.5">
            <div className="md:hidden"><Logo compact /></div>
            <FilterBar />
            <div className="ml-auto flex w-full items-center gap-2 sm:w-auto"><CustomerSearch /><ThemeToggle /></div>
          </div>
        </header>
        <main className="px-4 pt-6 sm:px-6">
          <AnimatePresence mode="wait">
            <motion.div key={location.pathname} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
              {children}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
      <MobileNav />
    </div>
  )
}
