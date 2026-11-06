import { createColumnHelper, flexRender, getCoreRowModel, useReactTable, type SortingState } from '@tanstack/react-table'
import { ArrowDown, ArrowUp, ChevronLeft, ChevronRight, ScanSearch } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCustomers, useFilters, type CustomerQuery, type S } from '@/api/client'
import { Card, PageHeader, RiskBadge, Skeleton, StateMessage, Tip } from '@/components/ui'
import { cn, fmt, riskBand, riskColor, segmentColor, type RiskBand } from '@/lib/utils'

const col = createColumnHelper<S['CustomerRow']>()
const columns = [
  col.accessor('customer_id', {
    header: 'Customer',
    cell: (c) => (
      <div className="whitespace-nowrap">
        <p className="font-medium tabular">#{c.getValue()}</p>
        <p className="text-xs text-ink-3">{c.row.original.country}</p>
      </div>
    ),
  }),
  col.accessor('segment', {
    header: 'Segment', enableSorting: false,
    cell: (c) => <span className="flex items-center gap-2 whitespace-nowrap"><span className="size-2 rounded-full" style={{ background: segmentColor(c.getValue()) }} />{c.getValue()}</span>,
  }),
  col.accessor('churn_prob', { header: 'Churn risk', cell: (c) => <RiskBadge band={riskBand(c.getValue())} probability={c.getValue()} /> }),
  col.accessor('revenue_at_risk', {
    header: 'Revenue at risk',
    cell: (c) => (
      <div className="w-32">
        <p className="font-semibold tabular">{fmt.gbp(c.getValue())}</p>
        <div className="mt-1 h-1 rounded-full bg-[var(--grid)]"><div className="h-full rounded-full" style={{ width: `${Math.min(100, c.row.original.churn_prob * 100)}%`, background: riskColor[riskBand(c.row.original.churn_prob)] }} /></div>
      </div>
    ),
  }),
  col.accessor('clv_pred', { header: 'Predicted CLV', cell: (c) => <span className="tabular">{fmt.gbp(c.getValue())}</span> }),
  col.accessor('recency_days', { header: 'Last order', cell: (c) => <span className="whitespace-nowrap tabular text-ink-2">{c.getValue() === 0 ? 'Today' : `${fmt.num(c.getValue())} days ago`}</span> }),
  col.accessor('next_purchase_prob', {
    header: () => <Tip content="Probability of another order within 90 days, from the purchase-sequence model."><span className="cursor-help">Next purchase</span></Tip>,
    cell: (c) => <span className="tabular text-ink-2">{fmt.pct(c.getValue())}</span>,
  }),
  col.accessor('top_driver', {
    header: 'Top churn driver', enableSorting: false,
    cell: (c) => (
      <span className="flex items-center gap-2 text-ink-2">
        {c.getValue()}
        {c.row.original.is_anomaly && <Tip content="Unusual spending pattern flagged by the anomaly detector — review manually."><ScanSearch className="size-4 text-mid" /></Tip>}
      </span>
    ),
  }),
]

export default function ChurnRisk() {
  const navigate = useNavigate()
  const { search } = useFilters()
  const [sorting, setSorting] = useState<SortingState>([{ id: 'revenue_at_risk', desc: true }])
  const [page, setPage] = useState(1)
  const [bands, setBands] = useState<RiskBand[]>(['High', 'Medium'])
  const [anomalies, setAnomalies] = useState(false)
  const pageSize = 15
  const q = useCustomers({
    page, page_size: pageSize, sort_by: (sorting[0]?.id ?? 'revenue_at_risk') as CustomerQuery['sort_by'],
    order: sorting[0]?.desc === false ? 'asc' : 'desc', risk_band: bands, anomalies_only: anomalies,
  })
  const table = useReactTable({
    data: q.data?.items ?? [], columns, manualSorting: true, manualPagination: true, state: { sorting },
    onSortingChange: (s) => { setSorting(s); setPage(1) }, getCoreRowModel: getCoreRowModel(),
  })
  const pages = Math.max(1, Math.ceil((q.data?.total ?? 0) / pageSize))
  const chip = (active: boolean) => cn('h-9 rounded-xl border px-3 text-sm transition', active ? 'border-iris/50 bg-iris/12 text-ink' : 'border-line text-ink-2 hover:text-ink')

  return (
    <>
      <PageHeader title="Churn risk" description="Customers ranked by revenue at risk — churn probability times predicted lifetime value — so the most valuable saves come first." />
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {(['High', 'Medium', 'Low'] as RiskBand[]).map((b) => (
          <button key={b} className={chip(bands.includes(b))} aria-pressed={bands.includes(b)}
            onClick={() => { setBands((v) => (v.includes(b) ? v.filter((x) => x !== b) : [...v, b])); setPage(1) }}>
            <span className="mr-2 inline-block size-2 rounded-full" style={{ background: riskColor[b] }} />{b} risk
          </button>
        ))}
        <button className={chip(anomalies)} aria-pressed={anomalies} onClick={() => { setAnomalies((a) => !a); setPage(1) }}>
          <ScanSearch className="mr-2 inline size-4 text-mid" />Unusual spending only
        </button>
        <span className="ml-auto text-sm text-ink-3">{q.data ? `${fmt.num(q.data.total)} customers` : ''}</span>
      </div>
      <Card className="overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[960px] text-sm">
            <thead>
              {table.getHeaderGroups().map((hg) => (
                <tr key={hg.id} className="border-b border-line text-left text-ink-3">
                  {hg.headers.map((h) => (
                    <th key={h.id} className="px-4 py-3 font-medium" aria-sort={h.column.getIsSorted() ? (h.column.getIsSorted() === 'desc' ? 'descending' : 'ascending') : undefined}>
                      {h.column.getCanSort() ? (
                        <button className="inline-flex items-center gap-1 hover:text-ink" onClick={h.column.getToggleSortingHandler()}>
                          {flexRender(h.column.columnDef.header, h.getContext())}
                          {h.column.getIsSorted() === 'desc' ? <ArrowDown className="size-3.5" /> : h.column.getIsSorted() === 'asc' ? <ArrowUp className="size-3.5" /> : null}
                        </button>
                      ) : flexRender(h.column.columnDef.header, h.getContext())}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody className={cn('transition-opacity', q.isPlaceholderData && 'opacity-60')}>
              {q.isLoading && Array.from({ length: 8 }, (_, i) => (
                <tr key={i}><td colSpan={columns.length} className="px-4 py-2"><Skeleton className="h-10" /></td></tr>
              ))}
              {table.getRowModel().rows.map((row) => (
                <tr key={row.id} tabIndex={0} onClick={() => navigate(`/customers/${row.original.customer_id}${search}`)}
                  onKeyDown={(e) => e.key === 'Enter' && navigate(`/customers/${row.original.customer_id}${search}`)}
                  className="cursor-pointer border-b border-line/60 transition last:border-0 hover:bg-iris/6 focus-visible:bg-iris/8">
                  {row.getVisibleCells().map((cell) => <td key={cell.id} className="px-4 py-3">{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
          {q.isError && <StateMessage kind="error" title="Couldn't load customers" detail={q.error.message} onRetry={() => q.refetch()} />}
          {q.data && !q.data.items.length && <StateMessage kind="empty" title="No customers match these filters" detail="Remove a risk band or widen the global filters." />}
        </div>
        <footer className="flex items-center justify-between border-t border-line px-4 py-3 text-sm text-ink-3">
          <span>Page {page} of {pages}</span>
          <div className="flex gap-2">
            <button className="grid size-9 place-items-center rounded-xl border border-line disabled:opacity-40" disabled={page <= 1} onClick={() => setPage((p) => p - 1)} aria-label="Previous page"><ChevronLeft className="size-4" /></button>
            <button className="grid size-9 place-items-center rounded-xl border border-line disabled:opacity-40" disabled={page >= pages} onClick={() => setPage((p) => p + 1)} aria-label="Next page"><ChevronRight className="size-4" /></button>
          </div>
        </footer>
      </Card>
    </>
  )
}
