import { CheckCircle2, Download, FileSpreadsheet, FileText, Loader2, UploadCloud } from 'lucide-react'
import { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { toast } from 'sonner'
import { STATIC, useBatchScore, type S } from '@/api/client'
import { Button, Card, CardTitle, PageHeader, RiskBadge, StateMessage } from '@/components/ui'
import { cn, downloadBlob, fmt, riskBand, segmentColor } from '@/lib/utils'

const REQUIRED = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country']

async function missingColumns(file: File) {
  const header = (await file.slice(0, 4096).text()).split(/\r?\n/)[0].split(',').map((h) => h.trim().replace(/^"|"$/g, ''))
  return REQUIRED.filter((c) => !header.includes(c))
}

function toCsv(items: S['Prediction'][]) {
  const head = ['customer_id', 'churn_probability', 'risk_band', 'predicted_clv', 'revenue_at_risk', 'next_purchase_probability', 'segment', 'unusual_spending', 'explanation']
  const esc = (v: unknown) => `"${String(v ?? '').replace(/"/g, '""')}"`
  return [head.join(','), ...items.map((i) => [i.customer_id, i.churn_prob.toFixed(4), i.risk_band, i.clv_pred.toFixed(2), i.revenue_at_risk.toFixed(2),
    i.next_purchase_prob?.toFixed(4) ?? '', i.segment, i.is_anomaly, i.explanation].map(esc).join(','))].join('\n')
}

async function toPdf(result: S['BatchResult'], fileName: string) {
  const [{ jsPDF }, { autoTable }] = await Promise.all([import('jspdf'), import('jspdf-autotable')])
  const doc = new jsPDF({ orientation: 'landscape' })
  doc.setFontSize(18).text('Vantara — batch churn report', 14, 18)
  const atRisk = result.items.reduce((s, i) => s + i.revenue_at_risk, 0)
  doc.setFontSize(10).setTextColor(90).text(
    `${fileName}: ${result.scored} customers scored as of ${result.cutoff}, ${result.items.filter((i) => i.risk_band === 'High').length} at high risk, ${fmt.gbp(atRisk)} revenue at risk`, 14, 26)
  autoTable(doc, {
    startY: 32, styles: { fontSize: 8, cellPadding: 2 }, headStyles: { fillColor: [107, 88, 240] },
    head: [['Customer', 'Churn risk', 'Band', 'Predicted CLV', 'Revenue at risk', 'Segment', 'Why']],
    body: result.items.map((i) => [i.customer_id ?? '', fmt.pct(i.churn_prob), i.risk_band, fmt.gbp(i.clv_pred), fmt.gbp(i.revenue_at_risk), i.segment, i.explanation.replace(/^.*?: /, '')]),
    columnStyles: { 6: { cellWidth: 110 } },
  })
  doc.save(`vantara-churn-report-${result.cutoff}.pdf`)
}

export default function BatchScoring() {
  const [progress, setProgress] = useState(0)
  const [problem, setProblem] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const score = useBatchScore()
  const onDrop = async (files: File[]) => {
    const f = files[0]
    if (!f) return
    setProblem(null); setFile(f); setProgress(0); score.reset()
    const missing = await missingColumns(f)
    if (missing.length) return setProblem(`This file is missing ${missing.length === 1 ? 'a required column' : 'required columns'}: ${missing.join(', ')}.`)
    score.mutate({ file: f, onProgress: setProgress }, {
      onSuccess: (r) => toast.success(`Scored ${fmt.num(r.scored)} customers`),
      onError: (e) => toast.error(e.message),
    })
  }
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { 'text/csv': ['.csv'] }, multiple: false, disabled: score.isPending })
  const r = score.data
  const uploading = score.isPending && progress < 100

  return (
    <>
      <PageHeader title="Batch scoring" description="Upload transaction lines for any group of customers. Every customer is scored with the production models as of the day after their latest transaction." />
      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          {STATIC ? (
            <StateMessage kind="empty" title="Uploads are switched off in this hosted snapshot"
              detail="This copy of the dashboard is read-only. Batch scoring runs in the full deployment, which includes the scoring API." />
          ) : <>
          <div {...getRootProps()} className={cn('grid cursor-pointer place-items-center rounded-2xl border-2 border-dashed px-6 py-14 text-center transition',
            isDragActive ? 'border-iris bg-iris/10' : 'border-line hover:border-iris/50 hover:bg-iris/5')}>
            <input {...getInputProps()} aria-label="Upload transactions CSV" />
            {score.isPending ? <Loader2 className="size-10 animate-spin text-iris" /> : <UploadCloud className="size-10 text-iris" />}
            <p className="mt-4 font-display text-lg font-semibold">
              {score.isPending ? (uploading ? `Uploading ${file?.name}… ${progress}%` : 'Scoring customers…') : isDragActive ? 'Drop the file to score it' : 'Drag a CSV here, or click to choose one'}
            </p>
            <p className="mt-1 text-sm text-ink-3">Online Retail II format, up to 60 MB</p>
            {score.isPending && (
              <div className="mt-5 h-1.5 w-64 overflow-hidden rounded-full bg-[var(--grid)]">
                <div className={cn('h-full rounded-full bg-[linear-gradient(90deg,var(--iris),var(--aqua))] transition-[width]', !uploading && 'shimmer w-full')} style={uploading ? { width: `${progress}%` } : undefined} />
              </div>
            )}
          </div>
          {problem && <StateMessage kind="error" title="This file can't be scored" detail={problem} className="py-6" />}
          {score.isError && <StateMessage kind="error" title="Scoring failed" detail={score.error.message} className="py-6" />}
          </>}
        </Card>
        <Card>
          <CardTitle title="File requirements" />
          <p className="text-sm text-ink-2">One row per invoice line, with these columns:</p>
          <ul className="mt-3 flex flex-wrap gap-1.5">{REQUIRED.map((c) => <li key={c} className="rounded-lg border border-line px-2 py-1 text-xs tabular">{c}</li>)}</ul>
          <p className="mt-4 text-sm text-ink-3">Cancellations (invoices starting with C) are kept as return signals. Rows without a Customer ID or with a non-positive price are skipped.</p>
          <a href={`${import.meta.env.BASE_URL}sample_transactions.csv`} download className="mt-4 inline-flex items-center gap-2 text-sm text-iris hover:underline"><Download className="size-4" />Download a sample file</a>
        </Card>
      </div>

      {r && (
        <Card className="mt-4">
          <CardTitle title="Results" hint={`${fmt.num(r.rows_used)} of ${fmt.num(r.rows_received)} rows used, scored as of ${fmt.date(r.cutoff)}`}
            action={<div className="flex gap-2">
              <Button variant="ghost" onClick={() => downloadBlob(toCsv(r.items), `vantara-scores-${r.cutoff}.csv`, 'text/csv')}><FileSpreadsheet className="size-4" />CSV</Button>
              <Button onClick={() => toPdf(r, file?.name ?? 'upload.csv')}><FileText className="size-4" />PDF report</Button>
            </div>} />
          <div className="mb-5 grid gap-3 sm:grid-cols-4">
            {[
              ['Customers scored', fmt.num(r.scored)],
              ['High risk', fmt.num(r.items.filter((i) => i.risk_band === 'High').length)],
              ['Revenue at risk', fmt.gbp(r.items.reduce((s, i) => s + i.revenue_at_risk, 0))],
              ['Inactive > 2 years, skipped', fmt.num(r.skipped_inactive)],
            ].map(([k, v]) => <div key={k} className="rounded-2xl border border-line p-3"><p className="text-xs text-ink-3">{k}</p><p className="mt-1 font-display text-xl font-semibold tabular">{v}</p></div>)}
          </div>
          {!r.items.length ? <StateMessage kind="empty" title="No customers to score" detail="Every customer in this file was inactive for more than two years." /> : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[820px] text-sm">
                <thead><tr className="border-b border-line text-left text-ink-3">{['Customer', 'Churn risk', 'Predicted CLV', 'Revenue at risk', 'Segment', 'Why'].map((h) => <th key={h} className="px-3 py-2.5 font-medium">{h}</th>)}</tr></thead>
                <tbody>
                  {r.items.slice(0, 25).map((i) => (
                    <tr key={i.customer_id} className="border-b border-line/60 last:border-0">
                      <td className="px-3 py-2.5 font-medium tabular">#{i.customer_id}</td>
                      <td className="px-3 py-2.5"><RiskBadge band={riskBand(i.churn_prob)} probability={i.churn_prob} /></td>
                      <td className="px-3 py-2.5 tabular">{fmt.gbp(i.clv_pred)}</td>
                      <td className="px-3 py-2.5 font-semibold tabular">{fmt.gbp(i.revenue_at_risk)}</td>
                      <td className="px-3 py-2.5"><span className="flex items-center gap-2 whitespace-nowrap"><span className="size-2 rounded-full" style={{ background: segmentColor(i.segment) }} />{i.segment}</span></td>
                      <td className="max-w-md px-3 py-2.5 text-ink-2">{i.explanation.replace(/^.*?: /, '')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {r.items.length > 25 && <p className="mt-3 flex items-center gap-2 text-sm text-ink-3"><CheckCircle2 className="size-4 text-low" />Showing the top 25 by revenue at risk. Download the CSV for all {fmt.num(r.items.length)}.</p>}
            </div>
          )}
        </Card>
      )}
    </>
  )
}
