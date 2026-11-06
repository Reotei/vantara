/**
 * Read-only snapshot mode (VITE_STATIC=1): answers API calls from JSON exported by scripts/export_static.py,
 * so the dashboard can be hosted without a backend. The customer list is filtered and sorted in the browser.
 */
type Row = Record<string, string | number | boolean>
const BASE = `${import.meta.env.BASE_URL}snapshot/`
const cache = new Map<string, Promise<unknown>>()
const load = <T,>(file: string) => {
  if (!cache.has(file)) cache.set(file, fetch(BASE + file).then((r) => (r.ok ? r.json() : Promise.reject(new Error(file)))))
  return cache.get(file) as Promise<T>
}

const fnv1a = (text: string) => {
  let h = 0x811c9dc5
  for (const byte of new TextEncoder().encode(text)) h = Math.imul(h ^ byte, 0x01000193) >>> 0
  return h.toString(16).padStart(8, '0')
}
const canonical = (path: string, q: URLSearchParams) =>
  `${path}?${new URLSearchParams([...q].sort(([a, x], [b, y]) => (a === b ? (x < y ? -1 : x > y ? 1 : 0) : a < b ? -1 : 1))).toString()}`

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
const unavailable = (what: string) => json({ detail: `${what} is not part of this hosted snapshot. It is available in the live deployment.` }, 404)

async function customers(q: URLSearchParams) {
  let rows = await load<Row[]>('customers.json')
  const any = (key: string, col: string) => {
    const values = q.getAll(key)
    if (values.length) rows = rows.filter((r) => values.includes(String(r[col])))
  }
  any('segment', 'segment'); any('country', 'country'); any('tier', 'value_tier'); any('risk_band', 'risk_band')
  if (q.get('date_from')) rows = rows.filter((r) => String(r.last_purchase) >= q.get('date_from')!)
  if (q.get('date_to')) rows = rows.filter((r) => String(r.last_purchase) <= q.get('date_to')!)
  if (q.get('search')) rows = rows.filter((r) => String(r.customer_id).startsWith(q.get('search')!.trim()))
  if (q.get('anomalies_only') === 'true') rows = rows.filter((r) => r.is_anomaly)
  const key = q.get('sort_by') ?? 'revenue_at_risk', dir = q.get('order') === 'asc' ? 1 : -1
  rows = [...rows].sort((a, b) => (a[key] > b[key] ? dir : a[key] < b[key] ? -dir : 0))
  const page = Number(q.get('page') ?? 1), size = Number(q.get('page_size') ?? 25)
  return json({ items: rows.slice((page - 1) * size, page * size), total: rows.length, page, page_size: size })
}

export async function staticFetch(input: Request): Promise<Response> {
  const url = new URL(input.url)
  const path = url.pathname.replace(/^.*?(?=\/(kpis|trends|segments|filters|models|health|customers|predict)\b)/, '')
  if (path.startsWith('/predict')) return unavailable('Scoring new data')
  if (path === '/customers') return customers(url.searchParams)
  const profile = path.match(/^\/customers\/(\d+)(\/lime)?$/)
  if (profile) {
    const id = Number(profile[1])
    const data = profile[2] ? await load<Record<string, unknown>>('lime.json') : await load<Record<string, unknown>>(`profiles-${id % 32}.json`)
    return data[id] ? json(data[id]) : unavailable(profile[2] ? 'The LIME check for this customer' : `Customer #${id}`)
  }
  try {
    return json(await load(`${fnv1a(canonical(path, url.searchParams))}.json`))
  } catch {
    return unavailable('This combination of filters')
  }
}
