import { keepPreviousData, useMutation, useQuery } from '@tanstack/react-query'
import createClient from 'openapi-fetch'
import { useSearchParams } from 'react-router-dom'
import type { components, paths } from './schema'
import { staticFetch } from './static'

export type S = components['schemas']
export const API_URL = import.meta.env.VITE_API_URL ?? '/api'
export const STATIC = import.meta.env.VITE_STATIC === '1'
const api = createClient<paths>({
  baseUrl: STATIC ? new URL('api', document.baseURI).href : API_URL,
  querySerializer: { array: { style: 'form', explode: true } },
  ...(STATIC && { fetch: staticFetch }),
})

export class ApiError extends Error {}

async function unwrap<T>(p: Promise<{ data?: T; error?: unknown; response: Response }>): Promise<T> {
  const { data, error, response } = await p
  if (error !== undefined || data === undefined) {
    const detail = (error as { detail?: unknown })?.detail
    throw new ApiError(typeof detail === 'string' ? detail : `Request failed (${response.status})`)
  }
  return data
}

export const FILTER_KEYS = ['segment', 'country', 'tier', 'date_from', 'date_to'] as const
export type Filters = { segment?: string[]; country?: string[]; tier?: string[]; date_from?: string; date_to?: string }

export function useFilters() {
  const [params, setParams] = useSearchParams()
  const filters: Filters = {
    segment: params.getAll('segment'), country: params.getAll('country'), tier: params.getAll('tier'),
    date_from: params.get('date_from') ?? undefined, date_to: params.get('date_to') ?? undefined,
  }
  const set = (key: keyof Filters, value: string[] | string | undefined) =>
    setParams((prev) => {
      const next = new URLSearchParams(prev)
      next.delete(key)
      ;[value ?? []].flat().filter(Boolean).forEach((v) => next.append(key, v))
      return next
    }, { replace: true })
  const clear = () => setParams((prev) => {
    const next = new URLSearchParams(prev)
    FILTER_KEYS.forEach((k) => next.delete(k))
    return next
  }, { replace: true })
  const active = FILTER_KEYS.reduce((n, k) => n + [filters[k] ?? []].flat().filter(Boolean).length, 0)
  const search = new URLSearchParams([...params].filter(([k]) => (FILTER_KEYS as readonly string[]).includes(k))).toString()
  return { filters, set, clear, active, search: search ? `?${search}` : '' }
}

const clean = (f: Filters) => Object.fromEntries(Object.entries(f).filter(([, v]) => (Array.isArray(v) ? v.length : v)))

export const useFilterOptions = () =>
  useQuery({ queryKey: ['filters'], queryFn: () => unwrap(api.GET('/filters')), staleTime: Infinity })

export function useKpis() {
  const { filters } = useFilters()
  return useQuery({ queryKey: ['kpis', filters], queryFn: () => unwrap(api.GET('/kpis', { params: { query: clean(filters) } })), placeholderData: keepPreviousData })
}

export function useTrends() {
  const { filters } = useFilters()
  return useQuery({ queryKey: ['trends', filters], queryFn: () => unwrap(api.GET('/trends', { params: { query: clean(filters) } })), placeholderData: keepPreviousData })
}

export type CustomerQuery = NonNullable<paths['/customers']['get']['parameters']['query']>
export function useCustomers(q: CustomerQuery) {
  const { filters } = useFilters()
  return useQuery({
    queryKey: ['customers', filters, q],
    queryFn: () => unwrap(api.GET('/customers', { params: { query: { ...clean(filters), ...q } } })),
    placeholderData: keepPreviousData,
  })
}

export const useCustomer = (id: number) =>
  useQuery({ queryKey: ['customer', id], queryFn: () => unwrap(api.GET('/customers/{customer_id}', { params: { path: { customer_id: id } } })), retry: false })

export const useLime = (id: number, enabled: boolean) =>
  useQuery({ queryKey: ['lime', id], enabled, staleTime: Infinity, queryFn: () => unwrap(api.GET('/customers/{customer_id}/lime', { params: { path: { customer_id: id } } })) })

export function useSegments() {
  const { filters } = useFilters()
  return useQuery({ queryKey: ['segments', filters], queryFn: () => unwrap(api.GET('/segments', { params: { query: clean(filters) } })), placeholderData: keepPreviousData })
}

export type ModelRow = {
  model: string; family: 'classical' | 'deep'; roc_auc: number; val_roc_auc: number; recall: number; precision: number
  f1: number; accuracy: number; threshold: number; confusion_matrix: number[][]; cv_roc_auc?: number
}
type Curve = { epoch: number; train: number; valid: number }[]
type RegressionScores = { mae: number; rmse: number; r2: number; r2_log: number }
export type ModelCard = {
  production_model: string; threshold: number; snapshot: string; churn_rate: number; recent_customers_roc_auc: number
  split_sizes: Record<string, number>; comparison: ModelRow[]; roc_curve: { fpr: number; tpr: number }[]
  global_importance: { feature: string; label: string; importance: number }[]
  pdp: Record<string, { x: number; y: number }[]>; feature_labels: Record<string, string>
  loss_curves: Record<'ann' | 'lstm', Curve>; error_correlation: Record<string, number>
  clv: { model: string; horizon_days: number; test: RegressionScores; validation: Record<string, RegressionScores & { cv_r2: number }> }
  anomaly: { threshold: number; test_flag_rate: number; flagged_median_spend: number | null; overall_median_spend: number }
  segmentation: {
    k: number; elbow: { k: number; inertia: number; silhouette: number; davies_bouldin: number }[]
    kmeans: { silhouette: number; davies_bouldin: number }
    gmm: { k: number; silhouette: number; davies_bouldin: number; ari_vs_kmeans: number }
  }
}
export const useModelCard = () =>
  useQuery({ queryKey: ['models'], staleTime: Infinity, queryFn: async () => (await unwrap(api.GET('/models'))) as unknown as ModelCard })

export const useHealth = () => useQuery({ queryKey: ['health'], queryFn: () => unwrap(api.GET('/health')), refetchInterval: 60_000 })

export function useBatchScore() {
  return useMutation({
    mutationFn: async ({ file, onProgress }: { file: File; onProgress: (p: number) => void }) => {
      const body = new FormData()
      body.append('file', file)
      const xhr = new XMLHttpRequest()
      return new Promise<S['BatchResult']>((resolve, reject) => {
        xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(Math.round((e.loaded / e.total) * 100))
        xhr.onload = () => {
          const json = JSON.parse(xhr.responseText || '{}')
          xhr.status < 300 ? resolve(json) : reject(new ApiError(typeof json.detail === 'string' ? json.detail : `Upload failed (${xhr.status})`))
        }
        xhr.onerror = () => reject(new ApiError('Could not reach the scoring service. Check your connection and try again.'))
        xhr.open('POST', `${API_URL}/predict/batch`)
        xhr.send(body)
      })
    },
  })
}
