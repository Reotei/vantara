import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TooltipProvider } from '@radix-ui/react-tooltip'
import { lazy, StrictMode, Suspense, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, MemoryRouter, Route, Routes } from 'react-router-dom'
import { STATIC } from '@/api/client'
import { Toaster } from 'sonner'
import { AppShell } from '@/components/layout/AppShell'
import { Skeleton } from '@/components/ui'
import './index.css'

const Overview = lazy(() => import('@/pages/Overview'))
const ChurnRisk = lazy(() => import('@/pages/ChurnRisk'))
const Segments = lazy(() => import('@/pages/Segments'))
const CustomerProfile = lazy(() => import('@/pages/CustomerProfile'))
const BatchScoring = lazy(() => import('@/pages/BatchScoring'))
const Model = lazy(() => import('@/pages/Model'))
const NotFound = lazy(() => import('@/pages/NotFound'))

// The hosted snapshot runs in a sandboxed frame where only in-page navigation is reliable.
const Router = STATIC ? MemoryRouter : ({ children }: { children: ReactNode }) => <BrowserRouter>{children}</BrowserRouter>
const queryClient = new QueryClient({ defaultOptions: { queries: { staleTime: 60_000, refetchOnWindowFocus: false, retry: 1 } } })

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Router>
          <AppShell>
            <Suspense fallback={<Skeleton className="h-[70vh]" />}>
              <Routes>
                <Route path="/" element={<Overview />} />
                <Route path="/churn" element={<ChurnRisk />} />
                <Route path="/segments" element={<Segments />} />
                <Route path="/customers/:id" element={<CustomerProfile />} />
                <Route path="/batch" element={<BatchScoring />} />
                <Route path="/model" element={<Model />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </Suspense>
          </AppShell>
        </Router>
        <Toaster theme="system" position="bottom-right" richColors />
      </TooltipProvider>
    </QueryClientProvider>
  </StrictMode>,
)
