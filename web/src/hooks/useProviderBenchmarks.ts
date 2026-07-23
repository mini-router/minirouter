import { useEffect, useState } from 'react'
import { fetchProviderBenchmarks } from '../lib/api'
import type { ProviderBenchmarkEntry } from '../lib/api'

export function useProviderBenchmarks(limit = 200) {
  const [entries, setEntries] = useState<ProviderBenchmarkEntry[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true

    fetchProviderBenchmarks(limit)
      .then((nextEntries) => {
        if (!active) return
        setEntries(nextEntries)
        setError(null)
      })
      .catch((err: unknown) => {
        if (!active) return
        setEntries([])
        setError(err instanceof Error ? err.message : 'Failed to load provider benchmarks')
      })

    return () => {
      active = false
    }
  }, [limit])

  return { entries, error }
}
