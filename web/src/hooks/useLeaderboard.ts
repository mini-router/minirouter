import { useEffect, useMemo, useState } from 'react'
import { LEADERBOARD_DATA } from '../data/leaderboard'
import { fetchLeaderboard } from '../lib/api'
import type { LeaderboardEntry } from '../types'

type LeaderboardSource = 'api' | 'static'

export function useLeaderboard(limit = 100) {
  const fallback = useMemo(() => LEADERBOARD_DATA.slice(0, limit), [limit])
  const [entries, setEntries] = useState<LeaderboardEntry[]>(fallback)
  const [source, setSource] = useState<LeaderboardSource>('static')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true

    fetchLeaderboard(limit)
      .then((nextEntries) => {
        if (!active) return
        setEntries(nextEntries)
        setSource('api')
        setError(null)
      })
      .catch((err: unknown) => {
        if (!active) return
        setEntries(fallback)
        setSource('static')
        setError(err instanceof Error ? err.message : 'Failed to load leaderboard')
      })

    return () => {
      active = false
    }
  }, [fallback, limit])

  return { entries, source, error }
}
