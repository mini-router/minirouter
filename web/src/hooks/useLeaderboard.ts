import { useEffect, useState } from 'react'
import { fetchLeaderboard } from '../lib/api'
import type { LeaderboardEntry } from '../types'

export function useLeaderboard(limit = 100) {
  const [entries, setEntries] = useState<LeaderboardEntry[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true

    fetchLeaderboard(limit)
      .then((nextEntries) => {
        if (!active) return
        setEntries(nextEntries)
        setError(null)
      })
      .catch((err: unknown) => {
        if (!active) return
        setEntries([])
        setError(err instanceof Error ? err.message : 'Failed to load leaderboard')
      })

    return () => {
      active = false
    }
  }, [limit])

  return { entries, error }
}
