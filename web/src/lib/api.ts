import type { LeaderboardEntry } from '../types'

const DEFAULT_API_BASE_URL = 'https://minirouter.work.gd'

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL?.trim() || DEFAULT_API_BASE_URL).replace(
  /\/+$/,
  '',
)

export function apiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) {
    return path
  }

  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${API_BASE_URL}${normalizedPath}`
}

interface BackendLeaderboardEntry {
  rank: number
  submission_id: string
  team: string
  accuracy: number | null
  gsm8k: number | null
  mmlu: number | null
  math: number | null
  humaneval: number | null
  bbh: number | null
  params: number | null
  submitted: string
  report: string
  status: string
}

interface BackendLeaderboardResponse {
  items: BackendLeaderboardEntry[]
}

function normalizeLeaderboardEntry(entry: BackendLeaderboardEntry): LeaderboardEntry {
  return {
    rank: entry.rank,
    team: entry.team,
    accuracy: entry.accuracy,
    gsm8k: entry.gsm8k,
    mmlu: entry.mmlu,
    math: entry.math,
    humaneval: entry.humaneval,
    bbh: entry.bbh,
    params: entry.params,
    submitted: entry.submitted,
    report: apiUrl(entry.report),
    status: entry.status,
  }
}

export async function fetchLeaderboard(limit = 100): Promise<LeaderboardEntry[]> {
  const response = await fetch(apiUrl(`/api/leaderboard?limit=${limit}`), {
    headers: {
      Accept: 'application/json',
    },
  })

  if (!response.ok) {
    throw new Error(`Leaderboard request failed with status ${response.status}`)
  }

  const payload = (await response.json()) as BackendLeaderboardResponse
  return payload.items.map(normalizeLeaderboardEntry)
}
