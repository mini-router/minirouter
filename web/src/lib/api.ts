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
  const report =
    entry.report.startsWith('/api/submissions/') ? `/submission/${entry.submission_id}` : entry.report

  return {
    rank: entry.rank,
    submission_id: entry.submission_id,
    team: entry.team,
    accuracy: entry.accuracy,
    gsm8k: entry.gsm8k,
    mmlu: entry.mmlu,
    math: entry.math,
    humaneval: entry.humaneval,
    bbh: entry.bbh,
    params: entry.params,
    submitted: entry.submitted,
    report,
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

interface BackendEvaluationOut {
  id: number
  submission_id: string
  status: string
  score: number | null
  metrics: Record<string, unknown>
  command: string | null
  stdout: string | null
  stderr: string | null
  results_path: string | null
  error: string | null
  started_at: string | null
  finished_at: string | null
  created_at: string
}

interface BackendSubmissionOut {
  id: string
  source: string
  team_name: string | null
  repo_full_name: string | null
  pr_number: number | null
  head_sha: string | null
  artifact_name: string
  artifact_path: string
  artifact_sha256: string
  checkpoint_path: string | null
  benchmark: string
  status: string
  latest_score: number | null
  best_run_id: number | null
  created_at: string
  updated_at: string
  evaluations: BackendEvaluationOut[]
}

export async function fetchSubmission(submissionId: string): Promise<BackendSubmissionOut> {
  const response = await fetch(apiUrl(`/api/submissions/${submissionId}`), {
    headers: {
      Accept: 'application/json',
    },
  })

  if (!response.ok) {
    throw new Error(`Submission request failed with status ${response.status}`)
  }

  return (await response.json()) as BackendSubmissionOut
}
