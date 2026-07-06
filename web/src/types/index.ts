export interface LeaderboardEntry {
  rank: number
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
  status?: string
}
