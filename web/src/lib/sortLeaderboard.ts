import type { LeaderboardEntry } from '../types'

export type SortKey = 'rank' | 'team' | 'accuracy' | 'math' | 'mmlu' | 'params' | 'submitted'
export type SortDir = 'asc' | 'desc'

export const SORTABLE_COLUMNS: { key: SortKey; label: string; align: 'left' | 'right' }[] = [
  { key: 'rank', label: 'Rank', align: 'left' },
  { key: 'team', label: 'Team', align: 'left' },
  { key: 'accuracy', label: 'Accuracy', align: 'right' },
  { key: 'math', label: 'Math', align: 'right' },
  { key: 'mmlu', label: 'MMLU', align: 'right' },
  { key: 'params', label: 'Params', align: 'right' },
  { key: 'submitted', label: 'Submitted', align: 'right' },
]

/** Numeric/date columns default to descending (best or newest first); text columns default to ascending. */
export function defaultSortDir(key: SortKey): SortDir {
  return key === 'team' || key === 'rank' ? 'asc' : 'desc'
}

function compareNullableNumbers(a: number | null, b: number | null, sign: 1 | -1): number {
  if (a == null && b == null) return 0
  if (a == null) return 1 // entries missing this metric always sort last
  if (b == null) return -1
  return sign * (a - b)
}

function compare(a: LeaderboardEntry, b: LeaderboardEntry, key: SortKey, sign: 1 | -1): number {
  switch (key) {
    case 'team':
      return sign * a.team.localeCompare(b.team)
    case 'submitted':
      return sign * (new Date(a.submitted).getTime() - new Date(b.submitted).getTime())
    case 'rank':
      return sign * (a.rank - b.rank)
    case 'accuracy':
      return compareNullableNumbers(a.accuracy, b.accuracy, sign)
    case 'math':
      return compareNullableNumbers(a.math, b.math, sign)
    case 'mmlu':
      return compareNullableNumbers(a.mmlu, b.mmlu, sign)
    case 'params':
      return compareNullableNumbers(a.params, b.params, sign)
    default:
      return 0
  }
}

/** Returns a new, sorted array; never mutates `entries`. Ties keep their original relative order. */
export function sortLeaderboardEntries(
  entries: LeaderboardEntry[],
  key: SortKey,
  dir: SortDir,
): LeaderboardEntry[] {
  const sign: 1 | -1 = dir === 'asc' ? 1 : -1
  return [...entries].sort((a, b) => compare(a, b, key, sign))
}
