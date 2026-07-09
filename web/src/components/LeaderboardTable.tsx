import { useState, useMemo, type ReactNode } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { useLeaderboard } from '../hooks/useLeaderboard'
import {
  SORTABLE_COLUMNS,
  defaultSortDir,
  sortLeaderboardEntries,
  type SortDir,
  type SortKey,
} from '../lib/sortLeaderboard'

function SortableHeader({
  label,
  sortKey,
  activeKey,
  activeDir,
  align = 'left',
  onSort,
}: {
  label: ReactNode
  sortKey: SortKey
  activeKey: SortKey
  activeDir: SortDir
  align?: 'left' | 'right'
  onSort: (key: SortKey) => void
}) {
  const active = sortKey === activeKey
  const ariaSort = active ? (activeDir === 'asc' ? 'ascending' : 'descending') : 'none'
  const indicator = active ? (activeDir === 'asc' ? '↑' : '↓') : null
  return (
    <th
      className={`p-4 font-medium ${align === 'right' ? 'text-right' : 'text-left'}`}
      aria-sort={ariaSort}
    >
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        className={`inline-flex items-center gap-1 uppercase tracking-[0.22em] transition-colors hover:text-text ${
          align === 'right' ? 'flex-row-reverse' : ''
        }`}
      >
        <span>{label}</span>
        <span className="w-3 text-accent-light">{indicator}</span>
      </button>
    </th>
  )
}

export default function LeaderboardTable() {
  const [search, setSearch] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('rank')
  const [sortDir, setSortDir] = useState<SortDir>('asc')
  const { entries, error } = useLeaderboard()

  const filtered = useMemo(() => {
    return entries.filter((e) => e.team.toLowerCase().includes(search.toLowerCase()))
  }, [entries, search])

  const sorted = useMemo(
    () => sortLeaderboardEntries(filtered, sortKey, sortDir),
    [filtered, sortKey, sortDir],
  )

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir(defaultSortDir(key))
    }
  }

  return (
    <div className="section-shell">
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-2xl">
          <p className="meta-label">Leaderboard</p>
          <h2 className="mt-2 text-2xl font-semibold text-text">
            Ranked by macro-average accuracy
          </h2>
          <p className="mt-2 text-sm text-text-dim">
            {sorted.length} entries currently shown. Search narrows by team name, and column
            headers sort.
            {' '}Live data is loaded from the backend.
          </p>
          {error && <p className="mt-2 text-xs text-text-dim">Backend fetch failed.</p>}
        </div>
        <div className="w-full max-w-md">
          <label className="meta-label mb-2 block" htmlFor="leaderboard-search">
            Search teams
          </label>
          <input
            id="leaderboard-search"
            type="text"
            placeholder="Baseline, random routing, your team..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-full border border-white/10 bg-white/5 px-4 py-3 text-sm text-text placeholder:text-text-dim/70 focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
          />
        </div>
      </div>

      <motion.div
        className="table-shell"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10 bg-white/4 text-left text-xs uppercase tracking-[0.22em] text-text-dim">
                {SORTABLE_COLUMNS.map((column) => (
                  <SortableHeader
                    key={column.key}
                    label={column.label}
                    sortKey={column.key}
                    activeKey={sortKey}
                    activeDir={sortDir}
                    align={column.align}
                    onSort={handleSort}
                  />
                ))}
                <th className="p-4 text-center font-medium">Report</th>
              </tr>
            </thead>
            <tbody>
              {sorted.length === 0 ? (
                <tr>
                  <td colSpan={8} className="p-8 text-center text-text-dim">
                    No entries found
                  </td>
                </tr>
              ) : (
                sorted.map((entry) => {
                  const rankClass =
                    entry.rank === 1
                      ? 'text-gold'
                      : entry.rank === 2
                        ? 'text-silver'
                        : entry.rank === 3
                          ? 'text-bronze'
                          : 'text-text-dim'
                  return (
                    <tr
                      key={entry.rank}
                      className="border-b border-white/8 last:border-0 transition-colors hover:bg-white/4"
                    >
                      <td className="p-4">
                        <span className={`rank-badge ${rankClass} bg-white/5`}>{entry.rank}</span>
                      </td>
                      <td className="p-4 font-medium text-text">{entry.team}</td>
                      <td className="p-4 text-right font-mono">
                        {entry.accuracy == null ? '—' : `${(entry.accuracy * 100).toFixed(1)}%`}
                      </td>
                      <td className="p-4 text-right font-mono text-text-dim">
                        {entry.math == null ? '—' : `${(entry.math * 100).toFixed(1)}%`}
                      </td>
                      <td className="p-4 text-right font-mono text-text-dim">
                        {entry.mmlu == null ? '—' : `${(entry.mmlu * 100).toFixed(1)}%`}
                      </td>
                      <td className="p-4 text-right font-mono text-text-dim">
                        {entry.params == null ? '—' : entry.params.toLocaleString()}
                      </td>
                      <td className="p-4 text-right text-text-dim">
                        {new Date(entry.submitted).toLocaleDateString('en-US', {
                          month: 'short',
                          day: 'numeric',
                          year: 'numeric',
                        })}
                      </td>
                      <td className="p-4 text-center">
                        {entry.report === '#' ? (
                          <span className="text-xs text-text-dim">Pending</span>
                        ) : (
                          <Link
                            to={entry.report}
                            className="text-xs font-medium text-accent-light hover:text-accent-light/80"
                          >
                            View
                          </Link>
                        )}
                      </td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
  )
}
