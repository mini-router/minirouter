import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { LEADERBOARD_DATA } from '../data/leaderboard'

export default function LeaderboardTable() {
  const [search, setSearch] = useState('')

  const filtered = useMemo(() => {
    return LEADERBOARD_DATA.filter((e) =>
      e.team.toLowerCase().includes(search.toLowerCase())
    )
  }, [search])

  return (
    <div className="section-shell">
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-2xl">
          <p className="meta-label">Leaderboard</p>
          <h2 className="mt-2 text-2xl font-semibold text-text">
            Ranked by macro-average accuracy
          </h2>
          <p className="mt-2 text-sm text-text-dim">
            {filtered.length} entries currently shown. Search narrows by team name.
          </p>
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
                <th className="p-4 font-medium">Rank</th>
                <th className="p-4 font-medium">Team</th>
                <th className="p-4 text-right font-medium">Accuracy</th>
                <th className="p-4 text-right font-medium">Math</th>
                <th className="p-4 text-right font-medium">MMLU</th>
                <th className="p-4 text-right font-medium">Params</th>
                <th className="p-4 text-right font-medium">Submitted</th>
                <th className="p-4 text-center font-medium">Report</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={8} className="p-8 text-center text-text-dim">
                    No entries found
                  </td>
                </tr>
              ) : (
                filtered.map((entry) => {
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
                        {(entry.accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="p-4 text-right font-mono text-text-dim">
                        {(entry.math * 100).toFixed(1)}%
                      </td>
                      <td className="p-4 text-right font-mono text-text-dim">
                        {(entry.mmlu * 100).toFixed(1)}%
                      </td>
                      <td className="p-4 text-right font-mono text-text-dim">
                        {entry.params.toLocaleString()}
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
                          <a
                            href={entry.report}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs font-medium text-accent-light hover:text-accent-light/80"
                          >
                            View
                          </a>
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
