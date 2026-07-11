import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { useLeaderboard } from '../hooks/useLeaderboard'

export default function LeaderboardPreview() {
  const { entries } = useLeaderboard(3)

  return (
    <section className="section-band">
      <div className="section-shell">
        <motion.h2
          className="section-title max-w-2xl"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.5 }}
        >
          Leaderboard preview
        </motion.h2>

        <motion.div
          className="table-shell mt-8"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10 bg-white/4 text-left text-xs uppercase tracking-[0.22em] text-text-dim">
                <th className="p-4 font-medium">Rank</th>
                <th className="p-4 font-medium">Team</th>
                <th className="p-4 text-right font-medium">Accuracy</th>
                <th className="p-4 text-right font-medium">Head params</th>
                <th className="p-4 text-right font-medium">Submitted</th>
              </tr>
            </thead>
            <tbody>
              {entries.slice(0, 3).map((entry) => {
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
                    <td className="p-4 font-medium text-text">{entry.miner_id || entry.team}</td>
                    <td className="p-4 text-right font-mono">
                      {entry.accuracy == null ? '—' : `${(entry.accuracy * 100).toFixed(1)}%`}
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
                  </tr>
                )
              })}
            </tbody>
          </table>
        </motion.div>

        <div className="mt-8 flex justify-start">
          <Link
            to="/leaderboard"
            className="button-secondary"
          >
            Full Leaderboard
          </Link>
        </div>
      </div>
    </section>
  )
}
