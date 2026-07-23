import { motion, type Variants } from 'framer-motion'
import { Link } from 'react-router-dom'
import { useLeaderboard } from '../hooks/useLeaderboard'
import brandVisual from '../assets/minirouter-brand.png'

const metrics = [
  { value: '~10K', label: 'Trainable params' },
  { value: 'Multi', label: 'Model routing' },
  { value: 'Live', label: 'Benchmark validation' },
  { value: 'TEE', label: 'MiniBridge calls' },
]

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.12, delayChildren: 0.2 },
  },
}

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: 'easeOut' } },
}

export default function Hero() {
  const { entries: leaderboard } = useLeaderboard(3)
  const topEntries = leaderboard.slice(0, 3)

  return (
    <section className="section-band pt-28 md:pt-32">
      <div className="section-shell">
        <motion.figure
          className="mb-10 overflow-hidden rounded-2xl border border-white/10 bg-surface-800/70 shadow-[0_24px_80px_rgba(2,8,23,0.32)]"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: 'easeOut', delay: 0.1 }}
        >
          <img
            src={brandVisual}
            alt="MiniRouter routing performance overview"
            className="w-full object-contain"
          />
        </motion.figure>

        <div className="grid gap-10 lg:grid-cols-[minmax(0,1.2fr)_minmax(340px,0.8fr)] lg:items-start">
          <motion.div
            className="space-y-7"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            <motion.div className="flex flex-wrap items-center gap-3" variants={itemVariants}>
              <span className="ui-chip">Benchmark-verified routing</span>
              <span className="meta-label">Frozen encoder, compact head, MiniBridge provider layer</span>
            </motion.div>

            <motion.div className="space-y-4" variants={itemVariants}>
              <h1 className="max-w-4xl text-5xl font-semibold leading-[1.02] tracking-tight md:text-6xl lg:text-7xl">
                MiniRouter
                <span className="gradient-text block">Efficient LLM Routing</span>
              </h1>
              <p className="section-copy max-w-2xl">
                MiniRouter selects the right model and role for each task with a small
                trainable head. The validator measures accuracy, runtime, and cost so
                routing improvements are judged by real benchmark performance.
              </p>
            </motion.div>

            <motion.div className="flex flex-wrap gap-3" variants={itemVariants}>
              <Link to="/leaderboard" className="button-secondary">
                View performance
              </Link>
              <Link to="/submit" className="button-primary">
                Submit router
              </Link>
            </motion.div>

            <motion.div
              className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4"
              variants={itemVariants}
            >
              {metrics.map((metric) => (
                <div key={metric.label} className="stat-card">
                  <span className="block text-3xl font-semibold text-text">{metric.value}</span>
                  <span className="mt-2 block text-sm text-text-dim">{metric.label}</span>
                </div>
              ))}
            </motion.div>
          </motion.div>

          <motion.aside
            className="panel p-6 lg:p-7"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: 'easeOut', delay: 0.15 }}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="meta-label">Performance snapshot</p>
                <h2 className="mt-2 text-xl font-semibold text-text">Measured router results</h2>
              </div>
              <span className="ui-chip">Live evals</span>
            </div>

            <div className="mt-6 divide-y divide-white/8">
              {topEntries.map((entry) => (
                <div key={entry.rank} className="flex items-center justify-between py-4">
                  <div>
                    <div className="flex items-center gap-3">
                      <span className={`rank-badge ${entry.rank === 1 ? 'bg-gold/15 text-gold' : entry.rank === 2 ? 'bg-silver/15 text-silver' : 'bg-bronze/15 text-bronze'}`}>
                        {entry.rank}
                      </span>
                      <span className="font-medium text-text">{entry.team}</span>
                    </div>
                    <div className="mt-1 text-sm text-text-dim">Validated accuracy</div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-semibold text-text font-mono">
                      {entry.accuracy == null ? '—' : `${(entry.accuracy * 100).toFixed(1)}%`}
                    </div>
                    <div className="text-sm text-text-dim">accuracy</div>
                  </div>
                </div>
              ))}
            </div>

            <div className="divider-line my-6" />

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-xl border border-white/8 bg-white/4 p-4">
                <div className="meta-label">Router artifact</div>
                <div className="mt-2 text-sm text-text">best_theta.npy plus summary metadata</div>
              </div>
              <div className="rounded-xl border border-white/8 bg-white/4 p-4">
                <div className="meta-label">Provider layer</div>
                <div className="mt-2 text-sm text-text">MiniBridge-backed model calls</div>
              </div>
            </div>
          </motion.aside>
        </div>
      </div>
    </section>
  )
}
