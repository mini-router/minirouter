import { motion } from 'framer-motion'
import { useProviderBenchmarks } from '../hooks/useProviderBenchmarks'
import ProviderBenchmarkChart from './ProviderBenchmarkChart'

export default function ProviderBenchmarkSection() {
  const { entries, error } = useProviderBenchmarks()

  return (
    <section className="section-band pt-0">
      <div className="section-shell">
        <motion.p
          className="section-kicker"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          Provider tests
        </motion.p>
        <motion.h2
          className="section-title mt-3 max-w-3xl"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.08 }}
        >
          Model benchmarks
        </motion.h2>
        <motion.p
          className="section-copy mt-4 max-w-2xl"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.16 }}
        >
          Direct single-route benchmark results for each upstream model, independent of any miner
          submission.
        </motion.p>
        <div className="divider-line mt-8 mb-8" />

        {error ? (
          <div className="panel-soft rounded-xl border border-rose-400/20 px-5 py-4 text-sm text-rose-200">
            {error}
          </div>
        ) : (
          <ProviderBenchmarkChart points={entries} />
        )}
      </div>
    </section>
  )
}
