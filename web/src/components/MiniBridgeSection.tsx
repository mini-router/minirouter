import { motion } from 'framer-motion'

const bridgePoints = [
  {
    label: 'Maintainer caller',
    value: 'minirouter-maintainer',
    desc: 'The validator identifies itself to MiniBridge as the allowed MiniRouter caller.',
  },
  {
    label: 'Miner provider',
    value: 'minirouter-miners',
    desc: 'Provider credentials stay on the MiniBridge side instead of being copied into validator jobs.',
  },
  {
    label: 'Measured calls',
    value: 'score, time, cost',
    desc: 'Each evaluation can record model responses, latency, token usage, and estimated spend.',
  },
]

export default function MiniBridgeSection() {
  return (
    <section className="section-band">
      <div className="section-shell grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-start">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.5 }}
        >
          <p className="section-kicker">MiniBridge integration</p>
          <h2 className="section-title mt-3">Provider access without exposing miner keys</h2>
          <p className="section-copy mt-5">
            MiniRouter can evaluate router submissions through MiniBridge. The validator
            sends approved model-call requests, MiniBridge applies its caller/provider
            policy, and the selected upstream model returns the answer used by the
            benchmark scorer.
          </p>
          <p className="section-copy mt-4">
            This lets the routing benchmark use real provider-backed LLMs while keeping
            credential ownership separate from the public validator workflow.
          </p>
        </motion.div>

        <motion.div
          className="panel p-6"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.5, delay: 0.12 }}
        >
          <div className="space-y-4">
            {bridgePoints.map((point) => (
              <div key={point.label} className="rounded-xl border border-white/8 bg-white/4 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <span className="meta-label">{point.label}</span>
                  <span className="font-mono text-sm text-accent-light">{point.value}</span>
                </div>
                <p className="mt-3 text-sm leading-6 text-text-dim">{point.desc}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}
