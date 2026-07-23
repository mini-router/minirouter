import { motion } from 'framer-motion'

export default function AboutSection() {
  return (
    <section className="section-band">
      <div className="section-shell grid gap-10 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] lg:items-start">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.5 }}
        >
          <p className="section-kicker">Overview</p>
          <h2 className="section-title mt-3">What is MiniRouter?</h2>
          <p className="section-copy mt-5">
            MiniRouter is a compact routing system for choosing which LLM should handle
            a task. It uses a frozen encoder and a small trainable head, then evaluates
            each routing policy against fixed benchmarks with measured score, runtime,
            and provider cost.
          </p>
          <p className="section-copy mt-4">
            The SN74 challenge is the improvement mechanism. Miners submit better router
            heads, but the product goal is broader: make model selection more effective
            without turning every request into a call to the largest available model.
          </p>
        </motion.div>

        <motion.div
          className="grid gap-4 sm:grid-cols-2"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.5, delay: 0.12 }}
        >
          {[
            { label: 'Head budget', value: '~10K parameters' },
            { label: 'Router inputs', value: '1024-dim hidden state' },
            { label: 'Model pool', value: 'Configurable provider models' },
            { label: 'Validation', value: 'Accuracy, cost, runtime' },
          ].map((item) => (
            <div key={item.label} className="stat-card">
              <div className="meta-label">{item.label}</div>
              <div className="mt-3 text-lg font-semibold text-text">{item.value}</div>
            </div>
          ))}
          <div className="sm:col-span-2 rounded-2xl border border-white/8 bg-white/4 p-5">
            <div className="meta-label">Success criterion</div>
            <div className="mt-3 text-base leading-7 text-text-dim">
              Better benchmark accuracy with transparent execution metrics. If two routers
              perform similarly, the smaller and cheaper route is easier to justify in
              production.
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
