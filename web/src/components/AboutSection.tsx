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
            MiniRouter is a compact routing system built around a frozen encoder and a
            small trainable head. The head decides which model to call and which role to
            assign to that model, then the benchmark suite scores the result.
          </p>
          <p className="section-copy mt-4">
            The challenge is narrow by design: the encoder, model pool, evaluation set,
            and reward function stay fixed. Your leverage is the head architecture,
            training setup, and the features you derive from the encoder output.
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
            { label: 'Model pool', value: '3 open-source LLMs' },
            { label: 'Training loop', value: 'Separable CMA-ES' },
          ].map((item) => (
            <div key={item.label} className="stat-card">
              <div className="meta-label">{item.label}</div>
              <div className="mt-3 text-lg font-semibold text-text">{item.value}</div>
            </div>
          ))}
          <div className="sm:col-span-2 rounded-2xl border border-white/8 bg-white/4 p-5">
            <div className="meta-label">Success criterion</div>
            <div className="mt-3 text-base leading-7 text-text-dim">
              Higher macro-average accuracy across the benchmark suite. If two submissions
              tie, the smaller head wins.
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
