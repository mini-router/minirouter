import { motion } from 'framer-motion'

const steps = [
  {
    num: '01',
    title: 'Encode the task',
    desc: 'The fixed encoder converts the incoming benchmark task into a hidden-state representation for the router head.',
  },
  {
    num: '02',
    title: 'Select the route',
    desc: 'The compact head chooses the model slot and role sequence that should solve or verify the task.',
  },
  {
    num: '03',
    title: 'Call through MiniBridge',
    desc: 'Provider calls can be routed through MiniBridge so validator workflows do not need direct miner API keys.',
  },
  {
    num: '04',
    title: 'Measure outcomes',
    desc: 'The evaluator records accuracy, runtime, token usage, and cost so router quality is visible in the leaderboard.',
  },
]

export default function HowItWorks() {
  return (
    <section className="section-band">
      <div className="section-shell">
        <motion.div
          className="max-w-2xl"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.5 }}
        >
          <p className="section-kicker">Process</p>
          <h2 className="section-title mt-3">Performance pipeline</h2>
          <p className="section-copy mt-4">
            MiniRouter is evaluated as a routing system, not as a standalone LLM. A
            router head chooses which external model should act at each turn, then the
            validator scores the final answer and records the operational cost.
          </p>
        </motion.div>

        <div className="mt-10 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {steps.map((step, i) => (
            <motion.div
              key={step.num}
              className="panel p-6"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-80px' }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
            >
              <div className="flex items-center justify-between">
                <span className="rank-badge bg-accent/10 text-accent-light">{step.num}</span>
                <span className="meta-label">Step {i + 1}</span>
              </div>
              <h3 className="mt-6 text-lg font-semibold text-text">{step.title}</h3>
              <p className="mt-3 text-sm leading-6 text-text-dim">{step.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}
