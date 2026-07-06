import { motion } from 'framer-motion'

const steps = [
  {
    num: '01',
    title: 'Prepare the baseline',
    desc: 'Clone the repository, install dependencies, and reproduce the reference score before making changes.',
  },
  {
    num: '02',
    title: 'Change the head',
    desc: 'Tune width, depth, activation, normalization, or derived features to improve routing quality.',
  },
  {
    num: '03',
    title: 'Validate locally',
    desc: 'Run the evaluation harness with a fixed seed so you can compare runs consistently.',
  },
  {
    num: '04',
    title: 'Submit the package',
    desc: 'Package weights, config, results, and report for organizer review and leaderboard refresh.',
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
          <h2 className="section-title mt-3">How it works</h2>
          <p className="section-copy mt-4">
            The competition is a closed-loop optimization problem. You change the head,
            test it against the fixed benchmark, and keep only the changes that move the
            score.
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
