import { motion } from 'framer-motion'

const prizes = [
  {
    place: 'King',
    amount: 'All emissions',
    desc: 'Daily evaluation rewards the strongest measured router. Emissions incentivize practical improvements to accuracy, cost, and runtime.',
    color: 'text-gold',
    border: 'border-gold/30',
  },
]

export default function PrizesSection() {
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
          <p className="section-kicker">Incentive layer</p>
          <h2 className="section-title mt-3">SN74 rewards useful routing</h2>
          <p className="section-copy mt-4">
            The competition layer exists to keep the router improving. Submissions are
            reviewed by the validator, ranked by measured performance, and rewarded when
            they become the best available routing policy.
          </p>
        </motion.div>

        <div className="mt-10 grid gap-4 md:grid-cols-1">
          {prizes.map((p, i) => (
            <motion.div
              key={p.place}
              className={`panel p-6 text-left border ${p.border}`}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-80px' }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
            >
              <div className="flex items-center justify-between">
                <span className={`section-kicker ${p.color}`}>{p.place}</span>
                <span className="meta-label">SN74 emissions</span>
              </div>
              <div className={`mt-5 text-4xl font-semibold ${p.color}`}>{p.amount}</div>
              <p className="mt-4 text-sm leading-6 text-text-dim">{p.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}
