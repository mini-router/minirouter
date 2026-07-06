import { motion } from 'framer-motion'

const prizes = [
  {
    place: 'King',
    amount: 'All emissions',
    desc: 'Daily evaluation chooses the king. The king receives all SN74 MiniRouter emissions.',
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
          <p className="section-kicker">Awards</p>
          <h2 className="section-title mt-3">Prizes</h2>
          <p className="section-copy mt-4">
            The leaderboard is ranked by macro-average accuracy. Daily evaluation chooses
            the king, and the king receives all SN74 MiniRouter emissions.
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
                <span className="meta-label">Cash prize</span>
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
