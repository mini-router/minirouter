import { motion } from 'framer-motion'
import PageHeader from '../components/PageHeader'

const sections = [
  {
    title: '1. Eligibility',
    body: 'Open to everyone. Organizers are not eligible for prizes.',
  },
  {
    title: '2. The Task',
    body: (
      <>
        <p>
          Optimize the <strong>routing head</strong> of MiniRouter — the ~10K-parameter
          MLP that maps the encoder's 1024-dim hidden state to a routing decision (model
          slot × role). The head is trained with separable CMA-ES against a binary reward
          (correct/incorrect).
        </p>
        <p className="font-semibold text-text mt-4">You may modify:</p>
        <ul className="list-disc space-y-2 pl-5">
          <li>Head architecture (width, depth, activation, normalization)</li>
          <li>Training hyperparameters (population size, sigma, elite ratio, budget)</li>
          <li>Input features derived from the encoder hidden state</li>
        </ul>
        <p className="font-semibold text-text mt-4">You may not modify:</p>
        <ul className="list-disc space-y-2 pl-5">
          <li>The 0.6B encoder (frozen)</li>
          <li>The three LLM pool models or their API</li>
          <li>The 120-question held-out evaluation set</li>
          <li>The reward function (binary correct/incorrect)</li>
        </ul>
      </>
    ),
  },
  {
    title: '3. Submission Requirements',
    body: (
      <ul className="list-disc space-y-2 pl-5">
        <li>
          <strong>Head weights</strong> in the standard{' '}
          <code className="text-accent">best_theta.npy</code> format
        </li>
        <li>
          <strong>Config file</strong> describing architecture and training setup
        </li>
        <li>
          <strong>Results JSON</strong> with per-task accuracy scores
        </li>
        <li>
          <strong>Brief report</strong> (max 2 pages) describing approach
        </li>
      </ul>
    ),
  },
  {
    title: '4. Evaluation',
    body: 'All submissions are re-evaluated by the organizers on a fixed GPU with a fixed random seed. The primary metric is macro-average accuracy across 2 benchmarks: math and mmlu.',
  },
  {
    title: '5. Winners',
    body: 'The king is determined by the highest macro-average accuracy. In case of a tie, the submission with fewer head parameters wins. Daily evaluations choose the king, and the king receives all SN74 MiniRouter emissions.',
  },
]

export default function Rules() {
  return (
    <>
      <PageHeader title="Competition Rules" />
      <section className="section-band pt-0">
        <div className="section-shell">
          <div className="max-w-4xl space-y-6">
            {sections.map((s, i) => (
              <motion.div
                key={s.title}
                className="panel p-7 md:p-8"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-80px' }}
                transition={{ duration: 0.4, delay: i * 0.05 }}
              >
                <h2 className="text-xl font-semibold text-text">{s.title}</h2>
                <div className="mt-4 space-y-3 text-text-dim leading-7">
                  {s.body}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </>
  )
}
