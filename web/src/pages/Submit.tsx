import { motion } from 'framer-motion'
import PageHeader from '../components/PageHeader'

const steps = [
  {
    title: '1. Work in your branch',
    content: (
      <>
        <p className="text-text-dim">
          Create a dedicated branch named{' '}
          <code className="text-accent">sn74-&lt;your-github-username&gt;</code>, for
          example <code className="text-accent">sn74-tmimmanuel</code>.
        </p>
        <pre className="mt-4 overflow-x-auto rounded-2xl border border-white/8 bg-surface-900/80 p-4 text-sm font-mono text-text">
          <code>{`git checkout main
git pull upstream main
git checkout -b sn74-your-github-username`}</code>
        </pre>
      </>
    ),
  },
  {
    title: '2. Add the model bundle',
    content: (
      <>
        <p className="text-text-dim">
          Put the trained model and metadata in `submissions/final_model/`, then run the
          local eval command before opening the PR.
        </p>
        <pre className="mt-4 overflow-x-auto rounded-2xl border border-white/8 bg-surface-900/80 p-4 text-sm font-mono text-text">
          <code>{`mkdir -p submissions/final_model
cp experiments/math500/<run-name>/best_theta.npy submissions/final_model/
cp experiments/math500/<run-name>/summary.json submissions/final_model/
cp experiments/math500/<run-name>/history.json submissions/final_model/  # optional

python -m trinity.eval \\
  --benchmark math500 \\
  --theta submissions/final_model/best_theta.npy \\
  --provider openrouter \\
  --models configs/models.openrouter.yaml \\
  --device cpu \\
  --dtype float32 \\
  --out submissions/final_model/eval.json

submissions/final_model/
├── best_theta.npy
├── summary.json
├── history.json  # optional
└── eval.json     # optional`}</code>
        </pre>
      </>
    ),
  },
  {
    title: '3. Open the PR',
    content: (
      <div className="space-y-4">
        <p className="text-text-dim">
          Push your branch and open a pull request from it. The maintainer bot will
          automatically check the PR and run evaluation.
        </p>
        <div className="grid gap-3 sm:grid-cols-2">
          {[
            'Use the sn74-<github-username> branch prefix',
            'Include best_theta.npy and metadata in the PR',
            'Let the maintainer bot trigger evaluation',
            'Keep the PR title and description clear',
          ].map((item) => (
            <div key={item} className="rounded-2xl border border-white/8 bg-white/4 p-4 text-sm text-text">
              {item}
            </div>
          ))}
        </div>
      </div>
    ),
  },
]

export default function Submit() {
  return (
    <>
      <PageHeader
        title="Submit Your Entry"
        subtitle="Use a sn74-<github-username> branch, include the trained model bundle, and submit it through a pull request."
      />
      <section className="section-band pt-0">
        <div className="section-shell">
          <div className="max-w-4xl space-y-6">
            {steps.map((step, i) => (
              <motion.div
                key={step.title}
                className="panel p-7 md:p-8"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-80px' }}
                transition={{ duration: 0.4, delay: i * 0.05 }}
              >
                <h2 className="text-xl font-semibold text-text">{step.title}</h2>
                <div className="mt-4 space-y-4 leading-7">{step.content}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </>
  )
}
