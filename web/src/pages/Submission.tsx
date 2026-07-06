import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import PageHeader from '../components/PageHeader'
import { fetchSubmission } from '../lib/api'

type SubmissionResponse = Awaited<ReturnType<typeof fetchSubmission>>

function formatDate(value: string | null | undefined) {
  if (!value) return '—'
  return new Date(value).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function formatMetricValue(value: unknown) {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toString() : value.toFixed(4)
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false'
  }
  if (value == null) {
    return '—'
  }
  if (typeof value === 'object') {
    return JSON.stringify(value, null, 2)
  }
  return String(value)
}

function isStructuredMetric(value: unknown) {
  return typeof value === 'object' && value !== null
}

export default function Submission() {
  const { id } = useParams()
  const [submission, setSubmission] = useState<SubmissionResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!id) {
      setLoading(false)
      setError('Missing submission id')
      return
    }

    let active = true
    setLoading(true)
    setError(null)

    fetchSubmission(id)
      .then((data) => {
        if (!active) return
        setSubmission(data)
      })
      .catch((err: unknown) => {
        if (!active) return
        setError(err instanceof Error ? err.message : 'Failed to load submission')
      })
      .finally(() => {
        if (!active) return
        setLoading(false)
      })

    return () => {
      active = false
    }
  }, [id])

  const bestEvaluation = useMemo(() => {
    if (!submission?.evaluations?.length) return null
    return [...submission.evaluations].sort((a, b) => (b.score ?? -1) - (a.score ?? -1))[0]
  }, [submission])

  return (
    <>
      <PageHeader
        title="Submission Report"
        subtitle="A structured view of the uploaded checkpoint, evaluation runs, and scoring output."
        eyebrow="MiniRouter Challenge"
      />

      <section className="section-band pt-0 pb-24">
        <div className="section-shell space-y-6">
          <div className="flex flex-wrap gap-3">
            <Link to="/leaderboard" className="button-secondary">
              Back to leaderboard
            </Link>
          </div>

          {loading ? (
            <div className="panel p-7 text-text-dim">Loading submission report...</div>
          ) : error ? (
            <div className="panel p-7">
              <p className="text-text">Could not load this submission.</p>
              <p className="mt-2 text-sm text-text-dim">{error}</p>
            </div>
          ) : submission ? (
            <>
              <motion.div
                className="panel p-7 md:p-8"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35 }}
              >
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div className="max-w-3xl">
                    <p className="meta-label">Submission {submission.id}</p>
                    <h2 className="mt-2 text-2xl font-semibold text-text">
                      {submission.team_name || submission.repo_full_name || 'Unnamed submission'}
                    </h2>
                    <p className="mt-2 text-sm text-text-dim">
                      Source: {submission.source} · Benchmark: {submission.benchmark} · Status: {submission.status}
                    </p>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2 lg:min-w-[320px]">
                    <div className="rounded-xl border border-white/8 bg-white/4 p-4">
                      <div className="meta-label">Latest score</div>
                      <div className="mt-2 text-2xl font-semibold text-text">
                        {submission.latest_score == null
                          ? 'Pending'
                          : `${(submission.latest_score * 100).toFixed(2)}%`}
                      </div>
                    </div>
                    <div className="rounded-xl border border-white/8 bg-white/4 p-4">
                      <div className="meta-label">Evaluations</div>
                      <div className="mt-2 text-2xl font-semibold text-text">
                        {submission.evaluations.length}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                  {[
                    ['Repository', submission.repo_full_name || '—'],
                    ['PR', submission.pr_number ? `#${submission.pr_number}` : '—'],
                    ['Head SHA', submission.head_sha || '—'],
                    ['Artifact', submission.artifact_name],
                  ].map(([label, value]) => (
                    <div key={label} className="rounded-xl border border-white/8 bg-white/4 p-4">
                      <div className="meta-label">{label}</div>
                      <div className="mt-2 break-words text-sm text-text">{value}</div>
                    </div>
                  ))}
                </div>
              </motion.div>

              {bestEvaluation && (
                <motion.div
                  className="panel p-7 md:p-8"
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35, delay: 0.05 }}
                >
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="meta-label">Best run</p>
                      <h3 className="mt-2 text-xl font-semibold text-text">Top evaluation result</h3>
                    </div>
                    <span className="ui-chip">{bestEvaluation.status}</span>
                  </div>

                  <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                    {[
                      ['Score', bestEvaluation.score == null ? 'Pending' : `${(bestEvaluation.score * 100).toFixed(2)}%`],
                      ['Started', formatDate(bestEvaluation.started_at)],
                      ['Finished', formatDate(bestEvaluation.finished_at)],
                      ['Created', formatDate(bestEvaluation.created_at)],
                    ].map(([label, value]) => (
                      <div key={label} className="rounded-xl border border-white/8 bg-white/4 p-4">
                        <div className="meta-label">{label}</div>
                        <div className="mt-2 text-sm text-text">{value}</div>
                      </div>
                    ))}
                  </div>

                  {Object.keys(bestEvaluation.metrics).length > 0 && (
                    <div className="mt-6">
                      <div className="meta-label">Metrics</div>
                      <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                        {Object.entries(bestEvaluation.metrics).map(([key, value]) => (
                          <div key={key} className="rounded-xl border border-white/8 bg-white/4 p-4">
                            <div className="text-xs uppercase tracking-[0.22em] text-text-dim">
                              {key}
                            </div>
                            {isStructuredMetric(value) ? (
                              <pre className="mt-2 max-h-44 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-5 text-text">
                                {formatMetricValue(value)}
                              </pre>
                            ) : (
                              <div className="mt-2 font-mono text-sm text-text">
                                {formatMetricValue(value)}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {bestEvaluation.error && (
                    <div className="mt-6 rounded-2xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-100">
                      {bestEvaluation.error}
                    </div>
                  )}
                </motion.div>
              )}

              <motion.div
                className="panel p-7 md:p-8"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35, delay: 0.1 }}
              >
                <p className="meta-label">Evaluations</p>
                <h3 className="mt-2 text-xl font-semibold text-text">All recorded runs</h3>

                <div className="mt-6 space-y-4">
                  {submission.evaluations.length === 0 ? (
                    <div className="rounded-xl border border-white/8 bg-white/4 p-4 text-sm text-text-dim">
                      No evaluation runs have been recorded yet.
                    </div>
                  ) : (
                    submission.evaluations.map((evaluation) => (
                      <div key={evaluation.id} className="rounded-2xl border border-white/8 bg-white/4 p-5">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div>
                            <div className="text-sm text-text-dim">Run #{evaluation.id}</div>
                            <div className="mt-1 text-base font-medium text-text">
                              {evaluation.status}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-sm text-text-dim">Score</div>
                            <div className="mt-1 font-mono text-text">
                              {evaluation.score == null ? 'Pending' : `${(evaluation.score * 100).toFixed(2)}%`}
                            </div>
                          </div>
                        </div>

                        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                          {[
                            ['Started', formatDate(evaluation.started_at)],
                            ['Finished', formatDate(evaluation.finished_at)],
                            ['Created', formatDate(evaluation.created_at)],
                          ].map(([label, value]) => (
                            <div key={label} className="rounded-xl border border-white/8 bg-surface-900/70 p-3">
                              <div className="text-xs uppercase tracking-[0.22em] text-text-dim">
                                {label}
                              </div>
                              <div className="mt-2 break-words text-sm text-text">{value}</div>
                            </div>
                          ))}
                        </div>

                        {Object.keys(evaluation.metrics).length > 0 && (
                          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                            {Object.entries(evaluation.metrics).map(([key, value]) => (
                              <div
                                key={key}
                                className="rounded-xl border border-white/8 bg-surface-900/70 p-3"
                              >
                                <div className="text-xs uppercase tracking-[0.22em] text-text-dim">
                                  {key}
                                </div>
                                {isStructuredMetric(value) ? (
                                  <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-5 text-text">
                                    {formatMetricValue(value)}
                                  </pre>
                                ) : (
                                  <div className="mt-2 font-mono text-sm text-text">
                                    {formatMetricValue(value)}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}

                        {evaluation.error && (
                          <div className="mt-4 rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-100">
                            {evaluation.error}
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </motion.div>
            </>
          ) : null}
        </div>
      </section>
    </>
  )
}
