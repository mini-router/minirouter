import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import PageHeader from '../components/PageHeader'
import {
  fetchSubmission,
  type BackendEvaluationOut,
  type BackendSubmissionOut,
  type BackendTrainOut,
} from '../lib/api'

type SubmissionResponse = BackendSubmissionOut

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

function formatMoney(value: number | null | undefined) {
  if (value == null) return '—'
  return `$${value.toFixed(4)}`
}

function formatSeconds(value: number | null | undefined) {
  if (value == null) return '—'
  return `${value.toFixed(2)}s`
}

function formatPercent(value: number | null | undefined) {
  if (value == null) return 'Pending'
  return `${(value * 100).toFixed(2)}%`
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

function RunMetaGrid({
  items,
}: {
  items: Array<[string, string]>
}) {
  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
      {items.map(([label, value]) => (
        <div key={label} className="rounded-xl border border-white/8 bg-white/4 p-4">
          <div className="meta-label">{label}</div>
          <div className="mt-2 break-words text-sm text-text">{value}</div>
        </div>
      ))}
    </div>
  )
}

function MetricsGrid({
  metrics,
}: {
  metrics: Record<string, unknown>
}) {
  const entries = Object.entries(metrics)
  if (entries.length === 0) return null

  return (
    <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {entries.map(([key, value]) => (
        <div key={key} className="rounded-xl border border-white/8 bg-surface-900/70 p-3">
          <div className="text-xs uppercase tracking-[0.22em] text-text-dim">{key}</div>
          {isStructuredMetric(value) ? (
            <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-5 text-text">
              {formatMetricValue(value)}
            </pre>
          ) : (
            <div className="mt-2 font-mono text-sm text-text">{formatMetricValue(value)}</div>
          )}
        </div>
      ))}
    </div>
  )
}

function RunCard({
  title,
  kind,
  id,
  status,
  score,
  startedAt,
  finishedAt,
  createdAt,
  progressCurrent,
  progressTotal,
  costUsd,
  durationSeconds,
  benchmarkNames,
  phase,
  message,
  command,
  error,
  metrics,
  artifactLabel,
  artifactId,
}: {
  title: string
  kind: string
  id: string | number
  status: string
  score?: number | null
  startedAt?: string | null
  finishedAt?: string | null
  createdAt?: string
  progressCurrent?: number | null
  progressTotal?: number | null
  costUsd?: number | null
  durationSeconds?: number | null
  benchmarkNames?: string[]
  phase?: string | null
  message?: string | null
  command?: string | null
  error?: string | null
  metrics: Record<string, unknown>
  artifactLabel: string
  artifactId?: string | null
}) {
  return (
    <div className="rounded-2xl border border-white/8 bg-white/4 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-sm text-text-dim">{title}</div>
          <div className="mt-1 text-base font-medium text-text">
            {kind} #{id}
          </div>
        </div>
        <div className="text-right">
          <div className="text-sm text-text-dim">Status</div>
          <div className="mt-1 font-mono text-text">{status}</div>
        </div>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {[
          ['Score', formatPercent(score ?? null)],
          ['Started', formatDate(startedAt)],
          ['Finished', formatDate(finishedAt)],
          ['Created', formatDate(createdAt)],
          ['Progress', `${progressCurrent ?? 'n/a'}/${progressTotal ?? 'n/a'}`],
          ['Duration', formatSeconds(durationSeconds)],
          ['Cost', formatMoney(costUsd)],
          ['Phase', phase || '—'],
          ['Benchmarks', benchmarkNames?.length ? benchmarkNames.join(', ') : '—'],
          ['Artifact', artifactId || '—'],
          ['Artifact kind', artifactLabel],
          ['Message', message || '—'],
        ].map(([label, value]) => (
          <div key={label as string} className="rounded-xl border border-white/8 bg-surface-900/70 p-3">
            <div className="text-xs uppercase tracking-[0.22em] text-text-dim">{label}</div>
            <div className="mt-2 break-words text-sm text-text">{value as string}</div>
          </div>
        ))}
      </div>

      {command && (
        <div className="mt-4 rounded-xl border border-white/8 bg-surface-900/70 p-4">
          <div className="text-xs uppercase tracking-[0.22em] text-text-dim">Command</div>
          <pre className="mt-2 overflow-x-auto whitespace-pre-wrap break-words font-mono text-xs leading-5 text-text">
            {command}
          </pre>
        </div>
      )}

      {error && (
        <div className="mt-4 rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-100">
          {error}
        </div>
      )}

      <MetricsGrid metrics={metrics} />
    </div>
  )
}

function pickRunById<T extends { id: number; created_at: string }>(runs: T[], id: number | null) {
  if (id == null) return null
  return runs.find((run) => run.id === id) ?? null
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

  const latestTrain = useMemo<BackendTrainOut | null>(() => {
    if (!submission?.trains?.length) return null
    return pickRunById(submission.trains, submission.latest_train_id) ?? submission.trains.at(-1) ?? null
  }, [submission])

  const bestEvaluation = useMemo<BackendEvaluationOut | null>(() => {
    if (!submission?.evaluations?.length) return null
    return (
      pickRunById(submission.evaluations, submission.best_eval_id) ??
      [...submission.evaluations].sort((a, b) => (b.score ?? -1) - (a.score ?? -1))[0]
    )
  }, [submission])

  return (
    <>
      <PageHeader
        title="Submission Report"
        subtitle="A structured view of the uploaded checkpoint, train runs, evaluation runs, and stored metrics."
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
                      {submission.team_name || submission.miner_id || submission.repo_full_name || 'Unnamed submission'}
                    </h2>
                    <p className="mt-2 text-sm text-text-dim">
                      Source: {submission.source} · Benchmarks:{' '}
                      {submission.benchmarks.length > 0 ? submission.benchmarks.join(', ') : submission.benchmark}{' '}
                      · Status: {submission.status}
                    </p>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2 lg:min-w-[360px]">
                    <div className="rounded-xl border border-white/8 bg-white/4 p-4">
                      <div className="meta-label">Latest score</div>
                      <div className="mt-2 text-2xl font-semibold text-text">
                        {submission.latest_score == null
                          ? 'Pending'
                          : `${(submission.latest_score * 100).toFixed(2)}%`}
                      </div>
                    </div>
                    <div className="rounded-xl border border-white/8 bg-white/4 p-4">
                      <div className="meta-label">Runs</div>
                      <div className="mt-2 text-2xl font-semibold text-text">
                        {submission.evaluations.length} eval / {submission.trains.length} train
                      </div>
                    </div>
                    <div className="rounded-xl border border-white/8 bg-white/4 p-4">
                      <div className="meta-label">Total duration</div>
                      <div className="mt-2 text-2xl font-semibold text-text">
                        {formatSeconds(submission.duration_seconds)}
                      </div>
                    </div>
                    <div className="rounded-xl border border-white/8 bg-white/4 p-4">
                      <div className="meta-label">Total cost</div>
                      <div className="mt-2 text-2xl font-semibold text-text">
                        {formatMoney(submission.cost_usd)}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                  {[
                    ['Repository', submission.repo_full_name || '—'],
                    ['PR', submission.pr_number ? `#${submission.pr_number}` : '—'],
                    ['Head SHA', submission.head_sha || '—'],
                    ['Submission artifact', submission.submission_artifact_id || '—'],
                    ['Latest train id', submission.latest_train_id == null ? '—' : `#${submission.latest_train_id}`],
                    ['Latest eval id', submission.latest_eval_id == null ? '—' : `#${submission.latest_eval_id}`],
                    ['Best eval id', submission.best_eval_id == null ? '—' : `#${submission.best_eval_id}`],
                    ['Current phase', submission.current_phase || '—'],
                    ['Current message', submission.current_message || '—'],
                    [
                      'Current progress',
                      `${submission.current_progress_current ?? 'n/a'}/${submission.current_progress_total ?? 'n/a'}`,
                    ],
                    ['Finished', formatDate(submission.finished_at)],
                    ['Updated', formatDate(submission.updated_at)],
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

                  <RunMetaGrid
                    items={[
                      ['Score', formatPercent(bestEvaluation.score)],
                      ['Started', formatDate(bestEvaluation.started_at)],
                      ['Finished', formatDate(bestEvaluation.finished_at)],
                      ['Created', formatDate(bestEvaluation.created_at)],
                      ['Benchmark', bestEvaluation.benchmark_names.length ? bestEvaluation.benchmark_names.join(', ') : '—'],
                      ['Provider', bestEvaluation.provider || '—'],
                      ['Models config', bestEvaluation.models_config || '—'],
                      ['Execution mode', bestEvaluation.execution_mode || '—'],
                      ['Device / dtype', `${bestEvaluation.device || '—'} / ${bestEvaluation.dtype || '—'}`],
                      ['Batch size', bestEvaluation.batch_size == null ? '—' : bestEvaluation.batch_size.toString()],
                      ['Max items', bestEvaluation.max_items == null ? '—' : bestEvaluation.max_items.toString()],
                      ['Cost', formatMoney(bestEvaluation.cost_usd)],
                    ]}
                  />

                  {bestEvaluation.error && (
                    <div className="mt-6 rounded-2xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-100">
                      {bestEvaluation.error}
                    </div>
                  )}

                  <MetricsGrid metrics={bestEvaluation.metrics} />
                </motion.div>
              )}

              {latestTrain && (
                <motion.div
                  className="panel p-7 md:p-8"
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35, delay: 0.08 }}
                >
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="meta-label">Latest train</p>
                      <h3 className="mt-2 text-xl font-semibold text-text">Training run details</h3>
                    </div>
                    <span className="ui-chip">{latestTrain.status}</span>
                  </div>

                  <RunMetaGrid
                    items={[
                      ['Train id', `#${latestTrain.id}`],
                      ['Started', formatDate(latestTrain.started_at)],
                      ['Finished', formatDate(latestTrain.finished_at)],
                      ['Created', formatDate(latestTrain.created_at)],
                      ['Benchmark', latestTrain.benchmark_names.length ? latestTrain.benchmark_names.join(', ') : '—'],
                      ['Warmstart artifact', latestTrain.warmstart_artifact_id || '—'],
                      ['Output artifact', latestTrain.output_artifact_id || '—'],
                      ['Cost', formatMoney(latestTrain.cost_usd)],
                      ['Duration', formatSeconds(latestTrain.duration_seconds)],
                      ['Progress', `${latestTrain.progress_current ?? 'n/a'}/${latestTrain.progress_total ?? 'n/a'}`],
                      ['Phase', latestTrain.phase || '—'],
                      ['Message', latestTrain.message || '—'],
                    ]}
                  />

                  {latestTrain.error && (
                    <div className="mt-6 rounded-2xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-100">
                      {latestTrain.error}
                    </div>
                  )}

                  <MetricsGrid metrics={latestTrain.metrics} />
                </motion.div>
              )}

              <motion.div
                className="panel p-7 md:p-8"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35, delay: 0.1 }}
              >
                <p className="meta-label">Evaluation history</p>
                <h3 className="mt-2 text-xl font-semibold text-text">All recorded evaluation runs</h3>

                <div className="mt-6 space-y-4">
                  {submission.evaluations.length === 0 ? (
                    <div className="rounded-xl border border-white/8 bg-white/4 p-4 text-sm text-text-dim">
                      No evaluation runs have been recorded yet.
                    </div>
                  ) : (
                    submission.evaluations.map((evaluation) => (
                      <RunCard
                        key={evaluation.id}
                        title="Evaluation"
                        kind="Run"
                        id={evaluation.id}
                        status={evaluation.status}
                        score={evaluation.score}
                        startedAt={evaluation.started_at}
                        finishedAt={evaluation.finished_at}
                        createdAt={evaluation.created_at}
                        progressCurrent={evaluation.progress_current}
                        progressTotal={evaluation.progress_total}
                        costUsd={evaluation.cost_usd}
                        durationSeconds={evaluation.duration_seconds}
                        benchmarkNames={evaluation.benchmark_names}
                        phase={evaluation.phase}
                        message={evaluation.message}
                        command={evaluation.command}
                        error={evaluation.error}
                        metrics={evaluation.metrics}
                        artifactLabel="results"
                        artifactId={evaluation.results_artifact_id}
                      />
                    ))
                  )}
                </div>
              </motion.div>

              <motion.div
                className="panel p-7 md:p-8"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35, delay: 0.14 }}
              >
                <p className="meta-label">Training history</p>
                <h3 className="mt-2 text-xl font-semibold text-text">All recorded train runs</h3>

                <div className="mt-6 space-y-4">
                  {submission.trains.length === 0 ? (
                    <div className="rounded-xl border border-white/8 bg-white/4 p-4 text-sm text-text-dim">
                      No train runs have been recorded yet.
                    </div>
                  ) : (
                    submission.trains.map((train) => (
                      <RunCard
                        key={train.id}
                        title="Train"
                        kind="Run"
                        id={train.id}
                        status={train.status}
                        startedAt={train.started_at}
                        finishedAt={train.finished_at}
                        createdAt={train.created_at}
                        progressCurrent={train.progress_current}
                        progressTotal={train.progress_total}
                        costUsd={train.cost_usd}
                        durationSeconds={train.duration_seconds}
                        benchmarkNames={train.benchmark_names}
                        phase={train.phase}
                        message={train.message}
                        command={train.command}
                        error={train.error}
                        metrics={train.metrics}
                        artifactLabel="training bundle"
                        artifactId={train.output_artifact_id}
                      />
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
