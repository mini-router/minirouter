import type { ProviderBenchmarkEntry } from '../lib/api'

// Grouped bar chart, faceted by benchmark. Hand-rolled SVG, no charting
// library. Mirrors the admin dashboard's chart 1:1 (same palette, same
// layout) so the two surfaces show the same picture -- see
// mini-router.github.io admin/src/components/ProviderBenchmarkChart.tsx.

// Validated dark-mode categorical palette (8 slots, fixed order -- never
// reassigned by rank/filter). Passes CVD + contrast checks against this
// site's dark surface (#060b14).
const ROUTE_PALETTE = [
  '#3987e5', // blue
  '#d95926', // orange
  '#199e70', // aqua
  '#c98500', // yellow
  '#d55181', // magenta
  '#008300', // green
  '#9085e9', // violet
  '#e66767', // red
]

function buildRouteColors(points: ProviderBenchmarkEntry[]): Map<string, string> {
  const routes = Array.from(new Set(points.map((p) => p.route))).sort()
  const map = new Map<string, string>()
  routes.forEach((route, idx) => {
    map.set(route, ROUTE_PALETTE[idx % ROUTE_PALETTE.length])
  })
  return map
}

function fmtPercent(value: number | null): string {
  return value == null || Number.isNaN(value) ? '—' : `${(value * 100).toFixed(1)}%`
}

const FACET_WIDTH = 420
const FACET_HEIGHT = 220
const PLOT_TOP = 16
const PLOT_BOTTOM = 36
const BAR_GAP = 6

export default function ProviderBenchmarkChart({ points }: { points: ProviderBenchmarkEntry[] }) {
  const scored = points.filter((p) => p.score != null)
  if (!scored.length) {
    return (
      <div className="panel-soft rounded-xl border border-dashed border-white/10 px-5 py-6 text-sm text-text-dim">
        No provider benchmark results yet.
      </div>
    )
  }

  const routeColors = buildRouteColors(scored)
  const benchmarks = Array.from(new Set(scored.map((p) => p.benchmark))).sort()

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        {Array.from(routeColors.entries()).map(([route, color]) => (
          <div key={route} className="flex items-center gap-1.5 text-xs text-text-dim">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
            {route}
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-4">
        {benchmarks.map((benchmark) => {
          const rows = scored
            .filter((p) => p.benchmark === benchmark)
            .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
          const plotHeight = FACET_HEIGHT - PLOT_TOP - PLOT_BOTTOM
          const barWidth = Math.max(28, (FACET_WIDTH - BAR_GAP * (rows.length - 1)) / rows.length - 4)

          return (
            <div key={benchmark} className="panel rounded-xl p-4">
              <div className="mb-2 text-xs font-medium uppercase tracking-[0.18em] text-text-dim">
                {benchmark}
              </div>
              <svg
                viewBox={`0 0 ${FACET_WIDTH} ${FACET_HEIGHT}`}
                width={FACET_WIDTH}
                height={FACET_HEIGHT}
                role="img"
                aria-label={`Score by model for ${benchmark}`}
              >
                {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
                  const y = PLOT_TOP + plotHeight * (1 - frac)
                  return (
                    <line
                      key={frac}
                      x1={0}
                      x2={FACET_WIDTH}
                      y1={y}
                      y2={y}
                      stroke="rgba(255,255,255,0.08)"
                      strokeWidth={1}
                    />
                  )
                })}
                {rows.map((row, idx) => {
                  const score = row.score ?? 0
                  const barHeight = plotHeight * Math.max(0, Math.min(1, score))
                  const x = idx * (barWidth + BAR_GAP)
                  const y = PLOT_TOP + (plotHeight - barHeight)
                  const color = routeColors.get(row.route) ?? ROUTE_PALETTE[0]
                  return (
                    <g key={`${row.id}-${row.route}`}>
                      <title>
                        {row.route} — {benchmark}: {fmtPercent(row.score)}
                      </title>
                      <rect x={x} y={y} width={barWidth} height={Math.max(1, barHeight)} rx={4} fill={color} />
                      <text
                        x={x + barWidth / 2}
                        y={y - 6}
                        textAnchor="middle"
                        fontSize={11}
                        fill="#e8eaf0"
                      >
                        {fmtPercent(row.score)}
                      </text>
                      <text
                        x={x + barWidth / 2}
                        y={FACET_HEIGHT - PLOT_BOTTOM + 16}
                        textAnchor="middle"
                        fontSize={9}
                        fill="#98a2b3"
                      >
                        {row.route.length > 12 ? `${row.route.slice(0, 11)}…` : row.route}
                      </text>
                    </g>
                  )
                })}
                <line
                  x1={0}
                  x2={FACET_WIDTH}
                  y1={PLOT_TOP + plotHeight}
                  y2={PLOT_TOP + plotHeight}
                  stroke="rgba(255,255,255,0.16)"
                  strokeWidth={1}
                />
              </svg>
            </div>
          )
        })}
      </div>
    </div>
  )
}
