import PageHeader from '../components/PageHeader'
import LeaderboardTable from '../components/LeaderboardTable'

export default function Leaderboard() {
  return (
    <>
      <PageHeader
        title="Leaderboard"
        subtitle="Ranked by macro-average accuracy across the 5-task benchmark suite."
      />
      <section className="pb-24">
        <LeaderboardTable />
      </section>
    </>
  )
}
