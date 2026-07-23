import PageHeader from '../components/PageHeader'
import LeaderboardTable from '../components/LeaderboardTable'
import ProviderBenchmarkSection from '../components/ProviderBenchmarkSection'

export default function Leaderboard() {
  return (
    <>
      <PageHeader
        title="Leaderboard"
        subtitle="Ranked by macro-average accuracy. Open any report to see a structured submission page."
      />
      <section className="pb-24">
        <LeaderboardTable />
      </section>
      <ProviderBenchmarkSection />
    </>
  )
}
