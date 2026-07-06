import Hero from '../components/Hero'
import AboutSection from '../components/AboutSection'
import HowItWorks from '../components/HowItWorks'
import PrizesSection from '../components/PrizesSection'
import LeaderboardPreview from '../components/LeaderboardPreview'

export default function Home() {
  return (
    <>
      <Hero />
      <AboutSection />
      <HowItWorks />
      <PrizesSection />
      <LeaderboardPreview />
    </>
  )
}
