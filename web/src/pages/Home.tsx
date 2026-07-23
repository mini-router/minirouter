import Hero from '../components/Hero'
import AboutSection from '../components/AboutSection'
import HowItWorks from '../components/HowItWorks'
import MiniBridgeSection from '../components/MiniBridgeSection'
import PrizesSection from '../components/PrizesSection'
import LeaderboardPreview from '../components/LeaderboardPreview'

export default function Home() {
  return (
    <>
      <Hero />
      <AboutSection />
      <HowItWorks />
      <MiniBridgeSection />
      <PrizesSection />
      <LeaderboardPreview />
    </>
  )
}
