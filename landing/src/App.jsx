import HeroSection from './components/HeroSection'
import TripleThreat from './components/TripleThreat'
import InteractiveScore from './components/InteractiveScore'
import Navbar from './components/Navbar'

export default function App() {
  return (
    <div className="min-h-screen bg-obsidian-950 grid-pattern">
      <Navbar />
      <HeroSection />
      <TripleThreat />
      <InteractiveScore />
      <footer className="py-16 text-center border-t border-white/5">
        <p className="text-sm text-white/30 font-mono tracking-wider">
          CERTUSDOC v2.0 &mdash; SECURE AI HACKATHON 2026
        </p>
        <p className="text-xs text-white/15 mt-2">
          IIT Madras &times; BITS Pilani Goa &middot; Blue Team
        </p>
      </footer>
    </div>
  )
}
