import { motion } from 'framer-motion'

export default function Navbar() {
  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className="fixed top-0 inset-x-0 z-50"
    >
      <div className="mx-auto max-w-7xl px-6 py-4">
        <div className="glass-strong rounded-2xl px-6 py-3 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-cyan to-accent-blue flex items-center justify-center">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 1L14 4.5V11.5L8 15L2 11.5V4.5L8 1Z" stroke="white" strokeWidth="1.5" fill="none" />
                <path d="M8 5L11 6.75V10.25L8 12L5 10.25V6.75L8 5Z" fill="white" fillOpacity="0.9" />
              </svg>
            </div>
            <span className="text-sm font-semibold tracking-wide text-white/90">
              CertusDoc
            </span>
          </div>

          {/* Nav links */}
          <div className="hidden md:flex items-center gap-8 text-sm text-white/40">
            <a href="#agents" className="hover:text-white/80 transition-colors">Agents</a>
            <a href="#score" className="hover:text-white/80 transition-colors">Live Score</a>
            <a href="#" className="hover:text-white/80 transition-colors">Docs</a>
          </div>

          {/* CTA */}
          <button className="text-xs font-medium px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white/70 hover:bg-white/10 hover:text-white transition-all">
            Try Demo
          </button>
        </div>
      </div>
    </motion.nav>
  )
}
