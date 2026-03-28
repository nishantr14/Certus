import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import {
  motion,
  useMotionValue,
  useSpring,
  useTransform,
  animate,
  AnimatePresence,
} from 'framer-motion'

/* ============================================================
   DATA
   ============================================================ */

const AGENTS = [
  {
    id: 'visual',
    num: '01',
    title: 'Visual Tamper',
    model: 'EfficientNet-B4 + ELA',
    finding: 'Splicing boundary detected around name field',
    score: 0.08,
    risk: 92,
  },
  {
    id: 'text',
    num: '02',
    title: 'Text Forensics',
    model: 'Tesseract OCR + Font CNN',
    finding: 'Font mismatch: Arial 11pt vs Noto Sans 10.5pt',
    score: 0.15,
    risk: 85,
  },
  {
    id: 'metadata',
    num: '03',
    title: 'Metadata',
    model: 'Isolation Forest',
    finding: 'PDF created with Adobe Photoshop CC 2024',
    score: 0.10,
    risk: 90,
  },
]

const PIPELINE = ['Ingestion', 'Detection', 'Fusion', 'Report']

const SESSION_HASH = '7f3a9c2e...d891b4a7'

/* ============================================================
   SECURITY SPHERE  (1.5x, laser, proximity glow)
   ============================================================ */

function SecuritySphere({ containerRef, isScanning, scanDone, onSpherePos }) {
  const mx = useMotionValue(0)
  const my = useMotionValue(0)
  const sx = useSpring(mx, { stiffness: 100, damping: 20 })
  const sy = useSpring(my, { stiffness: 100, damping: 20 })
  const rotY = useTransform(sx, [-400, 400], [-16, 16])
  const rotX = useTransform(sy, [-300, 300], [12, -12])

  // Report sphere position for proximity glow
  useEffect(() => {
    const unsub = sx.on('change', (x) => {
      onSpherePos?.({ x, y: sy.get() })
    })
    return unsub
  }, [sx, sy, onSpherePos])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onMove = (e) => {
      const r = el.getBoundingClientRect()
      mx.set(e.clientX - r.left - r.width / 2)
      my.set(e.clientY - r.top - r.height / 2)
    }
    el.addEventListener('mousemove', onMove)
    return () => el.removeEventListener('mousemove', onMove)
  }, [containerRef, mx, my])

  // When scanning triggers, animate sphere toward "forgery coordinates"
  useEffect(() => {
    if (!isScanning || scanDone) return
    const ctrl1 = animate(mx, 80, { duration: 1.5, ease: [0.22, 1, 0.36, 1] })
    const ctrl2 = animate(my, -30, { duration: 1.5, ease: [0.22, 1, 0.36, 1] })
    return () => { ctrl1.stop(); ctrl2.stop() }
  }, [isScanning, scanDone, mx, my])

  return (
    <motion.div
      className={`relative w-80 h-80 md:w-96 md:h-96 ${isScanning && !scanDone ? 'laser-active' : ''}`}
      style={{
        x: useTransform(sx, (v) => v * 0.07),
        y: useTransform(sy, (v) => v * 0.07),
        rotateX: rotX,
        rotateY: rotY,
        perspective: 1000,
      }}
    >
      {/* Ambient glow — gets brighter during scan */}
      <motion.div
        className="absolute -inset-20 rounded-full sphere-breathe"
        animate={{
          opacity: isScanning ? 1.4 : 1,
          scale: isScanning ? 1.1 : 1,
        }}
        transition={{ duration: 1.5 }}
      />

      {/* Outer dashed orbit */}
      <motion.div
        className="absolute inset-0"
        animate={{ rotate: 360 }}
        transition={{ duration: 55, repeat: Infinity, ease: 'linear' }}
      >
        <svg viewBox="0 0 200 200" className="w-full h-full">
          <circle cx="100" cy="100" r="96" fill="none" stroke="rgba(224,62,74,0.06)" strokeWidth="0.4" strokeDasharray="2 10" />
          <circle cx="100" cy="100" r="88" fill="none" stroke="rgba(224,62,74,0.08)" strokeWidth="0.5" strokeDasharray="4 8" />
        </svg>
      </motion.div>

      {/* Hex ring 1 */}
      <motion.div
        className="absolute inset-6"
        animate={{ rotate: -360 }}
        transition={{ duration: 42, repeat: Infinity, ease: 'linear' }}
      >
        <svg viewBox="0 0 200 200" className="w-full h-full">
          <polygon
            points="100,14 174,57 174,143 100,186 26,143 26,57"
            fill="none"
            stroke="rgba(224,62,74,0.09)"
            strokeWidth="0.6"
          />
        </svg>
      </motion.div>

      {/* Hex ring 2 */}
      <motion.div
        className="absolute inset-14"
        animate={{ rotate: 360 }}
        transition={{ duration: 65, repeat: Infinity, ease: 'linear' }}
      >
        <svg viewBox="0 0 200 200" className="w-full h-full">
          <polygon
            points="100,26 164,62 164,138 100,174 36,138 36,62"
            fill="rgba(224,62,74,0.02)"
            stroke="rgba(255,107,107,0.1)"
            strokeWidth="0.7"
          />
        </svg>
      </motion.div>

      {/* Innermost hex */}
      <motion.div
        className="absolute inset-[88px]"
        animate={{ rotate: -360 }}
        transition={{ duration: 80, repeat: Infinity, ease: 'linear' }}
      >
        <svg viewBox="0 0 100 100" className="w-full h-full">
          <polygon
            points="50,8 88,30 88,70 50,92 12,70 12,30"
            fill="rgba(224,62,74,0.03)"
            stroke="rgba(255,179,177,0.1)"
            strokeWidth="0.5"
          />
        </svg>
      </motion.div>

      {/* Center shield */}
      <motion.div
        className="absolute inset-0 flex items-center justify-center z-10"
        animate={{ scale: [1, 1.04, 1] }}
        transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
      >
        <div
          className="w-16 h-16 rounded-lg glass flex items-center justify-center"
          style={{ boxShadow: '0 0 30px rgba(224,62,74,0.12)' }}
        >
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" className="text-crimson-light">
            <path
              d="M12 2L3 7v5c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z"
              stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round"
            />
            <path d="M8.5 12.5l2.5 2.5 4.5-4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </motion.div>

      {/* Tick marks around perimeter */}
      {Array.from({ length: 24 }).map((_, i) => {
        const ang = (i / 24) * 360
        const major = i % 6 === 0
        return (
          <motion.div
            key={i}
            className="absolute"
            style={{
              width: 1,
              height: major ? 10 : 5,
              background: `rgba(224,62,74,${major ? 0.18 : 0.08})`,
              left: '50%',
              top: '50%',
              transformOrigin: 'center center',
              transform: `translate(-50%, -50%) rotate(${ang}deg) translateY(-${major ? 175 : 178}px)`,
            }}
            animate={{ opacity: [0.3, 0.7, 0.3] }}
            transition={{ duration: 3, delay: i * 0.12, repeat: Infinity }}
          />
        )
      })}
    </motion.div>
  )
}

/* ============================================================
   SCORE + LAB METADATA
   ============================================================ */

function ScoreDisplay({ from, to, running, scanDone }) {
  const [val, setVal] = useState(from)

  useEffect(() => {
    if (!running) { setVal(from); return }
    const ctrl = animate(from, to, {
      duration: 3,
      ease: [0.22, 1, 0.36, 1],
      onUpdate: (v) => setVal(v),
    })
    return () => ctrl.stop()
  }, [running, from, to])

  const risky = val < 0.4

  return (
    <div className="text-center space-y-1">
      {/* Micro-label above */}
      <p className="font-mono text-[9px] text-neutral-tertiary uppercase tracking-[0.2em]">
        Document Integrity Score
      </p>

      {/* THE SCORE — largest element */}
      <div className={`font-mono text-8xl md:text-[120px] font-extrabold tracking-tighter tabular-nums leading-none transition-colors duration-700 ${risky ? 'text-crimson' : 'text-neutral-text'}`}>
        {val.toFixed(2)}
      </div>

      {/* Lab metadata surrounding it */}
      <div className="flex items-center justify-center gap-6 pt-2">
        <span className="font-mono text-[9px] text-neutral-tertiary">
          DIS = &Sigma;(R<sub className="text-[7px]">i</sub>&times;S<sub className="text-[7px]">i</sub>) / &Sigma;R<sub className="text-[7px]">i</sub>
        </span>
        <span className="w-px h-3 bg-neutral-faint" />
        <span className="font-mono text-[9px] text-neutral-tertiary">
          {scanDone ? '3/3 AGENTS' : 'AWAITING SCAN'}
        </span>
        <span className="w-px h-3 bg-neutral-faint" />
        <span className="font-mono text-[9px] text-neutral-tertiary">
          {scanDone ? '1.2s ELAPSED' : '--'}
        </span>
      </div>

      {/* Risk badge */}
      <AnimatePresence>
        {risky && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="flex items-center justify-center gap-2 pt-2"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-crimson animate-pulse" />
            <span className="font-mono text-[10px] uppercase tracking-[0.25em] text-crimson font-semibold">
              High Risk
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ============================================================
   SIDEBAR (64px, Newsreader numbers)
   ============================================================ */

function Sidebar({ activeAgent, onSelect, sphereProximity }) {
  const items = [
    { id: 'overview', num: '00', label: 'SYS' },
    ...AGENTS.map((a) => ({ id: a.id, num: a.num, label: a.title.split(' ')[0].toUpperCase().slice(0, 3) })),
  ]

  // Proximity glow: sphere near left edge → sidebar gets subtle crimson tint
  const glowOpacity = Math.max(0, Math.min(0.05, (200 + sphereProximity.x * -1) / 4000))

  return (
    <aside
      className="fixed left-0 top-0 bottom-0 w-16 z-40 flex flex-col items-center pt-14 pb-4 border-r transition-all duration-300"
      style={{
        borderColor: `rgba(255,255,255,${0.04 + glowOpacity * 2})`,
        background: `linear-gradient(180deg, rgba(224,62,74,${glowOpacity}) 0%, rgba(6,6,8,1) 40%)`,
      }}
    >
      {/* Logo mark */}
      <div className="mb-8 mt-2">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-crimson/40">
          <path d="M12 2L3 7v5c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z"
            stroke="currentColor" strokeWidth="1.5" />
        </svg>
      </div>

      {/* Nav items — serif numbers */}
      <div className="flex-1 flex flex-col items-center gap-1">
        {items.map((item) => {
          const active = activeAgent === item.id
          return (
            <button
              key={item.id}
              onClick={() => onSelect(item.id)}
              className={`
                w-10 h-10 flex flex-col items-center justify-center rounded-sm transition-all duration-300
                ${active
                  ? 'bg-crimson/8 border border-crimson/15'
                  : 'hover:bg-white/[0.02] border border-transparent'
                }
              `}
            >
              <span className={`font-serif text-sm italic transition-colors ${active ? 'text-crimson-light' : 'text-neutral-faint'}`}>
                {item.num}
              </span>
              <span className={`font-mono text-[6px] uppercase tracking-wider transition-colors ${active ? 'text-neutral-tertiary' : 'text-neutral-faint/60'}`}>
                {item.label}
              </span>
            </button>
          )
        })}
      </div>

      {/* Bottom dot */}
      <div className="w-1.5 h-1.5 rounded-full bg-neutral-faint" />
    </aside>
  )
}

/* ============================================================
   GLOBAL STATUS BAR
   ============================================================ */

function StatusBar({ scanning, scanDone, sphereProximity }) {
  const glowOpacity = Math.max(0, Math.min(0.04, (200 + sphereProximity.y * -1) / 5000))

  return (
    <div
      className="fixed top-0 left-16 right-0 z-50 h-10 flex items-center justify-between px-6 border-b transition-all duration-300"
      style={{
        borderColor: `rgba(255,255,255,${0.04 + glowOpacity * 2})`,
        background: `linear-gradient(90deg, rgba(6,6,8,0.95) 0%, rgba(6,6,8,${0.95 - glowOpacity}) 50%, rgba(224,62,74,${glowOpacity}) 100%)`,
        backdropFilter: 'blur(12px)',
      }}
    >
      <div className="flex items-center gap-4">
        <span className="font-sans text-xs font-semibold text-neutral-text tracking-wide">
          CertusDoc
        </span>
        <span className="w-px h-3 bg-neutral-faint" />
        <div className="flex items-center gap-2">
          <span className={`w-1.5 h-1.5 rounded-full transition-colors duration-500 ${
            scanDone ? 'bg-crimson animate-pulse' : scanning ? 'bg-yellow-500 animate-pulse' : 'bg-green-600'
          }`} />
          <span className="font-mono text-[9px] text-neutral-tertiary uppercase tracking-[0.15em]">
            {scanDone ? 'THREAT_DETECTED' : scanning ? 'SCANNING' : 'SYSTEM_READY'}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <span className="font-mono text-[9px] text-neutral-faint">
          SHA-256: {SESSION_HASH}
        </span>
        <span className="w-px h-3 bg-neutral-faint" />
        <span className="font-mono text-[9px] text-neutral-tertiary uppercase tracking-wider">
          v2.0
        </span>
      </div>
    </div>
  )
}

/* ============================================================
   AGENT CARD
   ============================================================ */

function AgentCard({ agent, active, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: [0.22, 1, 0.36, 1] }}
      className="glass rounded-sm p-5 flex flex-col justify-between"
    >
      <div>
        {/* Number + Title */}
        <div className="flex items-baseline gap-3 mb-1">
          <span className="font-serif text-2xl italic text-neutral-faint">{agent.num}</span>
          <h3 className="text-sm font-semibold text-neutral-text">{agent.title}</h3>
        </div>
        <p className="font-mono text-[8px] text-neutral-faint uppercase tracking-[0.15em] mb-4">
          {agent.model}
        </p>

        {/* Finding */}
        <div className="flex items-start gap-2 mb-5">
          <span className={`mt-1 w-1.5 h-1.5 rounded-full shrink-0 transition-colors duration-700 ${active ? 'bg-crimson' : 'bg-neutral-faint'}`} />
          <p className={`text-xs leading-relaxed transition-colors duration-700 ${active ? 'text-neutral-secondary' : 'text-neutral-tertiary'}`}>
            {agent.finding}
          </p>
        </div>
      </div>

      {/* Risk bar */}
      <div>
        <div className="flex justify-between items-center mb-1.5">
          <span className={`font-mono text-[8px] uppercase tracking-[0.2em] font-semibold transition-colors duration-700 ${active ? 'text-crimson' : 'text-neutral-faint'}`}>
            {active ? 'High Risk' : 'Standby'}
          </span>
          <span className={`font-mono text-[11px] tabular-nums transition-colors duration-700 ${active ? 'text-crimson-light' : 'text-neutral-faint'}`}>
            {active ? `${agent.risk}%` : '--'}
          </span>
        </div>
        <div className="h-[2px] w-full bg-obsidian-high overflow-hidden">
          <motion.div
            className="h-full bg-crimson"
            initial={{ width: 0 }}
            animate={{ width: active ? `${agent.risk}%` : '0%' }}
            transition={{ duration: 1.5, ease: [0.22, 1, 0.36, 1] }}
          />
        </div>
      </div>
    </motion.div>
  )
}

/* ============================================================
   ACTION CARD
   ============================================================ */

function ActionCard({ visible }) {
  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 8 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="rounded-sm overflow-hidden"
          style={{
            background: 'linear-gradient(135deg, rgba(224,62,74,0.05) 0%, rgba(6,6,8,0.95) 100%)',
            border: '1px solid rgba(224,62,74,0.12)',
          }}
        >
          <div className="p-6 flex items-start gap-4">
            <motion.div
              animate={{ scale: [1, 1.3, 1], opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-2.5 h-2.5 rounded-full bg-crimson mt-1 shrink-0"
            />
            <div className="space-y-2">
              <p className="text-sm text-neutral-text leading-relaxed">
                <span className="font-semibold text-crimson-light">Flag immediately.</span>{' '}
                Do not accept as valid identity document.
                Escalate to forensic examiner.
                Request original document from issuing authority.
              </p>
              <p className="font-mono text-[9px] text-neutral-tertiary">
                3/3 agents converged &middot; DIS 0.11 &middot; confidence 99.2%
              </p>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

/* ============================================================
   COMMAND BAR (replaces scan button — machined hardware)
   ============================================================ */

function CommandBar({ scanning, scanDone, pipelineStep, onScan }) {
  return (
    <div className="fixed bottom-0 left-16 right-0 z-50 command-bar">
      <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
        {/* Pipeline stepper — inline */}
        <div className="flex items-center gap-1">
          {PIPELINE.map((stage, i) => {
            const done = i < pipelineStep
            const current = i === pipelineStep
            return (
              <div key={stage} className="flex items-center">
                <div className="flex items-center gap-1.5 px-2">
                  <div className={`
                    w-1.5 h-1.5 rounded-full transition-all duration-400
                    ${done ? 'bg-crimson' : current ? 'bg-crimson animate-pulse' : 'bg-neutral-faint'}
                  `} />
                  <span className={`
                    font-mono text-[8px] uppercase tracking-[0.12em] transition-colors duration-400
                    ${done || current ? 'text-neutral-secondary' : 'text-neutral-faint'}
                  `}>
                    {stage}
                  </span>
                </div>
                {i < PIPELINE.length - 1 && (
                  <div className="w-4 h-px relative">
                    <div className="absolute inset-0 bg-neutral-faint/30" />
                    <motion.div
                      className="absolute inset-y-0 left-0 bg-crimson/60"
                      animate={{ width: done ? '100%' : '0%' }}
                      transition={{ duration: 0.4 }}
                    />
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* File info */}
        <div className="hidden md:flex items-center gap-3">
          <span className="font-mono text-[9px] text-neutral-faint">
            Aadhaar_Card_Scan_2024.pdf
          </span>
          <span className="font-mono text-[9px] text-neutral-faint">
            2.4 MB &middot; 1 page
          </span>
        </div>

        {/* Command button */}
        <button
          onClick={onScan}
          disabled={scanning}
          className={`
            h-8 px-5 rounded-sm font-mono text-[9px] uppercase tracking-[0.2em] font-medium
            transition-all duration-300 border
            ${scanning
              ? scanDone
                ? 'bg-crimson/8 border-crimson/20 text-crimson cursor-default'
                : 'bg-obsidian-high border-neutral-faint/20 text-neutral-tertiary cursor-wait'
              : 'bg-obsidian-raised border-neutral-faint/30 text-neutral-secondary hover:border-crimson/30 hover:text-crimson active:scale-[0.97]'
            }
          `}
        >
          {scanning ? (scanDone ? '> SCAN COMPLETE' : '> ANALYZING...') : '> EXECUTE SCAN'}
        </button>
      </div>
    </div>
  )
}

/* ============================================================
   PROXIMITY GLOW BACKGROUND
   ============================================================ */

function ProximityGlow({ spherePos }) {
  // Map sphere XY → radial gradient position + color intensity
  const normX = useMemo(() => Math.max(0, Math.min(100, 50 + (spherePos.x / 5))), [spherePos.x])
  const normY = useMemo(() => Math.max(0, Math.min(100, 50 + (spherePos.y / 5))), [spherePos.y])
  const intensity = useMemo(() => {
    const dist = Math.sqrt(spherePos.x ** 2 + spherePos.y ** 2)
    return Math.min(0.06, dist / 5000)
  }, [spherePos.x, spherePos.y])

  return (
    <div
      className="fixed inset-0 pointer-events-none z-0 transition-all duration-200"
      style={{
        background: `radial-gradient(ellipse 600px 400px at ${normX}% ${normY}%, rgba(224,62,74,${intensity}), transparent 70%)`,
      }}
    />
  )
}

/* ============================================================
   MAIN APP
   ============================================================ */

export default function App() {
  const heroRef = useRef(null)
  const [scanning, setScanning] = useState(false)
  const [scanDone, setScanDone] = useState(false)
  const [pipelineStep, setPipelineStep] = useState(-1)
  const [activeAgent, setActiveAgent] = useState('overview')
  const [spherePos, setSpherePos] = useState({ x: 0, y: 0 })

  const handleScan = useCallback(() => {
    if (scanning) return
    setScanning(true)
    setScanDone(false)
    setPipelineStep(0)

    const timers = [
      setTimeout(() => setPipelineStep(1), 800),
      setTimeout(() => setPipelineStep(2), 1600),
      setTimeout(() => setPipelineStep(3), 2400),
      setTimeout(() => { setScanDone(true); setPipelineStep(4) }, 3200),
    ]
    return () => timers.forEach(clearTimeout)
  }, [scanning])

  return (
    <div className="min-h-screen bg-obsidian-base">
      {/* Proximity glow — follows sphere */}
      <ProximityGlow spherePos={spherePos} />

      {/* Chrome: sidebar + status bar */}
      <Sidebar
        activeAgent={activeAgent}
        onSelect={setActiveAgent}
        sphereProximity={spherePos}
      />
      <StatusBar scanning={scanning} scanDone={scanDone} sphereProximity={spherePos} />

      {/* -------- MAIN CONTENT -------- */}
      <div className="ml-16 pt-10 pb-16">
        {/* HERO: Sphere + Score */}
        <section
          ref={heroRef}
          className="min-h-[85vh] flex flex-col items-center justify-center px-6 relative"
        >
          {/* Sphere — 1.5x, centered */}
          <motion.div
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
            className="mb-8"
          >
            <SecuritySphere
              containerRef={heroRef}
              isScanning={scanning}
              scanDone={scanDone}
              onSpherePos={setSpherePos}
            />
          </motion.div>

          {/* Score — THE largest element */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <ScoreDisplay from={1.0} to={0.11} running={scanning} scanDone={scanDone} />
          </motion.div>
        </section>

        {/* AGENTS */}
        <section className="max-w-4xl mx-auto px-6 pb-20">
          <div className="flex items-baseline justify-between mb-6">
            <div className="flex items-baseline gap-3">
              <span className="font-serif text-lg italic text-neutral-faint">III</span>
              <h2 className="text-sm font-semibold text-neutral-text uppercase tracking-[0.1em]">
                Detection Agents
              </h2>
            </div>
            <span className="font-mono text-[8px] text-neutral-faint uppercase tracking-wider">
              Parallel Execution
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {AGENTS.map((agent, i) => (
              <AgentCard key={agent.id} agent={agent} active={scanDone} delay={0.05 + i * 0.08} />
            ))}
          </div>
        </section>

        {/* ACTION */}
        <section className="max-w-4xl mx-auto px-6 pb-24">
          <ActionCard visible={scanDone} />
        </section>
      </div>

      {/* COMMAND BAR — fixed bottom */}
      <CommandBar
        scanning={scanning}
        scanDone={scanDone}
        pipelineStep={pipelineStep}
        onScan={handleScan}
      />
    </div>
  )
}
