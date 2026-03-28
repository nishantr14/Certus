import { useRef, useEffect, useState } from 'react'
import { motion, useInView, animate } from 'framer-motion'

// Animated arc gauge
function DISGauge({ score, isActive }) {
  const radius = 110
  const strokeWidth = 10
  const center = 130
  const startAngle = 135
  const endAngle = 405
  const totalAngle = endAngle - startAngle
  const circumference = (totalAngle / 360) * 2 * Math.PI * radius

  const scoreAngle = startAngle + totalAngle * (1 - score)
  const fillLength = circumference * (1 - score)

  // Color based on score
  const getColor = (s) => {
    if (s < 0.4) return { stroke: '#ef4444', glow: 'rgba(239,68,68,0.3)', label: 'HIGH RISK', labelColor: '#ef4444' }
    if (s < 0.65) return { stroke: '#f59e0b', glow: 'rgba(245,158,11,0.3)', label: 'MEDIUM RISK', labelColor: '#f59e0b' }
    if (s < 0.8) return { stroke: '#3b82f6', glow: 'rgba(59,130,246,0.3)', label: 'LOW RISK', labelColor: '#3b82f6' }
    return { stroke: '#10b981', glow: 'rgba(16,185,129,0.3)', label: 'AUTHENTIC', labelColor: '#10b981' }
  }

  const color = getColor(score)

  // Tick marks
  const ticks = Array.from({ length: 28 }, (_, i) => {
    const angle = startAngle + (i / 27) * totalAngle
    const rad = (angle * Math.PI) / 180
    const isMajor = i % 9 === 0
    const innerR = isMajor ? radius - 22 : radius - 16
    const outerR = radius - 12
    return {
      x1: center + innerR * Math.cos(rad),
      y1: center + innerR * Math.sin(rad),
      x2: center + outerR * Math.cos(rad),
      y2: center + outerR * Math.sin(rad),
      isMajor,
    }
  })

  // Animated counter
  const [displayScore, setDisplayScore] = useState(0)
  useEffect(() => {
    if (!isActive) return
    const controls = animate(0, score, {
      duration: 2,
      ease: [0.22, 1, 0.36, 1],
      onUpdate: (v) => setDisplayScore(v),
    })
    return () => controls.stop()
  }, [isActive, score])

  const describeArc = (startA, endA, r) => {
    const start = {
      x: center + r * Math.cos((startA * Math.PI) / 180),
      y: center + r * Math.sin((startA * Math.PI) / 180),
    }
    const end = {
      x: center + r * Math.cos((endA * Math.PI) / 180),
      y: center + r * Math.sin((endA * Math.PI) / 180),
    }
    const largeArc = endA - startA > 180 ? 1 : 0
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`
  }

  return (
    <div className="relative">
      {/* Background glow */}
      <div
        className="absolute inset-0 rounded-full blur-3xl opacity-30"
        style={{ background: color.glow }}
      />

      <svg width="260" height="260" viewBox="0 0 260 260">
        {/* Track */}
        <path
          d={describeArc(startAngle, endAngle, radius)}
          fill="none"
          stroke="rgba(255,255,255,0.04)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />

        {/* Ticks */}
        {ticks.map((t, i) => (
          <line
            key={i}
            x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2}
            stroke={t.isMajor ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.06)'}
            strokeWidth={t.isMajor ? 1.5 : 0.8}
          />
        ))}

        {/* Active fill arc */}
        <motion.path
          d={describeArc(startAngle, endAngle, radius)}
          fill="none"
          stroke={color.stroke}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={isActive ? { strokeDashoffset: fillLength } : {}}
          transition={{ duration: 2, ease: [0.22, 1, 0.36, 1] }}
          style={{
            filter: `drop-shadow(0 0 8px ${color.glow})`,
          }}
        />

        {/* Score text */}
        <text
          x={center}
          y={center - 10}
          textAnchor="middle"
          className="fill-white font-mono text-4xl font-bold"
          style={{ fontSize: '42px' }}
        >
          {displayScore.toFixed(2)}
        </text>

        {/* Label */}
        <text
          x={center}
          y={center + 20}
          textAnchor="middle"
          className="font-mono font-semibold"
          style={{ fontSize: '11px', fill: color.labelColor, letterSpacing: '2px' }}
        >
          {color.label}
        </text>

        {/* Sub-label */}
        <text
          x={center}
          y={center + 40}
          textAnchor="middle"
          className="font-mono"
          style={{ fontSize: '9px', fill: 'rgba(255,255,255,0.2)', letterSpacing: '1px' }}
        >
          DOCUMENT INTEGRITY SCORE
        </text>
      </svg>
    </div>
  )
}

// Agent breakdown bars
function AgentBreakdown({ agents, isActive }) {
  return (
    <div className="space-y-4 w-full max-w-sm">
      {agents.map((agent, i) => (
        <motion.div
          key={agent.name}
          initial={{ opacity: 0, x: 30 }}
          animate={isActive ? { opacity: 1, x: 0 } : {}}
          transition={{ duration: 0.5, delay: 1.0 + i * 0.15 }}
        >
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs text-white/40 font-mono">{agent.name}</span>
            <span className="text-xs font-mono font-semibold" style={{ color: agent.color }}>
              {agent.score.toFixed(2)}
            </span>
          </div>
          <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{ background: agent.color }}
              initial={{ width: 0 }}
              animate={isActive ? { width: `${agent.score * 100}%` } : {}}
              transition={{ duration: 1.5, delay: 1.0 + i * 0.15, ease: [0.22, 1, 0.36, 1] }}
            />
          </div>
        </motion.div>
      ))}
    </div>
  )
}

// Finding pills
function Findings({ findings, isActive }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={isActive ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay: 1.8 }}
      className="w-full max-w-md"
    >
      <div className="text-[10px] font-mono text-white/20 uppercase tracking-wider mb-3">
        Key Findings
      </div>
      <div className="space-y-2">
        {findings.map((f, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: -10 }}
            animate={isActive ? { opacity: 1, x: 0 } : {}}
            transition={{ delay: 2.0 + i * 0.1 }}
            className="flex items-start gap-2 text-xs"
          >
            <span className={`mt-0.5 w-1.5 h-1.5 rounded-full shrink-0 ${f.severity === 'critical' ? 'bg-red-500' : f.severity === 'high' ? 'bg-amber-500' : 'bg-blue-500'}`} />
            <span className="text-white/35 leading-relaxed">{f.text}</span>
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}

export default function InteractiveScore() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: '-100px' })

  const disScore = 0.12

  const agentScores = [
    { name: 'Visual Tamper', score: 0.08, color: '#ef4444' },
    { name: 'Text Forensics', score: 0.15, color: '#f59e0b' },
    { name: 'Metadata', score: 0.10, color: '#ef4444' },
  ]

  const findings = [
    { severity: 'critical', text: 'ELA anomaly persists across ALL 3 quality levels (Q90/Q75/Q50). 8.2% of image affected.' },
    { severity: 'critical', text: 'Aadhaar created with Adobe Photoshop CC 2024 \u2014 government documents are never issued from editing software.' },
    { severity: 'high', text: 'Aadhaar number fails Verhoeff checksum validation.' },
    { severity: 'high', text: 'Baseline misalignment in 12 of 18 text lines. Worst deviation: 14.2px.' },
    { severity: 'high', text: 'Copy-move detected: 47 clustered matches, displacement (320, 0)px.' },
  ]

  return (
    <section id="score" ref={ref} className="relative py-32 px-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-16 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          className="inline-flex items-center gap-2 glass rounded-full px-4 py-1.5 mb-6"
        >
          <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          <span className="text-xs font-mono text-white/40 tracking-wider uppercase">
            Live Analysis
          </span>
        </motion.div>

        <motion.h2
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.1 }}
          className="text-4xl md:text-5xl font-bold text-gradient mb-4"
        >
          Document Integrity Score
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.2 }}
          className="text-lg text-white/30 max-w-xl mx-auto"
        >
          Real-time weighted fusion of all three agents.
          This forged Aadhaar didn't stand a chance.
        </motion.p>
      </div>

      {/* Score panel */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.3 }}
        className="max-w-5xl mx-auto"
      >
        <div className="glass-strong rounded-[2rem] p-10 md:p-14 glow-red">
          <div className="flex flex-col lg:flex-row items-center gap-12 lg:gap-16">
            {/* Gauge */}
            <div className="shrink-0">
              <DISGauge score={disScore} isActive={isInView} />
            </div>

            {/* Right panel */}
            <div className="flex flex-col items-start gap-8 flex-grow">
              {/* File info */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={isInView ? { opacity: 1 } : {}}
                transition={{ delay: 0.8 }}
                className="glass rounded-xl px-4 py-2.5 flex items-center gap-3"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" className="text-red-500/60" strokeWidth="1.5">
                  <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z" />
                  <path d="M14 2v6h6" />
                </svg>
                <div>
                  <div className="text-xs text-white/50 font-mono">forged_aadhaar_photoshop.pdf</div>
                  <div className="text-[10px] text-white/20 font-mono">2.4 MB &middot; 1 page &middot; analyzed in 1.2s</div>
                </div>
              </motion.div>

              {/* Agent breakdown */}
              <AgentBreakdown agents={agentScores} isActive={isInView} />

              {/* Findings */}
              <Findings findings={findings} isActive={isInView} />
            </div>
          </div>

          {/* Verdict bar */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 2.5 }}
            className="mt-10 pt-6 border-t border-white/5 flex flex-col sm:flex-row items-center justify-between gap-4"
          >
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
              <span className="text-sm font-semibold text-red-400 font-mono tracking-wide">
                VERDICT: FORGED DOCUMENT
              </span>
            </div>
            <span className="text-[10px] text-white/20 font-mono">
              3/3 agents converged &middot; confidence: 99.2% &middot; escalate immediately
            </span>
          </motion.div>
        </div>
      </motion.div>
    </section>
  )
}
