import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'

const agents = [
  {
    name: 'Visual Tamper Agent',
    model: 'EfficientNet-B4',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="12" r="3" />
        <path d="M2 12s4-8 10-8 10 8 10 8-4 8-10 8-10-8-10-8z" />
      </svg>
    ),
    gradient: 'from-accent-cyan to-blue-500',
    glowColor: 'rgba(0, 229, 255, 0.08)',
    borderColor: 'rgba(0, 229, 255, 0.15)',
    description: 'Multi-scale Error Level Analysis across Q90/Q75/Q50. Catches JPEG recompression artifacts, copy-move forgery, and splicing through pixel-level forensics.',
    capabilities: [
      'Multi-scale ELA (3 quality levels)',
      'Copy-move detection (ORB features)',
      'JPEG quantization analysis',
      'Noise consistency mapping',
    ],
    stat: { value: '0.12s', label: 'avg latency' },
  },
  {
    name: 'Text Forensics Agent',
    model: 'OCR + Font CNN',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M4 7V4h16v3" />
        <path d="M9 20h6" />
        <path d="M12 4v16" />
      </svg>
    ),
    gradient: 'from-blue-500 to-accent-violet',
    glowColor: 'rgba(59, 130, 246, 0.08)',
    borderColor: 'rgba(59, 130, 246, 0.15)',
    description: 'Analyzes OCR confidence variance, baseline alignment, character spacing, and font size consistency. Detects text splicing invisible to the human eye.',
    capabilities: [
      'OCR confidence variance analysis',
      'Baseline alignment forensics',
      'Inter-word spacing anomalies',
      'Regional confidence comparison',
    ],
    stat: { value: '6', label: 'detection methods' },
  },
  {
    name: 'Metadata Agent',
    model: 'Isolation Forest',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z" />
        <path d="M14 2v6h6" />
        <path d="M16 13H8M16 17H8M10 9H8" />
      </svg>
    ),
    gradient: 'from-accent-violet to-pink-500',
    glowColor: 'rgba(139, 92, 246, 0.08)',
    borderColor: 'rgba(139, 92, 246, 0.15)',
    description: 'Cross-references creation tools, timestamps, EXIF data, and embedded fonts. An Aadhaar made in Photoshop? Caught instantly.',
    capabilities: [
      'Creation tool provenance check',
      'EXIF consistency validation',
      'Indian document ID verification',
      'Verhoeff checksum (Aadhaar)',
    ],
    stat: { value: '50+', label: 'tool signatures' },
  },
]

function AgentCard({ agent, index }) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: '-80px' })

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 60 }}
      animate={isInView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.7, delay: index * 0.15, ease: [0.22, 1, 0.36, 1] }}
      className="group relative"
    >
      {/* Card */}
      <div
        className="relative glass rounded-3xl p-8 h-full flex flex-col transition-all duration-500 hover:scale-[1.02]"
        style={{
          borderColor: agent.borderColor,
          boxShadow: `0 0 40px ${agent.glowColor}`,
        }}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div
            className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${agent.gradient} flex items-center justify-center text-white/90`}
            style={{ boxShadow: `0 4px 20px ${agent.glowColor}` }}
          >
            {agent.icon}
          </div>
          <span className="text-[10px] font-mono text-white/25 tracking-wider uppercase px-3 py-1 rounded-full border border-white/5">
            {agent.model}
          </span>
        </div>

        {/* Title */}
        <h3 className="text-xl font-semibold text-white/90 mb-3">{agent.name}</h3>

        {/* Description */}
        <p className="text-sm text-white/35 leading-relaxed mb-6 flex-grow">
          {agent.description}
        </p>

        {/* Capabilities */}
        <div className="space-y-2 mb-6">
          {agent.capabilities.map((cap, i) => (
            <div key={i} className="flex items-center gap-2.5 text-xs text-white/40">
              <span className={`w-1 h-1 rounded-full bg-gradient-to-r ${agent.gradient}`} />
              <span>{cap}</span>
            </div>
          ))}
        </div>

        {/* Stat footer */}
        <div className="pt-4 border-t border-white/5 flex items-center justify-between">
          <span className={`text-lg font-bold font-mono bg-gradient-to-r ${agent.gradient} bg-clip-text text-transparent`}>
            {agent.stat.value}
          </span>
          <span className="text-[10px] font-mono text-white/20 uppercase tracking-wider">
            {agent.stat.label}
          </span>
        </div>
      </div>
    </motion.div>
  )
}

export default function TripleThreat() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: '-100px' })

  return (
    <section id="agents" className="relative py-32 px-6">
      {/* Section header */}
      <div ref={ref} className="max-w-7xl mx-auto mb-20 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="inline-flex items-center gap-2 glass rounded-full px-4 py-1.5 mb-6"
        >
          <span className="w-2 h-2 rounded-full bg-accent-blue animate-pulse" />
          <span className="text-xs font-mono text-white/40 tracking-wider uppercase">
            Detection Pipeline
          </span>
        </motion.div>

        <motion.h2
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7, delay: 0.1 }}
          className="text-4xl md:text-5xl font-bold text-gradient mb-4"
        >
          The Triple Threat
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7, delay: 0.2 }}
          className="text-lg text-white/30 max-w-2xl mx-auto"
        >
          Three independent agents analyze every document in parallel.
          When they converge on a verdict, there's nowhere to hide.
        </motion.p>
      </div>

      {/* Agent cards grid */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-6">
        {agents.map((agent, i) => (
          <AgentCard key={agent.name} agent={agent} index={i} />
        ))}
      </div>

      {/* Convergence line connecting the cards */}
      <motion.div
        initial={{ scaleX: 0 }}
        animate={isInView ? { scaleX: 1 } : {}}
        transition={{ duration: 1, delay: 0.8, ease: [0.22, 1, 0.36, 1] }}
        className="hidden md:block max-w-5xl mx-auto mt-12 h-px bg-gradient-to-r from-accent-cyan/0 via-accent-blue/20 to-accent-violet/0 origin-center"
      />

      <motion.p
        initial={{ opacity: 0 }}
        animate={isInView ? { opacity: 1 } : {}}
        transition={{ delay: 1.2 }}
        className="text-center text-xs font-mono text-white/20 mt-4 tracking-wider"
      >
        WEIGHTED TRUST FUSION &mdash; DIS = &Sigma;(R&#x1D62; &times; S&#x1D62;) / &Sigma;R&#x1D62;
      </motion.p>
    </section>
  )
}
