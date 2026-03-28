import { useRef, useState } from 'react'
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion'

// Floating particle dots around the core
function Particles() {
  const particles = Array.from({ length: 40 }, (_, i) => {
    const angle = (i / 40) * Math.PI * 2
    const radius = 120 + Math.random() * 80
    return {
      id: i,
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      size: 1.5 + Math.random() * 2,
      delay: Math.random() * 4,
      duration: 3 + Math.random() * 3,
    }
  })

  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
      {particles.map((p) => (
        <motion.div
          key={p.id}
          className="absolute rounded-full bg-accent-cyan/30"
          style={{ width: p.size, height: p.size }}
          initial={{ x: p.x, y: p.y, opacity: 0 }}
          animate={{
            x: [p.x, p.x + 15, p.x],
            y: [p.y, p.y - 10, p.y],
            opacity: [0, 0.6, 0],
          }}
          transition={{
            duration: p.duration,
            delay: p.delay,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  )
}

// The 3D hexagonal core that follows the cursor
function SecurityCore({ mouseX, mouseY }) {
  const rotateX = useSpring(useTransform(mouseY, [-300, 300], [15, -15]), {
    stiffness: 80,
    damping: 20,
  })
  const rotateY = useSpring(useTransform(mouseX, [-400, 400], [-15, 15]), {
    stiffness: 80,
    damping: 20,
  })

  return (
    <motion.div
      className="relative w-72 h-72 flex items-center justify-center"
      style={{
        perspective: 1000,
        rotateX,
        rotateY,
        transformStyle: 'preserve-3d',
      }}
    >
      {/* Outer glow ring */}
      <div className="absolute inset-0 rounded-full bg-gradient-to-br from-accent-cyan/10 via-accent-blue/5 to-transparent blur-2xl" />

      {/* Hexagonal layers */}
      {[0, 1, 2].map((layer) => (
        <motion.div
          key={layer}
          className="absolute"
          style={{ transform: `translateZ(${layer * 12 - 12}px)` }}
          animate={{ rotate: layer % 2 === 0 ? 360 : -360 }}
          transition={{ duration: 40 + layer * 10, repeat: Infinity, ease: 'linear' }}
        >
          <svg
            width={200 - layer * 30}
            height={200 - layer * 30}
            viewBox="0 0 200 200"
            className="opacity-100"
          >
            <polygon
              points="100,10 178,55 178,145 100,190 22,145 22,55"
              fill="none"
              stroke={layer === 0 ? 'rgba(0,229,255,0.15)' : layer === 1 ? 'rgba(59,130,246,0.2)' : 'rgba(139,92,246,0.25)'}
              strokeWidth={layer === 2 ? '2' : '1'}
            />
          </svg>
        </motion.div>
      ))}

      {/* Center shield icon */}
      <motion.div
        className="relative z-10 w-20 h-20 rounded-2xl bg-gradient-to-br from-accent-cyan/20 to-accent-blue/10 glass-strong flex items-center justify-center glow-cyan"
        animate={{ scale: [1, 1.04, 1] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
      >
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" className="text-accent-cyan">
          <path d="M12 2L3 7v5c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z" strokeWidth="1.5" strokeLinejoin="round" />
          <path d="M9 12l2 2 4-4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </motion.div>
    </motion.div>
  )
}

export default function HeroSection() {
  const containerRef = useRef(null)
  const mouseX = useMotionValue(0)
  const mouseY = useMotionValue(0)

  const handleMouseMove = (e) => {
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return
    mouseX.set(e.clientX - rect.left - rect.width / 2)
    mouseY.set(e.clientY - rect.top - rect.height / 2)
  }

  return (
    <section
      ref={containerRef}
      onMouseMove={handleMouseMove}
      className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-24 overflow-hidden"
    >
      {/* Ambient glow */}
      <div className="hero-glow" />

      {/* Badge */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.2 }}
        className="glass rounded-full px-4 py-1.5 mb-8 flex items-center gap-2"
      >
        <span className="w-2 h-2 rounded-full bg-accent-cyan animate-pulse" />
        <span className="text-xs font-mono text-white/50 tracking-wider uppercase">
          Blue Team &mdash; Secure AI Hackathon 2026
        </span>
      </motion.div>

      {/* 3D Security Core */}
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1, delay: 0.4, ease: [0.22, 1, 0.36, 1] }}
        className="relative mb-12"
      >
        <Particles />
        <SecurityCore mouseX={mouseX} mouseY={mouseY} />
      </motion.div>

      {/* Headline */}
      <motion.h1
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.6 }}
        className="text-5xl md:text-7xl font-bold text-center leading-tight tracking-tight max-w-4xl"
      >
        <span className="text-gradient">Forensic AI that</span>
        <br />
        <span className="text-gradient-cyan">sees forgeries</span>
      </motion.h1>

      {/* Subtitle */}
      <motion.p
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.8 }}
        className="mt-6 text-lg md:text-xl text-white/40 text-center max-w-2xl leading-relaxed"
      >
        Three specialized AI agents converge on every pixel, every glyph, every byte
        of metadata. If it's forged, CertusDoc will find it.
      </motion.p>

      {/* Stats bar */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 1.0 }}
        className="mt-12 flex flex-wrap items-center justify-center gap-8 md:gap-16"
      >
        {[
          { value: '3', label: 'Detection Agents' },
          { value: '<2s', label: 'Analysis Time' },
          { value: '96%', label: 'Recall on Forgeries' },
        ].map((stat, i) => (
          <div key={i} className="text-center">
            <div className="text-2xl md:text-3xl font-bold text-gradient-cyan font-mono">
              {stat.value}
            </div>
            <div className="text-xs text-white/30 mt-1 uppercase tracking-wider">
              {stat.label}
            </div>
          </div>
        ))}
      </motion.div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.4 }}
        className="absolute bottom-8"
      >
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="w-5 h-8 rounded-full border border-white/10 flex justify-center pt-1.5"
        >
          <div className="w-1 h-2 rounded-full bg-white/20" />
        </motion.div>
      </motion.div>
    </section>
  )
}
