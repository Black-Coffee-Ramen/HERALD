import React, { useState, useRef, useMemo, Suspense } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { 
  OrbitControls, 
  Points, 
  PointMaterial, 
  Float, 
  Text, 
  PerspectiveCamera, 
  Environment,
  Html,
  Line
} from '@react-three/drei'
import { EffectComposer, Bloom, ChromaticAberration } from '@react-three/postprocessing'
import * as THREE from 'three'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { Shield, Activity, Cpu, Zap, BarChart3, Milestone, Terminal, ChevronRight, Github, ExternalLink } from 'lucide-react'

gsap.registerPlugin(ScrollTrigger)

// --- 3D COMPONENTS ---

function NeuralNetwork() {
  const points = useMemo(() => {
    const p = new Float32Array(10000 * 3)
    for (let i = 0; i < 10000; i++) {
      p[i * 3] = (Math.random() - 0.5) * 15
      p[i * 3 + 1] = (Math.random() - 0.5) * 15
      p[i * 3 + 2] = (Math.random() - 0.5) * 15
    }
    return p
  }, [])

  const ref = useRef()
  useFrame((state) => {
    const t = state.clock.getElapsedTime()
    ref.current.rotation.y = t * 0.05
    ref.current.rotation.x = Math.sin(t * 0.1) * 0.1
    
    // Heartbeat pulse effect
    const pulse = 1 + Math.sin(t * 2) * 0.02
    ref.current.scale.set(pulse, pulse, pulse)
  })

  return (
    <group rotation={[0, 0, Math.PI / 4]}>
      <Points ref={ref} positions={points} stride={3} frustumCulled={false}>
        <PointMaterial
          transparent
          color="#00f2ff"
          size={0.015}
          sizeAttenuation={true}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </Points>
    </group>
  )
}

function AICore({ hovered }) {
  const meshRef = useRef()
  const innerRef = useRef()

  useFrame((state) => {
    const t = state.clock.getElapsedTime()
    meshRef.current.rotation.y = t * 0.5
    innerRef.current.rotation.y = -t * 0.8
    
    if (hovered) {
      meshRef.current.scale.lerp(new THREE.Vector3(1.3, 1.3, 1.3), 0.1)
    } else {
      meshRef.current.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1)
    }
  })

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={1}>
      <group ref={meshRef}>
        {/* Hexagonal Prism Outer Shell */}
        <mesh>
          <cylinderGeometry args={[1, 1, 0.5, 6]} />
          <meshStandardMaterial 
            color="#1e293b" 
            transparent 
            opacity={0.4} 
            roughness={0.1}
            metalness={0.8}
          />
        </mesh>
        <mesh>
          <cylinderGeometry args={[1, 1, 0.5, 6]} />
          <meshStandardMaterial 
            color="#00f2ff" 
            wireframe 
            emissive="#00f2ff"
            emissiveIntensity={0.5}
          />
        </mesh>

        {/* Interior AI Core */}
        <group ref={innerRef}>
          <mesh scale={0.6}>
            <octahedronGeometry />
            <meshStandardMaterial 
              color="#ff003c" 
              emissive="#ff003c"
              emissiveIntensity={2}
              wireframe
            />
          </mesh>
        </group>
      </group>
    </Float>
  )
}

function DataBar({ position, height, label, color, delay }) {
  const meshRef = useRef()
  
  useFrame((state) => {
    const t = state.clock.getElapsedTime()
    // Subtle breathing glow
    meshRef.current.material.emissiveIntensity = 0.5 + Math.sin(t * 3 + delay) * 0.2
  })

  return (
    <group position={position}>
      <mesh ref={meshRef} position={[0, height / 2, 0]}>
        <cylinderGeometry args={[0.3, 0.3, height, 32]} />
        <meshStandardMaterial 
          color={color} 
          transparent 
          opacity={0.6}
          emissive={color}
          emissiveIntensity={1}
        />
      </mesh>
      <mesh position={[0, height + 0.5, 0]}>
        <Text
          fontSize={0.2}
          color="white"
          font="https://fonts.gstatic.com/s/jetbrainsmono/v13/t6nu27PSqS93qj4V5OfD2u-n16B-W6G3.woff"
        >
          {label}
        </Text>
      </mesh>
    </group>
  )
}

function PipelineSplines() {
  const points1 = useMemo(() => [
    new THREE.Vector3(-4, 0, 0),
    new THREE.Vector3(-2, 1, 0),
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(2, -1, 0),
    new THREE.Vector3(4, 0, 0)
  ], [])

  return (
    <group position={[0, -5, 0]}>
      <Line points={points1} color="#00f2ff" lineWidth={2} dashed dashScale={10} />
      {/* Animated Packets */}
      <DataPacket path={points1} color="#00f2ff" />
      <DataPacket path={points1} color="#ff003c" delay={1.5} />
      
      {/* Nodes */}
      <PipelineNode position={[-4, 0, 0]} label="INGESTION" />
      <PipelineNode position={[-2, 1, 0]} label="ENRICH" />
      <PipelineNode position={[0, 0, 0]} label="INFERENCE" />
      <PipelineNode position={[2, -1, 0]} label="VERDICT" />
      <PipelineNode position={[4, 0, 0]} label="OUTPUT" />
    </group>
  )
}

function PipelineNode({ position, label }) {
  return (
    <group position={position}>
      <mesh>
        <boxGeometry args={[0.8, 0.4, 0.1]} />
        <meshStandardMaterial color="#1e293b" transparent opacity={0.8} />
      </mesh>
      <Text fontSize={0.15} position={[0, 0, 0.1]} color="#00f2ff">
        {label}
      </Text>
    </group>
  )
}

function DataPacket({ path, color, delay = 0 }) {
  const ref = useRef()
  const curve = useMemo(() => new THREE.CatmullRomCurve3(path), [path])

  useFrame((state) => {
    const t = (state.clock.getElapsedTime() + delay) % 3 / 3
    const pos = curve.getPointAt(t)
    ref.current.position.copy(pos)
  })

  return (
    <mesh ref={ref}>
      <sphereGeometry args={[0.1, 16, 16]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={2} />
    </mesh>
  )
}

// --- UI COMPONENTS ---

const Section = ({ title, subtitle, children, id }) => (
  <section id={id} className="min-h-screen py-24 px-6 md:px-24 flex flex-col justify-center relative z-10">
    <div className="max-w-4xl">
      <h2 className="font-display text-5xl md:text-7xl font-bold text-white mb-4 tracking-tighter uppercase italic">
        {title}
      </h2>
      <p className="font-mono text-cyber-cyan text-sm tracking-widest uppercase mb-12">
        {subtitle}
      </p>
      {children}
    </div>
  </section>
)

export default function App() {
  const [hoveredCore, setHoveredCore] = useState(false)
  const [lexicalCeiling, setLexicalCeiling] = useState(false)

  return (
    <div className="bg-cyber-black min-h-screen selection:bg-cyber-cyan selection:text-black">
      {/* 3D CANVAS LAYER */}
      <div className="fixed inset-0 z-0">
        <Canvas gl={{ antialias: false }}>
          <PerspectiveCamera makeDefault position={[0, 0, 10]} fov={50} />
          <color attach="background" args={['#050505']} />
          
          <Suspense fallback={null}>
            <NeuralNetwork />
            
            <group position={[3, 0, 0]}>
              <AICore hovered={hoveredCore} />
            </group>

            {/* v3/v5/v7 Comparison Section Visuals */}
            <group position={[0, -15, 0]}>
              <DataBar position={[-2, 0, 0]} height={3} label="v3: 87.7%" color="#1e293b" delay={0} />
              <DataBar position={[0, 0, 0]} height={4.5} label="v5: F1 0.81" color="#3b82f6" delay={0.5} />
              <DataBar position={[2, 0, 0]} height={6} label="v7: 98.1%" color="#00f2ff" delay={1} />
              
              {lexicalCeiling && (
                <mesh position={[0, 4, 0]}>
                  <planeGeometry args={[10, 0.1]} />
                  <meshStandardMaterial color="#ff003c" emissive="#ff003c" emissiveIntensity={2} />
                  <Text position={[0, 0.5, 0]} fontSize={0.2} color="#ff003c">LEXICAL CEILING</Text>
                </mesh>
              )}
            </group>

            <PipelineSplines />

            <Environment preset="city" />
            <ambientLight intensity={0.2} />
            <pointLight position={[10, 10, 10]} intensity={1} color="#00f2ff" />
            <pointLight position={[-10, -10, -10]} intensity={0.5} color="#ff003c" />
            
            <EffectComposer>
              <Bloom luminanceThreshold={1} luminanceSmoothing={0.9} height={300} intensity={1.5} />
              <ChromaticAberration offset={new THREE.Vector2(0.001, 0.001)} />
            </EffectComposer>
          </Suspense>
          
          <OrbitControls 
            enableZoom={false} 
            maxPolarAngle={Math.PI / 2} 
            minPolarAngle={Math.PI / 2.5}
            maxAzimuthAngle={Math.PI / 4}
            minAzimuthAngle={-Math.PI / 4}
          />
        </Canvas>
      </div>

      {/* UI OVERLAY LAYER */}
      <nav className="fixed top-0 w-full p-8 flex justify-between items-center z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 border border-cyber-cyan flex items-center justify-center">
            <Shield className="text-cyber-cyan w-6 h-6" />
          </div>
          <span className="font-display font-bold text-2xl tracking-tighter text-white">HERALD</span>
        </div>
        <div className="flex gap-8 font-mono text-[10px] tracking-widest text-slate-500 uppercase">
          <a href="#evolution" className="hover:text-cyber-cyan transition-colors">Evolution</a>
          <a href="#pipeline" className="hover:text-cyber-cyan transition-colors">Pipeline</a>
          <a href="#milestones" className="hover:text-cyber-cyan transition-colors">Milestones</a>
          <a href="https://github.com/Black-Coffee-Ramen/HERALD" className="text-white hover:text-cyber-cyan">GitHub</a>
        </div>
      </nav>

      {/* HERO SECTION */}
      <section className="min-h-screen relative flex items-center px-6 md:px-24 z-10 pointer-events-none">
        <div className="max-w-4xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 border border-cyber-cyan/30 text-cyber-cyan text-[10px] font-mono mb-8 animate-pulse">
            <Activity size={12} />
            SENTINEL ACTIVE: NODE_01
          </div>
          <h1 className="font-display text-7xl md:text-9xl font-bold text-white leading-[0.85] tracking-tighter uppercase italic mb-8">
            Digital<br/>Sentinel
          </h1>
          <p className="max-w-xl text-slate-400 font-mono text-sm leading-relaxed mb-12 pointer-events-auto">
            Distributed cyber intelligence platform for real-time phishing detection. 
            Monitoring 1M+ Certstream events daily with 98.1% precision.
          </p>
          <div className="flex gap-4 pointer-events-auto">
            <button 
              onMouseEnter={() => setHoveredCore(true)}
              onMouseLeave={() => setHoveredCore(false)}
              className="bg-cyber-cyan text-black px-8 py-4 font-bold uppercase tracking-widest text-xs flex items-center gap-2 hover:bg-white transition-all group"
            >
              Analyze Threat Core
              <ChevronRight className="group-hover:translate-x-1 transition-transform" />
            </button>
            <button className="border border-white/20 text-white px-8 py-4 font-bold uppercase tracking-widest text-xs hover:border-cyber-cyan transition-all">
              Documentation
            </button>
          </div>
        </div>
      </section>

      {/* EVOLUTION SECTION */}
      <Section id="evolution" title="The Evolution of Accuracy" subtitle="Model Performance Benchmark (v3 - v7)">
        <div className="grid md:grid-cols-2 gap-12 items-start">
          <div className="space-y-8">
            <div className="glass p-8 border-l-2 border-slate-500">
              <h4 className="font-mono text-slate-500 text-xs mb-2">v3 (Baseline)</h4>
              <p className="text-slate-400 text-sm">Initial lexical analysis model. Precision: 87.7%, Recall: 54.6%.</p>
            </div>
            <div className="glass p-8 border-l-2 border-blue-500">
              <h4 className="font-mono text-blue-500 text-xs mb-2">v5 (Operational)</h4>
              <p className="text-slate-400 text-sm">Added Tranco legitimate class integration. F1-Score improved to 0.814.</p>
            </div>
            <div className="glass p-8 border-l-2 border-cyber-cyan">
              <h4 className="font-mono text-cyber-cyan text-xs mb-2">v7 (Production)</h4>
              <p className="text-white text-sm font-bold">Current Gold Standard. Precision: 98.1%, Recall: 84.1% with OCR Fallback.</p>
            </div>
          </div>
          <div className="glass p-8 flex flex-col justify-center items-center gap-6">
            <p className="font-mono text-[10px] text-slate-500 text-center uppercase tracking-widest leading-relaxed">
              V7 utilizes a two-stage ensemble engine coupled with visual screenshot comparison to break the lexical similarity ceiling.
            </p>
            <button 
              onClick={() => setLexicalCeiling(!lexicalCeiling)}
              className={`w-full py-4 font-mono text-[10px] border tracking-widest transition-all ${lexicalCeiling ? 'bg-cyber-red border-cyber-red text-white' : 'border-white/20 text-white hover:border-cyber-red'}`}
            >
              {lexicalCeiling ? 'CEILING ACTIVE' : 'VISUALIZE LEXICAL CEILING'}
            </button>
          </div>
        </div>
      </Section>

      {/* PIPELINE SECTION */}
      <Section id="pipeline" title="Autonomous Pipeline" subtitle="3D Architecture Visualizer">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="glass p-6 text-center space-y-4 group hover:border-cyber-cyan transition-all">
            <Zap className="mx-auto text-slate-500 group-hover:text-cyber-cyan" />
            <h5 className="font-mono text-[10px] text-white">INGESTION</h5>
            <p className="text-[10px] text-slate-500">Certstream WS Inbound</p>
          </div>
          <div className="glass p-6 text-center space-y-4 group hover:border-cyber-cyan transition-all">
            <Activity className="mx-auto text-slate-500 group-hover:text-cyber-cyan" />
            <h5 className="font-mono text-[10px] text-white">ENRICHMENT</h5>
            <p className="text-[10px] text-slate-500">WHOIS / DNS / SSL</p>
          </div>
          <div className="glass p-6 text-center space-y-4 group hover:border-cyber-cyan transition-all">
            <Cpu className="mx-auto text-slate-500 group-hover:text-cyber-cyan" />
            <h5 className="font-mono text-[10px] text-white">INFERENCE</h5>
            <p className="text-[10px] text-slate-500">v7 Ensemble Core</p>
          </div>
          <div className="glass p-6 text-center space-y-4 group hover:border-cyber-cyan transition-all">
            <Shield className="mx-auto text-slate-500 group-hover:text-cyber-cyan" />
            <h5 className="font-mono text-[10px] text-white">VERDICT</h5>
            <p className="text-[10px] text-slate-500">STIX / TI Output</p>
          </div>
        </div>
      </Section>

      {/* MILESTONES SECTION */}
      <Section id="milestones" title="Project Milestones" subtitle="The Journey from P0 to P3">
        <div className="space-y-4">
          {[
            { p: 'P0', t: 'Foundational', d: 'Real-time Monitoring & JWT Security Implementation.' },
            { p: 'P1', t: 'Operational', d: 'React Dashboard & Forensic Evidence Export (PDF).' },
            { p: 'P2', t: 'Reliability', d: 'Horizontal Worker Scaling & Redis Job Management.' },
            { p: 'P3', t: 'Future', d: 'STIX Export & Campaign Clustering Analytics.' },
          ].map((m, i) => (
            <div key={i} className="glass p-8 flex items-center gap-8 group hover:bg-white/5 transition-all cursor-pointer">
              <span className="font-display text-4xl font-black text-slate-800 group-hover:text-cyber-cyan transition-colors">{m.p}</span>
              <div>
                <h4 className="font-bold text-white uppercase italic">{m.t}</h4>
                <p className="text-slate-500 text-xs font-mono">{m.d}</p>
              </div>
              <Milestone className="ml-auto text-slate-800 group-hover:text-cyber-cyan" />
            </div>
          ))}
        </div>
      </Section>

      {/* FOOTER */}
      <footer className="py-24 px-6 md:px-24 border-t border-white/5 relative z-10 flex flex-col md:flex-row justify-between items-center gap-8">
        <div className="flex items-center gap-3">
          <Shield className="text-cyber-cyan w-5 h-5" />
          <span className="font-display font-bold text-lg tracking-tighter text-white">HERALD</span>
        </div>
        <p className="font-mono text-[10px] text-slate-500 uppercase tracking-widest">
          Build by Athiyo · IIIT Delhi · MIT License
        </p>
        <div className="flex gap-6">
          <a href="https://github.com/Black-Coffee-Ramen/HERALD" className="text-slate-500 hover:text-white transition-colors"><Github size={18} /></a>
          <a href="#" className="text-slate-500 hover:text-white transition-colors"><ExternalLink size={18} /></a>
        </div>
      </footer>
    </div>
  )
}
