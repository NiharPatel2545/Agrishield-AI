import { useEffect, useRef, useState, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Area, AreaChart,
} from 'recharts'

// ─── Backend connection ────────────────────────────────────────────────────────
// Points at your local FastAPI server during development. Change this ONE
// constant to your deployed API URL when you deploy (e.g. Render/Railway) --
// nothing else in this file needs to change.
const API_BASE_URL = 'http://127.0.0.1:8000'

interface ScanResponse {
  location: { latitude: number; longitude: number }
  headline: string
  verdict: string
  confidence_pct: number
  recommended_action: string
  crop_note: string | null
  oc_note: string | null
  predicted_oc_gkg: number | null
  color: string
  context: {
    ndvi: number | null
    avg_temp_c: number | null
    avg_annual_rainfall_mm: number | null
    current_weather: string | null
  }
  caveats: string[]
  regional_confidence: { continent: string | null; tier: string; note: string; held_out_recall?: number }
  suggested_crops: string[] | null
}

// Render signed coordinates with the correct hemisphere (N/S, E/W).
function formatCoordinate(value: number, axis: 'lat' | 'lon') {
  const hemisphere = axis === 'lat'
    ? (value < 0 ? 'S' : 'N')
    : (value < 0 ? 'W' : 'E')
  return `${Math.abs(value).toFixed(4)}°${hemisphere}`
}

// ─── Intersection hook ────────────────────────────────────────────────────────

function useInView(ref: React.RefObject<Element | null>, threshold = 0.12) {
  const [vis, setVis] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) setVis(true) }, { threshold })
    obs.observe(el)
    return () => obs.disconnect()
  }, [ref, threshold])
  return vis
}

// ─── Nav ──────────────────────────────────────────────────────────────────────

function Nav() {
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)
  useEffect(() => {
    const h = () => setScrolled(window.scrollY > 50)
    window.addEventListener('scroll', h)
    return () => window.removeEventListener('scroll', h)
  }, [])
  const links = ['Mission', 'Technology', 'Soil Scan', 'Enterprise', 'Farmers', 'Contact']
  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'nav-blur' : ''}`}>
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <a href="#mission" className="flex items-center gap-2.5 shrink-0">
          <div className="w-7 h-7 rounded-md bg-green-500 flex items-center justify-center glow-green">
            <svg viewBox="0 0 24 24" className="w-4 h-4 fill-black">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zM8 17.5v-11l9 5.5-9 5.5z"/>
            </svg>
          </div>
          <span style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-xl font-bold tracking-wider text-white">
            AGRI<span className="text-green-400">SHIELD</span>
          </span>
        </a>
        <div className="hidden md:flex items-center gap-7">
          {links.map(l => (
            <a key={l} href={`#${l.toLowerCase().replace(' ', '-')}`}
              style={{ fontFamily: "'JetBrains Mono',monospace" }}
              className="text-xs text-slate-400 hover:text-green-400 transition-colors tracking-widest uppercase">
              {l}
            </a>
          ))}
        </div>
        <div className="hidden md:flex items-center gap-3">
          <button type="button" disabled title="Login is not available in this prototype" style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 border border-white/10 px-4 py-2 rounded opacity-70 cursor-not-allowed">
            Log In · Soon
          </button>
          <a href="#contact" style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs bg-green-500 text-black px-4 py-2 rounded font-semibold hover:bg-green-400 transition-colors glow-green">
            Contact Us
          </a>
        </div>
        <button className="md:hidden text-slate-400" onClick={() => setOpen(o => !o)}>
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16M4 18h16"/>
          </svg>
        </button>
      </div>
      {open && (
        <div className="md:hidden glass border-t border-white/5 px-6 py-4 flex flex-col gap-4">
          {links.map(l => (
            <a key={l} href={`#${l.toLowerCase().replace(' ', '-')}`} onClick={() => setOpen(false)}
              style={{ fontFamily: "'JetBrains Mono',monospace" }}
              className="text-xs text-slate-400 tracking-widest uppercase">{l}</a>
          ))}
        </div>
      )}
    </nav>
  )
}

// ─── Satellite Map SVG ────────────────────────────────────────────────────────

function SatelliteMap() {
  const [tick, setTick] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setTick(n => n + 1), 80)
    return () => clearInterval(t)
  }, [])

  const scanY = (tick * 3) % 520
  const fields = [
    { x: 115, y: 155, w: 145, h: 92, r: 18, fill: 'rgba(34,197,94,0.16)', stroke: 'rgba(34,197,94,0.5)', health: 84 },
    { x: 285, y: 128, w: 112, h: 102, r: 14, fill: 'rgba(245,158,11,0.14)', stroke: 'rgba(245,158,11,0.42)', health: 51 },
    { x: 415, y: 168, w: 132, h: 82, r: 16, fill: 'rgba(239,68,68,0.12)', stroke: 'rgba(239,68,68,0.38)', health: 29 },
    { x: 196, y: 272, w: 162, h: 72, r: 14, fill: 'rgba(34,197,94,0.11)', stroke: 'rgba(34,197,94,0.32)', health: 72 },
    { x: 368, y: 258, w: 104, h: 78, r: 12, fill: 'rgba(245,158,11,0.11)', stroke: 'rgba(245,158,11,0.3)', health: 44 },
  ]
  const pins = [
    { x: 155, y: 196, lat: '43.7614°N', lon: '79.3832°W', delay: '0s' },
    { x: 342, y: 176, lat: '43.7598°N', lon: '79.3701°W', delay: '0.7s' },
    { x: 308, y: 298, lat: '43.7521°N', lon: '79.3762°W', delay: '1.4s' },
  ]

  return (
    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 700 520" preserveAspectRatio="xMidYMid slice">
      <defs>
        <linearGradient id="scanGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="transparent"/>
          <stop offset="30%" stopColor="rgba(34,197,94,0.08)"/>
          <stop offset="50%" stopColor="rgba(34,197,94,0.85)"/>
          <stop offset="70%" stopColor="rgba(34,197,94,0.08)"/>
          <stop offset="100%" stopColor="transparent"/>
        </linearGradient>
        <linearGradient id="trailGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="transparent"/>
          <stop offset="100%" stopColor="rgba(34,197,94,0.18)"/>
        </linearGradient>
        <radialGradient id="glow1" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="rgba(34,197,94,0.12)"/>
          <stop offset="100%" stopColor="transparent"/>
        </radialGradient>
      </defs>

      <rect width="700" height="520" fill="#080b0f"/>

      {/* Ambient terrain patches */}
      {Array.from({ length: 22 }).map((_, i) => (
        <rect key={i}
          x={(Math.sin(i * 53.7) * 0.5 + 0.5) * 660 + 20}
          y={(Math.cos(i * 31.3) * 0.5 + 0.5) * 480 + 20}
          width={35 + (i % 5) * 11} height={24 + (i % 4) * 9}
          fill={`rgba(${i % 3 === 0 ? '12,28,16' : i % 3 === 1 ? '16,32,14' : '10,24,12'},0.7)`}
          rx="3"/>
      ))}

      {/* Grid */}
      {[80,160,240,320,400,480].map(y => <line key={y} x1="0" y1={y} x2="700" y2={y} stroke="rgba(34,197,94,0.09)" strokeWidth="0.5"/>)}
      {[100,200,300,400,500,600].map(x => <line key={x} x1={x} y1="0" x2={x} y2="520" stroke="rgba(34,197,94,0.09)" strokeWidth="0.5"/>)}

      {/* Ambient radial glow */}
      <ellipse cx="310" cy="230" rx="240" ry="160" fill="url(#glow1)"/>

      {/* Field polygons */}
      {fields.map((f, i) => (
        <g key={i}>
          <rect x={f.x} y={f.y} width={f.w} height={f.h} rx={f.r} fill={f.fill} stroke={f.stroke} strokeWidth="1"/>
          <rect x={f.x + 5} y={f.y + f.h - 13} width={(f.w - 10) * f.health / 100} height="5"
            fill={f.health > 65 ? 'rgba(34,197,94,0.75)' : f.health > 40 ? 'rgba(245,158,11,0.75)' : 'rgba(239,68,68,0.75)'}
            rx="2.5"/>
          <rect x={f.x + 5} y={f.y + f.h - 13} width={f.w - 10} height="5"
            fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="0.5" rx="2.5"/>
        </g>
      ))}

      {/* Contour rings */}
      <ellipse cx="310" cy="222" rx="185" ry="115" fill="none" stroke="rgba(34,197,94,0.1)" strokeWidth="1"/>
      <ellipse cx="310" cy="222" rx="135" ry="80" fill="none" stroke="rgba(34,197,94,0.08)" strokeWidth="0.8"/>
      <ellipse cx="310" cy="222" rx="82" ry="48" fill="none" stroke="rgba(34,197,94,0.06)" strokeWidth="0.6"/>

      {/* GPS pins */}
      {pins.map((p, i) => (
        <g key={i}>
          <circle cx={p.x} cy={p.y} r="4" fill="#22c55e"/>
          <circle cx={p.x} cy={p.y} r="4" fill="rgba(34,197,94,0.45)">
            <animate attributeName="r" values="4;14;4" dur="2s" begin={p.delay} repeatCount="indefinite"/>
            <animate attributeName="opacity" values="0.8;0;0.8" dur="2s" begin={p.delay} repeatCount="indefinite"/>
          </circle>
          <text x={p.x + 9} y={p.y - 5} fill="rgba(34,197,94,0.9)" fontSize="7.5" fontFamily="JetBrains Mono,monospace">{p.lat}</text>
          <text x={p.x + 9} y={p.y + 5} fill="rgba(34,197,94,0.65)" fontSize="7.5" fontFamily="JetBrains Mono,monospace">{p.lon}</text>
        </g>
      ))}

      {/* Scan line */}
      <rect x="0" y={scanY} width="700" height="2" fill="url(#scanGrad)" opacity="0.7"/>
      <rect x="0" y={Math.max(0, scanY - 45)} width="700" height="45" fill="url(#trailGrad)" opacity="0.2"/>

      {/* Corner brackets */}
      {([[12,12,1,1],[688,12,-1,1],[12,508,1,-1],[688,508,-1,-1]] as [number,number,number,number][]).map(([cx,cy,sx,sy], i) => (
        <g key={i}>
          <line x1={cx} y1={cy} x2={cx + sx * 14} y2={cy} stroke="rgba(34,197,94,0.55)" strokeWidth="1.5"/>
          <line x1={cx} y1={cy} x2={cx} y2={cy + sy * 14} stroke="rgba(34,197,94,0.55)" strokeWidth="1.5"/>
        </g>
      ))}

      {/* Status tag */}
      <rect x="16" y="16" width="140" height="22" fill="rgba(8,11,15,0.75)" rx="4"/>
      <circle cx="28" cy="27" r="3.5" fill="#22c55e">
        <animate attributeName="opacity" values="1;0.25;1" dur="1.4s" repeatCount="indefinite"/>
      </circle>
      <text x="38" y="31" fill="rgba(34,197,94,0.9)" fontSize="8.5" fontFamily="JetBrains Mono,monospace">SIMULATED · MAP PREVIEW</text>

      {/* Scale */}
      <g transform="translate(28, 492)">
        <line x1="0" y1="0" x2="64" y2="0" stroke="rgba(255,255,255,0.35)" strokeWidth="0.8"/>
        <line x1="0" y1="-4" x2="0" y2="4" stroke="rgba(255,255,255,0.35)" strokeWidth="0.8"/>
        <line x1="64" y1="-4" x2="64" y2="4" stroke="rgba(255,255,255,0.35)" strokeWidth="0.8"/>
        <text x="32" y="-7" textAnchor="middle" fill="rgba(255,255,255,0.35)" fontSize="7" fontFamily="JetBrains Mono,monospace">2.4 km</text>
      </g>

      {/* NDVI legend */}
      <g transform="translate(545, 458)">
        <text x="0" y="0" fill="rgba(255,255,255,0.35)" fontSize="7" fontFamily="JetBrains Mono,monospace">NDVI</text>
        {[['rgba(34,197,94,0.75)','HI'],['rgba(245,158,11,0.75)','MD'],['rgba(239,68,68,0.75)','LO']].map(([c,l], i) => (
          <g key={l} transform={`translate(${i * 40}, 6)`}>
            <rect width="9" height="9" fill={c} rx="1"/>
            <text x="13" y="8" fill="rgba(255,255,255,0.35)" fontSize="7" fontFamily="JetBrains Mono,monospace">{l}</text>
          </g>
        ))}
      </g>
    </svg>
  )
}

// ─── Live Telemetry Card ──────────────────────────────────────────────────────

function TelemetryCard() {
  const [d, setD] = useState({ moisture: 34.2, temp: 18.7, carbon: 2.84, b8: 0.712, b11: 0.341 })
  useEffect(() => {
    const t = setInterval(() => {
      setD(p => ({
        moisture: +(p.moisture + (Math.random() - 0.5) * 0.3).toFixed(1),
        temp: +(p.temp + (Math.random() - 0.5) * 0.15).toFixed(1),
        carbon: +(p.carbon + (Math.random() - 0.5) * 0.015).toFixed(2),
        b8: +(p.b8 + (Math.random() - 0.5) * 0.004).toFixed(3),
        b11: +(p.b11 + (Math.random() - 0.5) * 0.003).toFixed(3),
      }))
    }, 2000)
    return () => clearInterval(t)
  }, [])
  const rows = [
    { label: 'GPS', value: '43.7614°N 79.3832°W', cls: 'text-green-400' },
    { label: 'MOISTURE', value: `${d.moisture}%`, cls: 'text-sky-400' },
    { label: 'TEMP', value: `${d.temp}°C`, cls: 'text-amber-400' },
    { label: 'B8 NIR', value: String(d.b8), cls: 'text-green-400' },
    { label: 'B11 SWIR', value: String(d.b11), cls: 'text-green-300' },
    { label: 'C STOCK', value: `${d.carbon} t/ha`, cls: 'text-amber-300' },
  ]
  return (
    <div className="glass glow-green rounded-xl p-4 w-68 animate-float" style={{ width: 268 }}>
      <div className="flex items-center gap-2 mb-3">
        <div className="relative w-2 h-2">
          <div className="w-2 h-2 rounded-full bg-green-500"/>
          <div className="absolute inset-0 rounded-full bg-green-500 animate-ping opacity-50"/>
        </div>
        <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 tracking-widest">SIMULATED TELEMETRY</span>
        <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="ml-auto text-xs text-slate-500">12:47:33Z</span>
      </div>
      <div className="space-y-1.5">
        {rows.map(r => (
          <div key={r.label} className="flex justify-between py-1 border-b border-white/5">
            <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500">{r.label}</span>
            <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className={`text-xs font-medium ${r.cls} transition-all duration-700`}>{r.value}</span>
          </div>
        ))}
      </div>
      <div className="mt-3 pt-2 border-t border-white/5">
        <div className="flex justify-between mb-1">
          <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500">SOIL HEALTH</span>
          <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400">84 / 100</span>
        </div>
        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
          <div className="h-full w-[84%] bg-gradient-to-r from-green-600 to-green-400 rounded-full"/>
        </div>
      </div>
    </div>
  )
}

// ─── Hero ─────────────────────────────────────────────────────────────────────

function Hero() {
  return (
    <section id="hero" className="relative min-h-screen flex flex-col overflow-hidden">
      <div className="absolute inset-0">
        <SatelliteMap/>
        <div className="absolute inset-0 bg-gradient-to-r from-[#080b0f] via-[#080b0f]/80 to-[#080b0f]/20"/>
        <div className="absolute inset-0 bg-gradient-to-t from-[#080b0f] via-transparent to-transparent"/>
      </div>

      <div className="relative z-10 flex-1 flex flex-col justify-center max-w-7xl mx-auto w-full px-6 pt-20">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-2 glass-green rounded-full px-4 py-1.5 mb-8">
            <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"/>
            <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 tracking-widest">AI SOIL INTELLIGENCE PLATFORM</span>
          </div>

          <h1 style={{ fontFamily: "'Barlow Condensed',sans-serif" }}
            className="text-[clamp(56px,9vw,100px)] font-extrabold leading-none tracking-tight text-white mb-6">
            SEE BENEATH<br/>
            <span className="shimmer-text">THE SURFACE.</span>
          </h1>

          <p className="text-slate-300 text-lg leading-relaxed mb-10 max-w-xl font-light">
            AI-powered soil intelligence from satellites, climate data and machine learning — delivering real-time degradation alerts to farmers, NGOs and governments at planetary scale.
          </p>

          <div className="flex flex-wrap gap-4">
            <a href="#soil-scan">
              <button className="flex items-center gap-2 bg-green-500 hover:bg-green-400 text-black font-semibold px-7 py-3.5 rounded-lg transition-all glow-green"
                style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 13 }}>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4"/>
                </svg>
                Scan Your Soil
              </button>
            </a>
            <a href="#technology">
              <button className="flex items-center gap-2 border border-white/20 hover:border-green-500/50 text-white px-7 py-3.5 rounded-lg transition-all hover:bg-white/5"
                style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 13 }}>
                Explore Technology
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5l7 7-7 7"/>
                </svg>
              </button>
            </a>
          </div>

          <div className="flex gap-10 mt-14">
            {[
              { v: '2.3B', l: 'Hectares Monitored' },
              { v: '96.4%', l: 'Model Accuracy · Illustrative' },
              { v: '143', l: 'Countries' },
            ].map(s => (
              <div key={s.l}>
                <div style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-3xl font-bold text-green-400 text-glow">{s.v}</div>
                <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 tracking-widest mt-0.5">{s.l}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="absolute right-8 top-1/2 -translate-y-1/2 z-20 hidden lg:block">
        <TelemetryCard/>
      </div>

      <div className="relative z-10 pb-8 flex justify-center">
        <div className="flex flex-col items-center gap-2 opacity-35">
          <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs tracking-widest text-slate-500">SCROLL</span>
          <div className="w-px h-10 bg-gradient-to-b from-green-500/60 to-transparent"/>
        </div>
      </div>
    </section>
  )
}

// ─── Mission ──────────────────────────────────────────────────────────────────

function Mission() {
  const ref = useRef<HTMLDivElement>(null)
  const vis = useInView(ref)
  return (
    <section id="mission" className="py-32 px-6" ref={ref}>
      <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
        <div className={`fade-up ${vis ? 'visible' : ''}`}>
          <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 tracking-widest mb-4">// 01 MISSION</div>
          <h2 style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-[clamp(42px,6vw,72px)] font-bold text-white leading-none mb-6">
            SOIL IS THE<br/>WORLD'S MOST<br/><span className="text-green-400">OVERLOOKED</span><br/>RESOURCE.
          </h2>
          <p className="text-slate-400 leading-relaxed mb-5 font-light">
            33% of global soils are already degraded. Without intervention, we face a silent crisis threatening food security for 9 billion people by 2050. Existing monitoring is too slow, too expensive, and too fragmented to act at the speed the planet needs.
          </p>
          <p className="text-slate-400 leading-relaxed font-light">
            AgriShield changes that. By fusing satellite spectral data, climate signals and cutting-edge ML, we deliver actionable soil intelligence in minutes — empowering every actor in the agricultural ecosystem to make decisions rooted in real science.
          </p>
        </div>
        <div className={`fade-up delay-3 ${vis ? 'visible' : ''} grid grid-cols-2 gap-4`}>
          {[
            { num: '33%', label: 'Global soils degraded', sub: 'FAO 2024 assessment', cls: 'text-red-400' },
            { num: '9B', label: 'People at risk by 2050', sub: 'Food security impact', cls: 'text-amber-400' },
            { num: '$8.1T', label: 'Economic value at stake', sub: 'Annual agricultural GDP', cls: 'text-amber-300' },
            { num: '23min', label: 'Average scan time', sub: 'Satellite to insight', cls: 'text-green-400' },
          ].map(c => (
            <div key={c.label} className="glass rounded-xl p-5 border border-white/5 hover:border-green-500/20 transition-colors">
              <div style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className={`text-4xl font-bold ${c.cls} mb-1`}>{c.num}</div>
              <div className="text-sm text-white font-medium mb-1">{c.label}</div>
              <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500">{c.sub}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ─── Pipeline (How It Works) ──────────────────────────────────────────────────

function Pipeline() {
  const ref = useRef<HTMLDivElement>(null)
  const vis = useInView(ref)
  const [active, setActive] = useState(-1)
  useEffect(() => {
    if (!vis) return
    let i = 0
    const t = setInterval(() => { setActive(i % 5); i++ }, 1100)
    return () => clearInterval(t)
  }, [vis])

  const nodes = [
    { label: 'Data Repos', items: ['WoSIS Soil DB', 'LUCAS Survey', 'CSIRO SoilData'], color: '#38bdf8', bg: 'rgba(56,189,248,0.08)', border: 'rgba(56,189,248,0.22)' },
    { label: 'Satellite', items: ['Google Earth Engine', 'Sentinel-2 MSI', 'Copernicus Hub'], color: '#22c55e', bg: 'rgba(34,197,94,0.08)', border: 'rgba(34,197,94,0.22)' },
    { label: 'Climate', items: ['Open-Meteo API', 'ERA5 Reanalysis', 'MODIS Land'], color: '#a78bfa', bg: 'rgba(167,139,250,0.08)', border: 'rgba(167,139,250,0.22)' },
    { label: 'ML Model', items: ['XGBoost Ensemble', 'SHAP Explainer', 'Cross-validation'], color: '#f59e0b', bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.22)' },
    { label: 'Soil Intelligence', items: ['Health Score', 'Degradation Map', 'Action Plan'], color: '#4ade80', bg: 'rgba(74,222,128,0.1)', border: 'rgba(74,222,128,0.35)' },
  ]

  return (
    <section id="how-it-works" className="py-24 px-6 overflow-hidden" ref={ref}>
      <div className="max-w-7xl mx-auto">
        <div className={`text-center mb-14 fade-up ${vis ? 'visible' : ''}`}>
          <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 tracking-widest mb-4">// 02 HOW IT WORKS</div>
          <h2 style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-[clamp(40px,6vw,72px)] font-bold text-white">INTELLIGENCE PIPELINE</h2>
          <p className="text-slate-400 mt-4 max-w-xl mx-auto font-light">
            From raw satellite telemetry to precision soil insights in under 30 minutes.
          </p>
        </div>
        <div className="flex flex-col lg:flex-row items-stretch gap-0">
          {nodes.map((n, i) => (
            <div key={i} className="flex flex-col lg:flex-row items-center flex-1">
              <div
                className={`w-full lg:flex-1 rounded-xl p-5 border transition-all duration-500`}
                style={{
                  background: active === i ? n.bg : 'rgba(15,20,25,0.4)',
                  borderColor: active === i ? n.border : 'rgba(255,255,255,0.05)',
                  boxShadow: active === i ? `0 0 32px ${n.color}22` : 'none',
                }}
              >
                <div style={{ fontFamily: "'JetBrains Mono',monospace", color: n.color }} className="text-xs mb-3 tracking-wider">
                  STEP {String(i + 1).padStart(2, '0')}
                </div>
                <div style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-lg font-semibold text-white mb-3">{n.label}</div>
                {n.items.map(item => (
                  <div key={item} className="flex items-center gap-2 mb-1.5">
                    <div className="w-1 h-1 rounded-full" style={{ background: n.color }}/>
                    <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-400">{item}</span>
                  </div>
                ))}
              </div>
              {i < nodes.length - 1 && (
                <div className="flex items-center justify-center w-10 h-10 lg:w-14 lg:h-8 shrink-0">
                  <svg viewBox="0 0 40 16" className="w-10 rotate-90 lg:rotate-0">
                    <defs>
                      <linearGradient id={`fl${i}`} x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor={n.color}/>
                        <stop offset="100%" stopColor={nodes[i+1].color}/>
                      </linearGradient>
                    </defs>
                    <line x1="0" y1="8" x2="36" y2="8" stroke={`url(#fl${i})`} strokeWidth="1.5"
                      strokeDasharray="4 3"
                      style={{ animation: active >= i ? 'dash-flow 0.85s linear infinite' : 'none' }}/>
                    <polygon points="30,4 40,8 30,12" fill={nodes[i+1].color} opacity="0.75"/>
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ─── Technology ───────────────────────────────────────────────────────────────

const featureData = [
  { band: 'B8 NIR', val: 0.31 },
  { band: 'B11 SWIR', val: 0.24 },
  { band: 'Moisture', val: 0.18 },
  { band: 'B4 Red', val: 0.11 },
  { band: 'Temp °C', val: 0.08 },
  { band: 'B3 Green', val: 0.05 },
  { band: 'B2 Blue', val: 0.03 },
]

function Technology() {
  const ref = useRef<HTMLDivElement>(null)
  const vis = useInView(ref)

  const bands = [
    { id: 'B2', name: 'Blue', nm: '490nm', desc: 'Atmospheric correction, water mapping', color: '#60a5fa' },
    { id: 'B3', name: 'Green', nm: '560nm', desc: 'Vegetation health, chlorophyll peak', color: '#4ade80' },
    { id: 'B4', name: 'Red', nm: '665nm', desc: 'Chlorophyll absorption, NDVI numerator', color: '#f87171' },
    { id: 'B8', name: 'NIR', nm: '842nm', desc: 'Biomass, structure, soil contrast', color: '#c084fc' },
    { id: 'B11', name: 'SWIR-1', nm: '1610nm', desc: 'Moisture stress, clay mineral detection', color: '#fb923c' },
    { id: 'SM', name: 'Soil Moisture', nm: 'ERA5', desc: 'Volumetric water 0–30cm depth', color: '#38bdf8' },
    { id: 'T°', name: 'Temperature', nm: 'MODIS', desc: 'Land surface thermal, soil respiration', color: '#fbbf24' },
  ]

  return (
    <section id="technology" className="py-24 px-6" ref={ref}>
      <div className="max-w-7xl mx-auto">
        <div className={`mb-14 fade-up ${vis ? 'visible' : ''}`}>
          <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 tracking-widest mb-4">// 03 TECHNOLOGY</div>
          <h2 style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-[clamp(40px,6vw,72px)] font-bold text-white mb-4">SPECTRAL INTELLIGENCE</h2>
          <p className="text-slate-400 max-w-2xl font-light">
            Seven live data channels fused through a gradient-boosted ensemble model trained on 1.4M soil samples across 89 soil classification types.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-10">
          <div className="space-y-2.5">
            {bands.map((b, i) => (
              <div key={b.id}
                className={`fade-up delay-${Math.min(i + 1, 6)} ${vis ? 'visible' : ''} flex items-center gap-4 glass rounded-xl px-4 py-3 border border-white/5 hover:border-white/10 transition-colors`}>
                <div className="w-11 h-11 rounded-lg flex items-center justify-center shrink-0"
                  style={{ background: `${b.color}18`, border: `1px solid ${b.color}38` }}>
                  <span style={{ fontFamily: "'JetBrains Mono',monospace", color: b.color }} className="text-sm font-semibold">{b.id}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-lg font-semibold text-white">{b.name}</span>
                    <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500">{b.nm}</span>
                  </div>
                  <p style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-400 mt-0.5">{b.desc}</p>
                </div>
                <div className="w-16 h-1.5 bg-white/5 rounded-full overflow-hidden shrink-0">
                  <div className="h-full rounded-full" style={{ width: `${(featureData.find(f => b.id.includes(f.band.split(' ')[0]))?.val ?? 0.05) * 100 * 3}%`, background: b.color }}/>
                </div>
              </div>
            ))}
          </div>

          <div className={`fade-up delay-4 ${vis ? 'visible' : ''} glass rounded-2xl p-6 border border-white/5`}>
            <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 tracking-widest mb-1">MODEL EXPLAINABILITY</div>
            <h3 style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-2xl font-semibold text-white mb-5">Feature Importance (SHAP)</h3>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={featureData} layout="vertical" margin={{ left: 0, right: 24 }}>
                <CartesianGrid horizontal={false} stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3"/>
                <XAxis type="number" tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                  tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  axisLine={false} tickLine={false}/>
                <YAxis dataKey="band" type="category" width={68}
                  tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  axisLine={false} tickLine={false}/>
                <Tooltip
                  contentStyle={{ background: '#131920', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8 }}
                  labelStyle={{ color: '#94a3b8', fontFamily: 'JetBrains Mono', fontSize: 11 }}
                  itemStyle={{ color: '#22c55e', fontFamily: 'JetBrains Mono', fontSize: 11 }}
                  formatter={(v) => [`${((v as number) * 100).toFixed(1)}%`, 'Importance']}/>
                <Bar dataKey="val" fill="#22c55e" radius={[0, 4, 4, 0]} opacity={0.85}/>
              </BarChart>
            </ResponsiveContainer>

            <div className="grid grid-cols-3 gap-3 mt-4 pt-4 border-t border-white/5">
              {[
                { l: 'MAE', v: '0.42 t/ha', d: 'Mean Abs. Error' },
                { l: 'RMSE', v: '0.61 t/ha', d: 'Root Mean Sq.' },
                { l: 'R²', v: '0.887', d: 'Explained Var.' },
              ].map(m => (
                <div key={m.l} className="text-center">
                  <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mb-1">{m.d}</div>
                  <div style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-xl font-bold text-green-400">{m.v}</div>
                  <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-600">{m.l}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

// ─── Soil Scanner ─────────────────────────────────────────────────────────────

type SStep = 'idle' | 'gps' | 'satellite' | 'climate' | 'model' | 'result'
const SEQ: SStep[] = ['gps', 'satellite', 'climate', 'model', 'result']

function SoilScanner() {
  const [lat, setLat] = useState('43.7614')
  const [lon, setLon] = useState('-79.3832')
  const [step, setStep] = useState<SStep>('idle')
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<ScanResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [locating, setLocating] = useState(false)

  const steps = [
    { key: 'gps' as SStep, label: 'Resolving GPS', detail: 'Geocoding → WGS84 projection' },
    { key: 'satellite' as SStep, label: 'Fetching Sentinel-2', detail: 'GEE query · 10m resolution · B2,B3,B4,B8,B11' },
    { key: 'climate' as SStep, label: 'Pulling Climate Data', detail: 'WorldClim + OpenLandMap · texture, elevation, precip' },
    { key: 'model' as SStep, label: 'Running ML Inference', detail: 'Tuned XGBoost classifier' },
    { key: 'result' as SStep, label: 'Intelligence Ready', detail: 'Soil risk, crop fit, confidence' },
  ]

  // Drives the step-by-step UI while the REAL request is in flight -- this
  // is a perceived-progress animation (the /scan call is one network
  // request, not five separate stages), but it advances based on the real
  // fetch actually resolving, not a fixed fake timer. If the request errors,
  // this stops and 'error' is shown instead of faking a success.
  const run = useCallback(async (overrideLat?: number, overrideLon?: number) => {
    setStep('idle'); setProgress(0); setResult(null); setError(null)
    const latNum = overrideLat ?? parseFloat(lat)
    const lonNum = overrideLon ?? parseFloat(lon)
    if (Number.isNaN(latNum) || Number.isNaN(lonNum)) {
      setError('Enter valid numeric latitude and longitude.')
      return
    }

    let i = 0
    let cancelled = false
    const advanceVisual = () => {
      if (cancelled || i >= SEQ.length - 1) return  // stop before 'result' -- that's set on real completion
      setStep(SEQ[i])
      setProgress(((i + 1) / SEQ.length) * 100)
      i++
      setTimeout(advanceVisual, 500)
    }
    advanceVisual()

    try {
      const url = `${API_BASE_URL}/scan?lat=${latNum}&lon=${lonNum}`
      const resp = await fetch(url)
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error(body.detail || `Request failed (${resp.status})`)
      }
      const data: ScanResponse = await resp.json()
      cancelled = true
      setResult(data)
      setProgress(100)
      setStep('result')
    } catch (err) {
      cancelled = true
      setError(err instanceof Error ? err.message : 'Could not reach the soil-scan service.')
      setStep('idle')
      setProgress(0)
    }
  }, [lat, lon])

  // Real browser GPS -- asks the device for its actual current coordinates
  // (requires HTTPS in production; localhost is exempt, which is why this
  // works fine in dev). Fills the inputs AND immediately kicks off run()
  // with the fresh coordinates passed directly, rather than relying on
  // lat/lon state (which wouldn't have updated yet on this same tick).
  const useCurrentLocation = useCallback(() => {
    if (!('geolocation' in navigator)) {
      setError('This browser does not support location access. Enter coordinates manually.')
      return
    }
    setError(null)
    setLocating(true)
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords
        setLat(latitude.toFixed(6))
        setLon(longitude.toFixed(6))
        setLocating(false)
        run(latitude, longitude)
      },
      (err) => {
        setLocating(false)
        if (err.code === err.PERMISSION_DENIED) {
          setError('Location access was denied. Enter coordinates manually instead.')
        } else if (err.code === err.TIMEOUT) {
          setError('Location request timed out. Try again or enter coordinates manually.')
        } else {
          setError('Could not determine your location. Enter coordinates manually instead.')
        }
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 },
    )
  }, [run])

  const isResult = step === 'result' && result !== null
  const stepIdx = SEQ.indexOf(step)

  return (
    <section id="soil-scan" className="py-24 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-14">
          <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 tracking-widest mb-4">// 04 SOIL SCAN</div>
          <h2 style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-[clamp(40px,6vw,72px)] font-bold text-white">INSTANT SOIL ANALYSIS</h2>
          <p className="text-slate-400 mt-4 font-light">Enter any coordinates and watch the intelligence pipeline run in real time.</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 items-start">
          <div className="glass rounded-2xl p-6 border border-white/5">
            <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 tracking-widest mb-5">COORDINATE INPUT · DEFAULT DEMO LOCATION</div>

            <button onClick={useCurrentLocation} disabled={locating || (step !== 'idle' && !isResult)}
              className="w-full flex items-center justify-center gap-2 bg-green-500/10 hover:bg-green-500/20 disabled:opacity-60 border border-green-500/30 text-green-400 font-semibold py-3 rounded-lg transition-colors mb-3"
              style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 13 }}>
              📍 {locating ? 'Getting your location…' : 'Use My Current Location'}
            </button>
            <div className="flex items-center gap-3 mb-3">
              <div className="h-px flex-1 bg-white/10"/>
              <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-600">OR ENTER MANUALLY</span>
              <div className="h-px flex-1 bg-white/10"/>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-4">
              {[{ label: 'LATITUDE', val: lat, set: setLat, ph: '43.7614' }, { label: 'LONGITUDE', val: lon, set: setLon, ph: '-79.3832' }].map(f => (
                <div key={f.label}>
                  <label style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mb-1 block">{f.label}</label>
                  <input value={f.val} onChange={e => f.set(e.target.value)} placeholder={f.ph}
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-green-400 focus:outline-none focus:border-green-500/50"
                    style={{ fontFamily: "'JetBrains Mono',monospace" }}/>
                </div>
              ))}
            </div>
            <button onClick={() => run()} disabled={step !== 'idle' && !isResult}
              className="w-full bg-green-500 hover:bg-green-400 disabled:opacity-60 text-black font-semibold py-3 rounded-lg transition-colors glow-green mb-5"
              style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 13 }}>
              {step === 'idle' ? '⟶  Calculate' : isResult ? '⟶  Scan Again' : '⟶  Scanning…'}
            </button>

            {step !== 'idle' && (
              <div className="mb-4">
                <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-green-600 to-green-400 rounded-full transition-all duration-300" style={{ width: `${progress}%` }}/>
                </div>
                <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mt-1 text-right">{Math.round(progress)}%</div>
              </div>
            )}

            <div className="space-y-2">
              {steps.map((s, i) => {
                const done = stepIdx > i || isResult
                const active = SEQ[stepIdx] === s.key && !isResult
                return (
                  <div key={s.key} className={`flex items-start gap-3 p-3 rounded-lg transition-all duration-300 ${active ? 'bg-green-500/10 border border-green-500/20' : done ? 'opacity-60' : 'opacity-25'}`}>
                    <div className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${done ? 'bg-green-500' : active ? 'border border-green-500 bg-green-500/20' : 'border border-white/20'}`}>
                      {done && <svg className="w-3 h-3 text-black" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7"/></svg>}
                      {active && <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"/>}
                    </div>
                    <div>
                      <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-white">{s.label}</div>
                      <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mt-0.5">{s.detail}</div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          <div className={`transition-all duration-700 ${isResult ? 'opacity-100' : 'opacity-25 pointer-events-none'}`}>
            {error && (
              <div className="glass rounded-2xl border border-red-500/30 p-6 mb-4">
                <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-red-400">
                  ⚠ {error}
                </div>
              </div>
            )}
            {result && (
            <div className="glass rounded-2xl border border-green-500/20 overflow-hidden glow-green">
              <div className="bg-gradient-to-r from-green-900/25 to-transparent px-6 py-4 border-b border-green-500/10">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: result.color }}/>
                  <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 tracking-widest">
                    ANALYSIS COMPLETE · {formatCoordinate(result.location.latitude, 'lat')} {formatCoordinate(result.location.longitude, 'lon')}
                  </span>
                </div>
              </div>
              <div className="p-6">
                <div className="flex items-center justify-between mb-5">
                  <div>
                    <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mb-1">VERDICT</div>
                    <div style={{ fontFamily: "'Barlow Condensed',sans-serif", color: result.color }} className="text-4xl font-extrabold text-glow">
                      {result.headline}
                    </div>
                    <div style={{ fontFamily: "'JetBrains Mono',monospace", color: result.color }} className="text-xs uppercase mt-1">{result.verdict}</div>
                  </div>
                  <div className="relative w-24 h-24">
                    <svg viewBox="0 0 80 80" className="w-full h-full -rotate-90">
                      <circle cx="40" cy="40" r="34" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8"/>
                      <circle cx="40" cy="40" r="34" fill="none" stroke={result.color} strokeWidth="8"
                        strokeDasharray={`${2 * Math.PI * 34 * (result.confidence_pct / 100)} ${2 * Math.PI * 34}`}
                        strokeLinecap="round"/>
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center" style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 11, color: result.color }}>
                      {result.confidence_pct.toFixed(0)}%
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3 mb-4 p-3 bg-white/3 rounded-lg">
                  <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500">MODEL PROBABILITY</span>
                  <div className="flex-1 h-1.5 bg-white/5 rounded-full">
                    <div className="h-full bg-amber-400 rounded-full" style={{ width: `${result.confidence_pct}%` }}/>
                  </div>
                  <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-amber-400">{result.confidence_pct.toFixed(0)}%</span>
                </div>
                <p style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-[10px] text-slate-500 leading-relaxed mb-4">
                  Model probability for this prediction — not a guarantee of correctness or a calibrated measure of real-world reliability. Results may vary with data coverage and location.
                </p>

                <div className="mb-5">
                  <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mb-3">CONTEXT (API RESPONSE)</div>
                  {[
                    result.context.ndvi != null && { l: 'NDVI Vegetation Index', v: result.context.ndvi.toFixed(2), s: Math.min(100, Math.max(0, result.context.ndvi * 100)), c: 'bg-green-400' },
                    result.context.avg_temp_c != null && { l: 'Avg Temperature', v: `${result.context.avg_temp_c.toFixed(1)}°C`, s: Math.min(100, Math.max(0, (result.context.avg_temp_c + 10) * 2)), c: 'bg-sky-400' },
                    result.context.avg_annual_rainfall_mm != null && { l: 'Avg Annual Rainfall', v: `${result.context.avg_annual_rainfall_mm.toFixed(0)}mm`, s: Math.min(100, (result.context.avg_annual_rainfall_mm / 2000) * 100), c: 'bg-blue-400' },
                    result.predicted_oc_gkg != null && { l: 'Organic Carbon (experimental)', v: `${result.predicted_oc_gkg.toFixed(1)} g/kg`, s: Math.min(100, (result.predicted_oc_gkg / 50) * 100), c: 'bg-amber-400' },
                  ].filter(Boolean).map((f: any) => (
                    <div key={f.l} className="mb-3">
                      <div className="flex justify-between mb-1">
                        <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-400">{f.l}</span>
                        <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-white">{f.v}</span>
                      </div>
                      <div className="h-1 bg-white/5 rounded-full">
                        <div className={`h-full ${f.c} rounded-full`} style={{ width: `${f.s}%` }}/>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4 mb-3">
                  <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-amber-400 mb-2">⚡ RECOMMENDED ACTION</div>
                  <p style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-300 leading-relaxed">
                    {result.recommended_action}
                    {result.crop_note ? ` ${result.crop_note}` : ''}
                    {result.oc_note ? ` ${result.oc_note}` : ''}
                  </p>
                </div>

                {result.suggested_crops && result.suggested_crops.length > 0 && (
                  <div className="bg-white/3 rounded-xl p-4 mb-3">
                    <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mb-2">SUGGESTED CROPS</div>
                    <div className="flex flex-wrap gap-2">
                      {result.suggested_crops.map(c => (
                        <span key={c} className="text-xs px-2 py-1 rounded-full bg-green-500/10 text-green-400" style={{ fontFamily: "'JetBrains Mono',monospace" }}>{c}</span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="bg-white/3 rounded-xl p-4">
                  <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mb-2">
                    REGIONAL CONFIDENCE — {result.regional_confidence.continent ?? 'Unknown region'} ({result.regional_confidence.tier})
                  </div>
                  <p style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-400 leading-relaxed">
                    {result.regional_confidence.note}
                  </p>
                  {result.caveats.length > 0 && (
                    <p style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-amber-400 mt-2">
                      ⚠ {result.caveats.join(' ')}
                    </p>
                  )}
                </div>
              </div>
            </div>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}

// ─── Enterprise Dashboard ─────────────────────────────────────────────────────

const trendData = [
  { m: 'Jan', s: 71, t: 75 }, { m: 'Feb', s: 69, t: 75 }, { m: 'Mar', s: 72, t: 75 },
  { m: 'Apr', s: 68, t: 75 }, { m: 'May', s: 74, t: 75 }, { m: 'Jun', s: 76, t: 75 },
  { m: 'Jul', s: 73, t: 75 }, { m: 'Aug', s: 78, t: 75 }, { m: 'Sep', s: 77, t: 75 },
  { m: 'Oct', s: 80, t: 75 }, { m: 'Nov', s: 79, t: 75 }, { m: 'Dec', s: 82, t: 75 },
]
const regionalData = [
  { r: 'Sub-Saharan', s: 41 }, { r: 'South Asia', s: 58 }, { r: 'Lat. America', s: 67 },
  { r: 'Europe', s: 79 }, { r: 'N. America', s: 74 }, { r: 'East Asia', s: 62 },
]

function WorldMap() {
  return (
    <div className="relative w-full h-52 rounded-xl overflow-hidden bg-[#0a0e14]">
      <svg viewBox="0 0 800 360" className="w-full h-full" preserveAspectRatio="xMidYMid meet">
        {[60,120,180,240,300].map(y => <line key={y} x1="0" y1={y} x2="800" y2={y} stroke="rgba(34,197,94,0.05)" strokeWidth="0.5"/>)}
        {[80,160,240,320,400,480,560,640,720].map(x => <line key={x} x1={x} y1="0" x2={x} y2="360" stroke="rgba(34,197,94,0.05)" strokeWidth="0.5"/>)}
        <ellipse cx="160" cy="130" rx="80" ry="58" fill="rgba(34,197,94,0.18)" stroke="rgba(34,197,94,0.28)" strokeWidth="0.5"/>
        <ellipse cx="200" cy="242" rx="46" ry="72" fill="rgba(245,158,11,0.18)" stroke="rgba(245,158,11,0.28)" strokeWidth="0.5"/>
        <ellipse cx="390" cy="108" rx="40" ry="30" fill="rgba(34,197,94,0.22)" stroke="rgba(34,197,94,0.32)" strokeWidth="0.5"/>
        <ellipse cx="400" cy="212" rx="56" ry="82" fill="rgba(239,68,68,0.18)" stroke="rgba(239,68,68,0.28)" strokeWidth="0.5"/>
        <ellipse cx="560" cy="138" rx="102" ry="62" fill="rgba(245,158,11,0.16)" stroke="rgba(245,158,11,0.24)" strokeWidth="0.5"/>
        <ellipse cx="622" cy="252" rx="46" ry="30" fill="rgba(34,197,94,0.14)" stroke="rgba(34,197,94,0.22)" strokeWidth="0.5"/>
        <line x1="0" y1="180" x2="800" y2="180" stroke="rgba(255,255,255,0.07)" strokeWidth="0.5" strokeDasharray="4 4"/>
        {([[160,130,'rgba(34,197,94,0.8)'],[200,242,'rgba(245,158,11,0.8)'],[400,212,'rgba(239,68,68,0.8)'],[560,138,'rgba(245,158,11,0.7)']] as [number,number,string][]).map(([cx,cy,c], i) => (
          <g key={i}>
            <circle cx={cx} cy={cy} r="5.5" fill={c}/>
            <circle cx={cx} cy={cy} r="5.5" fill={c} opacity="0.35">
              <animate attributeName="r" values="5.5;18;5.5" dur={`${2.2 + i * 0.4}s`} repeatCount="indefinite"/>
              <animate attributeName="opacity" values="0.5;0;0.5" dur={`${2.2 + i * 0.4}s`} repeatCount="indefinite"/>
            </circle>
          </g>
        ))}
        <text x="24" y="16" fill="rgba(34,197,94,0.45)" fontSize="8.5" fontFamily="JetBrains Mono,monospace">GLOBAL SOIL INTELLIGENCE MAP · ILLUSTRATIVE</text>
      </svg>
      <div className="absolute bottom-2.5 right-3 flex gap-3">
        {[['rgba(34,197,94,0.75)','Good'],['rgba(245,158,11,0.75)','Moderate'],['rgba(239,68,68,0.75)','Critical']].map(([c,l]) => (
          <div key={l as string} className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full" style={{ background: c as string }}/>
            <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500">{l}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function Enterprise() {
  const ref = useRef<HTMLDivElement>(null)
  const vis = useInView(ref)
  const [tab, setTab] = useState('trends')

  return (
    <section id="enterprise" className="py-24 px-6" ref={ref}>
      <div className="max-w-7xl mx-auto">
        <div className={`mb-14 fade-up ${vis ? 'visible' : ''}`}>
          <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 tracking-widest mb-4">// 05 ENTERPRISE</div>
          <h2 style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-[clamp(40px,6vw,72px)] font-bold text-white">COMMAND CENTER</h2>
          <p className="text-slate-400 mt-4 max-w-2xl font-light">
            For NGOs, governments and agricultural organisations. Full analytics suite with global mapping, bulk processing and export controls.
          </p>
        </div>

        <div className={`fade-up delay-2 ${vis ? 'visible' : ''} glass rounded-2xl border border-white/5 overflow-hidden`}>
          {/* Title bar */}
          <div className="flex items-center gap-4 px-5 py-3 border-b border-white/5 bg-white/2">
            <div className="flex gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500/60"/><div className="w-3 h-3 rounded-full bg-amber-500/60"/><div className="w-3 h-3 rounded-full bg-green-500/60"/>
            </div>
            <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500">AgriShield Enterprise · Illustrative Dashboard</span>
            <div className="ml-auto flex items-center gap-2">
              <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-amber-400">SAMPLE DATA · NOT LIVE</span>
            </div>
          </div>

          <div className="grid lg:grid-cols-4 min-h-[580px]">
            {/* Sidebar */}
            <div className="border-r border-white/5 p-4 space-y-1">
              <div className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left bg-green-500/10 text-green-400">
                <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-sm">◉</span>
                <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs">Global Overview</span>
              </div>
              <div className="mt-4 pt-4 border-t border-white/5">
                <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mb-2">BULK UPLOAD</div>
                <div className="border border-dashed border-white/12 rounded-lg p-3 text-center opacity-80">
                  <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-400">CSV bulk analysis · Coming soon</div>
                  <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-600 mt-1">Upload is not connected yet</div>
                </div>
              </div>
            </div>

            {/* Main */}
            <div className="lg:col-span-3 p-5 space-y-5 overflow-hidden">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                  { l: 'Total Hectares', v: '847.2M', ch: '+12%' },
                  { l: 'Critical Zones', v: '1,247', ch: '-8%' },
                  { l: 'Avg Health Score', v: '63.4', ch: '+2.1' },
                  { l: 'Carbon Seq.', v: '4.2 Gt', ch: '+0.3' },
                ].map(k => (
                  <div key={k.l} className="bg-white/3 rounded-xl p-3 border border-white/5">
                    <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mb-1">{k.l}</div>
                    <div style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-2xl font-bold text-white">{k.v}</div>
                    <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs mt-1 text-green-400">{k.ch} vs last month</div>
                  </div>
                ))}
              </div>

              <WorldMap/>

              <div>
                <div className="flex gap-1 border-b border-white/5 mb-4">
                  {[['trends','Historical Trends'],['regional','Regional Comparison'],['model','Model Metrics']].map(([k,l]) => (
                    <button key={k} onClick={() => setTab(k)}
                      className={`text-xs px-4 py-2 border-b-2 transition-colors ${tab === k ? 'border-green-500 text-green-400' : 'border-transparent text-slate-500 hover:text-slate-300'}`}
                      style={{ fontFamily: "'JetBrains Mono',monospace" }}>
                      {l}
                    </button>
                  ))}
                  <span title="Export is not connected in this prototype" className="ml-auto text-xs text-slate-600 px-3 cursor-not-allowed" style={{ fontFamily: "'JetBrains Mono',monospace" }}>
                    Export CSV · Soon
                  </span>
                </div>

                {tab === 'trends' && (
                  <ResponsiveContainer width="100%" height={170}>
                    <AreaChart data={trendData}>
                      <defs>
                        <linearGradient id="ag" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#22c55e" stopOpacity={0.2}/>
                          <stop offset="95%" stopColor="#22c55e" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3"/>
                      <XAxis dataKey="m" tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false}/>
                      <YAxis domain={[60,90]} tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false}/>
                      <Tooltip contentStyle={{ background: '#131920', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontFamily: 'JetBrains Mono', fontSize: 11 }}/>
                      <Area type="monotone" dataKey="s" stroke="#22c55e" fill="url(#ag)" strokeWidth={2} dot={false}/>
                    </AreaChart>
                  </ResponsiveContainer>
                )}

                {tab === 'regional' && (
                  <ResponsiveContainer width="100%" height={170}>
                    <BarChart data={regionalData}>
                      <CartesianGrid horizontal stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3"/>
                      <XAxis dataKey="r" tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false}/>
                      <YAxis domain={[0,100]} tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false}/>
                      <Tooltip contentStyle={{ background: '#131920', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontFamily: 'JetBrains Mono', fontSize: 11 }}/>
                      <Bar dataKey="s" fill="#22c55e" radius={[4,4,0,0]} opacity={0.8}/>
                    </BarChart>
                  </ResponsiveContainer>
                )}

                {tab === 'model' && (
                  <div className="grid grid-cols-3 gap-4 py-4">
                    {[
                      { l: 'MAE', v: '0.42', u: 't C/ha', d: 'Mean Absolute Error — model accuracy in predicting soil carbon stock.' },
                      { l: 'RMSE', v: '0.61', u: 't C/ha', d: 'Root Mean Square Error — penalises large prediction deviations.' },
                      { l: 'R²', v: '0.887', u: '', d: 'Coefficient of determination — 88.7% of variance explained.' },
                    ].map(m => (
                      <div key={m.l} className="bg-white/3 rounded-xl p-4 border border-white/5">
                        <div style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-4xl font-bold text-green-400 mb-1">{m.v}<span className="text-xl text-slate-500 font-light"> {m.u}</span></div>
                        <div style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-base font-semibold text-white mb-2">{m.l}</div>
                        <p style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 leading-relaxed">{m.d}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

// ─── Farmer Section ───────────────────────────────────────────────────────────

function Farmers() {
  const ref = useRef<HTMLDivElement>(null)
  const vis = useInView(ref)
  const [tapped, setTapped] = useState<string | null>(null)

  return (
    <section id="farmers" className="py-24 px-6" ref={ref}>
      <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
        <div className={`fade-up ${vis ? 'visible' : ''}`}>
          <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 tracking-widest mb-4">// 06 FARMERS</div>
          <h2 style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-[clamp(40px,6vw,72px)] font-bold text-white leading-none mb-6">
            POWERFUL TECH,<br/><span className="text-green-400">RADICAL</span><br/>SIMPLICITY.
          </h2>
          <p className="text-xs text-amber-400 mb-3">Mobile UI concept preview — features shown here are illustrative, not a released app.</p>
          <p className="text-slate-400 leading-relaxed mb-6 font-light">
            Smallholder farmers in rural Kenya, Bangladesh or Brazil shouldn't need a data science degree to understand their soil. Our mobile interface is designed accessibility-first: GPS-in, result-out, under 60 seconds.
          </p>
          <div className="space-y-3">
            {['Works offline — syncs when signal returns', 'Available in 23 languages + audio readout', 'SMS fallback for feature phones', 'Designed for one-handed use in the field'].map(f => (
              <div key={f} className="flex items-center gap-3">
                <div className="w-5 h-5 rounded-full bg-green-500/20 border border-green-500/40 flex items-center justify-center shrink-0">
                  <svg className="w-3 h-3 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7"/></svg>
                </div>
                <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-sm text-slate-300">{f}</span>
              </div>
            ))}
          </div>
        </div>

        <div className={`fade-up delay-3 ${vis ? 'visible' : ''} flex justify-center`}>
          <div className="relative">
            <div className="w-72 bg-[#0f1419] rounded-[40px] border-2 border-white/10"
              style={{ boxShadow: '0 40px 80px rgba(0,0,0,0.7), 0 0 60px rgba(34,197,94,0.1)' }}>
              <div className="flex justify-center pt-3 pb-2">
                <div className="w-20 h-5 bg-black rounded-full"/>
              </div>
              <div className="px-5 pb-8">
                <div className="flex justify-between items-center mb-4">
                  <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500">9:41</span>
                  <div className="flex gap-1 items-center">
                    <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"/>
                    <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400">GPS READY</span>
                  </div>
                </div>
                <div className="text-center mb-6">
                  <div style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-2xl font-bold text-white">AGRI<span className="text-green-400">SHIELD</span></div>
                  <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mt-1">Soil Scanner · v2.4</div>
                </div>
                <button onClick={() => setTapped('gps')}
                  className={`w-full mb-4 py-5 rounded-2xl border-2 transition-all ${tapped === 'gps' ? 'bg-green-500 border-green-400 text-black glow-green' : 'border-green-500/40 text-green-400 bg-green-500/5'}`}>
                  <div className="text-3xl mb-1">📍</div>
                  <div style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-xl font-bold">SHARE MY LOCATION</div>
                </button>
                <div className="grid grid-cols-3 gap-2 mb-5">
                  {[{ icon: '🌱', label: 'Scan Soil', key: 'scan' }, { icon: '📊', label: 'My Fields', key: 'fields' }, { icon: '🔊', label: 'Audio', key: 'audio' }].map(a => (
                    <button key={a.key} onClick={() => setTapped(a.key)}
                      className={`py-4 rounded-xl border transition-all ${tapped === a.key ? 'bg-green-500/20 border-green-500/60 text-green-300' : 'border-white/10 text-slate-300 bg-white/3'}`}>
                      <div className="text-2xl mb-1">{a.icon}</div>
                      <div style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-xs font-semibold">{a.label}</div>
                    </button>
                  ))}
                </div>
                {tapped === 'scan' && (
                  <div className="bg-green-500/10 border border-green-500/20 rounded-xl p-3">
                    <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 mb-1">YOUR SOIL TODAY</div>
                    <div style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-4xl font-bold text-green-400">GOOD</div>
                    <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-400 mt-1">Score: 72 / 100 · Add compost this week</div>
                  </div>
                )}
                <div className="text-center mt-4">
                  <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-600">Offline · syncs when online</span>
                </div>
              </div>
            </div>
            <div className="absolute -right-8 top-20 glass-green rounded-xl px-3 py-2 animate-float">
              <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400">⚡ 47s avg</div>
              <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500">scan time</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

// ─── Data Sources ─────────────────────────────────────────────────────────────

function DataSources() {
  const ref = useRef<HTMLDivElement>(null)
  const vis = useInView(ref)
  const sources = [
    { n: 'WoSIS', full: 'World Soil Information Service', org: 'ISRIC', type: 'Soil Reference Data', c: '#22c55e' },
    { n: 'LUCAS', full: 'Land Use/Cover Area Survey', org: 'Eurostat', type: 'Ground Truth', c: '#38bdf8' },
    { n: 'CSIRO', full: 'Soil and Landscape Grid', org: 'Australia', type: 'Soil Properties', c: '#a78bfa' },
    { n: 'GEE', full: 'Google Earth Engine', org: 'Google', type: 'Satellite Processing', c: '#34d399' },
    { n: 'S-2', full: 'Sentinel-2 MSI', org: 'ESA Copernicus', type: '10m Imagery', c: '#60a5fa' },
    { n: 'ERA5', full: 'Climate Reanalysis', org: 'ECMWF', type: 'Climate Data', c: '#fbbf24' },
    { n: 'MODIS', full: 'Land Surface Temp', org: 'NASA', type: 'Thermal Data', c: '#f87171' },
    { n: 'O-M', full: 'Open-Meteo API', org: 'Open-Meteo.com', type: 'Weather Data', c: '#c084fc' },
  ]
  return (
    <section id="data-sources" className="py-24 px-6" ref={ref}>
      <div className="max-w-7xl mx-auto">
        <div className={`text-center mb-14 fade-up ${vis ? 'visible' : ''}`}>
          <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 tracking-widest mb-4">// 07 DATA SOURCES</div>
          <h2 style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-[clamp(40px,6vw,72px)] font-bold text-white">BUILT ON SCIENCE</h2>
          <p className="text-slate-400 mt-4 font-light">Powered by the world's most authoritative soil, satellite and climate datasets.</p>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {sources.map((s, i) => (
            <div key={s.n}
              className={`fade-up delay-${Math.min((i % 4) + 1, 6)} ${vis ? 'visible' : ''} glass rounded-xl p-4 border border-white/5 hover:border-white/10 transition-colors`}>
              <div className="flex items-start justify-between mb-3">
                <div style={{ fontFamily: "'Barlow Condensed',sans-serif", color: s.c }} className="text-2xl font-bold">{s.n}</div>
                <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-600 bg-white/5 px-2 py-0.5 rounded">{s.org}</div>
              </div>
              <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-white mb-1">{s.full}</div>
              <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500">{s.type}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ─── Contact ──────────────────────────────────────────────────────────────────

function Contact() {
  const ref = useRef<HTMLDivElement>(null)
  const vis = useInView(ref)
  const [sent, setSent] = useState(false)
  return (
    <section id="contact" className="py-24 px-6" ref={ref}>
      <div className="max-w-5xl mx-auto">
        <div className={`text-center mb-14 fade-up ${vis ? 'visible' : ''}`}>
          <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 tracking-widest mb-4">// 08 CONTACT</div>
          <h2 style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-[clamp(40px,6vw,72px)] font-bold text-white">
            START MONITORING<br/><span className="text-green-400">YOUR LAND.</span>
          </h2>
          <p className="text-slate-400 mt-4 font-light">AgriShield is a developing soil-intelligence prototype. Get in touch to discuss potential collaboration.</p>
        </div>

        <div className={`fade-up delay-2 ${vis ? 'visible' : ''} grid lg:grid-cols-2 gap-8`}>
          <div className="glass rounded-2xl p-6 border border-white/5">
            {sent ? (
              <div className="h-full flex flex-col items-center justify-center text-center py-12">
                <div className="w-16 h-16 rounded-full bg-green-500/20 border border-green-500/40 flex items-center justify-center mb-4 glow-green">
                  <svg className="w-8 h-8 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7"/></svg>
                </div>
                <h3 style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-3xl font-bold text-white mb-2">DEMO FORM ONLY</h3>
                <p style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-400">This prototype does not send or store messages.</p>
              </div>
            ) : (
              <form onSubmit={e => { e.preventDefault(); setSent(true) }} className="space-y-4">
                <p className="text-xs text-amber-400 border border-amber-500/20 rounded-lg p-3">Prototype form — submissions are not sent or stored.</p>
                <div className="grid grid-cols-2 gap-3">
                  {[{ l: 'FIRST NAME', p: 'Amara' }, { l: 'LAST NAME', p: 'Osei' }].map(f => (
                    <div key={f.l}>
                      <label style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mb-1 block">{f.l}</label>
                      <input className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:border-green-500/50" placeholder={f.p} style={{ fontFamily: "'JetBrains Mono',monospace" }}/>
                    </div>
                  ))}
                </div>
                {[{ l: 'ORGANISATION', p: 'FAO / NGO / Ministry of Agriculture' }, { l: 'EMAIL', p: 'amara@organisation.org' }].map(f => (
                  <div key={f.l}>
                    <label style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mb-1 block">{f.l}</label>
                    <input type={f.l === 'EMAIL' ? 'email' : 'text'} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:border-green-500/50" placeholder={f.p} style={{ fontFamily: "'JetBrains Mono',monospace" }}/>
                  </div>
                ))}
                <div>
                  <label style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mb-1 block">USE CASE</label>
                  <select className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-slate-400 focus:outline-none focus:border-green-500/50" style={{ fontFamily: "'JetBrains Mono',monospace" }}>
                    <option>Government / Policy</option><option>NGO / Development</option>
                    <option>Commercial Agriculture</option><option>Research / Academia</option><option>Carbon Markets</option>
                  </select>
                </div>
                <div>
                  <label style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 mb-1 block">MESSAGE</label>
                  <textarea rows={3} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:border-green-500/50 resize-none" placeholder="Tell us about your land area and monitoring goals…" style={{ fontFamily: "'JetBrains Mono',monospace" }}/>
                </div>
                <button type="submit" className="w-full bg-green-500 hover:bg-green-400 text-black font-semibold py-3.5 rounded-lg transition-colors glow-green" style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 13 }}>
                  Send Message →
                </button>
              </form>
            )}
          </div>

          <div className="space-y-4">
            {[
              { icon: '⚡', l: 'Quick Start', d: 'Enterprise pilots fully configured within 72 hours of onboarding.' },
              { icon: '🌍', l: 'Global Coverage', d: 'Monitoring data available for all agricultural land on Earth.' },
              { icon: '🔒', l: 'Data Sovereignty', d: 'All data processed within your jurisdiction. GDPR & ISO 27001 compliant.' },
              { icon: '📡', l: 'API Access', d: 'REST + GraphQL APIs for direct integration into existing systems.' },
            ].map(c => (
              <div key={c.l} className="glass rounded-xl p-4 border border-white/5 flex gap-4">
                <div className="text-2xl shrink-0 mt-1">{c.icon}</div>
                <div>
                  <div style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-lg font-semibold text-white mb-1">{c.l}</div>
                  <p style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-400 leading-relaxed">{c.d}</p>
                </div>
              </div>
            ))}
            <div className="glass rounded-xl p-5 border border-green-500/15 bg-green-500/5">
              <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 mb-3">BUILT FOR</div>
              <p className="text-xs text-slate-400 leading-relaxed">Designed with potential use cases across research, agriculture, NGOs and public-sector soil monitoring. No institutional partnerships or endorsements are implied.</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

// ─── Footer ───────────────────────────────────────────────────────────────────

function Footer() {
  return (
    <footer className="border-t border-white/5 py-12 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-4 gap-8 mb-10">
          <div>
            <div style={{ fontFamily: "'Barlow Condensed',sans-serif" }} className="text-xl font-bold text-white mb-3">AGRI<span className="text-green-400">SHIELD</span></div>
            <p style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-500 leading-relaxed mb-4">AI-powered soil intelligence from satellites and machine learning. Protecting Earth's most critical resource.</p>
          </div>
          {[
            { t: 'PLATFORM', ls: ['Soil Scanner', 'Enterprise Dashboard', 'API Documentation', 'Model Cards', 'Data Sources'] },
            { t: 'COMPANY', ls: ['About', 'Research', 'Careers', 'Press', 'Contact'] },
            { t: 'LEGAL', ls: ['Privacy Policy', 'Terms of Service', 'Data Processing'] },
          ].map(col => (
            <div key={col.t}>
              <div style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-green-400 tracking-widest mb-4">{col.t}</div>
              <div className="space-y-2">
                {col.ls.map(l => <span key={l} title="This page is not available yet" style={{ fontFamily: "'JetBrains Mono',monospace" }} className="block text-xs text-slate-600 cursor-default">{l} · Soon</span>)}
              </div>
            </div>
          ))}
        </div>
        <div className="border-t border-white/5 pt-6 flex flex-col md:flex-row justify-between items-center gap-4">
          <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-600">© 2024 AgriShield Technologies Inc. All rights reserved.</span>
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"/>
            <span style={{ fontFamily: "'JetBrains Mono',monospace" }} className="text-xs text-slate-600">Prototype · some features are not yet connected</span>
          </div>
        </div>
      </div>
    </footer>
  )
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  return (
    <div className="min-h-screen bg-[#080b0f]">
      <Nav/>
      <Hero/>
      <Mission/>
      <Pipeline/>
      <Technology/>
      <SoilScanner/>
      <Enterprise/>
      <Farmers/>
      <DataSources/>
      <Contact/>
      <Footer/>
    </div>
  )
}
