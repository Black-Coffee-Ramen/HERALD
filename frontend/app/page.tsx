"use client";

import { useTelemetry } from "@/hooks/useTelemetry";
import { LiveThreatFeed } from "@/components/threat-feed/LiveThreatFeed";
import { SystemHealth } from "@/components/system-health/SystemHealth";
import { Shield, Activity, Share2, AlertTriangle } from "lucide-react";
import Link from "next/link";

export default function Dashboard() {
  const { threatEvents, queueMetrics, infrastructure, circuitBreakers, connectionState } = useTelemetry();

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 flex flex-col font-sans">
      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="w-5 h-5 text-indigo-500" />
            <h1 className="font-bold tracking-widest text-zinc-100">HERALD <span className="text-zinc-500 font-mono text-xs ml-2 uppercase">Ops Console</span></h1>
          </div>
          <nav className="flex items-center gap-6 text-sm font-mono">
            <Link href="/" className="text-indigo-400 flex items-center gap-2 border-b-2 border-indigo-500 pb-1 pt-1"><Activity className="w-4 h-4"/> Dashboard</Link>
            <Link href="/observability" className="text-zinc-400 hover:text-zinc-200 transition-colors flex items-center gap-2"><Share2 className="w-4 h-4"/> Observability</Link>
            <Link href="/dlq" className="text-zinc-400 hover:text-zinc-200 transition-colors flex items-center gap-2"><AlertTriangle className="w-4 h-4"/> DLQ</Link>
          </nav>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-4 py-6">
        <SystemHealth 
          queue={queueMetrics} 
          infra={infrastructure} 
          breakers={circuitBreakers} 
        />
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            <LiveThreatFeed events={threatEvents} connectionState={connectionState ?? "DISCONNECTED"} />
          </div>
          <div className="flex flex-col gap-4">
            {/* Future Placeholder for mini charts or recent alerts */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 h-full flex flex-col items-center justify-center text-center">
              <Activity className="w-8 h-8 text-zinc-700 mb-2" />
              <h3 className="text-zinc-400 font-mono text-sm mb-1">Queue Backlog History</h3>
              <p className="text-zinc-600 text-xs">Waiting for detailed chart component implementation...</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
