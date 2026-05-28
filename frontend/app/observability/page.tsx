"use client";

import { useTelemetry } from "@/hooks/useTelemetry";
import { QueueBacklogChart, ChartDataPoint } from "@/components/charts/QueueBacklogChart";
import { ThroughputChart } from "@/components/charts/ThroughputChart";
import { MetricCard, SectionHeader, StatusBadge, LoadingState } from "@/components/shared/Primitives";
import { Activity, Gauge, Server, AlertTriangle, Monitor, XCircle } from "lucide-react";
import { useEffect, useState } from "react";
import Link from "next/link";
import { Shield, Share2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
export default function ObservabilityPage() {
  const { queueMetrics, infrastructure, circuitBreakers, browserTelemetry } = useTelemetry();
  const [backlogData, setBacklogData] = useState<ChartDataPoint[]>([]);
  const [throughputData, setThroughputData] = useState<ChartDataPoint[]>([]);

  useEffect(() => {
    if (queueMetrics) {
      setBacklogData(prev => [...prev, { timestamp: queueMetrics.timestamp, value: queueMetrics.queueDepth }].slice(-20));
      setThroughputData(prev => [...prev, { timestamp: queueMetrics.timestamp, value: queueMetrics.processingThroughput }].slice(-20));
    }
  }, [queueMetrics]);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 flex flex-col font-sans">
      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="w-5 h-5 text-indigo-500" />
            <h1 className="font-bold tracking-widest text-zinc-100">HERALD <span className="text-zinc-500 font-mono text-xs ml-2 uppercase">Ops Console</span></h1>
          </div>
          <nav className="flex items-center gap-6 text-sm font-mono">
            <Link href="/" className="text-zinc-400 hover:text-zinc-200 transition-colors flex items-center gap-2"><Activity className="w-4 h-4"/> Dashboard</Link>
            <Link href="/observability" className="text-indigo-400 flex items-center gap-2 border-b-2 border-indigo-500 pb-1 pt-1"><Share2 className="w-4 h-4"/> Observability</Link>
            <Link href="/dlq" className="text-zinc-400 hover:text-zinc-200 transition-colors flex items-center gap-2"><AlertTriangle className="w-4 h-4"/> DLQ</Link>
          </nav>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-4 py-6">
        <SectionHeader title="Infrastructure Observability" description="Real-time telemetry and subsystem performance" />
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
           <MetricCard title="API Latency" value={`${infrastructure?.apiLatencyMs || 0}ms`} icon={Gauge} alert={(infrastructure?.apiLatencyMs || 0) > 1000} />
           <MetricCard title="Throughput" value={`${queueMetrics?.processingThroughput || 0}/s`} icon={Activity} />
           <MetricCard title="DLQ Pressure" value={queueMetrics?.dlqSize || 0} icon={AlertTriangle} alert={(queueMetrics?.dlqSize || 0) > 20} />
           <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 flex flex-col justify-center">
             <div className="text-xs font-mono text-zinc-400 uppercase mb-2">Degraded Mode</div>
             {infrastructure?.degradedModeActive ? 
               <StatusBadge status="DOWN" /> : 
               <StatusBadge status="HEALTHY" />
             }
           </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 h-[300px] flex flex-col">
            <h3 className="text-xs font-mono text-zinc-400 uppercase mb-4 flex items-center gap-2"><Activity className="w-4 h-4"/> Queue Backlog History</h3>
            <div className="flex-1 min-h-0">
               <QueueBacklogChart data={backlogData} />
            </div>
          </div>
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 h-[300px] flex flex-col">
            <h3 className="text-xs font-mono text-zinc-400 uppercase mb-4 flex items-center gap-2"><Gauge className="w-4 h-4"/> Worker Throughput</h3>
            <div className="flex-1 min-h-0">
               <ThroughputChart data={throughputData} />
            </div>
          </div>
        </div>

        <SectionHeader title="Browser Fleet Isolation" description="Real-time telemetry for headless browser fleet" />
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 mb-6">
          {browserTelemetry ? (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="flex flex-col p-3 rounded-lg border border-zinc-800 bg-zinc-950">
                <span className="text-[10px] text-zinc-500 font-mono uppercase mb-1 flex items-center gap-1"><Monitor className="w-3 h-3"/> Active Sessions</span>
                <span className="text-xl font-mono text-zinc-200">{browserTelemetry.activeSessions}</span>
              </div>
              <div className="flex flex-col p-3 rounded-lg border border-zinc-800 bg-zinc-950">
                <span className="text-[10px] text-zinc-500 font-mono uppercase mb-1">Launch Latency (avg)</span>
                <span className={`text-xl font-mono ${browserTelemetry.browserLaunchTimeMs > 2000 ? "text-yellow-400" : "text-zinc-200"}`}>{browserTelemetry.browserLaunchTimeMs}ms</span>
              </div>
              <div className="flex flex-col p-3 rounded-lg border border-zinc-800 bg-zinc-950">
                <span className="text-[10px] text-zinc-500 font-mono uppercase mb-1">Capture Latency (avg)</span>
                <span className="text-xl font-mono text-zinc-200">{browserTelemetry.screenshotDurationMs}ms</span>
              </div>
              <div className="flex flex-col p-3 rounded-lg border border-zinc-800 bg-zinc-950">
                <span className="text-[10px] text-zinc-500 font-mono uppercase mb-1">Memory Pressure</span>
                <div className="flex items-center gap-2">
                  <span className={`text-xl font-mono ${browserTelemetry.memoryPressurePercent > 85 ? "text-red-400" : "text-zinc-200"}`}>{browserTelemetry.memoryPressurePercent}%</span>
                  {browserTelemetry.memoryPressurePercent > 85 && <AlertTriangle className="w-4 h-4 text-red-500" />}
                </div>
              </div>
              <div className="flex flex-col p-3 rounded-lg border border-zinc-800 bg-zinc-950">
                <span className="text-[10px] text-zinc-500 font-mono uppercase mb-1">Timeout Frequency</span>
                <span className={`text-xl font-mono ${browserTelemetry.timeoutFrequencyRate > 5 ? "text-yellow-400" : "text-zinc-200"}`}>{browserTelemetry.timeoutFrequencyRate}/min</span>
              </div>
              <div className="flex flex-col p-3 rounded-lg border border-red-900/50 bg-red-950/20">
                <span className="text-[10px] text-zinc-500 font-mono uppercase mb-1">Isolation Failures / Crashes</span>
                <div className="flex items-center gap-2">
                   <span className={`text-xl font-mono ${browserTelemetry.isolationFailures > 0 ? "text-red-400" : "text-green-400"}`}>
                     {browserTelemetry.isolationFailures}
                   </span>
                   <span className="text-zinc-500 font-mono text-xs">/ {browserTelemetry.crashFrequencyRate}</span>
                   {browserTelemetry.isolationFailures > 0 && <XCircle className="w-4 h-4 text-red-500" />}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-32 flex items-center justify-center">
               <LoadingState message="Connecting to browser fleet..." />
            </div>
          )}
        </div>

        <SectionHeader title="Circuit Breakers" />
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
           {circuitBreakers.map(b => (
             <div key={b.name} className={`p-3 rounded border ${b.state === 'CLOSED' ? 'bg-zinc-900 border-zinc-800' : 'bg-red-950/20 border-red-900/50'}`}>
                <div className="text-[10px] font-mono text-zinc-500 uppercase mb-1">{b.name}</div>
                <StatusBadge status={b.state} />
                <div className="text-[10px] text-zinc-500 mt-2 font-mono">Failures: <span className={b.recentFailures > 0 ? "text-red-400" : ""}>{b.recentFailures}</span></div>
             </div>
           ))}
        </div>
      </main>
    </div>
  );
}
