"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { getMockTrace } from "@/services/mock-traces";
import { TraceEvent } from "@/types";
import { TraceHeader } from "@/components/traces/TraceHeader";
import { LinearTimeline } from "@/components/traces/LinearTimeline";
import { WorkerExecutionPanel } from "@/components/traces/WorkerExecutionPanel";
import { TraceEventLog } from "@/components/traces/TraceEventLog";
import { LoadingState } from "@/components/shared/Primitives";
import { Shield, Activity, Share2, AlertTriangle, ArrowLeft } from "lucide-react";
import Link from "next/link";

export default function TraceVisualizationPage() {
  const params = useParams();
  const traceId = params.trace_id as string;
  const [trace, setTrace] = useState<TraceEvent | null>(null);

  useEffect(() => {
    // Determine a random domain based on trace ID string length to make it deterministic for mock
    const mockDomain = traceId.length % 2 === 0 ? "secure-login-apple.com" : "update-billing.net";
    const data = getMockTrace(traceId, mockDomain);
    
    const timer = setTimeout(() => setTrace(data), 500);
    return () => clearTimeout(timer);
  }, [traceId]);

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
            <Link href="/observability" className="text-zinc-400 hover:text-zinc-200 transition-colors flex items-center gap-2"><Share2 className="w-4 h-4"/> Observability</Link>
            <Link href="/dlq" className="text-zinc-400 hover:text-zinc-200 transition-colors flex items-center gap-2"><AlertTriangle className="w-4 h-4"/> DLQ</Link>
          </nav>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-4 py-6 flex flex-col">
        <div className="mb-4">
          <Link href="/observability" className="text-xs font-mono text-zinc-500 hover:text-indigo-400 transition-colors flex items-center gap-1 w-max">
            <ArrowLeft className="w-3 h-3" /> Back to Observability
          </Link>
        </div>

        {!trace ? (
          <div className="h-[60vh] flex items-center justify-center">
            <LoadingState message="Retrieving distributed trace payload..." />
          </div>
        ) : (
          <div className="flex-1 flex flex-col min-h-0">
            <TraceHeader trace={trace} />
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
              
              {/* Left Column: Lineage (70%) */}
              <div className="lg:col-span-2 h-full min-h-[500px]">
                <LinearTimeline spans={trace.spans} />
              </div>

              {/* Right Column: Execution Panels (30%) */}
              <div className="h-full flex flex-col gap-6 min-h-[500px]">
                <div className="h-[250px] shrink-0">
                  <WorkerExecutionPanel spans={trace.spans} />
                </div>
                <div className="flex-1 min-h-[250px]">
                  <TraceEventLog spans={trace.spans} />
                </div>
              </div>

            </div>
          </div>
        )}
      </main>
    </div>
  );
}
