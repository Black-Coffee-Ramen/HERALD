import { TraceEvent } from "@/types";
import { StatusBadge, MetricCard } from "@/components/shared/Primitives";
import { Badge } from "@/components/ui/badge";
import { ShieldAlert, AlertTriangle, CheckCircle, Activity, Timer, RefreshCcw, Network } from "lucide-react";

export function TraceHeader({ trace }: { trace: TraceEvent }) {
  const getVerdictBadge = () => {
    switch (trace.verdict) {
      case "MALICIOUS": return <Badge variant="destructive" className="bg-red-900 text-red-100 border-red-800">MALICIOUS</Badge>;
      case "SUSPICIOUS": return <Badge className="bg-yellow-700 text-yellow-100 border-yellow-800">SUSPICIOUS</Badge>;
      default: return <Badge className="bg-green-900 text-green-100 border-green-800">BENIGN</Badge>;
    }
  };

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 mb-6">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <Network className="w-5 h-5 text-indigo-400" />
            <h1 className="text-xl font-bold text-zinc-100 font-mono tracking-wide">TRACE: {trace.traceId}</h1>
          </div>
          <div className="flex items-center gap-3 text-xs font-mono text-zinc-400">
            <span>Domain: <strong className="text-zinc-200">{trace.domain}</strong></span>
            <span>•</span>
            <span>Outcome: <span className={trace.outcome === "VERDICTED" ? "text-green-400" : "text-yellow-400"}>{trace.outcome}</span></span>
          </div>
        </div>

        <div className="flex flex-col items-end gap-2">
           <div className="flex items-center gap-3">
              {trace.isDegraded && <Badge variant="outline" className="border-yellow-700 text-yellow-500 font-mono text-[10px] bg-yellow-950/20">DEGRADED STREAM</Badge>}
              {getVerdictBadge()}
           </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-zinc-950 border border-zinc-800 p-3 rounded-lg flex flex-col justify-between">
           <div className="text-[10px] text-zinc-500 font-mono uppercase flex items-center gap-1 mb-1"><Timer className="w-3 h-3"/> Total Duration</div>
           <div className="text-lg font-mono text-zinc-200">{trace.overallDurationMs}ms</div>
        </div>
        <div className="bg-zinc-950 border border-zinc-800 p-3 rounded-lg flex flex-col justify-between">
           <div className="text-[10px] text-zinc-500 font-mono uppercase flex items-center gap-1 mb-1"><RefreshCcw className="w-3 h-3"/> Total Retries</div>
           <div className={`text-lg font-mono ${trace.totalRetries > 0 ? "text-yellow-400" : "text-green-400"}`}>{trace.totalRetries}</div>
        </div>
        <div className="bg-zinc-950 border border-zinc-800 p-3 rounded-lg flex flex-col justify-between">
           <div className="text-[10px] text-zinc-500 font-mono uppercase flex items-center gap-1 mb-1"><Activity className="w-3 h-3"/> Worker Spans</div>
           <div className="text-lg font-mono text-indigo-400">{trace.spans.length}</div>
        </div>
        <div className="bg-zinc-950 border border-zinc-800 p-3 rounded-lg flex flex-col justify-between">
           <div className="text-[10px] text-zinc-500 font-mono uppercase flex items-center gap-1 mb-1"><AlertTriangle className="w-3 h-3"/> Errors</div>
           <div className={`text-lg font-mono ${trace.spans.filter(s => s.status === 'ERROR').length > 0 ? "text-red-400" : "text-green-400"}`}>{trace.spans.filter(s => s.status === 'ERROR').length}</div>
        </div>
      </div>
    </div>
  );
}
