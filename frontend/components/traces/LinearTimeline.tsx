import { TraceSpan } from "@/types";
import { format } from "date-fns";
import { CheckCircle2, AlertCircle, CircleDashed, Clock, Server, ArrowDownRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export function LinearTimeline({ spans }: { spans: TraceSpan[] }) {
  // Sort spans by start time just in case
  const sortedSpans = [...spans].sort((a, b) => new Date(a.startTime).getTime() - new Date(b.startTime).getTime());

  const getStatusIcon = (status: string, state: string) => {
    if (state === "TIMEOUT") return <AlertCircle className="w-5 h-5 text-red-500" />;
    if (state === "DEGRADED") return <AlertCircle className="w-5 h-5 text-yellow-500" />;
    if (status === "ERROR") return <AlertCircle className="w-5 h-5 text-red-500" />;
    if (status === "PENDING") return <CircleDashed className="w-5 h-5 text-yellow-500 animate-spin-slow" />;
    return <CheckCircle2 className="w-5 h-5 text-green-500" />;
  };

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 h-full flex flex-col">
      <h3 className="text-xs font-mono text-zinc-400 uppercase mb-6 tracking-wider flex items-center gap-2">
        <ArrowDownRight className="w-4 h-4" />
        Processing Lineage
      </h3>
      
      <div className="relative border-l border-zinc-800 ml-2.5 space-y-8 flex-1 overflow-auto pr-4">
        {sortedSpans.map((span, idx) => (
          <div key={span.id} className="relative pl-8">
            <div className="absolute -left-2.5 top-0.5 bg-zinc-900 rounded-full">
              {getStatusIcon(span.status, span.executionState)}
            </div>
            
            <div className="flex flex-col gap-2">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-zinc-800/50 pb-2">
                <div className="flex items-center gap-3">
                  <span className={`text-sm font-bold font-mono ${span.status === 'ERROR' ? 'text-red-400' : 'text-zinc-200'}`}>
                    {span.name}
                  </span>
                  <Badge variant="outline" className="font-mono text-[9px] h-4 border-zinc-700 text-zinc-400 px-1.5 flex items-center gap-1 bg-zinc-950">
                     <Server className="w-3 h-3"/> {span.workerType}
                  </Badge>
                </div>
                <div className="flex items-center gap-4 text-[10px] font-mono">
                  <span className="text-zinc-500 flex items-center gap-1" title="Queue Wait Time">
                    Q: <span className={span.queueWaitMs > 500 ? "text-yellow-500" : "text-zinc-300"}>{span.queueWaitMs}ms</span>
                  </span>
                  <span className="text-zinc-500 flex items-center gap-1" title="Execution Duration">
                    E: <span className="text-zinc-300">{span.durationMs}ms</span>
                  </span>
                  <span className="text-zinc-500 w-24 text-right">
                    {format(new Date(span.startTime), "HH:mm:ss.SSS")}
                  </span>
                </div>
              </div>
              
              {(span.retryCount > 0 || span.message) && (
                <div className="flex items-start justify-between bg-zinc-950 p-2 rounded text-xs font-mono border border-zinc-800">
                  <span className={`${span.status === 'ERROR' ? 'text-red-400' : 'text-zinc-400'}`}>
                    {span.message || "Execution completed"}
                  </span>
                  {span.retryCount > 0 && (
                    <Badge variant="outline" className="border-yellow-700 text-yellow-500 text-[9px] h-4">
                      {span.retryCount} RETRIES
                    </Badge>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
