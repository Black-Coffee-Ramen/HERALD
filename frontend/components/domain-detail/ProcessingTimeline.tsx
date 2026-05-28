import { DomainIntelligence } from "@/types";
import { format } from "date-fns";
import { CheckCircle2, CircleDashed, AlertCircle, Clock } from "lucide-react";

export function ProcessingTimeline({ intel }: { intel: DomainIntelligence }) {
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "SUCCESS": return <CheckCircle2 className="w-4 h-4 text-green-500" />;
      case "FAILED": return <AlertCircle className="w-4 h-4 text-red-500" />;
      case "PENDING": return <CircleDashed className="w-4 h-4 text-yellow-500 animate-spin-slow" />;
      default: return <Clock className="w-4 h-4 text-zinc-500" />;
    }
  };

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6">
      <h3 className="text-xs font-mono text-zinc-400 uppercase mb-6 tracking-wider">Processing Timeline</h3>
      
      <div className="relative border-l border-zinc-800 ml-2 space-y-6">
        {intel.timeline.map((event, idx) => (
          <div key={idx} className="relative pl-6">
            <div className="absolute -left-2 top-0.5 bg-zinc-900 rounded-full">
              {getStatusIcon(event.status)}
            </div>
            
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-3">
                <span className={`text-sm font-bold font-mono ${event.status === 'FAILED' ? 'text-red-400' : 'text-zinc-200'}`}>
                  {event.stage}
                </span>
                <span className="text-xs font-mono text-zinc-500">
                  {format(new Date(event.timestamp), "HH:mm:ss.SSS")}
                </span>
                {event.durationMs && (
                  <span className="text-[10px] font-mono text-zinc-600 bg-zinc-950 px-1.5 py-0.5 rounded">
                    {event.durationMs}ms
                  </span>
                )}
              </div>
              
              {event.message && (
                <p className="text-xs font-mono text-zinc-400 mt-1">
                  {event.message}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
