import { TraceSpan } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Terminal } from "lucide-react";
import { format } from "date-fns";

export function TraceEventLog({ spans }: { spans: TraceSpan[] }) {
  // Synthesize an event stream from spans to simulate SOC event log
  const events = spans.flatMap(span => {
    const stream = [];
    
    // Ingestion / Queue event
    stream.push({
      time: new Date(new Date(span.startTime).getTime() - span.queueWaitMs),
      msg: `[${span.workerType}] received job for ${span.name}`,
      level: "info"
    });

    if (span.queueWaitMs > 500) {
      stream.push({
        time: new Date(new Date(span.startTime).getTime() - (span.queueWaitMs / 2)),
        msg: `[${span.workerType}] queue delay detected (${span.queueWaitMs}ms)`,
        level: "warn"
      });
    }

    if (span.retryCount > 0) {
      stream.push({
        time: new Date(new Date(span.startTime).getTime() + (span.durationMs / 3)),
        msg: `[${span.workerType}] ${span.message || "execution failed, triggered retry"}`,
        level: "warn"
      });
    }

    if (span.status === "ERROR") {
      stream.push({
        time: new Date(span.endTime || span.startTime),
        msg: `[${span.workerType}] fatal error during ${span.name}: ${span.message}`,
        level: "error"
      });
    } else {
      stream.push({
        time: new Date(new Date(span.startTime).getTime() + span.durationMs),
        msg: `[${span.workerType}] ${span.name} completed in ${span.durationMs}ms`,
        level: "success"
      });
    }

    return stream;
  }).sort((a, b) => a.time.getTime() - b.time.getTime());

  return (
    <Card className="bg-zinc-950 border-zinc-800 h-full flex flex-col font-mono">
      <CardHeader className="pb-3 shrink-0 border-b border-zinc-800/50 bg-zinc-900/50">
        <CardTitle className="text-xs text-zinc-400 uppercase tracking-wider flex items-center gap-2">
          <Terminal className="w-4 h-4" />
          Event Stream
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-auto p-4 pt-4 space-y-1.5 text-[11px]">
        {events.map((evt, idx) => {
          let color = "text-zinc-400";
          if (evt.level === "warn") color = "text-yellow-400";
          if (evt.level === "error") color = "text-red-400";
          if (evt.level === "success") color = "text-green-400";

          return (
             <div key={idx} className="flex gap-3 hover:bg-zinc-900/50 px-1 py-0.5 rounded transition-colors">
               <span className="text-zinc-600 shrink-0">[{format(evt.time, "HH:mm:ss.SSS")}]</span>
               <span className={color}>{evt.msg}</span>
             </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
