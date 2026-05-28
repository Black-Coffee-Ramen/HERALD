import { TraceSpan } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Server, Clock, RefreshCw, AlertCircle } from "lucide-react";

export function WorkerExecutionPanel({ spans }: { spans: TraceSpan[] }) {
  // Group spans by worker type for aggregate view
  const workers = Array.from(new Set(spans.map(s => s.workerType)));

  return (
    <Card className="bg-zinc-900 border-zinc-800 h-full flex flex-col">
      <CardHeader className="pb-3 shrink-0">
        <CardTitle className="text-xs font-mono text-zinc-400 uppercase tracking-wider flex items-center gap-2">
          <Server className="w-4 h-4" />
          Worker Execution Summary
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-auto p-4 pt-0">
        <div className="space-y-3">
          {workers.map(worker => {
            const workerSpans = spans.filter(s => s.workerType === worker);
            const totalDur = workerSpans.reduce((acc, s) => acc + s.durationMs, 0);
            const totalRetries = workerSpans.reduce((acc, s) => acc + s.retryCount, 0);
            const hasError = workerSpans.some(s => s.status === 'ERROR');

            return (
              <div key={worker} className={`p-3 rounded-lg border ${hasError ? 'border-red-900/50 bg-red-950/10' : 'border-zinc-800 bg-zinc-950'} flex flex-col gap-2`}>
                 <div className="flex justify-between items-center">
                    <span className="font-mono text-xs text-zinc-300 font-bold">{worker}</span>
                    {hasError ? (
                      <Badge variant="destructive" className="h-4 text-[9px] font-mono px-1">ERROR</Badge>
                    ) : totalRetries > 0 ? (
                      <Badge variant="outline" className="h-4 text-[9px] font-mono px-1 border-yellow-700 text-yellow-500">RETRY</Badge>
                    ) : (
                      <Badge variant="outline" className="h-4 text-[9px] font-mono px-1 border-green-800 text-green-500">OK</Badge>
                    )}
                 </div>
                 
                 <div className="flex gap-4 mt-1">
                   <div className="flex items-center gap-1 text-[10px] font-mono text-zinc-500">
                     <Clock className="w-3 h-3" />
                     {totalDur}ms
                   </div>
                   <div className={`flex items-center gap-1 text-[10px] font-mono ${totalRetries > 0 ? 'text-yellow-500' : 'text-zinc-500'}`}>
                     <RefreshCw className="w-3 h-3" />
                     {totalRetries} retries
                   </div>
                 </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
