"use client";

import { QueueMetrics, InfrastructureMetadata, CircuitBreakerStatus } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Activity, Database, Server, Cpu } from "lucide-react";

interface SystemHealthProps {
  queue: QueueMetrics | null;
  infra: InfrastructureMetadata | null;
  breakers: CircuitBreakerStatus[];
}

export function SystemHealth({ queue, infra, breakers }: SystemHealthProps) {
  const getStatusColor = (status: string) => {
    if (status === "HEALTHY" || status === "CLOSED") return "text-green-400";
    if (status === "DEGRADED" || status === "HALF_OPEN") return "text-yellow-400";
    return "text-red-400";
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 mb-4">
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-xs font-mono text-zinc-400 uppercase tracking-wider">Queue Pressure</CardTitle>
          <Activity className="h-4 w-4 text-zinc-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold font-mono text-zinc-100">
            {queue && typeof queue.queueDepth === "number" ? queue.queueDepth.toLocaleString() : "..."}
          </div>
          <p className="text-xs text-zinc-500 font-mono mt-1">
            Lat: {queue && typeof queue.eventLatencyMs === "number" ? `${queue.eventLatencyMs}ms` : "-"} | Tput: {queue && typeof queue.processingThroughput === "number" ? `${queue.processingThroughput}/s` : "-"}
          </p>
        </CardContent>
      </Card>

      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-xs font-mono text-zinc-400 uppercase tracking-wider">Infrastructure</CardTitle>
          <Database className="h-4 w-4 text-zinc-500" />
        </CardHeader>
        <CardContent className="flex flex-col gap-1">
          <div className="flex justify-between items-center text-xs font-mono">
            <span className="text-zinc-300">PostgreSQL</span>
            <span className={getStatusColor(infra?.postgresStatus || "DOWN")}>{infra?.postgresStatus || "UNKNOWN"}</span>
          </div>
          <div className="flex justify-between items-center text-xs font-mono">
            <span className="text-zinc-300">Redis</span>
            <span className={getStatusColor(infra?.redisStatus || "DOWN")}>{infra?.redisStatus || "UNKNOWN"}</span>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-xs font-mono text-zinc-400 uppercase tracking-wider">Circuit Breakers</CardTitle>
          <Server className="h-4 w-4 text-zinc-500" />
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-2">
            {breakers.slice(0, 4).map(b => (
              <div key={b.name} className="flex justify-between items-center text-[10px] font-mono bg-zinc-950 p-1 rounded border border-zinc-800">
                <span className="text-zinc-400 truncate w-16" title={b.name}>{b.name.split(" ")[0]}</span>
                <span className={getStatusColor(b.state)}>{b.state === "CLOSED" ? "OK" : b.state}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-xs font-mono text-zinc-400 uppercase tracking-wider">System State</CardTitle>
          <Cpu className="h-4 w-4 text-zinc-500" />
        </CardHeader>
        <CardContent>
           <div className="flex flex-col gap-2">
             <div className="flex justify-between items-center text-sm">
                <span className="text-zinc-400 font-mono text-xs">DLQ Size</span>
                <span className="font-mono text-red-400">{queue?.dlqSize || 0}</span>
             </div>
             <div className="flex justify-between items-center text-sm">
                <span className="text-zinc-400 font-mono text-xs">Degraded Mode</span>
                {infra?.degradedModeActive ? 
                  <Badge variant="destructive" className="h-4 text-[10px]">ACTIVE</Badge> : 
                  <Badge variant="outline" className="h-4 text-[10px] text-zinc-500 border-zinc-700">INACTIVE</Badge>
                }
             </div>
           </div>
        </CardContent>
      </Card>
    </div>
  );
}
