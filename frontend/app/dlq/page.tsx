"use client";

import { useTelemetry } from "@/hooks/useTelemetry";
import { SectionHeader, EmptyState } from "@/components/shared/Primitives";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, Activity, Share2, Shield, SearchX } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { DLQEntry } from "@/types";
import { format } from "date-fns";

const generateMockDlq = (): DLQEntry[] => {
  return Array.from({ length: 8 }).map((_, i) => ({
    jobId: `job-${Math.random().toString(36).substring(2, 8)}`,
    traceId: `tr-${Math.random().toString(36).substring(2, 8)}`,
    failureClass: Math.random() > 0.5 ? "TimeoutError" : "ParseError",
    workerType: Math.random() > 0.5 ? "visual_worker" : "lexical_worker",
    retryAttempts: Math.floor(Math.random() * 5) + 1,
    retryEligible: Math.random() > 0.3,
    timestamp: new Date(Date.now() - Math.floor(Math.random() * 10000000)).toISOString(),
    dependencyFailureSource: Math.random() > 0.5 ? "BrowserTimeout" : undefined
  }));
};

export default function DLQPage() {
  const { queueMetrics } = useTelemetry();
  const [dlqItems, setDlqItems] = useState<DLQEntry[]>([]);

  useEffect(() => {
    if (process.env.NEXT_PUBLIC_TELEMETRY_MODE === 'REAL') {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
      fetch(`${backendUrl}/api/admin/failed-jobs`)
        .then(res => res.json())
        .then(data => {
          if (data && data.jobs) {
            setDlqItems(data.jobs.map((j: any) => ({
              jobId: j.job_id || "unknown",
              traceId: j.trace_id || "unknown",
              failureClass: j.last_error || "UnknownError",
              workerType: j.source || "unknown",
              retryAttempts: j.attempts || 0,
              retryEligible: (j.attempts || 0) < 3,
              timestamp: j.last_failed_at ? new Date(j.last_failed_at * 1000).toISOString() : new Date().toISOString(),
              dependencyFailureSource: undefined
            })));
          }
        })
        .catch(err => {
          console.error("Failed to fetch real DLQ", err);
          setDlqItems(generateMockDlq());
        });
    } else {
      setDlqItems(generateMockDlq());
    }
  }, []);

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
            <Link href="/dlq" className="text-indigo-400 flex items-center gap-2 border-b-2 border-indigo-500 pb-1 pt-1"><AlertTriangle className="w-4 h-4"/> DLQ</Link>
          </nav>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-4 py-6">
        <div className="flex justify-between items-end mb-6">
          <SectionHeader title="Dead Letter Queue" description="Failed processing events requiring manual intervention or review" />
          <div className="bg-red-950/20 border border-red-900/50 text-red-400 px-4 py-2 rounded-lg font-mono text-sm flex items-center gap-2">
             <AlertTriangle className="w-4 h-4" />
             {queueMetrics?.dlqSize || dlqItems.length} Unresolved Items
          </div>
        </div>

        <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
          {dlqItems.length === 0 ? (
            <EmptyState message="DLQ is currently empty." icon={SearchX} />
          ) : (
            <Table>
              <TableHeader className="bg-zinc-900/50">
                <TableRow className="border-zinc-800 hover:bg-transparent">
                  <TableHead className="text-zinc-500 font-mono text-xs">TIMESTAMP</TableHead>
                  <TableHead className="text-zinc-500 font-mono text-xs">JOB ID</TableHead>
                  <TableHead className="text-zinc-500 font-mono text-xs">WORKER</TableHead>
                  <TableHead className="text-zinc-500 font-mono text-xs">FAILURE CLASS</TableHead>
                  <TableHead className="text-zinc-500 font-mono text-xs text-right">RETRIES</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {dlqItems.map(item => (
                  <TableRow key={item.jobId} className="border-zinc-800/50 hover:bg-zinc-800/30">
                    <TableCell className="font-mono text-xs text-zinc-400">
                      {format(new Date(item.timestamp), "yyyy-MM-dd HH:mm:ss")}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-zinc-300">
                      {item.jobId}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-zinc-400">
                      {item.workerType}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs text-red-400">{item.failureClass}</span>
                        {item.dependencyFailureSource && (
                          <Badge variant="outline" className="text-[9px] h-4 border-zinc-700 text-zinc-500">{item.dependencyFailureSource}</Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                       <Badge variant={item.retryEligible ? "outline" : "destructive"} className="font-mono text-[10px]">
                         {item.retryAttempts} / {item.retryEligible ? 'MAX' : 'LIMIT'}
                       </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </div>
      </main>
    </div>
  );
}
