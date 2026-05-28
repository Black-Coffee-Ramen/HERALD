"use client";

import { ThreatEvent } from "@/types";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { format } from "date-fns";
import { ShieldAlert, Fingerprint, Eye, Globe } from "lucide-react";
import Link from "next/link";

export function LiveThreatFeed({ events, connectionState }: { events: ThreatEvent[], connectionState: string }) {
  const getVerdictBadge = (verdict: string) => {
    switch (verdict) {
      case "MALICIOUS":
        return <Badge variant="destructive" className="bg-red-900 text-red-100">Malicious</Badge>;
      case "SUSPICIOUS":
        return <Badge className="bg-yellow-700 text-yellow-100">Suspicious</Badge>;
      case "BENIGN":
        return <Badge className="bg-green-900 text-green-100">Benign</Badge>;
      default:
        return <Badge variant="outline" className="text-zinc-400">Pending</Badge>;
    }
  };

  const getStageIcon = (stage: string) => {
    switch (stage) {
      case "LEXICAL": return <Globe className="w-4 h-4 text-blue-400" />;
      case "DNS": return <Fingerprint className="w-4 h-4 text-purple-400" />;
      case "VISUAL": return <Eye className="w-4 h-4 text-indigo-400" />;
      case "OCR": return <ShieldAlert className="w-4 h-4 text-orange-400" />;
      default: return <span className="text-xs text-zinc-500">{stage}</span>;
    }
  };

  const getConnectionColor = () => {
    if (connectionState === "CONNECTED") return "bg-green-500";
    if (connectionState === "DEGRADED" || connectionState === "RECONNECTING") return "bg-yellow-500";
    return "bg-red-500";
  };

  return (
    <Card className="bg-zinc-900 border-zinc-800 h-full flex flex-col">
      <CardHeader className="pb-2 shrink-0">
        <CardTitle className="text-sm font-semibold tracking-wider text-zinc-300 flex items-center justify-between">
          <span>LIVE THREAT FEED</span>
          <span className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              {connectionState === "CONNECTED" && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>}
              {(connectionState === "RECONNECTING" || connectionState === "DEGRADED") && <span className="animate-pulse absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>}
              <span className={`relative inline-flex rounded-full h-2 w-2 ${getConnectionColor()}`}></span>
            </span>
            <span className={`text-xs font-mono text-[10px] ${connectionState === "DISCONNECTED" ? "text-red-400" : "text-zinc-500"}`}>
              {connectionState}
            </span>
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0 flex-1 overflow-hidden">
        <ScrollArea className="h-[400px]">
          <Table>
            <TableHeader className="bg-zinc-900/50 sticky top-0 z-10">
              <TableRow className="border-zinc-800 hover:bg-transparent">
                <TableHead className="text-zinc-500 font-mono text-xs w-[180px]">TIMESTAMP</TableHead>
                <TableHead className="text-zinc-500 font-mono text-xs">DOMAIN</TableHead>
                <TableHead className="text-zinc-500 font-mono text-xs">STAGE</TableHead>
                <TableHead className="text-zinc-500 font-mono text-xs w-[120px]">SCORE</TableHead>
                <TableHead className="text-zinc-500 font-mono text-xs text-right w-[120px]">VERDICT</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {events.map((evt) => (
                <TableRow key={evt.id} className="border-zinc-800/50 hover:bg-zinc-800/30 transition-colors">
                  <TableCell className="font-mono text-xs text-zinc-400">
                    {format(new Date(evt.timestamp), "HH:mm:ss.SSS")}
                  </TableCell>
                  <TableCell className="font-mono text-sm text-zinc-200">
                    <Link href={`/domain/${evt.id}`} className="hover:text-indigo-400 hover:underline transition-colors">
                      {evt.domain}
                    </Link>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      {getStageIcon(evt.workerStage)}
                    </div>
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    <span className={typeof evt.confidenceScore === "number" && evt.confidenceScore > 80 ? "text-red-400" : "text-zinc-400"}>
                      {typeof evt.confidenceScore === "number" ? evt.confidenceScore.toFixed(1) : "--"}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    {getVerdictBadge(evt.verdict)}
                  </TableCell>
                </TableRow>
              ))}
              {events.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="text-center text-zinc-500 py-8">
                    Waiting for telemetry...
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
