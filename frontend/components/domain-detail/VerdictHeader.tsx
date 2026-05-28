import { DomainIntelligence } from "@/types";
import { Badge } from "@/components/ui/badge";
import { StatusBadge } from "@/components/shared/Primitives";
import { format } from "date-fns";
import { ShieldAlert, AlertTriangle, CheckCircle, Clock, Hash, Zap, Network } from "lucide-react";
import Link from "next/link";

export function VerdictHeader({ intel }: { intel: DomainIntelligence }) {
  const getVerdictIcon = () => {
    switch(intel.verdict) {
      case "MALICIOUS": return <ShieldAlert className="w-8 h-8 text-red-500" />;
      case "SUSPICIOUS": return <AlertTriangle className="w-8 h-8 text-yellow-500" />;
      case "BENIGN": return <CheckCircle className="w-8 h-8 text-green-500" />;
      default: return <Clock className="w-8 h-8 text-zinc-500" />;
    }
  };

  const getVerdictColor = () => {
    switch(intel.verdict) {
      case "MALICIOUS": return "border-red-900 bg-red-950/20";
      case "SUSPICIOUS": return "border-yellow-900 bg-yellow-950/20";
      case "BENIGN": return "border-green-900 bg-green-950/20";
      default: return "border-zinc-800 bg-zinc-900/50";
    }
  };

  return (
    <div className={`p-6 rounded-xl border ${getVerdictColor()} mb-6`}>
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800 shadow-inner">
            {getVerdictIcon()}
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-wider text-zinc-100 font-mono mb-1">{intel.domain}</h1>
            <div className="flex items-center gap-3 text-xs font-mono text-zinc-400">
              <span className="flex items-center gap-1"><Hash className="w-3 h-3"/> {intel.id}</span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <Network className="w-3 h-3"/>
                <Link href={`/traces/${intel.traceId}`} className="hover:text-indigo-400 hover:underline transition-colors">
                  {intel.traceId}
                </Link>
              </span>
              <span>•</span>
              <span>Analyzed: {format(new Date(intel.lastAnalyzed), "yyyy-MM-dd HH:mm:ss")}</span>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-end gap-2">
           <div className="flex items-center gap-2">
             <span className="text-xs font-mono text-zinc-500 uppercase">Confidence</span>
             <span className={`text-xl font-mono font-bold ${intel.confidenceScore > 80 ? "text-red-400" : intel.confidenceScore > 50 ? "text-yellow-400" : "text-green-400"}`}>
               {typeof intel.confidenceScore === "number" ? intel.confidenceScore.toFixed(1) : "--"}%
             </span>
           </div>
           <div className="flex gap-2">
             <StatusBadge status={intel.verdict === "MALICIOUS" ? "OPEN" : "HEALTHY"} />
             <Badge variant="outline" className="font-mono text-[10px] border-zinc-700 text-zinc-400 flex items-center gap-1">
               <Zap className="w-3 h-3" /> {intel.queueStage}
             </Badge>
           </div>
        </div>
      </div>
    </div>
  );
}
