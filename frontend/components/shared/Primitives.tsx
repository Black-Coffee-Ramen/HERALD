import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, CheckCircle2, Loader2, ServerCrash } from "lucide-react";
import { cn } from "@/lib/utils";

export const StatusBadge = ({ status, className }: { status: "HEALTHY" | "DEGRADED" | "DOWN" | "CLOSED" | "OPEN" | "HALF_OPEN", className?: string }) => {
  if (status === "HEALTHY" || status === "CLOSED") {
    return <Badge className={cn("bg-green-900/50 text-green-400 hover:bg-green-900/50 border-green-800", className)}>OK</Badge>;
  }
  if (status === "DEGRADED" || status === "HALF_OPEN") {
    return <Badge className={cn("bg-yellow-900/50 text-yellow-400 hover:bg-yellow-900/50 border-yellow-800", className)}>WARN</Badge>;
  }
  return <Badge variant="destructive" className={cn("bg-red-900/50 text-red-400 hover:bg-red-900/50 border-red-800", className)}>CRIT</Badge>;
};

export const MetricCard = ({ title, value, subtext, icon: Icon, alert }: { title: string, value: string | number, subtext?: React.ReactNode, icon?: any, alert?: boolean }) => (
  <Card className={cn("bg-zinc-900 border-zinc-800", alert && "border-red-900 bg-red-950/20")}>
    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
      <CardTitle className="text-xs font-mono text-zinc-400 uppercase tracking-wider">{title}</CardTitle>
      {Icon && <Icon className={cn("h-4 w-4", alert ? "text-red-500" : "text-zinc-500")} />}
    </CardHeader>
    <CardContent>
      <div className={cn("text-2xl font-bold font-mono", alert ? "text-red-400" : "text-zinc-100")}>{value}</div>
      {subtext && <p className="text-xs text-zinc-500 font-mono mt-1">{subtext}</p>}
    </CardContent>
  </Card>
);

export const HealthIndicator = ({ isHealthy }: { isHealthy: boolean }) => (
  <div className="flex items-center gap-2">
    <span className="relative flex h-2 w-2">
      {isHealthy && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>}
      <span className={cn("relative inline-flex rounded-full h-2 w-2", isHealthy ? "bg-green-500" : "bg-red-500")}></span>
    </span>
    <span className="text-xs font-mono text-zinc-500">{isHealthy ? "ONLINE" : "OFFLINE"}</span>
  </div>
);

export const EmptyState = ({ message, icon: Icon = AlertCircle }: { message: string, icon?: any }) => (
  <div className="flex flex-col items-center justify-center py-12 text-zinc-500 border border-dashed border-zinc-800 rounded-lg bg-zinc-900/50">
    <Icon className="h-8 w-8 mb-2 opacity-50" />
    <p className="font-mono text-sm">{message}</p>
  </div>
);

export const LoadingState = ({ message = "Loading telemetry..." }: { message?: string }) => (
  <div className="flex flex-col items-center justify-center py-12 text-zinc-500">
    <Loader2 className="h-8 w-8 mb-4 animate-spin text-indigo-500" />
    <p className="font-mono text-sm animate-pulse">{message}</p>
  </div>
);

export const SectionHeader = ({ title, description }: { title: string, description?: string }) => (
  <div className="mb-4">
    <h2 className="text-lg font-bold tracking-widest text-zinc-100 uppercase">{title}</h2>
    {description && <p className="text-sm font-mono text-zinc-500">{description}</p>}
  </div>
);
