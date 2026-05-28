"use client";

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { format } from "date-fns";

export interface ChartDataPoint {
  timestamp: string;
  value: number;
  secondaryValue?: number;
}

interface QueueBacklogChartProps {
  data: ChartDataPoint[];
}

export function QueueBacklogChart({ data }: QueueBacklogChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="h-full w-full flex items-center justify-center text-zinc-500 font-mono text-xs">
        No telemetry available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
        <XAxis 
          dataKey="timestamp" 
          tickFormatter={(tick) => format(new Date(tick), "HH:mm:ss")} 
          stroke="#52525b" 
          fontSize={10} 
          tickMargin={10}
        />
        <YAxis stroke="#52525b" fontSize={10} tickFormatter={(val) => typeof val === "number" && val >= 1000 ? `${(val/1000).toFixed(1)}k` : val} />
        <Tooltip 
          contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", fontSize: "12px", fontFamily: "monospace" }}
          labelFormatter={(label) => format(new Date(label), "HH:mm:ss")}
        />
        <Area type="monotone" dataKey="value" stroke="#818cf8" fillOpacity={1} fill="url(#colorValue)" isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}
