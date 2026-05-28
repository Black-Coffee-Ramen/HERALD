"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { format } from "date-fns";
import { ChartDataPoint } from "./QueueBacklogChart";

interface ThroughputChartProps {
  data: ChartDataPoint[];
}

export function ThroughputChart({ data }: ThroughputChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="h-full w-full flex items-center justify-center text-zinc-500 font-mono text-xs">
        No telemetry available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
        <XAxis 
          dataKey="timestamp" 
          tickFormatter={(tick) => format(new Date(tick), "HH:mm:ss")} 
          stroke="#52525b" 
          fontSize={10} 
          tickMargin={10}
        />
        <YAxis stroke="#52525b" fontSize={10} />
        <Tooltip 
          contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", fontSize: "12px", fontFamily: "monospace" }}
          labelFormatter={(label) => format(new Date(label), "HH:mm:ss")}
        />
        <Bar dataKey="value" fill="#10b981" radius={[2, 2, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}
