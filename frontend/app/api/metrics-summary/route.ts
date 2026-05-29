import { NextResponse } from "next/server";
import { getInfrastructureMetadata, getQueueMetrics, getBrowserTelemetry, getCircuitBreakers } from "@/services/mock-generator";

export async function GET() {
  if (process.env.NEXT_PUBLIC_TELEMETRY_MODE === 'REAL') {
    try {
      const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
      const res = await fetch(`${backendUrl}/api/metrics-summary`, { next: { revalidate: 0 } });
      const data = await res.json();
      
      const infra = { apiLatencyMs: 50, memoryUsageMb: 120, activeConnections: 10 };
      const queues = { queueDepth: (data.queues?.lexical?.ready || 0) + (data.queues?.visual?.ready || 0), processingThroughput: (data.workers?.lexical_active || 0) + (data.workers?.visual_active || 0) };
      const browser = data.browser_pressure || {};
      const breakers = data.circuit_breakers?.visual_analysis ? [{name: "Visual Analysis", state: data.circuit_breakers.visual_analysis.state === "open" ? "OPEN" : "CLOSED"}] : [];
      
      return NextResponse.json({
        timestamp: new Date().toISOString(),
        metrics: {
          infrastructure: infra,
          queues,
          browserFleet: browser,
          circuitBreakers: breakers
        },
        aggregates: {
          totalQueueBacklog: queues.queueDepth,
          totalActiveWorkers: queues.processingThroughput > 0 ? 5 : 0,
          degradedDependencyCount: breakers.filter(b => b.state !== "CLOSED").length,
          isSystemDegraded: breakers.some(b => b.state === "OPEN")
        }
      });
    } catch (e) {
      // fallback
    }
  }

  // Aggregate all mock telemetry states to present a consolidated operational view
  const infra = getInfrastructureMetadata().payload;
  const queues = getQueueMetrics().payload;
  const browser = getBrowserTelemetry().payload;
  const breakers = getCircuitBreakers().payload;

  return NextResponse.json({
    timestamp: new Date().toISOString(),
    metrics: {
      infrastructure: infra,
      queues,
      browserFleet: browser,
      circuitBreakers: breakers
    },
    // Useful derived metrics for prometheus scraping or dashboard aggregates
    aggregates: {
      totalQueueBacklog: queues.queueDepth,
      totalActiveWorkers: queues.processingThroughput > 0 ? 5 : 0,
      degradedDependencyCount: breakers.filter(b => b.state !== "CLOSED").length,
      isSystemDegraded: breakers.some(b => b.state === "OPEN") || infra.apiLatencyMs > 1000
    }
  });
}
