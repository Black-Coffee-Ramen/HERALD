import { NextResponse } from "next/server";
import { getInfrastructureMetadata, getQueueMetrics, getBrowserTelemetry, getCircuitBreakers } from "@/services/mock-generator";

export async function GET() {
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
