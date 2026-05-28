import { NextResponse } from "next/server";
import { getCircuitBreakers, getQueueMetrics } from "@/services/mock-generator";

export async function GET() {
  const breakers = getCircuitBreakers().payload;
  const queues = getQueueMetrics().payload;

  // Readiness probe: Can we accept traffic right now?
  // We're ready if we have active workers and dependencies are closed/half-open.
  
  const hasWorkers = queues.processingThroughput > 0;
  
  const postgresStatus = breakers.find(b => b.name === "PostgreSQL")?.state;
  const redisStatus = breakers.find(b => b.name === "Redis")?.state;
  const workerStatus = breakers.find(b => b.name === "Browser Worker")?.state;

  // We are NOT ready if workers are disconnected or databases are totally failed (OPEN).
  const isReady = hasWorkers && postgresStatus !== "OPEN" && redisStatus !== "OPEN" && workerStatus !== "OPEN";

  return NextResponse.json({
    ready: isReady,
    timestamp: new Date().toISOString(),
    details: {
      activeWorkers: queues.processingThroughput > 0 ? 5 : 0,
      databaseConnections: postgresStatus !== "OPEN",
      cacheConnections: redisStatus !== "OPEN",
      workerAvailability: workerStatus !== "OPEN"
    }
  }, {
    status: isReady ? 200 : 503
  });
}
