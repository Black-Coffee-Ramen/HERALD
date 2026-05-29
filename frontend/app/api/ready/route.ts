import { NextResponse } from "next/server";
import { getCircuitBreakers, getQueueMetrics } from "@/services/mock-generator";

export async function GET() {
  if (process.env.NEXT_PUBLIC_TELEMETRY_MODE === 'REAL') {
    try {
      const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
      const res = await fetch(`${backendUrl}/api/ready`, { next: { revalidate: 0 } });
      const data = await res.json();
      return NextResponse.json({
        ready: data.status === "ok",
        timestamp: new Date().toISOString(),
        details: {
          activeWorkers: data.details?.activeWorkers || 0,
          databaseConnections: data.database === "connected",
          cacheConnections: data.redis === "connected",
          workerAvailability: true
        }
      }, { status: data.status === "ok" ? 200 : 503 });
    } catch (e) {
      return NextResponse.json({ ready: false, timestamp: new Date().toISOString(), details: {} }, { status: 503 });
    }
  }

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
