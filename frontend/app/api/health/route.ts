import { NextResponse } from "next/server";
import { getCircuitBreakers } from "@/services/mock-generator";

export async function GET() {
  if (process.env.NEXT_PUBLIC_TELEMETRY_MODE === 'REAL') {
    try {
      const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
      const res = await fetch(`${backendUrl}/api/health`, { next: { revalidate: 0 } });
      const data = await res.json();
      return NextResponse.json({
        status: data.status === "ok" ? "OK" : "DEGRADED",
        timestamp: new Date().toISOString(),
        version: "1.0.0",
        dependencies: { postgres: "UNKNOWN", redis: "UNKNOWN" }
      }, { status: data.status === "ok" ? 200 : 503 });
    } catch (e) {
      return NextResponse.json({ status: "DEGRADED", timestamp: new Date().toISOString(), version: "1.0.0", dependencies: {} }, { status: 503 });
    }
  }

  const breakers = getCircuitBreakers().payload;
  
  // A basic liveness probe. If the API container is running, it's alive.
  // We optionally check critical dependencies like PostgreSQL or Redis.
  const postgresStatus = breakers.find(b => b.name === "PostgreSQL")?.state;
  const redisStatus = breakers.find(b => b.name === "Redis")?.state;

  const isDegraded = postgresStatus === "OPEN" || redisStatus === "OPEN";

  const status = isDegraded ? "DEGRADED" : "OK";

  return NextResponse.json({
    status,
    timestamp: new Date().toISOString(),
    version: "1.0.0",
    dependencies: {
      postgres: postgresStatus || "UNKNOWN",
      redis: redisStatus || "UNKNOWN"
    }
  }, {
    status: status === "DEGRADED" ? 503 : 200 // 503 Service Unavailable if critical infrastructure is down
  });
}
