import { NextResponse } from "next/server";
import { getCircuitBreakers } from "@/services/mock-generator";

export async function GET() {
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
