import { ThreatEvent, QueueMetrics, InfrastructureMetadata, CircuitBreakerStatus } from "../../types";

// Scenarios define how the base telemetry should be mutated to simulate operational states

export type ScenarioState = "NORMAL" | "PHISHING_BURST" | "REDIS_PRESSURE" | "RETRY_STORM" | "DEGRADED";

export interface ScenarioMutator {
  mutateThreats: (events: ThreatEvent[]) => ThreatEvent[];
  mutateQueue: (metrics: QueueMetrics) => QueueMetrics;
  mutateInfra: (infra: InfrastructureMetadata) => InfrastructureMetadata;
  mutateBreakers: (breakers: CircuitBreakerStatus[]) => CircuitBreakerStatus[];
}

export const scenarios: Record<ScenarioState, ScenarioMutator> = {
  NORMAL: {
    mutateThreats: (e) => e,
    mutateQueue: (m) => m,
    mutateInfra: (i) => i,
    mutateBreakers: (b) => b
  },
  PHISHING_BURST: {
    mutateThreats: (e) => e.map(evt => ({ ...evt, verdict: "MALICIOUS", confidenceScore: 95 })),
    mutateQueue: (m) => ({ ...m, queueDepth: m.queueDepth + 5000, processingThroughput: Math.floor(m.processingThroughput * 1.5) }),
    mutateInfra: (i) => i,
    mutateBreakers: (b) => b
  },
  REDIS_PRESSURE: {
    mutateThreats: (e) => e,
    mutateQueue: (m) => ({ ...m, eventLatencyMs: m.eventLatencyMs + 2000, queueDepth: m.queueDepth + 1000 }),
    mutateInfra: (i) => ({ ...i, redisStatus: "DEGRADED", apiLatencyMs: i.apiLatencyMs + 500 }),
    mutateBreakers: (b) => b.map(cb => cb.name === "Redis" ? { ...cb, state: "HALF_OPEN", recentFailures: 12 } : cb)
  },
  RETRY_STORM: {
    mutateThreats: (e) => e.map(evt => ({ ...evt, workerStage: "FAILED" })),
    mutateQueue: (m) => ({ ...m, retryQueueSize: m.retryQueueSize + 800, dlqSize: m.dlqSize + 50, processingThroughput: Math.floor(m.processingThroughput * 0.2) }),
    mutateInfra: (i) => ({ ...i, degradedModeActive: true }),
    mutateBreakers: (b) => b.map(cb => cb.name.includes("Worker") ? { ...cb, state: "OPEN", recentFailures: 45 } : cb)
  },
  DEGRADED: {
    mutateThreats: (e) => e,
    mutateQueue: (m) => ({ ...m, eventLatencyMs: 5000, processingThroughput: 10 }),
    mutateInfra: (i) => ({ ...i, postgresStatus: "DEGRADED", degradedModeActive: true }),
    mutateBreakers: (b) => b.map(cb => cb.name === "PostgreSQL" ? { ...cb, state: "HALF_OPEN", recentFailures: 5 } : cb)
  }
};

let currentScenario: ScenarioState = "NORMAL";

export const setScenario = (s: ScenarioState) => {
  currentScenario = s;
  console.log(`[Telemetry] Scenario shifted to: ${s}`);
};

export const applyScenario = () => scenarios[currentScenario];

// Periodically shift scenarios to simulate alive system
setInterval(() => {
  const states: ScenarioState[] = ["NORMAL", "NORMAL", "NORMAL", "PHISHING_BURST", "REDIS_PRESSURE", "RETRY_STORM", "DEGRADED"];
  setScenario(states[Math.floor(Math.random() * states.length)]);
}, 30000); // Shift every 30s
