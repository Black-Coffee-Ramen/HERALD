import { useEffect, useState } from "react";
import { telemetryClient } from "../services/websocket";
import { ThreatEvent, QueueMetrics, InfrastructureMetadata, CircuitBreakerStatus, BrowserWorkerTelemetry } from "../types";

export function useTelemetry() {
  const [threatEvents, setThreatEvents] = useState<ThreatEvent[]>([]);
  const [queueMetrics, setQueueMetrics] = useState<QueueMetrics | null>(null);
  const [infrastructure, setInfrastructure] = useState<InfrastructureMetadata | null>(null);
  const [browserTelemetry, setBrowserTelemetry] = useState<BrowserWorkerTelemetry | null>(null);
  const [circuitBreakers, setCircuitBreakers] = useState<CircuitBreakerStatus[]>([]);
  const [connectionState, setConnectionState] = useState<string>("DISCONNECTED");

  useEffect(() => {
    telemetryClient.connect();

    // To handle throttled bursts properly without dropping intermediate updates in React state,
    // we should ideally batch, but setState with a callback safely accumulates them.
    // The telemetryClient flushes bursts every 200ms.
    const unsubThreats = telemetryClient.subscribe("threatEvent", (env) => {
      setThreatEvents((prev) => {
        // Drop policy on the UI side to prevent DOM memory overflow
        const newArr = [env.payload, ...prev];
        return newArr.slice(0, 100); 
      });
    });

    const unsubQueue = telemetryClient.subscribe("queueMetrics", (env) => setQueueMetrics(env.payload));
    const unsubInfra = telemetryClient.subscribe("infrastructure", (env) => setInfrastructure(env.payload));
    const unsubBrowser = telemetryClient.subscribe("browserTelemetry", (env) => setBrowserTelemetry(env.payload));
    const unsubCircuit = telemetryClient.subscribe("circuitBreakers", (env) => setCircuitBreakers(env.payload));
    const unsubConn = telemetryClient.subscribe("connectionState", (state: any) => setConnectionState(state));

    return () => {
      unsubThreats();
      unsubQueue();
      unsubInfra();
      unsubBrowser();
      unsubCircuit();
      unsubConn();
      telemetryClient.disconnect();
    };
  }, []);

  return { threatEvents, queueMetrics, infrastructure, browserTelemetry, circuitBreakers, connectionState };
}
