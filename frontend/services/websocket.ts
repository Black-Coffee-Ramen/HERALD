import { ThreatEvent, QueueMetrics, InfrastructureMetadata, CircuitBreakerStatus, EventEnvelope, BrowserWorkerTelemetry } from "../types";
import { generateThreatEvent, getQueueMetrics, getInfrastructureMetadata, getCircuitBreakers, getBrowserTelemetry } from "./mock-generator";

type EventHandler<T> = (data: EventEnvelope<T>) => void;

export type TelemetryMode = "REAL" | "MOCK" | "HYBRID";
const TELEMETRY_MODE: TelemetryMode = (process.env.NEXT_PUBLIC_TELEMETRY_MODE as TelemetryMode) || "MOCK";
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/telemetry";

type ConnectionState = "CONNECTED" | "DISCONNECTED" | "RECONNECTING" | "DEGRADED" | "CONNECTING";

interface Listeners {
  threatEvent: EventHandler<ThreatEvent>[];
  queueMetrics: EventHandler<QueueMetrics>[];
  infrastructure: EventHandler<InfrastructureMetadata>[];
  browserTelemetry: EventHandler<BrowserWorkerTelemetry>[];
  circuitBreakers: EventHandler<CircuitBreakerStatus[]>[];
  connectionState: EventHandler<ConnectionState>[];
}

class TelemetryClient {
  private listeners: Listeners = {
    threatEvent: [],
    queueMetrics: [],
    infrastructure: [],
    browserTelemetry: [],
    circuitBreakers: [],
    connectionState: []
  };

  // Buffer Protection Configuration
  private readonly MAX_BUFFER_SIZE = 50;
  private readonly THROTTLE_MS = 200; // max 5 renders per second per channel
  private eventBuffers: Record<string, EventEnvelope<any>[]> = {};
  private emitTimers: Record<string, NodeJS.Timeout | null> = {};

  private intervals: NodeJS.Timeout[] = [];
  private ws: WebSocket | null = null;
  public connectionState: ConnectionState = "DISCONNECTED";

  private updateConnectionState(state: ConnectionState) {
    this.connectionState = state;
    // connectionState bypasses the buffer since it's critical control-plane data
    this.listeners.connectionState.forEach(fn => fn(state as any));
  }

  // Buffered emit with Drop Policy (LIFO/FIFO based on operational priority)
  private emitBuffered(channel: keyof Listeners, data: EventEnvelope<any>) {
    if (!this.eventBuffers[channel]) this.eventBuffers[channel] = [];
    
    this.eventBuffers[channel].push(data);

    // Drop policy: If buffer exceeds MAX_BUFFER_SIZE, drop oldest low-priority events first
    if (this.eventBuffers[channel].length > this.MAX_BUFFER_SIZE) {
       // Sort buffer: LOW priority first, so they get dropped. 
       // We NEVER drop HIGH priority events (like threat verdicts) unless the entire buffer is HIGH priority.
       const dropCount = Math.floor(this.MAX_BUFFER_SIZE * 0.1);
       
       // If it's the threatEvent channel, we dynamically expand the buffer to prevent dropping HIGH priority verdicts
       if (channel === "threatEvent" || data.telemetry_priority === "HIGH") {
          // Allow buffer to grow larger for HIGH priority, but hard cap at 1000 to prevent OOM
          if (this.eventBuffers[channel].length > 1000) {
            this.eventBuffers[channel] = this.eventBuffers[channel].slice(dropCount);
          }
       } else {
          // For metrics and spans (LOW/MEDIUM), aggressively drop oldest
          this.eventBuffers[channel] = this.eventBuffers[channel].slice(dropCount);
       }
    }

    this.scheduleFlush(channel);
  }

  private scheduleFlush(channel: keyof Listeners) {
    if (this.emitTimers[channel]) return; // already scheduled

    this.emitTimers[channel] = setTimeout(() => {
      this.flushBuffer(channel);
      this.emitTimers[channel] = null;
    }, this.THROTTLE_MS);
  }

  private flushBuffer(channel: keyof Listeners) {
    if (!this.eventBuffers[channel] || this.eventBuffers[channel].length === 0) return;
    
    // We could batch them to the UI, but React hooks currently expect single events for some channels.
    // So we'll emit them linearly but throttled, or just emit the latest if it's metrics.
    // For threatEvents, we want to emit all buffered events to the listener to update the array at once.
    // But since the signature is single event, we'll emit them fast but only once per THROTTLE_MS.
    
    const events = [...this.eventBuffers[channel]];
    this.eventBuffers[channel] = [];

    // Notify listeners
    if (this.listeners[channel]) {
       events.forEach(evt => {
         (this.listeners[channel] as any).forEach((fn: any) => fn(evt));
       });
    }
  }

  connect() {
    if (this.connectionState !== "DISCONNECTED") return;
    this.updateConnectionState("CONNECTING");

    if (TELEMETRY_MODE === "REAL" || TELEMETRY_MODE === "HYBRID") {
        this.connectRealWebSocket();
    }

    if (TELEMETRY_MODE === "MOCK" || TELEMETRY_MODE === "HYBRID") {
        this.startMockGenerators();
    }
  }

  private connectRealWebSocket() {
      try {
          this.ws = new WebSocket(WS_URL);
          
          this.ws.onopen = () => {
              this.updateConnectionState("CONNECTED");
              console.log("[TelemetryClient] Real WebSocket Connected");
          };

          this.ws.onmessage = (event) => {
              try {
                  const data: EventEnvelope<any> = JSON.parse(event.data);
                  if (data.event_type === "THREAT_DETECTED") this.emitBuffered("threatEvent", data);
                  else if (data.event_type === "QUEUE_METRICS_UPDATED") this.emitBuffered("queueMetrics", data);
                  else if (data.event_type === "INFRA_METRICS_UPDATED") this.emitBuffered("infrastructure", data);
                  else if (data.event_type === "BROWSER_TELEMETRY_UPDATED") this.emitBuffered("browserTelemetry", data);
                  else if (data.event_type === "CIRCUIT_BREAKERS_UPDATED") this.emitBuffered("circuitBreakers", data);
              } catch (err) {
                  console.error("[TelemetryClient] Message parsing failed", err);
              }
          };

          this.ws.onclose = () => {
              this.updateConnectionState("DISCONNECTED");
              console.log("[TelemetryClient] Real WebSocket Closed. Reconnecting in 3s...");
              setTimeout(() => this.connectRealWebSocket(), 3000);
          };
      } catch (err) {
          console.error("WebSocket connection error:", err);
          this.updateConnectionState("DISCONNECTED");
      }
  }

  private startMockGenerators() {
    if (TELEMETRY_MODE === "MOCK") {
        setTimeout(() => this.updateConnectionState("CONNECTED"), 500);
    }
    
    this.intervals.push(
      setInterval(() => {
        if (this.connectionState === "CONNECTED" || this.connectionState === "DEGRADED") {
          const burstSize = Math.random() > 0.8 ? (Math.floor(Math.random() * 16) + 5) : 1;
          for (let i = 0; i < burstSize; i++) {
            this.emitBuffered("threatEvent", generateThreatEvent());
          }
        }
      }, TELEMETRY_MODE === "HYBRID" ? 15000 : 800)
    );

    this.intervals.push(
      setInterval(() => {
        if (this.connectionState === "CONNECTED" || this.connectionState === "DEGRADED") {
          this.emitBuffered("queueMetrics", getQueueMetrics());
          this.emitBuffered("infrastructure", getInfrastructureMetadata());
          this.emitBuffered("browserTelemetry", getBrowserTelemetry());
          this.emitBuffered("circuitBreakers", getCircuitBreakers());
        }
      }, TELEMETRY_MODE === "HYBRID" ? 10000 : 3000)
    );
  }

  get isConnected(): boolean {
    return this.connectionState === "CONNECTED" || this.connectionState === "DEGRADED";
  }

  disconnect() {
    console.log("[TelemetryClient] Disconnecting...");
    this.updateConnectionState("DISCONNECTED");
    this.intervals.forEach(clearInterval);
    this.intervals = [];
  }

  subscribe<K extends keyof Listeners>(event: K, callback: Listeners[K][number]) {
    // @ts-ignore
    this.listeners[event].push(callback);
    return () => {
      // @ts-ignore
      this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
    };
  }

  private emit<K extends keyof Listeners>(event: K, data: Parameters<Listeners[K][number]>[0]) {
    if (!this.isConnected) return;
    this.listeners[event].forEach(cb => (cb as any)(data));
  }
}

export const telemetryClient = new TelemetryClient();
