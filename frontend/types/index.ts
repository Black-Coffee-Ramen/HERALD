export type ThreatVerdict = "BENIGN" | "SUSPICIOUS" | "MALICIOUS" | "PENDING";
export type WorkerStage = "RECEIVED" | "LEXICAL" | "DNS" | "VISUAL" | "OCR" | "COMPLETED" | "FAILED";

export interface EventEnvelope<T> {
  event_id: string;
  event_type: string;
  trace_id?: string;
  timestamp: string;
  worker_type: string;
  severity: "INFO" | "WARNING" | "ERROR" | "CRITICAL";
  telemetry_priority: "HIGH" | "MEDIUM" | "LOW";
  degraded_state: boolean;
  source_service: string;
  payload: T;
  version: string;
}

export interface ThreatEvent {
  id: string;
  domain: string;
  verdict: ThreatVerdict;
  confidenceScore: number;
  timestamp: string; // ISO format
  sourceStream: string;
  workerStage: WorkerStage;
  ocrDetections: number;
  hasScreenshot: boolean;
  traceId: string;
}

export interface QueueMetrics {
  timestamp: string;
  queueDepth: number;
  retryQueueSize: number;
  dlqSize: number;
  processingThroughput: number; // events per second
  eventLatencyMs: number;
}

export interface WorkerMetrics {
  workerType: string;
  concurrency: number;
  activeJobs: number;
  errorRate: number;
  cpuUtilization: number;
  memoryUtilization: number;
}

export interface BrowserWorkerTelemetry {
  activeSessions: number;
  browserLaunchTimeMs: number;
  screenshotDurationMs: number;
  memoryPressurePercent: number;
  timeoutFrequencyRate: number;
  crashFrequencyRate: number;
  isolationFailures: number;
}

export type CircuitBreakerState = "CLOSED" | "OPEN" | "HALF_OPEN";

export interface CircuitBreakerStatus {
  name: string;
  state: CircuitBreakerState;
  recentFailures: number;
  lastFailureTime?: string;
  nextRetryTime?: string;
}

export interface OCRFinding {
  text: string;
  confidence: number;
  bbox: { x: number; y: number; w: number; h: number };
}

export interface BrowserAnalysisResult {
  screenshotAvailable: boolean;
  timeoutOccurred: boolean;
  loadTimeMs: number;
  domElementCount: number;
}

export interface InfrastructureMetadata {
  postgresStatus: "HEALTHY" | "DEGRADED" | "DOWN";
  redisStatus: "HEALTHY" | "DEGRADED" | "DOWN";
  apiLatencyMs: number;
  degradedModeActive: boolean;
}

export interface TraceSpan {
  id: string; // span_id
  parentSpanId?: string;
  name: string;
  workerType: string;
  startTime: string;
  endTime?: string;
  durationMs: number;
  queueWaitMs: number;
  retryCount: number;
  executionState: "COMPLETED" | "FAILED" | "RETRYING" | "TIMEOUT" | "DEGRADED";
  status: "SUCCESS" | "ERROR" | "PENDING";
  message?: string;
}

export interface TraceEvent {
  traceId: string;
  domain: string;
  verdict: ThreatVerdict;
  overallDurationMs: number;
  totalRetries: number;
  isDegraded: boolean;
  outcome: "VERDICTED" | "DROPPED" | "IN_PROGRESS";
  spans: TraceSpan[];
}

export interface DLQEntry {
  jobId: string;
  traceId: string;
  failureClass: string;
  workerType: string;
  retryAttempts: number;
  retryEligible: boolean;
  timestamp: string;
  dependencyFailureSource?: string;
}

export interface DnsRecord {
  type: string;
  value: string;
}

export interface TlsMetadata {
  issuer: string;
  validFrom: string;
  validTo: string;
  subjectAltNames: string[];
}

export interface WhoisMetadata {
  registrar: string;
  creationDate: string;
  expirationDate: string;
  nameServers: string[];
}

export interface TimelineEvent {
  stage: string;
  timestamp: string;
  status: "SUCCESS" | "PENDING" | "FAILED";
  durationMs?: number;
  message?: string;
}

export interface DomainIntelligence {
  id: string;
  domain: string;
  verdict: ThreatVerdict;
  confidenceScore: number;
  firstSeen: string;
  lastAnalyzed: string;
  traceId: string;
  queueStage: WorkerStage;
  dns: DnsRecord[];
  tls?: TlsMetadata;
  whois?: WhoisMetadata;
  ocrFindings: OCRFinding[];
  screenshotAvailable: boolean;
  screenshotUrl?: string; // mock URL or placeholder
  timeline: TimelineEvent[];
  relatedDomains: string[];
}
