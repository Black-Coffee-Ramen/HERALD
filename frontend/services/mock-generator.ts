import { ThreatEvent, QueueMetrics, InfrastructureMetadata, CircuitBreakerStatus, EventEnvelope, BrowserWorkerTelemetry } from "../types";
import { applyScenario } from "./mock-scenarios";

export const generateId = () => Math.random().toString(36).substring(2, 15);
const randomInt = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1) + min);
const randomChoice = <T>(arr: T[]): T => arr[randomInt(0, arr.length - 1)];

export const createEnvelope = <T>(eventType: string, payload: T, priority: "HIGH" | "MEDIUM" | "LOW" = "MEDIUM", traceId?: string): EventEnvelope<T> => ({
  event_id: `evt-${generateId()}`,
  event_type: eventType,
  timestamp: new Date().toISOString(),
  trace_id: traceId,
  worker_type: "mock_worker",
  severity: priority === "HIGH" ? "WARNING" : "INFO",
  telemetry_priority: priority,
  degraded_state: Math.random() > 0.9,
  source_service: "mock_generator",
  version: "1.0",
  payload
});

const domains = [
  "secure-login-paypal.com",
  "netflix-update-billing.net",
  "microsoft-auth-portal.com",
  "apple-id-verify.support",
  "chase-online-banking.info",
  "amazon-prime-rewards.co",
  "dhl-package-tracking.net",
  "wellsfargo-secure.com",
  "github-security-alert.io",
  "google-drive-share.link",
  "linkedin-profile-update.com",
  "dropbox-file-access.net",
  "adobe-account-verify.com",
  "twitter-login-check.info",
  "facebook-security-notice.co",
  "instagram-password-reset.com",
  "yahoo-mail-support.net",
  "paypal-dispute-resolution.com",
  "netflix-account-suspension.info",
  "microsoft-office365-update.co"
];

export const generateThreatEvent = (): EventEnvelope<ThreatEvent> => {
  const isMalicious = Math.random() > 0.7;
  const traceId = `tr-${generateId()}`;
  const payload: ThreatEvent = {
    id: generateId(),
    domain: randomChoice(domains),
    verdict: isMalicious ? "MALICIOUS" : Math.random() > 0.5 ? "SUSPICIOUS" : "BENIGN",
    confidenceScore: isMalicious ? randomInt(80, 99) : randomInt(5, 50),
    timestamp: new Date().toISOString(),
    sourceStream: randomChoice(["certstream", "daily_feed", "telegram"]),
    workerStage: randomChoice(["LEXICAL", "DNS", "VISUAL", "OCR", "COMPLETED"]),
    ocrDetections: isMalicious ? randomInt(1, 5) : 0,
    hasScreenshot: Math.random() > 0.3,
    traceId
  };
  return createEnvelope("THREAT_DETECTED", payload, isMalicious ? "HIGH" : "MEDIUM", traceId);
};

export const getQueueMetrics = (): EventEnvelope<QueueMetrics> => {
  const base: QueueMetrics = {
    timestamp: new Date().toISOString(),
    queueDepth: randomInt(100, 5000),
    retryQueueSize: randomInt(10, 50),
    dlqSize: randomInt(0, 50),
    processingThroughput: randomInt(50, 200),
    eventLatencyMs: randomInt(10, 300)
  };
  return createEnvelope("QUEUE_METRICS_UPDATED", applyScenario().mutateQueue(base));
};

export const getInfrastructureMetadata = (): EventEnvelope<InfrastructureMetadata> => {
  const base: InfrastructureMetadata = {
    postgresStatus: randomChoice(["HEALTHY", "DEGRADED"]) as any,
    redisStatus: randomChoice(["HEALTHY", "DEGRADED"]) as any,
    apiLatencyMs: randomInt(10, 150),
    degradedModeActive: Math.random() > 0.9
  };
  return createEnvelope("INFRA_METRICS_UPDATED", applyScenario().mutateInfra(base));
};

export const getBrowserTelemetry = (): EventEnvelope<BrowserWorkerTelemetry> => {
  const degraded = Math.random() > 0.85;
  const browserCrash = degraded && Math.random() > 0.5;
  const base: BrowserWorkerTelemetry = {
    activeSessions: randomInt(15, 60),
    browserLaunchTimeMs: degraded ? randomInt(800, 3500) : randomInt(150, 450),
    screenshotDurationMs: degraded ? randomInt(2500, 8000) : randomInt(800, 1800),
    memoryPressurePercent: degraded ? randomInt(80, 99) : randomInt(30, 65),
    timeoutFrequencyRate: degraded ? randomInt(2, 12) : 0,
    crashFrequencyRate: browserCrash ? randomInt(1, 5) : 0,
    isolationFailures: browserCrash ? 1 : 0
  };
  return createEnvelope("BROWSER_TELEMETRY_UPDATED", base);
};

export const getCircuitBreakers = (): EventEnvelope<CircuitBreakerStatus[]> => {
  const base: CircuitBreakerStatus[] = [
    { name: "DNS Resolver", state: "CLOSED", recentFailures: 0 },
    { name: "WHOIS API", state: Math.random() > 0.9 ? "HALF_OPEN" : "CLOSED", recentFailures: randomInt(0, 5) },
    { name: "Browser Worker", state: "CLOSED", recentFailures: 0 },
    { name: "ML Inference", state: "CLOSED", recentFailures: 0 },
    { name: "PostgreSQL", state: "CLOSED", recentFailures: 0 },
    { name: "Redis", state: "CLOSED", recentFailures: 0 }
  ];
  return createEnvelope("CIRCUIT_BREAKERS_UPDATED", applyScenario().mutateBreakers(base));
};

export const getMockDomainIntelligence = (id: string, domainName?: string): any => {
  const isMalicious = Math.random() > 0.4; // 60% chance malicious for testing
  const domain = domainName || randomChoice(domains);
  
  return {
    id,
    domain,
    verdict: isMalicious ? "MALICIOUS" : "BENIGN",
    confidenceScore: isMalicious ? randomInt(85, 99) + Math.random() : randomInt(5, 25) + Math.random(),
    firstSeen: new Date(Date.now() - randomInt(100000, 10000000)).toISOString(),
    lastAnalyzed: new Date().toISOString(),
    traceId: `tr-${generateId()}-${generateId()}`,
    queueStage: "COMPLETED",
    dns: [
      { type: "A", value: `${randomInt(1,255)}.${randomInt(1,255)}.${randomInt(1,255)}.${randomInt(1,255)}` },
      { type: "A", value: `${randomInt(1,255)}.${randomInt(1,255)}.${randomInt(1,255)}.${randomInt(1,255)}` },
      { type: "MX", value: `mail.${domain}` }
    ],
    tls: {
      issuer: isMalicious ? "Let's Encrypt Authority X3" : "DigiCert SHA2 Secure Server CA",
      validFrom: new Date(Date.now() - randomInt(1000, 1000000)).toISOString(),
      validTo: new Date(Date.now() + randomInt(1000000, 10000000)).toISOString(),
      subjectAltNames: [domain, `www.${domain}`]
    },
    whois: {
      registrar: isMalicious ? "NAMECHEAP INC" : "MarkMonitor Inc.",
      creationDate: isMalicious ? new Date(Date.now() - randomInt(1000, 5000000)).toISOString() : "1999-01-01T00:00:00Z",
      expirationDate: new Date(Date.now() + randomInt(1000000, 10000000)).toISOString(),
      nameServers: [`ns1.${isMalicious ? 'cheap-dns.net' : 'awsdns.com'}`, `ns2.${isMalicious ? 'cheap-dns.net' : 'awsdns.com'}`]
    },
    ocrFindings: isMalicious ? [
      { text: "Sign in to your account", confidence: 98.5, bbox: { x: 100, y: 150, w: 200, h: 20 } },
      { text: "Verify your identity", confidence: 95.2, bbox: { x: 100, y: 180, w: 180, h: 20 } },
      { text: "Enter your password to continue", confidence: 92.1, bbox: { x: 100, y: 220, w: 250, h: 20 } }
    ] : [],
    screenshotAvailable: true,
    screenshotUrl: "https://images.unsplash.com/photo-1555066931-4365d14bab8c?q=80&w=800&auto=format&fit=crop", // generic code/hacker image for mock
    timeline: [
      { stage: "DOMAIN_OBSERVED", timestamp: new Date(Date.now() - 5000).toISOString(), status: "SUCCESS", durationMs: 12 },
      { stage: "LEXICAL_ANALYSIS", timestamp: new Date(Date.now() - 4800).toISOString(), status: "SUCCESS", durationMs: 45, message: "Suspicious keyword count: 2" },
      { stage: "DNS_ENRICHMENT", timestamp: new Date(Date.now() - 4500).toISOString(), status: "SUCCESS", durationMs: 250, message: "Resolved 3 records" },
      { stage: "VISUAL_ANALYSIS", timestamp: new Date(Date.now() - 3000).toISOString(), status: "SUCCESS", durationMs: 1450, message: "Screenshot captured successfully" },
      { stage: "OCR_PROCESSING", timestamp: new Date(Date.now() - 1000).toISOString(), status: "SUCCESS", durationMs: 850, message: isMalicious ? "Detected 3 high-risk phrases" : "No high-risk text found" },
      { stage: "VERDICT_PERSISTED", timestamp: new Date().toISOString(), status: "SUCCESS", durationMs: 45 }
    ],
    relatedDomains: isMalicious ? [`login-${domain}`, `auth-${domain}`] : []
  };
};
