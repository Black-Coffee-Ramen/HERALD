import { TraceEvent, TraceSpan } from "../types";

const generateId = () => Math.random().toString(36).substring(2, 9);
const randomInt = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1) + min);

export const getMockTrace = (traceId: string, domainName: string): TraceEvent => {
  const isMalicious = Math.random() > 0.4;
  const isDegraded = Math.random() > 0.8;
  const rootSpanId = generateId();
  
  const startTimeMs = Date.now() - randomInt(10000, 60000);
  let currentMs = startTimeMs;

  const spans: TraceSpan[] = [];
  let totalRetries = 0;

  const addSpan = (name: string, workerType: string, baseDuration: number, retryProb: number, canTimeout: boolean = false) => {
    const queueWait = isDegraded ? randomInt(500, 2500) : randomInt(10, 50);
    currentMs += queueWait;
    
    let retries = 0;
    let state: TraceSpan["executionState"] = "COMPLETED";
    let status: TraceSpan["status"] = "SUCCESS";
    let message = "";

    if (Math.random() < retryProb) {
      retries = randomInt(1, 3);
      totalRetries += retries;
      currentMs += retries * 1000; // time spent retrying
      state = "RETRYING";
      message = `Retried ${retries} times due to upstream timeout`;
    }

    if (canTimeout && isDegraded && Math.random() > 0.5) {
      status = "ERROR";
      state = "TIMEOUT";
      message = "Worker execution timed out after 30s";
    }

    const duration = baseDuration + (isDegraded ? randomInt(100, 500) : randomInt(0, 50));
    
    spans.push({
      id: generateId(),
      parentSpanId: rootSpanId,
      name,
      workerType,
      startTime: new Date(currentMs).toISOString(),
      endTime: new Date(currentMs + duration).toISOString(),
      durationMs: duration,
      queueWaitMs: queueWait,
      retryCount: retries,
      executionState: state === "RETRYING" && status === "SUCCESS" ? "COMPLETED" : state,
      status,
      message
    });

    currentMs += duration;
    return status === "SUCCESS";
  };

  spans.push({
    id: rootSpanId,
    name: "domain_observed",
    workerType: "ingestion_api",
    startTime: new Date(currentMs).toISOString(),
    durationMs: 15,
    queueWaitMs: 0,
    retryCount: 0,
    executionState: "COMPLETED",
    status: "SUCCESS"
  });
  currentMs += 15;

  addSpan("lexical_analysis", "lexical_worker", 45, 0.05);
  addSpan("dns_enrichment", "enrichment_worker", 120, 0.2);
  addSpan("whois_lookup", "enrichment_worker", 350, 0.1);
  
  const visualSuccess = addSpan("visual_analysis", "browser_worker", 1500, 0.4, true);
  
  if (visualSuccess) {
    addSpan("ocr_processing", "ml_worker", 850, 0.1);
  }

  addSpan("verdict_persisted", "persistence_worker", 35, 0.05);

  const overallDuration = currentMs - startTimeMs;

  return {
    traceId,
    domain: domainName,
    verdict: isMalicious ? "MALICIOUS" : "BENIGN",
    overallDurationMs: overallDuration,
    totalRetries,
    isDegraded,
    outcome: "VERDICTED",
    spans
  };
};
