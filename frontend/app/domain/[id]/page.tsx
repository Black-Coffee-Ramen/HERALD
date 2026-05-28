"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { getMockDomainIntelligence } from "@/services/mock-generator";
import { DomainIntelligence } from "@/types";
import { VerdictHeader } from "@/components/domain-detail/VerdictHeader";
import { ScreenshotPanel } from "@/components/domain-detail/ScreenshotPanel";
import { OCRPanel } from "@/components/domain-detail/OCRPanel";
import { IntelligenceTable } from "@/components/domain-detail/IntelligenceTable";
import { ProcessingTimeline } from "@/components/domain-detail/ProcessingTimeline";
import { LoadingState, SectionHeader } from "@/components/shared/Primitives";
import { Shield, Activity, Share2, AlertTriangle, ArrowLeft } from "lucide-react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function DomainIntelligencePage() {
  const params = useParams();
  const id = params.id as string;
  const [intel, setIntel] = useState<DomainIntelligence | null>(null);

  useEffect(() => {
    let timer: NodeJS.Timeout;
    async function fetchData() {
      try {
        const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const res = await fetch(`${baseUrl}/api/export/${id}/json`);
        if (res.ok) {
          const data = await res.json();
          const transformed: DomainIntelligence = {
            id: data.domain,
            domain: data.domain,
            verdict: data.label ? (data.label.toUpperCase() as any) : "PENDING",
            confidenceScore: data.confidence <= 1 ? data.confidence * 100 : data.confidence,
            firstSeen: data.scan_date,
            lastAnalyzed: data.scan_date,
            traceId: "real-trace",
            queueStage: data.lifecycle_state || "COMPLETED",
            dns: data.dns_records ? JSON.parse(data.dns_records) : [],
            ocrFindings: data.ocr_text ? [{text: data.ocr_text, confidence: 100, bbox: {x:0, y:0, w:0, h:0}}] : [],
            screenshotAvailable: !!data.screenshot_path,
            screenshotUrl: data.screenshot_path ? `${baseUrl}/${data.screenshot_path.replace(/\\/g, '/')}` : undefined,
            timeline: [],
            relatedDomains: []
          };
          setIntel(transformed);
          return;
        }
      } catch (e) {
        console.error("Failed to fetch real data, falling back to mock", e);
      }
      
      const mockData = getMockDomainIntelligence(id);
      timer = setTimeout(() => setIntel(mockData), 600);
    }
    fetchData();
    return () => clearTimeout(timer);
  }, [id]);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 flex flex-col font-sans">
      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="w-5 h-5 text-indigo-500" />
            <h1 className="font-bold tracking-widest text-zinc-100">HERALD <span className="text-zinc-500 font-mono text-xs ml-2 uppercase">Ops Console</span></h1>
          </div>
          <nav className="flex items-center gap-6 text-sm font-mono">
            <Link href="/" className="text-zinc-400 hover:text-zinc-200 transition-colors flex items-center gap-2"><Activity className="w-4 h-4"/> Dashboard</Link>
            <Link href="/observability" className="text-zinc-400 hover:text-zinc-200 transition-colors flex items-center gap-2"><Share2 className="w-4 h-4"/> Observability</Link>
            <Link href="/dlq" className="text-zinc-400 hover:text-zinc-200 transition-colors flex items-center gap-2"><AlertTriangle className="w-4 h-4"/> DLQ</Link>
          </nav>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-4 py-6">
        <div className="mb-4">
          <Link href="/" className="text-xs font-mono text-zinc-500 hover:text-indigo-400 transition-colors flex items-center gap-1 w-max">
            <ArrowLeft className="w-3 h-3" /> Back to Dashboard
          </Link>
        </div>

        {!intel ? (
          <div className="h-[60vh] flex items-center justify-center">
            <LoadingState message="Retrieving domain intelligence..." />
          </div>
        ) : (
          <>
            <VerdictHeader intel={intel} />
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              <div className="lg:col-span-2 space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="h-[350px]">
                    <ScreenshotPanel intel={intel} />
                  </div>
                  <div className="h-[350px]">
                    <OCRPanel intel={intel} />
                  </div>
                </div>

                <div className="pt-2">
                  <SectionHeader title="Infrastructure Intelligence" description="DNS, WHOIS, and TLS Certificate details" />
                  <IntelligenceTable intel={intel} />
                </div>
              </div>

              <div className="space-y-6">
                <ProcessingTimeline intel={intel} />
                
                {intel.relatedDomains.length > 0 && (
                  <Card className="bg-zinc-900 border-zinc-800">
                    <CardHeader className="pb-3 shrink-0">
                      <CardTitle className="text-xs font-mono text-zinc-400 uppercase tracking-wider">
                        Infrastructure Relationships
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex flex-wrap gap-2">
                        {intel.relatedDomains.map(rd => (
                          <Badge key={rd} variant="outline" className="font-mono text-xs border-zinc-700 text-zinc-300">
                            {rd}
                          </Badge>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>

            </div>
          </>
        )}
      </main>
    </div>
  );
}
