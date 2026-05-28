import { DomainIntelligence } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { FileText, ShieldAlert } from "lucide-react";

export function OCRPanel({ intel }: { intel: DomainIntelligence }) {
  if (!intel.ocrFindings || intel.ocrFindings.length === 0) {
    return (
      <Card className="bg-zinc-900 border-zinc-800 h-full flex flex-col">
        <CardHeader className="pb-3 shrink-0">
          <CardTitle className="text-xs font-mono text-zinc-400 uppercase tracking-wider flex items-center gap-2">
            <FileText className="w-4 h-4" />
            Extracted Text & Brands
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 min-h-[150px] flex items-center justify-center text-zinc-500 font-mono text-xs">
          No OCR findings available for this domain.
        </CardContent>
      </Card>
    );
  }

  const getSuspiciousClass = (confidence: number) => {
    if (confidence > 80) return "bg-red-950/40 text-red-400 border border-red-900/50 p-1 rounded-sm font-bold";
    if (confidence > 50) return "bg-yellow-950/40 text-yellow-400 border border-yellow-900/50 p-1 rounded-sm";
    return "text-zinc-300";
  };

  return (
    <Card className="bg-zinc-900 border-zinc-800 h-full flex flex-col">
      <CardHeader className="pb-3 shrink-0 flex flex-row items-center justify-between">
        <CardTitle className="text-xs font-mono text-zinc-400 uppercase tracking-wider flex items-center gap-2">
          <FileText className="w-4 h-4" />
          Extracted Text & Brands
        </CardTitle>
        <Badge variant="destructive" className="bg-red-900/50 text-red-400 border-red-800 font-mono text-[10px]">
          {intel.ocrFindings.length} FINDINGS
        </Badge>
      </CardHeader>
      <CardContent className="flex-1 p-0 overflow-hidden">
        <ScrollArea className="h-[300px] px-4 pb-4">
          <div className="space-y-3">
            {intel.ocrFindings.map((finding, idx) => (
              <div key={idx} className="flex flex-col gap-2 p-3 rounded-lg border border-zinc-800 bg-zinc-950">
                <div className="flex justify-between items-start">
                  <span className={`font-mono text-sm break-all ${getSuspiciousClass(finding.confidence)}`}>
                    "{finding.text}"
                  </span>
                </div>
                <div className="flex justify-between items-center text-[10px] font-mono mt-2">
                   <div className="text-zinc-500">
                     BBOX: [{finding.bbox.x}, {finding.bbox.y}, {finding.bbox.w}, {finding.bbox.h}]
                   </div>
                   <div className="flex items-center gap-1 text-zinc-400">
                     Confidence: <span className={typeof finding.confidence === "number" && finding.confidence > 80 ? "text-red-400" : ""}>{typeof finding.confidence === "number" ? finding.confidence.toFixed(1) : "--"}%</span>
                   </div>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
