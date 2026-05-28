import { DomainIntelligence } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Camera, AlertCircle, Maximize2 } from "lucide-react";
import Image from "next/image";

export function ScreenshotPanel({ intel }: { intel: DomainIntelligence }) {
  return (
    <Card className="bg-zinc-900 border-zinc-800 h-full flex flex-col">
      <CardHeader className="pb-3 shrink-0 flex flex-row items-center justify-between">
        <CardTitle className="text-xs font-mono text-zinc-400 uppercase tracking-wider flex items-center gap-2">
          <Camera className="w-4 h-4" />
          Visual Evidence
        </CardTitle>
        {intel.screenshotAvailable && (
          <button className="text-zinc-500 hover:text-zinc-300 transition-colors">
            <Maximize2 className="w-4 h-4" />
          </button>
        )}
      </CardHeader>
      <CardContent className="flex-1 min-h-[300px] flex items-center justify-center p-4 pt-0">
        <div className="w-full h-full rounded border border-zinc-800 bg-zinc-950 flex flex-col items-center justify-center overflow-hidden relative group cursor-pointer">
          {intel.screenshotAvailable && intel.screenshotUrl ? (
             // Using standard img tag instead of next/image for mock URLs to avoid strict domain config issues
             <img 
               src={intel.screenshotUrl} 
               alt={`Screenshot of ${intel.domain}`} 
               className="object-cover w-full h-full opacity-80 group-hover:opacity-100 transition-opacity blur-[1px] hover:blur-none"
             />
          ) : (
            <div className="flex flex-col items-center justify-center text-zinc-500 p-6 text-center">
              <AlertCircle className="w-8 h-8 mb-2 opacity-50" />
              <p className="font-mono text-xs uppercase mb-1">Evidence Unavailable</p>
              <p className="text-[10px] font-mono text-zinc-600">Browser timeout or degraded processing</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
