import { DomainIntelligence } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import { Globe, Lock, Info } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export function IntelligenceTable({ intel }: { intel: DomainIntelligence }) {
  return (
    <div className="flex flex-col gap-4">
      {/* DNS Records */}
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader className="pb-2 pt-4 px-4">
          <CardTitle className="text-xs font-mono text-zinc-400 uppercase tracking-wider flex items-center gap-2">
            <Globe className="w-4 h-4" /> DNS Records
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-4 pt-0">
          <Table>
            <TableBody>
              {intel.dns.map((record, idx) => (
                <TableRow key={idx} className="border-zinc-800/50">
                  <TableCell className="font-mono text-xs text-zinc-500 w-[60px] py-2">{record.type}</TableCell>
                  <TableCell className="font-mono text-xs text-zinc-300 py-2 break-all">{record.value}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* WHOIS Metadata */}
      {intel.whois && (
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-xs font-mono text-zinc-400 uppercase tracking-wider flex items-center gap-2">
              <Info className="w-4 h-4" /> WHOIS Metadata
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-4 pt-0">
            <Table>
              <TableBody>
                <TableRow className="border-zinc-800/50">
                  <TableCell className="font-mono text-xs text-zinc-500 py-2 w-[120px]">Registrar</TableCell>
                  <TableCell className="font-mono text-xs text-zinc-300 py-2">{intel.whois.registrar}</TableCell>
                </TableRow>
                <TableRow className="border-zinc-800/50">
                  <TableCell className="font-mono text-xs text-zinc-500 py-2">Creation Date</TableCell>
                  <TableCell className="font-mono text-xs text-zinc-300 py-2">{intel.whois.creationDate}</TableCell>
                </TableRow>
                <TableRow className="border-zinc-800/50">
                  <TableCell className="font-mono text-xs text-zinc-500 py-2">Nameservers</TableCell>
                  <TableCell className="font-mono text-xs text-zinc-300 py-2">
                    <div className="flex flex-col gap-1">
                      {intel.whois.nameServers.map(ns => (
                        <span key={ns} className="text-zinc-400">{ns}</span>
                      ))}
                    </div>
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* TLS Metadata */}
      {intel.tls && (
        <Card className="bg-zinc-900 border-zinc-800">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-xs font-mono text-zinc-400 uppercase tracking-wider flex items-center gap-2">
              <Lock className="w-4 h-4" /> TLS Certificate
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-4 pt-0">
            <Table>
              <TableBody>
                <TableRow className="border-zinc-800/50">
                  <TableCell className="font-mono text-xs text-zinc-500 py-2 w-[120px]">Issuer</TableCell>
                  <TableCell className="font-mono text-xs text-zinc-300 py-2">{intel.tls.issuer}</TableCell>
                </TableRow>
                <TableRow className="border-zinc-800/50">
                  <TableCell className="font-mono text-xs text-zinc-500 py-2">Valid From</TableCell>
                  <TableCell className="font-mono text-xs text-zinc-300 py-2">{intel.tls.validFrom}</TableCell>
                </TableRow>
                <TableRow className="border-zinc-800/50">
                  <TableCell className="font-mono text-xs text-zinc-500 py-2">Subject Alt Names</TableCell>
                  <TableCell className="font-mono text-xs text-zinc-300 py-2">
                    <div className="flex flex-wrap gap-1">
                      {intel.tls.subjectAltNames.map(san => (
                        <Badge key={san} variant="outline" className="text-[9px] font-mono bg-zinc-950 border-zinc-700">{san}</Badge>
                      ))}
                    </div>
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
