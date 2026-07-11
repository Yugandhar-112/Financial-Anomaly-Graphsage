"use client";

import { useMemo, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";

export type GraphNode = {
  id: string;
  fraud_probability: number;
  flagged: boolean;
  true_label?: string;
};

export type GraphEdge = {
  source: string;
  target: string;
};

// Flagged (above-threshold) nodes are always red, regardless of exact
// probability -- that's the operationally meaningful cutoff. Below
// threshold, color shifts green -> amber as probability rises, so a human
// reviewer can still spot "close calls" the model didn't flag.
function riskColor(probability: number, flagged: boolean): string {
  if (flagged) return "#dc2626"; // red-600
  const g = Math.round(200 - probability * 100);
  const r = Math.round(80 + probability * 150);
  return `rgb(${r}, ${g}, 60)`;
}

export default function GraphView({
  nodes,
  edges,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
}) {
  const fgRef = useRef<any>(null);

  // react-force-graph mutates the objects it's given (adds x/y/vx/vy), so we
  // pass fresh copies each render rather than the raw props directly.
  const graphData = useMemo(
    () => ({
      nodes: nodes.map((n) => ({ ...n })),
      links: edges.map((e) => ({ source: e.source, target: e.target })),
    }),
    [nodes, edges]
  );

  return (
    <div className="w-full h-[600px] rounded-lg border border-gray-700 overflow-hidden bg-gray-950">
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        nodeId="id"
        nodeLabel={(node: any) =>
          `${node.id}\nFraud probability: ${(node.fraud_probability * 100).toFixed(1)}%\n` +
          `${node.flagged ? "FLAGGED" : "not flagged"}` +
          `${node.true_label ? `\nTrue label: ${node.true_label}` : ""}`
        }
        nodeColor={(node: any) => riskColor(node.fraud_probability, node.flagged)}
        nodeVal={(node: any) => 2 + node.fraud_probability * 10}
        linkColor={() => "rgba(148, 163, 184, 0.4)"}
        backgroundColor="#030712"
        height={600}
      />
    </div>
  );
}
