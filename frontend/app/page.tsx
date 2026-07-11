"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { fetchSampleSubgraph, predictSubgraph, NodeInput, EdgeInput } from "@/lib/api";

// react-force-graph touches `window`/`document` at import time, so it must
// never be part of the server-rendered bundle.
const GraphView = dynamic(() => import("@/components/GraphView"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[600px] flex items-center justify-center text-gray-400">
      Loading graph...
    </div>
  ),
});

const MODEL_OPTIONS = [
  { value: "graphsage", label: "GraphSAGE" },
  { value: "gat", label: "GAT" },
  { value: "mlp", label: "MLP (non-graph baseline)" },
  { value: "xgboost", label: "XGBoost (non-graph baseline)" },
];

type PredictionMap = Record<string, { fraud_probability: number; flagged: boolean }>;

export default function Home() {
  const [rawNodes, setRawNodes] = useState<NodeInput[]>([]);
  const [rawEdges, setRawEdges] = useState<EdgeInput[]>([]);
  const [modelType, setModelType] = useState("graphsage");
  const [predictions, setPredictions] = useState<PredictionMap>({});
  const [threshold, setThreshold] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runPrediction(nodes: NodeInput[], edges: EdgeInput[], model: string) {
    setLoading(true);
    setError(null);
    try {
      const result = await predictSubgraph(nodes, edges, model);
      const map: PredictionMap = {};
      for (const p of result.predictions) {
        map[p.id] = { fraud_probability: p.fraud_probability, flagged: p.flagged };
      }
      setPredictions(map);
      setThreshold(result.threshold);
    } catch (e: any) {
      setError(e.message || "Prediction failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  async function handleLoadSample() {
    setLoading(true);
    setError(null);
    try {
      const sample = await fetchSampleSubgraph();
      setRawNodes(sample.nodes);
      setRawEdges(sample.edges);
      await runPrediction(sample.nodes, sample.edges, modelType);
    } catch (e: any) {
      setError(e.message || "Failed to load sample subgraph. Is the backend running?");
      setLoading(false);
    }
  }

  async function handleModelChange(newModel: string) {
    setModelType(newModel);
    if (rawNodes.length > 0) {
      await runPrediction(rawNodes, rawEdges, newModel);
    }
  }

  function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const parsed = JSON.parse(reader.result as string);
        if (!Array.isArray(parsed.nodes)) {
          throw new Error('Uploaded JSON must have a "nodes" array (each with "id" and "features").');
        }
        setRawNodes(parsed.nodes);
        setRawEdges(parsed.edges || []);
        await runPrediction(parsed.nodes, parsed.edges || [], modelType);
      } catch (err: any) {
        setError(err.message || "Failed to parse uploaded file");
      }
    };
    reader.readAsText(file);
    e.target.value = ""; // allow re-uploading the same filename later
  }

  const graphNodes = rawNodes.map((n) => ({
    id: n.id,
    true_label: n.true_label,
    fraud_probability: predictions[n.id]?.fraud_probability ?? 0,
    flagged: predictions[n.id]?.flagged ?? false,
  }));

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-8">
      <div className="max-w-5xl mx-auto space-y-6">
        <header>
          <h1 className="text-2xl font-semibold">Financial Anomaly Detection — Live Demo</h1>
          <p className="text-gray-400 mt-1">
            Transaction subgraphs scored by a GraphSAGE / GAT / MLP / XGBoost model trained on
            the Elliptic Bitcoin dataset. Nodes are sized and colored by predicted fraud risk;
            flagged (above-threshold) nodes are red.
          </p>
        </header>

        <div className="flex flex-wrap items-center gap-4">
          <button
            onClick={handleLoadSample}
            disabled={loading}
            className="px-4 py-2 rounded-md bg-blue-600 hover:bg-blue-500 disabled:opacity-50 transition"
          >
            Load sample subgraph
          </button>

          <label className="px-4 py-2 rounded-md bg-gray-800 hover:bg-gray-700 cursor-pointer transition">
            Upload subgraph JSON
            <input type="file" accept="application/json" onChange={handleFileUpload} className="hidden" />
          </label>

          <select
            value={modelType}
            onChange={(e) => handleModelChange(e.target.value)}
            className="px-3 py-2 rounded-md bg-gray-800 border border-gray-700"
          >
            {MODEL_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>

          {threshold !== null && (
            <span className="text-sm text-gray-400">Decision threshold: {threshold.toFixed(3)}</span>
          )}
        </div>

        {error && (
          <div className="px-4 py-2 rounded-md bg-red-900/40 border border-red-700 text-red-200 text-sm">
            {error}
          </div>
        )}

        {loading && <p className="text-gray-400 text-sm">Scoring subgraph…</p>}

        {rawNodes.length > 0 ? (
          <GraphView nodes={graphNodes} edges={rawEdges} />
        ) : (
          <div className="w-full h-[600px] rounded-lg border border-dashed border-gray-700 flex items-center justify-center text-gray-500">
            Load the sample subgraph or upload your own to get started.
          </div>
        )}

        <p className="text-xs text-gray-500">
          Note: on the real time-based held-out test set, XGBoost (no graph structure) currently
          outperforms the graph models -- see the project README for the full comparison table
          and an explanation (a documented distribution shift in the Elliptic dataset around a
          real dark-market shutdown event).
        </p>
      </div>
    </main>
  );
}
