export type NodeInput = {
  id: string;
  features: number[];
  true_label?: string;
};

export type EdgeInput = {
  source: string;
  target: string;
};

export type SampleSubgraphResponse = {
  nodes: NodeInput[];
  edges: EdgeInput[];
};

export type PredictionResult = {
  id: string;
  fraud_probability: number;
  flagged: boolean;
};

export type SubgraphPredictionResponse = {
  predictions: PredictionResult[];
  threshold: number;
  model_type: string;
};

// Set NEXT_PUBLIC_API_URL in frontend/.env.local for local dev, and as an
// environment variable on Vercel pointing at the deployed backend (step 7).
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function fetchSampleSubgraph(): Promise<SampleSubgraphResponse> {
  const res = await fetch(`${API_URL}/sample-subgraph`);
  if (!res.ok) {
    throw new Error(`Failed to fetch sample subgraph (${res.status})`);
  }
  return res.json();
}

export async function predictSubgraph(
  nodes: NodeInput[],
  edges: EdgeInput[],
  modelType: string
): Promise<SubgraphPredictionResponse> {
  const res = await fetch(`${API_URL}/predict/subgraph`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // Strip any extra fields (e.g. true_label) -- the API only wants id + features.
      nodes: nodes.map((n) => ({ id: n.id, features: n.features })),
      edges,
      model_type: modelType,
    }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Prediction failed (${res.status}): ${detail}`);
  }
  return res.json();
}
