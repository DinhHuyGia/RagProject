export type Strategy =
  | "auto"
  | "simple"
  | "multi_query"
  | "decomposition"
  | "step_back"
  | "hyde";

export interface DocumentRecord {
  document_id: string;
  filename: string;
  page_count: number;
  chunk_count: number;
  status: string;
  created_at: string;
}

export interface SourceReference {
  document_id: string | null;
  filename: string;
  page_number: number | null;
}

export interface RagAnswer {
  question: string;
  answer: string;
  strategy: Exclude<Strategy, "auto">;
  indexes: string[];
  routing_reason: string;
  retrieval_queries: string[];
  sources: SourceReference[];
}
