import type { DocumentRecord, RagAnswer, Strategy } from "./types";

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "");
const API_BASE_URL = configuredBaseUrl || "/api";

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, options);

  if (!response.ok) {
    let message = `Request failed with status ${response.status}.`;

    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) {
        message = body.detail;
      }
    } catch {
      // The fallback status message is useful when the API returns no JSON.
    }

    throw new Error(message);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export function listDocuments(): Promise<DocumentRecord[]> {
  return request<DocumentRecord[]>("/documents");
}

export function uploadDocument(file: File): Promise<DocumentRecord> {
  const body = new FormData();
  body.append("file", file);

  return request<DocumentRecord>("/documents", {
    method: "POST",
    body,
  });
}

export function deleteDocument(documentId: string): Promise<void> {
  return request<void>(`/documents/${documentId}`, {
    method: "DELETE",
  });
}

export function askQuestion(
  question: string,
  strategy: Strategy,
): Promise<RagAnswer> {
  return request<RagAnswer>("/rag/ask", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      strategy,
      indexes: ["chunks"],
    }),
  });
}
