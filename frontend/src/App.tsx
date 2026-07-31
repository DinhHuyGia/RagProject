import { useCallback, useEffect, useState } from "react";

import {
  askQuestion,
  deleteDocument,
  listDocuments,
  uploadDocument,
} from "./api";
import { DocumentPanel } from "./components/DocumentPanel";
import { QuestionPanel } from "./components/QuestionPanel";
import type { DocumentRecord, RagAnswer, Strategy } from "./types";

interface Notice {
  kind: "success" | "error";
  message: string;
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return "Something unexpected happened. Please try again.";
}

export default function App() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [answer, setAnswer] = useState<RagAnswer | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);

  const loadDocuments = useCallback(async () => {
    try {
      setDocuments(await listDocuments());
    } catch (error) {
      setNotice({
        kind: "error",
        message: `Could not reach the document API. ${getErrorMessage(error)}`,
      });
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDocuments();
  }, [loadDocuments]);

  useEffect(() => {
    if (!notice) {
      return;
    }
    const timer = window.setTimeout(() => setNotice(null), 5000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  async function handleUpload(file: File) {
    setIsUploading(true);
    setNotice(null);
    try {
      const uploaded = await uploadDocument(file);
      setDocuments((current) => [uploaded, ...current]);
      setNotice({ kind: "success", message: `${file.name} is ready to search.` });
    } catch (error) {
      setNotice({ kind: "error", message: getErrorMessage(error) });
    } finally {
      setIsUploading(false);
    }
  }

  async function handleDelete(documentId: string) {
    setDeletingId(documentId);
    setNotice(null);
    try {
      await deleteDocument(documentId);
      setDocuments((current) =>
        current.filter((document) => document.document_id !== documentId),
      );
      setAnswer(null);
      setNotice({ kind: "success", message: "Document removed." });
    } catch (error) {
      setNotice({ kind: "error", message: getErrorMessage(error) });
    } finally {
      setDeletingId(null);
    }
  }

  async function handleAsk(question: string, strategy: Strategy) {
    setIsAsking(true);
    setAnswer(null);
    setNotice(null);
    try {
      setAnswer(await askQuestion(question, strategy));
    } catch (error) {
      setNotice({ kind: "error", message: getErrorMessage(error) });
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="wordmark" href="/" aria-label="Grounded home">
          Grounded<span>.</span>
        </a>
        <div className="api-status">
          <span className={isLoading ? "status-dot checking" : "status-dot"} />
          {isLoading ? "Checking workspace" : "Adaptive RAG workspace"}
        </div>
      </header>

      <div className="workspace">
        <DocumentPanel
          documents={documents}
          isLoading={isLoading}
          isUploading={isUploading}
          deletingId={deletingId}
          onUpload={handleUpload}
          onDelete={handleDelete}
        />
        <QuestionPanel
          documentCount={documents.length}
          answer={answer}
          isAsking={isAsking}
          onAsk={handleAsk}
        />
      </div>

      {notice && (
        <div className={`notice ${notice.kind}`} role="status">
          <span>{notice.kind === "success" ? "Done" : "Notice"}</span>
          {notice.message}
          <button type="button" onClick={() => setNotice(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}
    </div>
  );
}
