import { useRef, useState } from "react";

import type { DocumentRecord } from "../types";

interface DocumentPanelProps {
  documents: DocumentRecord[];
  isLoading: boolean;
  isUploading: boolean;
  deletingId: string | null;
  onUpload: (file: File) => Promise<void>;
  onDelete: (documentId: string) => Promise<void>;
}

const ACCEPTED_FILE_TYPES = ".pdf,.docx,.txt,.md";

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Recently added";
  }

  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
  }).format(date);
}

export function DocumentPanel({
  documents,
  isLoading,
  isUploading,
  deletingId,
  onUpload,
  onDelete,
}: DocumentPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  async function handleFile(file: File | undefined) {
    if (!file || isUploading) {
      return;
    }
    await onUpload(file);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  return (
    <aside className="document-panel" aria-labelledby="documents-heading">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Knowledge base</p>
          <h2 id="documents-heading">Your documents</h2>
        </div>
        <span className="count-badge" aria-label={`${documents.length} documents`}>
          {documents.length}
        </span>
      </div>

      <button
        className={`upload-zone ${isDragging ? "is-dragging" : ""}`}
        type="button"
        disabled={isUploading}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          void handleFile(event.dataTransfer.files[0]);
        }}
      >
        <input
          ref={inputRef}
          className="visually-hidden"
          type="file"
          accept={ACCEPTED_FILE_TYPES}
          onChange={(event) => void handleFile(event.target.files?.[0])}
          tabIndex={-1}
        />
        <span className="upload-mark" aria-hidden="true">+</span>
        <span className="upload-title">
          {isUploading ? "Indexing document…" : "Add a document"}
        </span>
        <span className="upload-caption">PDF, DOCX, TXT or MD · up to 10 MB</span>
      </button>

      <div className="document-list" aria-live="polite">
        {isLoading ? (
          <p className="empty-state">Loading your library…</p>
        ) : documents.length === 0 ? (
          <div className="empty-state">
            <span className="empty-index">01</span>
            <p>Your library is empty.</p>
            <small>Add a document to begin asking grounded questions.</small>
          </div>
        ) : (
          documents.map((document, index) => (
            <article className="document-item" key={document.document_id}>
              <span className="document-index">
                {String(index + 1).padStart(2, "0")}
              </span>
              <div className="document-details">
                <strong title={document.filename}>{document.filename}</strong>
                <span>
                  {document.page_count} {document.page_count === 1 ? "page" : "pages"}
                  {" · "}
                  {document.chunk_count} chunks
                  {" · "}
                  {formatDate(document.created_at)}
                </span>
              </div>
              <button
                className="text-button danger"
                type="button"
                disabled={deletingId === document.document_id}
                onClick={() => void onDelete(document.document_id)}
                aria-label={`Delete ${document.filename}`}
              >
                {deletingId === document.document_id ? "…" : "Remove"}
              </button>
            </article>
          ))
        )}
      </div>
    </aside>
  );
}
