import { type FormEvent, useState } from "react";

import type { RagAnswer, Strategy } from "../types";
import { AnswerCard } from "./AnswerCard";

interface QuestionPanelProps {
  documentCount: number;
  answer: RagAnswer | null;
  isAsking: boolean;
  onAsk: (question: string, strategy: Strategy) => Promise<void>;
}

const STRATEGIES: { value: Strategy; label: string }[] = [
  { value: "auto", label: "Auto" },
  { value: "simple", label: "Simple" },
  { value: "multi_query", label: "Multi-query" },
  { value: "decomposition", label: "Decompose" },
  { value: "step_back", label: "Step-back" },
  { value: "hyde", label: "HyDE" },
];

export function QuestionPanel({
  documentCount,
  answer,
  isAsking,
  onAsk,
}: QuestionPanelProps) {
  const [question, setQuestion] = useState("");
  const [strategy, setStrategy] = useState<Strategy>("auto");
  const canAsk = documentCount > 0 && question.trim().length >= 3 && !isAsking;

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (canAsk) {
      void onAsk(question.trim(), strategy);
    }
  }

  return (
    <main className="question-panel">
      <div className="hero-copy">
        <p className="eyebrow">Adaptive retrieval workspace</p>
        <h1>Ask what matters.<br />Trace every answer.</h1>
        <p className="hero-description">
          Explore your documents with answers grounded in the material
          you choose—not the open web.
        </p>
      </div>

      <form className="question-form" onSubmit={handleSubmit}>
        <label htmlFor="question">What would you like to know?</label>
        <textarea
          id="question"
          value={question}
          rows={3}
          maxLength={1000}
          placeholder={
            documentCount > 0
              ? "Ask a specific question about your documents…"
              : "Add a document before asking a question…"
          }
          disabled={documentCount === 0 || isAsking}
          onChange={(event) => setQuestion(event.target.value)}
        />
        <div className="form-actions">
          <div className="strategy-field">
            <label htmlFor="strategy">Retrieval</label>
            <select
              id="strategy"
              value={strategy}
              disabled={isAsking}
              onChange={(event) => setStrategy(event.target.value as Strategy)}
            >
              {STRATEGIES.map((option) => (
                <option value={option.value} key={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <button className="ask-button" type="submit" disabled={!canAsk}>
            {isAsking ? "Finding answer…" : "Ask Grounded"}
            <span aria-hidden="true">↗</span>
          </button>
        </div>
      </form>

      {isAsking && (
        <div className="thinking-card" role="status">
          <span className="thinking-line" />
          <span className="thinking-line short" />
          <p>Reading across your documents…</p>
        </div>
      )}

      {!isAsking && answer && <AnswerCard answer={answer} />}

      {!answer && !isAsking && (
        <div className="guide-row" aria-label="How it works">
          <div><span>01</span><p>Add your source material</p></div>
          <div><span>02</span><p>Ask in natural language</p></div>
          <div><span>03</span><p>Review the cited answer</p></div>
        </div>
      )}
    </main>
  );
}
