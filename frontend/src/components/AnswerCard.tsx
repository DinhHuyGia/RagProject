import type { RagAnswer } from "../types";

interface AnswerCardProps {
  answer: RagAnswer;
}

function readableStrategy(strategy: string): string {
  return strategy.replace("_", " ");
}

export function AnswerCard({ answer }: AnswerCardProps) {
  return (
    <article className="answer-card">
      <div className="answer-heading">
        <p className="eyebrow">Grounded answer</p>
        <span className="strategy-pill">
          {readableStrategy(answer.strategy)}
        </span>
      </div>

      <p className="answer-text">{answer.answer}</p>

      {answer.sources.length > 0 && (
        <section className="sources" aria-labelledby="sources-heading">
          <h3 id="sources-heading">Sources</h3>
          <div className="source-list">
            {answer.sources.map((source, index) => (
              <span
                className="source-chip"
                key={`${source.document_id}-${source.page_number}-${index}`}
              >
                <b>{index + 1}</b>
                {source.filename}
                {source.page_number ? ` · p. ${source.page_number}` : ""}
              </span>
            ))}
          </div>
        </section>
      )}

      <details className="answer-details">
        <summary>How this answer was found</summary>
        <p>{answer.routing_reason}</p>
        {answer.retrieval_queries.length > 0 && (
          <>
            <h4>Search queries</h4>
            <ul>
              {answer.retrieval_queries.map((query) => (
                <li key={query}>{query}</li>
              ))}
            </ul>
          </>
        )}
      </details>
    </article>
  );
}
