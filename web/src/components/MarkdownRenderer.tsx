import type { Component } from "solid-js";
import { createMemo } from "solid-js";
import { marked } from "marked";

interface MarkdownRendererProps {
  content: string;
  class?: string;
}

/**
 * MarkdownRenderer
 *
 * Lightweight markdown renderer using marked.
 * Used for both historical messages and the live agent stream.
 */
const MarkdownRenderer: Component<MarkdownRendererProps> = (props) => {
  // Guard against undefined/null content; downstream libraries expect a string.
  const safeContent = () => props.content ?? "";

  // Parse markdown; memoized to avoid unnecessary work.
  const renderedHTML = createMemo(() => {
    return marked.parse(safeContent(), { async: false }) as string;
  });

  return (
    <div
      class={`mmcp-markdown text-[15px] leading-relaxed ${
        props.class || ""
      }`.trim()}
      innerHTML={renderedHTML()}
    />
  );
};

export default MarkdownRenderer;
