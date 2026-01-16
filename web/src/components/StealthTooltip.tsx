import { createSignal, onCleanup, type JSX } from "solid-js";

interface StealthTooltipProps {
  text: string;
  children: JSX.Element;
  class?: string;
  position?: "top" | "bottom" | "left" | "right";
}

export default function StealthTooltip(props: StealthTooltipProps) {
  const [isVisible, setIsVisible] = createSignal(false);
  let showTimeout: number | undefined;
  let hideTimeout: number | undefined;

  const position = () => props.position || "top";

  const showTooltip = () => {
    if (hideTimeout) {
      clearTimeout(hideTimeout);
      hideTimeout = undefined;
    }

    showTimeout = window.setTimeout(() => {
      setIsVisible(true);
      showTimeout = undefined;
    });
  };

  const hideTooltip = () => {
    if (showTimeout) {
      clearTimeout(showTimeout);
      showTimeout = undefined;
    }

    hideTimeout = window.setTimeout(() => {
      setIsVisible(false);
      hideTimeout = undefined;
    }); // 0.2s fade out
  };

  onCleanup(() => {
    if (showTimeout) clearTimeout(showTimeout);
    if (hideTimeout) clearTimeout(hideTimeout);
  });

  const getPositionClass = () => {
    const pos = position();

    switch (pos) {
      case "top":
        return "-top-1 left-1/2 transform -translate-x-1/2 -translate-y-full";
      case "bottom":
        return "-bottom-1 left-1/2 transform -translate-x-1/2 translate-y-full";
      case "left":
        return "-left-1 top-1/2 transform -translate-x-full -translate-y-1/2";
      case "right":
        return "-right-1 top-1/2 transform translate-x-full -translate-y-1/2";
      default:
        return "";
    }
  };

  return (
    <div class={`relative inline-block ${props.class || ""}`}>
      <div
        onMouseEnter={showTooltip}
        onMouseLeave={hideTooltip}
        onFocus={showTooltip}
        onBlur={hideTooltip}
      >
        {props.children}
      </div>

      <div
        class={`absolute z-50 pointer-events-none transition-opacity duration-200 ease-in-out ${getPositionClass()} ${
          isVisible() ? "opacity-100" : "opacity-0"
        }`}
      >
        <div
          class="px-3 py-2 text-sm rounded-md shadow-lg whitespace-nowrap select-none"
          style={{
            background: "var(--bg-primary)",
            color: "var(--color-primary)",
          }}
        >
          {props.text}
        </div>
      </div>
    </div>
  );
}
