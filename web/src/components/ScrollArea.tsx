import { createSignal, onMount, onCleanup, type JSX } from "solid-js";

interface ScrollAreaProps {
  children: JSX.Element;
  class?: string;
  orientation?: "vertical" | "horizontal" | "both";
  // Expose the underlying scroll container element for consumers that
  // need custom behavior (they own any higher-level logic).
  ref?: (el: HTMLDivElement) => void;
}

export default function ScrollArea(props: ScrollAreaProps) {
  const [isScrolling, setIsScrolling] = createSignal(false);
  const [isHoveringScrollbar, setIsHoveringScrollbar] = createSignal(false);
  let scrollContainerRef: HTMLDivElement | undefined;
  let scrollTimeout: number | undefined;

  // Expose scroll methods and ref to parent
  const setRef = (el: HTMLDivElement) => {
    scrollContainerRef = el;
    if (props.ref) {
      props.ref(el);
    }
  };

  const handleScroll = () => {
    setIsScrolling(true);

    // Clear existing timeout
    if (scrollTimeout) {
      clearTimeout(scrollTimeout);
    }

    // Set new timeout to hide scrollbar after 1 second of no scrolling
    scrollTimeout = window.setTimeout(() => {
      setIsScrolling(false);
    }, 3000);
  };

  const handleMouseMove = (event: MouseEvent) => {
    if (!scrollContainerRef) return;

    const rect = scrollContainerRef.getBoundingClientRect();
    const isVertical = props.orientation !== "horizontal";
    const scrollbarWidth = 17; // Standard scrollbar width

    let isInScrollbarArea = false;

    if (isVertical) {
      // Check if mouse is in the rightmost scrollbarWidth pixels
      isInScrollbarArea =
        event.clientX >= rect.right - scrollbarWidth &&
        event.clientX <= rect.right &&
        event.clientY >= rect.top &&
        event.clientY <= rect.bottom;
    } else {
      // For horizontal scrollbar, check bottom area
      isInScrollbarArea =
        event.clientY >= rect.bottom - scrollbarWidth &&
        event.clientY <= rect.bottom &&
        event.clientX >= rect.left &&
        event.clientX <= rect.right;
    }

    setIsHoveringScrollbar(isInScrollbarArea);
  };

  const handleMouseLeave = () => {
    setIsHoveringScrollbar(false);
  };

  onMount(() => {
    if (scrollContainerRef) {
      scrollContainerRef.addEventListener("scroll", handleScroll, {
        passive: true,
      });
      scrollContainerRef.addEventListener("mousemove", handleMouseMove);
      scrollContainerRef.addEventListener("mouseleave", handleMouseLeave);
    }
  });

  onCleanup(() => {
    if (scrollTimeout) {
      clearTimeout(scrollTimeout);
    }
    if (scrollContainerRef) {
      scrollContainerRef.removeEventListener("scroll", handleScroll);
      scrollContainerRef.removeEventListener("mousemove", handleMouseMove);
      scrollContainerRef.removeEventListener("mouseleave", handleMouseLeave);
    }
  });

  const containerClass = () => {
    const baseClass = "mmcp-scrollbar";
    const orientationClass =
      props.orientation === "horizontal"
        ? "overflow-x-auto overflow-y-hidden"
        : props.orientation === "vertical"
        ? "overflow-y-auto overflow-x-hidden"
        : "overflow-auto";

    return `${baseClass} ${orientationClass} ${props.class || ""}`.trim();
  };

  const showScrollbar = () => isScrolling() || isHoveringScrollbar();

  return (
    <div
      ref={setRef}
      class={containerClass()}
      style={{
        "--scrollbar-opacity": showScrollbar() ? "1" : "0",
        "--scrollbar-bg": showScrollbar() ? "#151515" : "transparent",
      }}
    >
      {props.children}
    </div>
  );
}
