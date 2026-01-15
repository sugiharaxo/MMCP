import { splitProps, type JSX } from "solid-js";

interface StealthButtonProps
  extends JSX.ButtonHTMLAttributes<HTMLButtonElement> {
  children: JSX.Element;
  highlight?: boolean;
}

export default function StealthButton(props: StealthButtonProps) {
  const [local, others] = splitProps(props, ["children", "class", "highlight"]);

  return (
    <button
      {...others}
      class={`cursor-pointer border-none bg-transparent outline-none select-none transition-colors text-[var(--color-primary)] hover:text-[var(--color-primary-highlight)] ${
        local.highlight ? "text-[var(--color-primary-highlight)]" : ""
      } ${local.class || ""}`}
    >
      {local.children}
    </button>
  );
}
