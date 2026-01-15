import { createSignal, type JSX } from "solid-js";
import Sidebar from "./Sidebar";

interface LayoutProps {
  children: JSX.Element;
}

export default function Layout(props: LayoutProps) {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = createSignal(false);

  return (
    <div class="flex h-screen bg-[var(--bg-primary)] text-zinc-100">
      <aside
        class={`bg-[var(--bg-secondary)] border-r border-[var(--border-primary)] flex-shrink-0 transition-all duration-300 overflow-hidden ${
          isSidebarCollapsed() ? "w-[64px]" : "w-[280px]"
        }`}
      >
        <Sidebar
          isCollapsed={isSidebarCollapsed()}
          onToggle={() => setIsSidebarCollapsed(!isSidebarCollapsed())}
        />
      </aside>
      <main class="flex-1 overflow-auto">
        <div class="max-w-4xl mx-auto h-full">{props.children}</div>
      </main>
    </div>
  );
}
