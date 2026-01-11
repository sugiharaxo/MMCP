import type { Component } from "solid-js";
import { createSignal } from "solid-js";

const App: Component = () => {
  const [status] = createSignal("MMCP Web Shell Active");

  return (
    <div class="flex min-h-screen items-center justify-center bg-zinc-950 text-zinc-100">
      <div class="text-center">
        <h1 class="text-2xl font-medium tracking-tight">{status()}</h1>
        <p class="text-zinc-500 mt-2">vite + solid + tailwind</p>
      </div>
    </div>
  );
};

export default App;
