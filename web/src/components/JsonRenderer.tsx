import { createSignal, For, Show } from "solid-js";
import StealthButton from "./StealthButton";

interface JsonValueProps {
  value: unknown;
  depth?: number;
  onCollapse?: () => void;
}

interface JsonObjectProps {
  value: Record<string, unknown>;
  depth: number;
  onCollapse?: () => void;
}

interface JsonArrayProps {
  value: unknown[];
  depth: number;
  onCollapse?: () => void;
}

interface JsonPrimitiveProps {
  value: unknown;
}

export const JsonValue = (props: JsonValueProps) => {
  const depth = () => props.depth ?? 0;

  if (Array.isArray(props.value)) {
    return (
      <JsonArray
        value={props.value}
        depth={depth()}
        onCollapse={props.onCollapse}
      />
    );
  }

  if (typeof props.value === "object" && props.value !== null) {
    return (
      <JsonObject
        value={props.value as Record<string, unknown>}
        depth={depth()}
        onCollapse={props.onCollapse}
      />
    );
  }

  return <JsonPrimitive value={props.value} />;
};

interface JsonPropertyProps {
  key: string;
  value: unknown;
  depth: number;
  onCollapse?: () => void;
}

const JsonProperty = (props: JsonPropertyProps) => {
  const isNested = () =>
    Array.isArray(props.value) ||
    (typeof props.value === "object" && props.value !== null);

  const isArray = () => Array.isArray(props.value);

  if (!isNested()) {
    if (props.onCollapse) {
      return (
        <div class="block">
          <StealthButton onClick={() => props.onCollapse?.()}>
            <div class="block">
              <span
                class={`${
                  !isNested()
                    ? "text-[var(--color-primary)]"
                    : "text-[var(--json-key-color)]"
                } font-medium `}
              >
                {props.key}:
              </span>{" "}
              <JsonPrimitive value={props.value} />
            </div>
          </StealthButton>
        </div>
      );
    }
    return (
      <div class="block">
        <span
          class={`${
            !isNested()
              ? "text-[var(--color-primary)]"
              : "text-[var(--json-key-color)]"
          } font-medium `}
        >
          {props.key}:
        </span>{" "}
        <JsonPrimitive value={props.value} />
      </div>
    );
  }

  const [open, setOpen] = createSignal(false);

  return (
    <div>
      <div class="mb-1">
        <StealthButton onClick={() => setOpen(!open())}>
          <div class="flex items-center gap-2">
            <span class="text-[var(--json-key-color)] font-medium ">
              {props.key}
            </span>
            <Show
              when={!open()}
              fallback={
                <span class="font-mono text-xs">{isArray() ? "[" : "{"}</span>
              }
            >
              <span class="text-xs font-mono">
                {isArray() ? "[ ... ]" : "{ ... }"}
              </span>
            </Show>
          </div>
        </StealthButton>
      </div>
      <Show when={open()}>
        <div class="ml-4">
          <JsonValue
            value={props.value}
            depth={props.depth + 1}
            onCollapse={() => setOpen(false)}
          />
          <StealthButton onClick={() => setOpen(false)}>
            <div class="font-mono text-xs mt-1">{isArray() ? "]" : "}"}</div>
          </StealthButton>
        </div>
      </Show>
    </div>
  );
};

const JsonObject = (props: JsonObjectProps) => {
  const entries = () => Object.entries(props.value);

  return (
    <div class="space-y-3">
      <For each={entries()}>
        {([key, value]) => (
          <JsonProperty
            key={key}
            value={value}
            depth={props.depth}
            onCollapse={props.onCollapse}
          />
        )}
      </For>
    </div>
  );
};

interface JsonArrayItemProps {
  item: unknown;
  index: number;
  depth: number;
  onCollapse?: () => void;
}

const JsonArrayItem = (props: JsonArrayItemProps) => {
  const isNested = () =>
    Array.isArray(props.item) ||
    (typeof props.item === "object" && props.item !== null);

  const isArray = () => Array.isArray(props.item);

  if (!isNested()) {
    if (props.onCollapse) {
      return (
        <div class="block">
          <StealthButton onClick={() => props.onCollapse?.()}>
            <div class="block">
              <span class="text-[var(--color-primary)]/60 ">
                {props.index}.
              </span>{" "}
              <JsonPrimitive value={props.item} />
            </div>
          </StealthButton>
        </div>
      );
    }
    return (
      <div class="block">
        <span class="text-[var(--color-primary)]/60 ">{props.index}.</span>{" "}
        <JsonPrimitive value={props.item} />
      </div>
    );
  }

  const [open, setOpen] = createSignal(false);

  return (
    <div>
      <div class="mb-1">
        <StealthButton onClick={() => setOpen(!open())}>
          <div class="flex items-center gap-2">
            <span class="text-[var(--color-primary)]/60  flex-shrink-0">
              {props.index}
            </span>
            <Show
              when={!open()}
              fallback={
                <span class="font-mono text-xs">{isArray() ? "[" : "{"}</span>
              }
            >
              <span class="text-xs font-mono">
                {isArray() ? "[ ... ]" : "{ ... }"}
              </span>
            </Show>
          </div>
        </StealthButton>
      </div>
      <Show when={open()}>
        <div class="ml-4">
          <JsonValue
            value={props.item}
            depth={props.depth + 1}
            onCollapse={() => setOpen(false)}
          />
          <StealthButton onClick={() => setOpen(false)}>
            <div class="font-mono text-xs mt-1">{isArray() ? "]" : "}"}</div>
          </StealthButton>
        </div>
      </Show>
    </div>
  );
};

const JsonArray = (props: JsonArrayProps) => {
  return (
    <div class="space-y-3">
      <For each={props.value}>
        {(item, index) => (
          <JsonArrayItem
            item={item}
            index={index()}
            depth={props.depth}
            onCollapse={props.onCollapse}
          />
        )}
      </For>
    </div>
  );
};

const JsonPrimitive = (props: JsonPrimitiveProps) => {
  if (props.value === null) {
    return <span class="text-gray-500 italic font-mono">null</span>;
  }

  if (typeof props.value === "string") {
    return (
      <span class="text-[var(--color-primary-highlight)] font-mono">
        "{props.value}"
      </span>
    );
  }

  if (typeof props.value === "boolean") {
    return (
      <span
        class={`${props.value ? "text-green-300" : "text-red-300"} font-mono`}
      >
        {String(props.value)}
      </span>
    );
  }

  if (typeof props.value === "number") {
    return (
      <span class="text-[var(--color-primary)] font-mono">{props.value}</span>
    );
  }

  return (
    <span class="text-[var(--color-primary)] font-mono">
      {String(props.value)}
    </span>
  );
};
