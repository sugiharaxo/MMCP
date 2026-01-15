import { z } from "zod";

export const RoutingSchema = z.object({
  address: z.enum(["session", "user"]),
  target: z.enum(["user", "agent"]),
  handler: z.enum(["system", "agent"]),
});

export const NotificationSchema = z.object({
  type: z.literal("notification"),
  id: z.string(),
  session_id: z.string().nullable().optional(),
  content: z.string(),
  routing: RoutingSchema,
  owner_lease: z.number(),
  metadata: z.record(z.string(), z.unknown()).nullable().optional(),
  timestamp: z.string().nullable().optional(),
});

export type Notification = z.infer<typeof NotificationSchema>;

export interface ChatResponse {
  response?: string;
  type: "regular" | "action_request" | "notification" | "tool_use";
  session_id: string;
  approval_id?: string;
  tool_name?: string;
  tool_args?: Record<string, unknown>;
  explanation?: string;
  content?: string;
  routing?: {
    address: "session" | "user";
    target: "user" | "agent";
    handler: "system" | "agent";
  };
}
