export interface SessionListItem {
  id: string;
  title: string;
  updated_at: string;
}

export async function fetchSessions(
  limit: number = 50,
  cursor?: string
): Promise<SessionListItem[]> {
  const params = new URLSearchParams();
  params.set("limit", limit.toString());
  if (cursor) {
    params.set("cursor", cursor);
  }

  const response = await fetch(`/api/v1/sessions?${params.toString()}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch sessions: ${response.statusText}`);
  }

  return response.json();
}

export async function renameSession(
  sessionId: string,
  title: string
): Promise<void> {
  const response = await fetch(`/api/v1/sessions/${sessionId}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title }),
  });

  if (!response.ok) {
    throw new Error(`Failed to rename session: ${response.statusText}`);
  }
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/v1/sessions/${sessionId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error(`Failed to delete session: ${response.statusText}`);
  }
}
