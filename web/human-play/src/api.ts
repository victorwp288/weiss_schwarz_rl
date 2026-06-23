import type {
  ApiHealth,
  CardInfo,
  CreateSessionPayload,
  DeckSummary,
  PolicySummary,
  RunSummary,
  SessionState,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

export function cardArtUrl(cardNo: string): string {
  return apiUrl(`/api/card-art/${encodeURIComponent(cardNo)}`);
}

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function getHealth(): Promise<ApiHealth> {
  return requestJson<ApiHealth>("/api/health");
}

export async function getDecks(): Promise<DeckSummary[]> {
  const payload = await requestJson<{ decks: DeckSummary[] }>("/api/decks");
  return payload.decks;
}

export async function getRuns(): Promise<RunSummary[]> {
  const payload = await requestJson<{ runs: RunSummary[] }>("/api/runs");
  return payload.runs;
}

export async function getPolicies(runDir: string): Promise<PolicySummary[]> {
  const payload = await requestJson<{ policies: PolicySummary[] }>(
    `/api/policies?run_dir=${encodeURIComponent(runDir)}`,
  );
  return payload.policies;
}

export async function createSession(payload: CreateSessionPayload): Promise<SessionState> {
  return requestJson<SessionState>("/api/sessions", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getSession(sessionId: string): Promise<SessionState> {
  return requestJson<SessionState>(`/api/sessions/${sessionId}`);
}

export async function submitAction(
  sessionId: string,
  actionId: number,
  clientViewHash64?: string,
): Promise<SessionState> {
  return requestJson<SessionState>(`/api/sessions/${sessionId}/actions`, {
    method: "POST",
    body: JSON.stringify({ action_id: actionId, client_view_hash64: clientViewHash64 ?? null }),
  });
}

export async function stepSession(sessionId: string): Promise<SessionState> {
  return requestJson<SessionState>(`/api/sessions/${sessionId}/step`, {
    method: "POST",
    body: "{}",
  });
}

const cardInfoCache = new Map<string, Promise<CardInfo>>();

/** Card reference text + engine flags; cached per card number for the session. */
export function getCardInfo(cardNo: string, cardId?: string | number | null): Promise<CardInfo> {
  const key = cardNo;
  let cached = cardInfoCache.get(key);
  if (!cached) {
    const query = cardId != null && `${cardId}`.match(/^\d+$/) ? `?card_id=${cardId}` : "";
    cached = requestJson<CardInfo>(`/api/card-info/${encodeURIComponent(cardNo)}${query}`);
    cardInfoCache.set(key, cached);
    cached.catch(() => cardInfoCache.delete(key));
  }
  return cached;
}

export async function closeSession(sessionId: string): Promise<void> {
  await requestJson<{ closed: boolean }>(`/api/sessions/${sessionId}/close`, {
    method: "POST",
    body: "{}",
  });
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
  });
  const text = await response.text();
  const payload = text ? (JSON.parse(text) as unknown) : {};
  if (!response.ok) {
    const message =
      payload && typeof payload === "object" && "error" in payload ? String(payload.error) : response.statusText;
    throw new ApiError(message, response.status);
  }
  return payload as T;
}
