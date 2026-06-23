import { afterEach, describe, expect, it, vi } from "vitest";

describe("api URL helpers", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it("keeps API paths same-origin when no hosted backend is configured", async () => {
    vi.stubEnv("VITE_API_BASE", "");
    vi.resetModules();

    const { apiUrl, cardArtUrl } = await import("./api");

    expect(apiUrl("/api/health")).toBe("/api/health");
    expect(cardArtUrl("5HY/W83-E032")).toBe("/api/card-art/5HY%2FW83-E032");
  });

  it("points API and card art requests at VITE_API_BASE for split deploys", async () => {
    vi.stubEnv("VITE_API_BASE", "https://play-api.example.com/");
    vi.resetModules();

    const { apiUrl, cardArtUrl } = await import("./api");

    expect(apiUrl("/api/health")).toBe("https://play-api.example.com/api/health");
    expect(cardArtUrl("5HY/W83-E032")).toBe("https://play-api.example.com/api/card-art/5HY%2FW83-E032");
  });
});
