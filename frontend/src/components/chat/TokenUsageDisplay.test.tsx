import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const i18nMock = vi.hoisted(() => ({ locale: "en" as "en" | "zh" }))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", () => ({
  cn: (...classes: Array<string | false | null | undefined>) =>
    classes.filter(Boolean).join(" "),
  getApiUrl: () => "http://api.local",
}))

const interpolate = (
  message: string,
  vars?: Record<string, string | number>,
) =>
  Object.entries(vars ?? {}).reduce(
    (result, [name, value]) => result.replaceAll(`{${name}}`, String(value)),
    message,
  )

const messages: Record<string, string> = {
  "chatPage.tokenUsage.input": "Input tokens",
  "chatPage.tokenUsage.output": "Output tokens",
  "chatPage.tokenUsage.cached": "Cached input tokens",
  "chatPage.tokenUsage.inputShort": "Input",
  "chatPage.tokenUsage.outputShort": "Output",
  "chatPage.tokenUsage.cachedShort": "Cached",
  "chatPage.tokenUsage.cachedShare": "{pct}% cached",
  "chatPage.tokenUsage.oneModel": "{count} model",
  "chatPage.tokenUsage.models": "{count} models",
  "chatPage.tokenUsage.oneModelWithUnattributed": "{count} model + {unattributed} unattributed",
  "chatPage.tokenUsage.modelsWithUnattributed": "{count} models + {unattributed} unattributed",
  "chatPage.tokenUsage.unattributedCount": "{count} unattributed",
  "chatPage.tokenUsage.byModel": "Usage by model",
  "chatPage.tokenUsage.model": "Model",
  "chatPage.tokenUsage.unknownModel": "Unknown model",
  "chatPage.tokenUsage.unattributed": "Unattributed",
  "chatPage.tokenUsage.mediaByModel": "Media usage",
  "chatPage.tokenUsage.mediaCall": "{count} media call",
  "chatPage.tokenUsage.mediaCalls": "{count} media calls",
  "chatPage.tokenUsage.quantity": "Amount",
  "chatPage.tokenUsage.callType": "Type",
  "chatPage.tokenUsage.unit.images": "images",
  "chatPage.tokenUsage.unit.seconds": "sec",
  "chatPage.tokenUsage.unit.characters": "chars",
  "chatPage.tokenUsage.unit.requests": "requests",
  "chatPage.tokenUsage.unit.tokens": "tokens",
  "chatPage.tokenUsage.mediaType.generate_image": "Image generation",
  "chatPage.tokenUsage.mediaType.tts": "Text-to-speech",
  "chatPage.tokenUsage.mediaType.video": "Video",
}

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: i18nMock.locale,
    t: (key: string, vars?: Record<string, string | number>) =>
      interpolate(messages[key] ?? key, vars),
    tDynamic: (
      key: string,
      fallback: string,
      vars?: Record<string, string | number>,
    ) => interpolate(messages[key] ?? fallback, vars),
  }),
}))

import {
  formatExactTokenCount,
  formatTokenCount,
  TokenUsageDisplay,
} from "./TokenUsageDisplay"

describe("TokenUsageDisplay", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    i18nMock.locale = "en"
  })

  afterEach(() => {
    cleanup()
  })

  it("formats large token counts with compact lowercase suffixes", () => {
    expect(formatTokenCount(999)).toBe("999")
    expect(formatTokenCount(37_499)).toBe("37.5k")
    expect(formatTokenCount(2_755_525)).toBe("2.76m")
    expect(formatExactTokenCount(2_755_525)).toBe("2,755,525")
    expect(formatTokenCount(2_755_525, "zh")).toBe("275.55万")
  })

  it.each([-1, Number.NaN, Number.POSITIVE_INFINITY])(
    "normalizes invalid token count %s to zero",
    (value) => {
      expect(formatTokenCount(value)).toBe("0")
      expect(formatExactTokenCount(value)).toBe("0")
    },
  )

  it("uses the active locale when rendering token counts", async () => {
    i18nMock.locale = "zh"
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 2_755_525,
          output_tokens: 0,
          total_tokens: 2_755_525,
          llm_calls: 1,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={6} isRunning={false} />)

    expect(await screen.findByText("275.55万")).toHaveAttribute("title", "2,755,525")
  })

  it("shows aggregate counts and exposes each model in a popover", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 2_755_525,
          output_tokens: 37_499,
          total_tokens: 2_793_024,
          llm_calls: 3,
          model_usage: [
            {
              model_id: "main",
              model_name: "deepseek/deepseek-v4-pro",
              input_tokens: 2_700_000,
              output_tokens: 35_000,
            },
            {
              model_id: "compact",
              model_name: "deepseek/deepseek-v4-flash",
              input_tokens: 55_525,
              output_tokens: 2_499,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={7} isRunning={false} />)

    await waitFor(() => {
      expect(screen.getByText("2.76m")).toHaveAttribute("title", "2,755,525")
    })
    expect(screen.getByText("37.5k")).toHaveAttribute("title", "37,499")
    expect(screen.getByText("Input")).toHaveAttribute("title", "Input tokens")
    expect(screen.getByText("Output")).toHaveAttribute("title", "Output tokens")

    fireEvent.click(screen.getByRole("button", { name: /2 models/ }))

    const modelUsageDialog = await screen.findByRole("dialog")
    expect(modelUsageDialog).toHaveClass("w-[32rem]")
    expect(screen.getAllByText("Input")).toHaveLength(2)
    expect(screen.getAllByText("Output")).toHaveLength(2)
    expect(screen.getByText("deepseek/deepseek-v4-pro")).toBeInTheDocument()
    expect(screen.getByText("deepseek/deepseek-v4-flash")).toBeInTheDocument()
    expect(screen.getByText("main")).toBeInTheDocument()
    expect(screen.getByText("compact")).toBeInTheDocument()
    expect(screen.getByText("2.7m")).toHaveAttribute("title", "2,700,000")
    expect(screen.getByText("55.53k")).toHaveAttribute("title", "55,525")
  })

  it("uses the singular label and renders an id-only model without a sub-label", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 12,
          output_tokens: 3,
          total_tokens: 15,
          llm_calls: 1,
          model_usage: [
            {
              model_id: "router:model-only",
              model_name: "",
              input_tokens: 12,
              output_tokens: 3,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={8} isRunning={false} />)

    const modelsButton = await screen.findByRole("button", { name: /^1 model$/ })
    expect(modelsButton).toHaveAccessibleName("1 model")
    expect(screen.queryByRole("button", { name: "1 models" })).not.toBeInTheDocument()
    fireEvent.click(modelsButton)

    expect(await screen.findByText("router:model-only")).toBeInTheDocument()
    expect(screen.queryByText("Unattributed")).not.toBeInTheDocument()
  })

  it("counts and labels unknown model usage as unattributed", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 12,
          output_tokens: 3,
          total_tokens: 15,
          llm_calls: 1,
          model_usage: [
            {
              model_id: "",
              model_name: "",
              input_tokens: 12,
              output_tokens: 3,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={10} isRunning={false} />)

    fireEvent.click(await screen.findByRole("button", { name: /^1 unattributed$/ }))
    expect(await screen.findByText("Unknown model")).toBeInTheDocument()
  })

  it("separates attributed models from name-only usage in the trigger count", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 50,
          output_tokens: 0,
          total_tokens: 50,
          llm_calls: 2,
          model_usage: [
            {
              model_id: "main",
              model_name: "shared-name",
              input_tokens: 20,
              output_tokens: 0,
            },
            {
              model_id: "",
              model_name: "shared-name",
              input_tokens: 30,
              output_tokens: 0,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={11} isRunning={false} />)

    fireEvent.click(
      await screen.findByRole("button", { name: /^1 model \+ 1 unattributed$/ }),
    )
    expect(await screen.findAllByText("shared-name")).toHaveLength(2)
    expect(screen.getByText("Unattributed")).toBeInTheDocument()
  })

  it.each([undefined, []])(
    "does not render a model popover without model usage (%s)",
    async (modelUsage) => {
      apiRequestMock.mockResolvedValue(
        new Response(
          JSON.stringify({
            input_tokens: 12,
            output_tokens: 3,
            total_tokens: 15,
            llm_calls: 1,
            model_usage: modelUsage,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      )

      render(<TokenUsageDisplay taskId={9} isRunning={false} />)

      await screen.findByText("12")
      expect(screen.queryByRole("button")).not.toBeInTheDocument()
    },
  )
})

describe("TokenUsageDisplay media usage", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    i18nMock.locale = "en"
  })

  afterEach(() => {
    cleanup()
  })

  it("exposes media usage in its own popover with unit-formatted amounts", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 100,
          output_tokens: 20,
          total_tokens: 120,
          llm_calls: 1,
          model_usage: [],
          media_usage: [
            {
              model_id: "sd",
              model_name: "stable-diffusion-xl",
              unit: "images",
              call_type: "generate_image",
              resolution: "1K",
              quantity: 3,
              calls: 2,
              tokens: 0,
            },
            {
              model_id: "tts-1",
              model_name: "elevenlabs-tts",
              unit: "seconds",
              call_type: "tts",
              quantity: 12.5,
              calls: 1,
              tokens: 0,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={20} isRunning={false} />)

    // 3 media calls total (2 image + 1 tts).
    fireEvent.click(await screen.findByRole("button", { name: /3 media calls/ }))

    expect(await screen.findByText("Media usage")).toBeInTheDocument()
    expect(screen.getByText("stable-diffusion-xl")).toBeInTheDocument()
    expect(screen.getByText("Image generation")).toBeInTheDocument()
    expect(screen.getByText("3 images")).toBeInTheDocument()
    expect(screen.getByText("1K")).toBeInTheDocument()
    expect(screen.getByText("elevenlabs-tts")).toBeInTheDocument()
    expect(screen.getByText("Text-to-speech")).toBeInTheDocument()
    expect(screen.getByText("12.5 sec")).toBeInTheDocument()
  })

  it("uses the singular label for a single media call", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "sd",
              model_name: "sd",
              unit: "images",
              call_type: "generate_image",
              quantity: 1,
              calls: 1,
              tokens: 0,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={21} isRunning={false} />)

    expect(
      await screen.findByRole("button", { name: /^1 media call$/ }),
    ).toBeInTheDocument()
  })

  it("does not render a media popover without media usage", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 10,
          output_tokens: 2,
          total_tokens: 12,
          llm_calls: 1,
          model_usage: [],
          media_usage: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={22} isRunning={false} />)

    await screen.findByText("Input")
    expect(
      screen.queryByRole("button", { name: /media call/ }),
    ).not.toBeInTheDocument()
  })

  it("falls back to the raw call type and unit when no translation exists", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "x",
              model_name: "custom-model",
              unit: "widgets",
              call_type: "custom_op",
              quantity: 4,
              calls: 1,
              tokens: 0,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={23} isRunning={false} />)

    fireEvent.click(await screen.findByRole("button", { name: /1 media call/ }))
    expect(await screen.findByText("custom_op")).toBeInTheDocument()
    expect(screen.getByText("4 widgets")).toBeInTheDocument()
  })

  it("renders an empty unit without a dangling trailing space", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "x",
              model_name: "no-unit-model",
              unit: "",
              call_type: "video",
              quantity: 4,
              calls: 1,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={24} isRunning={false} />)

    fireEvent.click(await screen.findByRole("button", { name: /1 media call/ }))
    // Exact match: "4 " with a trailing space would not satisfy this.
    expect(await screen.findByText("4")).toBeInTheDocument()
  })

  it("renders provider tokens and marks estimated counts", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
          llm_calls: 0,
          model_usage: [],
          media_usage: [
            {
              model_id: "g",
              model_name: "gemini-image",
              unit: "images",
              call_type: "generate_image",
              quantity: 1,
              calls: 1,
              provider_tokens: 1120,
              tokens_estimated: false,
            },
            {
              model_id: "e",
              model_name: "text-embed",
              unit: "texts",
              call_type: "embedding",
              quantity: 3,
              calls: 1,
              provider_tokens: 40,
              tokens_estimated: true,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={25} isRunning={false} />)

    fireEvent.click(await screen.findByRole("button", { name: /2 media calls/ }))
    // Real Gemini image tokens are surfaced, not silently dropped.
    expect(await screen.findByText(/1\.12k/)).toBeInTheDocument()
    // The estimate is visibly marked so it is not mistaken for a measurement.
    expect(screen.getByText(/40~/)).toBeInTheDocument()
  })
})

describe("TokenUsageDisplay cached tokens", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    i18nMock.locale = "en"
  })

  afterEach(() => {
    cleanup()
  })

  it("shows the cached share and a per-model cached column", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 100_000,
          output_tokens: 5_000,
          total_tokens: 105_000,
          llm_calls: 2,
          cached_input_tokens: 75_000,
          model_usage: [
            {
              model_id: "main",
              model_name: "claude-sonnet-5",
              input_tokens: 100_000,
              output_tokens: 5_000,
              cached_input_tokens: 75_000,
              cache_write_input_tokens: 1_000,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={11} isRunning={false} />)

    const share = await screen.findByText("75% cached")
    expect(share).toHaveAttribute("title", "Cached input tokens: 75,000")

    fireEvent.click(screen.getByRole("button", { name: /^1 model$/ }))
    await screen.findByRole("dialog")
    expect(screen.getByText("Cached")).toHaveAttribute(
      "title",
      "Cached input tokens",
    )
    expect(screen.getByText("75k")).toHaveAttribute("title", "75,000")
  })

  it("suppresses the cached share when input tokens are zero", async () => {
    // Malformed/partial backend data: cached > 0 with input == 0 must not
    // render a NaN/Infinity percentage.
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 0,
          output_tokens: 5,
          total_tokens: 5,
          llm_calls: 1,
          cached_input_tokens: 75_000,
          model_usage: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={13} isRunning={false} />)

    await screen.findByText("Input")
    expect(screen.queryByText(/% cached/)).not.toBeInTheDocument()
    expect(screen.queryByText(/NaN|Infinity/)).not.toBeInTheDocument()
  })

  it("hides the cached share when the backend reports no cache usage", async () => {
    apiRequestMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          input_tokens: 100,
          output_tokens: 5,
          total_tokens: 105,
          llm_calls: 1,
          model_usage: [
            {
              model_id: "main",
              model_name: "gpt-4.1",
              input_tokens: 100,
              output_tokens: 5,
              cached_input_tokens: 0,
              cache_write_input_tokens: 0,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )

    render(<TokenUsageDisplay taskId={12} isRunning={false} />)

    await screen.findByText("Input")
    expect(screen.queryByText(/% cached/)).not.toBeInTheDocument()
  })
})
