import { describe, expect, it } from "vitest"

import {
  bedrockAuthModesForUser,
  bedrockCredentialsChainRequiresAdmin,
  buildModelRequestPayload,
  providerAllowsEmptyApiKey,
  shouldFetchProviderModelCatalog,
} from "./model-provider-settings"

describe("Bedrock credential-chain permissions", () => {
  it("offers the credential chain only to administrators creating a model", () => {
    expect(bedrockAuthModesForUser(false)).toEqual(["api_key"])
    expect(bedrockAuthModesForUser(true)).toEqual(["api_key", "credentials_chain"])
  })

  it("keeps a historical credential-chain value visible so it can be changed", () => {
    expect(bedrockAuthModesForUser(false, "credentials_chain")).toEqual([
      "api_key",
      "credentials_chain",
    ])
  })

  it("blocks only a non-admin Bedrock credential-chain submission", () => {
    expect(bedrockCredentialsChainRequiresAdmin("bedrock", "credentials_chain", false)).toBe(true)
    expect(bedrockCredentialsChainRequiresAdmin("bedrock", "credentials_chain", true)).toBe(false)
    expect(bedrockCredentialsChainRequiresAdmin("bedrock", "api_key", false)).toBe(false)
    expect(bedrockCredentialsChainRequiresAdmin("openai", undefined, false)).toBe(false)
  })
})

describe("providerAllowsEmptyApiKey", () => {
  it("allows the AWS credential chain without an API key", () => {
    expect(providerAllowsEmptyApiKey("bedrock", "credentials_chain")).toBe(true)
    expect(providerAllowsEmptyApiKey("bedrock", "api_key")).toBe(false)
  })

  it("preserves Xinference's optional API key behavior", () => {
    expect(providerAllowsEmptyApiKey("xinference")).toBe(true)
    expect(providerAllowsEmptyApiKey("openai")).toBe(false)
  })
})

describe("shouldFetchProviderModelCatalog", () => {
  it("skips Bedrock without requiring provider metadata", () => {
    expect(shouldFetchProviderModelCatalog("bedrock")).toBe(false)
    expect(shouldFetchProviderModelCatalog("claude")).toBe(true)
    expect(shouldFetchProviderModelCatalog("openai")).toBe(true)
  })
})

describe("buildModelRequestPayload", () => {
  it.each([
    "anthropic.claude-sonnet-4-5-v1:0",
    "us.anthropic.claude-sonnet-4-5-v1:0",
    "arn:aws:bedrock:us-west-2:123456789012:inference-profile/example",
  ])("preserves a manually entered Bedrock model or profile identifier: %s", (modelName) => {
    const payload = buildModelRequestPayload({
      model_provider: "bedrock",
      model_name: modelName,
      bedrock_region: " us-west-2 ",
      bedrock_auth_mode: "credentials_chain" as const,
      base_url: "https://bedrock-runtime.us-west-2.amazonaws.com",
      api_key: "",
    })

    expect(payload).toEqual({
      model_provider: "bedrock",
      model_name: modelName,
      bedrock_region: "us-west-2",
      bedrock_auth_mode: "credentials_chain",
      base_url: "https://bedrock-runtime.us-west-2.amazonaws.com",
      api_key: "",
    })
  })

  it("defaults Bedrock authentication to an API key", () => {
    expect(buildModelRequestPayload({
      model_provider: "bedrock",
      model_name: "anthropic.claude-sonnet-4-5-v1:0",
      bedrock_region: "us-east-1",
      api_key: "test-token",
    })).toEqual({
      model_provider: "bedrock",
      model_name: "anthropic.claude-sonnet-4-5-v1:0",
      bedrock_region: "us-east-1",
      bedrock_auth_mode: "api_key",
      api_key: "test-token",
    })
  })

  it("removes Bedrock-only settings from other provider payloads", () => {
    expect(buildModelRequestPayload({
      model_provider: "openai",
      model_name: "gpt-5",
      bedrock_region: undefined,
      bedrock_auth_mode: "api_key" as const,
      api_key: "test-token",
    })).toEqual({
      model_provider: "openai",
      model_name: "gpt-5",
      api_key: "test-token",
    })
  })
})
