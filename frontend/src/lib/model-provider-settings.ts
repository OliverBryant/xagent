export type BedrockAuthMode = "api_key" | "credentials_chain"

type ModelProviderSettings = {
  model_provider: string
  bedrock_region?: string
  bedrock_auth_mode?: BedrockAuthMode
}

export function providerAllowsEmptyApiKey(
  providerId: string,
  bedrockAuthMode?: BedrockAuthMode,
): boolean {
  return providerId === "xinference"
    || (providerId === "bedrock" && bedrockAuthMode === "credentials_chain")
}

export function bedrockAuthModesForUser(
  isAdmin: boolean,
  currentMode?: BedrockAuthMode,
): BedrockAuthMode[] {
  if (isAdmin || currentMode === "credentials_chain") {
    return ["api_key", "credentials_chain"]
  }
  return ["api_key"]
}

export function bedrockCredentialsChainRequiresAdmin(
  providerId: string,
  authMode: BedrockAuthMode | undefined,
  isAdmin: boolean,
): boolean {
  return providerId === "bedrock"
    && authMode === "credentials_chain"
    && !isAdmin
}

export function shouldFetchProviderModelCatalog(
  providerId: string,
): boolean {
  return providerId !== "bedrock"
}

export function buildModelRequestPayload<T extends ModelProviderSettings>(data: T): T {
  if (data.model_provider !== "bedrock") {
    const payload = { ...data }
    delete payload.bedrock_region
    delete payload.bedrock_auth_mode
    return payload
  }

  return {
    ...data,
    bedrock_region: data.bedrock_region?.trim(),
    bedrock_auth_mode: data.bedrock_auth_mode || "api_key",
  }
}
