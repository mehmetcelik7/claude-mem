export interface NormalizedHookInput {
  sessionId: string;
  cwd: string;
  platform?: string;   
  prompt?: string;
  toolName?: string;
  toolInput?: unknown;
  toolResponse?: unknown;
  transcriptPath?: string;
  filePath?: string;   
  edits?: unknown[];   
  metadata?: Record<string, unknown>;
  agentId?: string;
  agentType?: string;
  // Pre-extracted last assistant message — set by adapters that already have
  // it inline (e.g. Gemini CLI's AfterAgent.prompt_response). Used as a
  // fallback by the summarize handler when transcriptPath isn't available.
  lastAssistantMessage?: string;
}

export interface HookResult {
  continue?: boolean;
  suppressOutput?: boolean;
  hookSpecificOutput?: {
    hookEventName: string;
    additionalContext: string;
    permissionDecision?: 'allow' | 'deny';
    updatedInput?: Record<string, unknown>;
  };
  systemMessage?: string;
  exitCode?: number;
}

export interface PlatformAdapter {
  normalizeInput(raw: unknown): NormalizedHookInput;
  formatOutput(result: HookResult): unknown;
}

export interface EventHandler {
  execute(input: NormalizedHookInput): Promise<HookResult>;
}
