
import path from 'path';
import { homedir } from 'os';
import { execSync, spawn } from 'child_process';
import { existsSync } from 'fs';
import { DatabaseManager } from './DatabaseManager.js';
import { SessionManager } from './SessionManager.js';
import { logger } from '../../utils/logger.js';
import { buildInitPrompt, buildObservationPrompt, buildSummaryPrompt, buildContinuationPrompt } from '../../sdk/prompts.js';
import { SettingsDefaultsManager } from '../../shared/SettingsDefaultsManager.js';
import { getCredential, buildIsolatedEnv } from '../../shared/EnvManager.js';
import { USER_SETTINGS_PATH } from '../../shared/paths.js';
import { estimateTokens } from '../../shared/timeline-formatting.js';
import type { ActiveSession, ConversationMessage } from '../worker-types.js';
import { ModeManager } from '../domain/ModeManager.js';
import type { ModeConfig } from '../domain/types.js';
import {
  processAgentResponse,
  isAbortError,
  type WorkerRef
} from './agents/index.js';

const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1/models';

export type GeminiAuthMethod = 'api' | 'cli';

// Whitelisted models for the REST/API path. Kept narrow because each entry needs
// an RPM mapping below. The `cli` auth path skips this whitelist and accepts any
// model the local Gemini CLI exposes (preview models included).
export type GeminiModel =
  | 'gemini-2.5-flash-lite'
  | 'gemini-2.5-flash'
  | 'gemini-2.5-pro'
  | 'gemini-2.0-flash'
  | 'gemini-2.0-flash-lite'
  | 'gemini-3-flash'
  | 'gemini-3-flash-preview';

const GEMINI_RPM_LIMITS: Record<GeminiModel, number> = {
  'gemini-2.5-flash-lite': 10,
  'gemini-2.5-flash': 10,
  'gemini-2.5-pro': 5,
  'gemini-2.0-flash': 15,
  'gemini-2.0-flash-lite': 30,
  'gemini-3-flash': 10,
  'gemini-3-flash-preview': 5,
};

let lastRequestTime = 0;

const DEFAULT_MAX_CONTEXT_MESSAGES = 20;  
const DEFAULT_MAX_ESTIMATED_TOKENS = 100000;  

async function enforceRateLimitForModel(model: GeminiModel, rateLimitingEnabled: boolean): Promise<void> {
  if (!rateLimitingEnabled) {
    return;
  }

  const rpm = GEMINI_RPM_LIMITS[model] || 5;
  const minimumDelayMs = Math.ceil(60000 / rpm) + 100; 

  const now = Date.now();
  const timeSinceLastRequest = now - lastRequestTime;

  if (timeSinceLastRequest < minimumDelayMs) {
    const waitTime = minimumDelayMs - timeSinceLastRequest;
    logger.debug('SDK', `Rate limiting: waiting ${waitTime}ms before Gemini request`, { model, rpm });
    await new Promise(resolve => setTimeout(resolve, waitTime));
  }

  lastRequestTime = Date.now();
}

interface GeminiResponse {
  candidates?: Array<{
    content?: {
      parts?: Array<{
        text?: string;
      }>;
    };
  }>;
  usageMetadata?: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    totalTokenCount?: number;
  };
}

interface GeminiContent {
  role: 'user' | 'model';
  parts: Array<{ text: string }>;
}

interface GeminiConfig {
  authMethod: GeminiAuthMethod;
  // Validated GeminiModel when authMethod=='api', any string when 'cli' (the local CLI accepts whatever Google currently exposes, including preview models).
  model: string;
  apiKey: string;            // populated only when authMethod=='api'
  geminiPath: string;        // populated only when authMethod=='cli' (resolved or empty for auto-detect)
  rateLimitingEnabled: boolean;  // honored only when authMethod=='api'; the CLI enforces its own server-side throttling
  cliTimeoutMs: number;      // honored only when authMethod=='cli'
}

// Shape of `gemini -p ... -o json` stdout. The actual schema (verified against
// gemini-cli 0.40.x) nests token totals at stats.models[<modelName>].tokens.total
// — no top-level totalTokens field. We sum across model entries for safety.
interface GeminiCliJsonResult {
  response?: string;
  stats?: {
    models?: Record<string, {
      tokens?: {
        total?: number;
        input?: number;
        prompt?: number;
        candidates?: number;
        cached?: number;
        thoughts?: number;
        tool?: number;
      };
      [key: string]: unknown;
    }>;
    [key: string]: unknown;
  };
  error?: { message?: string; [key: string]: unknown } | string | null;
}

function extractTotalTokens(stats: GeminiCliJsonResult['stats']): number | undefined {
  const models = stats?.models;
  if (!models || typeof models !== 'object') return undefined;
  let total = 0;
  let found = false;
  for (const m of Object.values(models)) {
    const t = m?.tokens?.total;
    if (typeof t === 'number') {
      total += t;
      found = true;
    }
  }
  return found ? total : undefined;
}

export class GeminiProvider {
  private dbManager: DatabaseManager;
  private sessionManager: SessionManager;

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager) {
    this.dbManager = dbManager;
    this.sessionManager = sessionManager;
  }

  async startSession(session: ActiveSession, worker?: WorkerRef): Promise<void> {
    const cfg = this.getGeminiConfig();
    const { authMethod, model, rateLimitingEnabled } = cfg;

    // Validate auth-method-specific prerequisites up front so we fail fast with
    // a clear message instead of inside the message loop.
    if (authMethod === 'api' && !cfg.apiKey) {
      throw new Error('Gemini API key not configured. Set CLAUDE_MEM_GEMINI_API_KEY in settings or GEMINI_API_KEY environment variable, or switch to CLAUDE_MEM_GEMINI_AUTH_METHOD=cli to use the local Gemini CLI subscription.');
    }
    if (authMethod === 'cli') {
      // Resolve once at session start; throws if not found.
      this.findGeminiExecutable();
    }

    if (!session.memorySessionId) {
      const syntheticMemorySessionId = `gemini-${session.contentSessionId}-${Date.now()}`;
      session.memorySessionId = syntheticMemorySessionId;
      this.dbManager.getSessionStore().updateMemorySessionId(session.sessionDbId, syntheticMemorySessionId);
      logger.info('SESSION', `MEMORY_ID_GENERATED | sessionDbId=${session.sessionDbId} | provider=Gemini | authMethod=${authMethod}`);
    }

    const mode = ModeManager.getInstance().getActiveMode();
    const initPrompt = session.lastPromptNumber === 1
      ? buildInitPrompt(session.project, session.contentSessionId, session.userPrompt, mode)
      : buildContinuationPrompt(session.userPrompt, session.lastPromptNumber, session.contentSessionId, mode);

    session.conversationHistory.push({ role: 'user', content: initPrompt });
    let initResponse: { content: string; tokensUsed?: number };
    try {
      initResponse = await this.queryGemini(session.conversationHistory, cfg);
    } catch (error: unknown) {
      if (error instanceof Error) {
        logger.error('SDK', 'Gemini init query failed', { sessionId: session.sessionDbId, model }, error);
      } else {
        logger.error('SDK', 'Gemini init query failed with non-Error', { sessionId: session.sessionDbId, model }, new Error(String(error)));
      }
      return this.handleGeminiError(error, session, worker);
    }

    if (initResponse.content) {
      session.conversationHistory.push({ role: 'assistant', content: initResponse.content });
      const tokensUsed = initResponse.tokensUsed || 0;
      session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
      session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
      await processAgentResponse(initResponse.content, session, this.dbManager, this.sessionManager, worker, tokensUsed, null, 'Gemini', undefined, model);
    } else {
      logger.error('SDK', 'Empty Gemini init response - session may lack context', { sessionId: session.sessionDbId, model, authMethod });
    }

    try {
      await this.processMessageLoop(session, worker, cfg, mode);
    } catch (error: unknown) {
      if (error instanceof Error) {
        logger.error('SDK', 'Gemini message loop failed', { sessionId: session.sessionDbId, model }, error);
      } else {
        logger.error('SDK', 'Gemini message loop failed with non-Error', { sessionId: session.sessionDbId, model }, new Error(String(error)));
      }
      return this.handleGeminiError(error, session, worker);
    }

    const sessionDuration = Date.now() - session.startTime;
    logger.success('SDK', 'Gemini agent completed', {
      sessionId: session.sessionDbId,
      duration: `${(sessionDuration / 1000).toFixed(1)}s`,
      historyLength: session.conversationHistory.length
    });
  }

  private async processMessageLoop(
    session: ActiveSession,
    worker: WorkerRef | undefined,
    cfg: GeminiConfig,
    mode: ModeConfig
  ): Promise<void> {
    let lastCwd: string | undefined;

    for await (const message of this.sessionManager.getMessageIterator(session.sessionDbId)) {
      session.pendingAgentId = message.agentId ?? null;
      session.pendingAgentType = message.agentType ?? null;

      if (message.cwd) {
        lastCwd = message.cwd;
      }
      const originalTimestamp = session.earliestPendingTimestamp;

      if (message.type === 'observation') {
        await this.processObservationMessage(session, message, worker, cfg, originalTimestamp, lastCwd);
      } else if (message.type === 'summarize') {
        await this.processSummaryMessage(session, message, worker, cfg, mode, originalTimestamp, lastCwd);
      }
    }
  }

  private async processObservationMessage(
    session: ActiveSession,
    message: { type: string; prompt_number?: number; tool_name?: string; tool_input?: unknown; tool_response?: unknown; cwd?: string },
    worker: WorkerRef | undefined,
    cfg: GeminiConfig,
    originalTimestamp: number | null,
    lastCwd: string | undefined
  ): Promise<void> {
    if (message.prompt_number !== undefined) {
      session.lastPromptNumber = message.prompt_number;
    }

    if (!session.memorySessionId) {
      throw new Error('Cannot process observations: memorySessionId not yet captured. This session may need to be reinitialized.');
    }

    const obsPrompt = buildObservationPrompt({
      id: 0,
      tool_name: message.tool_name!,
      tool_input: JSON.stringify(message.tool_input),
      tool_output: JSON.stringify(message.tool_response),
      created_at_epoch: originalTimestamp ?? Date.now(),
      cwd: message.cwd
    });

    session.conversationHistory.push({ role: 'user', content: obsPrompt });
    const obsResponse = await this.queryGemini(session.conversationHistory, cfg);

    let tokensUsed = 0;
    if (obsResponse.content) {
      session.conversationHistory.push({ role: 'assistant', content: obsResponse.content });
      tokensUsed = obsResponse.tokensUsed || 0;
      session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
      session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
    }

    if (obsResponse.content) {
      await processAgentResponse(obsResponse.content, session, this.dbManager, this.sessionManager, worker, tokensUsed, originalTimestamp, 'Gemini', lastCwd, cfg.model);
    } else {
      logger.warn('SDK', 'Empty Gemini observation response, leaving queue intact', {
        sessionId: session.sessionDbId
      });
    }
  }

  private async processSummaryMessage(
    session: ActiveSession,
    message: { type: string; last_assistant_message?: string },
    worker: WorkerRef | undefined,
    cfg: GeminiConfig,
    mode: ModeConfig,
    originalTimestamp: number | null,
    lastCwd: string | undefined
  ): Promise<void> {
    if (!session.memorySessionId) {
      throw new Error('Cannot process summary: memorySessionId not yet captured. This session may need to be reinitialized.');
    }

    const summaryPrompt = buildSummaryPrompt({
      id: session.sessionDbId,
      memory_session_id: session.memorySessionId,
      project: session.project,
      user_prompt: session.userPrompt,
      last_assistant_message: message.last_assistant_message || ''
    }, mode);

    session.conversationHistory.push({ role: 'user', content: summaryPrompt });
    const summaryResponse = await this.queryGemini(session.conversationHistory, cfg);

    let tokensUsed = 0;
    if (summaryResponse.content) {
      session.conversationHistory.push({ role: 'assistant', content: summaryResponse.content });
      tokensUsed = summaryResponse.tokensUsed || 0;
      session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
      session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
    }

    if (summaryResponse.content) {
      await processAgentResponse(summaryResponse.content, session, this.dbManager, this.sessionManager, worker, tokensUsed, originalTimestamp, 'Gemini', lastCwd, cfg.model);
    } else {
      logger.warn('SDK', 'Empty Gemini summary response, leaving queue intact', {
        sessionId: session.sessionDbId
      });
    }
  }

  private handleGeminiError(error: unknown, session: ActiveSession, _worker?: WorkerRef): never {
    if (isAbortError(error)) {
      logger.warn('SDK', 'Gemini agent aborted', { sessionId: session.sessionDbId });
      throw error;
    }

    logger.failure('SDK', 'Gemini agent error', { sessionDbId: session.sessionDbId }, error instanceof Error ? error : new Error(String(error)));
    throw error;
  }

  private truncateHistory(history: ConversationMessage[]): ConversationMessage[] {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const MAX_CONTEXT_MESSAGES = parseInt(settings.CLAUDE_MEM_GEMINI_MAX_CONTEXT_MESSAGES) || DEFAULT_MAX_CONTEXT_MESSAGES;
    const MAX_ESTIMATED_TOKENS = parseInt(settings.CLAUDE_MEM_GEMINI_MAX_TOKENS) || DEFAULT_MAX_ESTIMATED_TOKENS;

    if (history.length <= MAX_CONTEXT_MESSAGES) {
      const totalTokens = history.reduce((sum, m) => sum + estimateTokens(m.content), 0);
      if (totalTokens <= MAX_ESTIMATED_TOKENS) {
        return history;
      }
    }

    const truncated: ConversationMessage[] = [];
    let tokenCount = 0;

    for (let i = history.length - 1; i >= 0; i--) {
      const msg = history[i];
      const msgTokens = estimateTokens(msg.content);

      if (truncated.length > 0 && (truncated.length >= MAX_CONTEXT_MESSAGES || tokenCount + msgTokens > MAX_ESTIMATED_TOKENS)) {
        logger.warn('SDK', 'Context window truncated to prevent runaway costs', {
          originalMessages: history.length,
          keptMessages: truncated.length,
          droppedMessages: i + 1,
          estimatedTokens: tokenCount,
          tokenLimit: MAX_ESTIMATED_TOKENS
        });
        break;
      }

      truncated.unshift(msg);  
      tokenCount += msgTokens;
    }

    return truncated;
  }

  private conversationToGeminiContents(history: ConversationMessage[]): GeminiContent[] {
    return history.map(msg => ({
      role: msg.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: msg.content }]
    }));
  }

  // Auth-aware dispatcher. Picks the REST/API path or the local Gemini CLI
  // subprocess path based on cfg.authMethod. Both paths return the same
  // {content, tokensUsed} shape so the caller stays uniform.
  private async queryGemini(
    history: ConversationMessage[],
    cfg: GeminiConfig
  ): Promise<{ content: string; tokensUsed?: number }> {
    if (cfg.authMethod === 'cli') {
      return this.queryGeminiCli(history, cfg);
    }
    return this.queryGeminiMultiTurn(history, cfg.apiKey, cfg.model, cfg.rateLimitingEnabled);
  }

  private async queryGeminiMultiTurn(
    history: ConversationMessage[],
    apiKey: string,
    model: string,
    rateLimitingEnabled: boolean
  ): Promise<{ content: string; tokensUsed?: number }> {
    const truncatedHistory = this.truncateHistory(history);
    const contents = this.conversationToGeminiContents(truncatedHistory);
    const totalChars = truncatedHistory.reduce((sum, m) => sum + m.content.length, 0);

    logger.debug('SDK', `Querying Gemini multi-turn (${model})`, {
      turns: truncatedHistory.length,
      totalTurns: history.length,
      totalChars
    });

    const url = `${GEMINI_API_URL}/${model}:generateContent?key=${apiKey}`;

    // Rate limiting for the REST path only — the Gemini CLI enforces its own
    // server-side throttling on the OAuth-billed entitlement.
    await enforceRateLimitForModel(model as GeminiModel, rateLimitingEnabled);

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents,
        generationConfig: {
          temperature: 0.3,  // Lower temperature for structured extraction
          maxOutputTokens: 4096,
        },
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Gemini API error: ${response.status} - ${error}`);
    }

    const data = await response.json() as GeminiResponse;

    if (!data.candidates?.[0]?.content?.parts?.[0]?.text) {
      logger.error('SDK', 'Empty response from Gemini');
      return { content: '' };
    }

    const content = data.candidates[0].content.parts[0].text;
    const tokensUsed = data.usageMetadata?.totalTokenCount;

    return { content, tokensUsed };
  }

  // Spawn `gemini -p <prompt> -m <model> -o json` and parse its JSON result.
  // Inherits the OAuth session from ~/.gemini/oauth_creds.json — no API key
  // needed and the request is billed against the user's paid Gemini plan.
  private async queryGeminiCli(
    history: ConversationMessage[],
    cfg: GeminiConfig
  ): Promise<{ content: string; tokensUsed?: number }> {
    const truncatedHistory = this.truncateHistory(history);
    // Gemini CLI's `-p` is a single-turn invocation. Flatten the conversation
    // history into a single prompt, mirroring the role markers Anthropic-style
    // chats use so the model can still infer turns.
    const flattenedPrompt = truncatedHistory
      .map(m => `[${m.role === 'assistant' ? 'assistant' : 'user'}]\n${m.content}`)
      .join('\n\n');

    const geminiPath = this.findGeminiExecutable();

    logger.debug('SDK', `Querying Gemini CLI (${cfg.model})`, {
      turns: truncatedHistory.length,
      totalTurns: history.length,
      promptChars: flattenedPrompt.length,
      geminiPath,
    });

    const args = ['-m', cfg.model, '-o', 'json'];
    // Pass prompt via stdin; stdin is large in summarization workloads and the
    // -p arg gets argv-truncated on some shells. Headless mode is implied by
    // non-TTY stdin, but pass an explicit empty -p as belt-and-braces to keep
    // CLI from waiting for terminal input.
    args.push('-p', '');

    const stdoutText = await new Promise<string>((resolve, reject) => {
      // Use buildIsolatedEnv() so the subprocess inherits CLAUDE_MEM_INTERNAL=1 —
      // this is the canonical loop guard read by shouldTrackProject() and the
      // hook handlers; without it, any claude-mem hooks the user has installed
      // in ~/.gemini/settings.json would re-enter the worker for every spawned
      // summarization and snowball pending observations.
      // GEMINI_CLI_TRUST_WORKSPACE=true silences the "directory not trusted"
      // headless prompt that exits with code 55 outside ~/.gemini/trustedFolders.
      // OAuth creds are file-based (~/.gemini/oauth_creds.json), so we don't
      // need to forward credentials through env.
      const child = spawn(geminiPath, args, {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...buildIsolatedEnv(false),
          GEMINI_CLI_TRUST_WORKSPACE: 'true',
        },
      });

      const stdoutChunks: Buffer[] = [];
      const stderrChunks: Buffer[] = [];
      let settled = false;

      const finish = (fn: () => void) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        fn();
      };

      const timer = setTimeout(() => {
        finish(() => {
          try { child.kill('SIGTERM'); } catch { /* already gone */ }
          reject(new Error(`Gemini CLI timed out after ${cfg.cliTimeoutMs}ms`));
        });
      }, cfg.cliTimeoutMs);

      child.stdout.on('data', (chunk: Buffer) => stdoutChunks.push(chunk));
      child.stderr.on('data', (chunk: Buffer) => stderrChunks.push(chunk));
      child.on('error', err => finish(() => reject(err)));
      child.on('close', (code, signal) => {
        finish(() => {
          const stdout = Buffer.concat(stdoutChunks).toString('utf-8');
          const stderr = Buffer.concat(stderrChunks).toString('utf-8');

          // Documented Gemini CLI exit codes: 0 ok, 1 general/API, 42 input
          // validation, 53 turn limit. Anything non-zero → throw with stderr
          // for context.
          if (code !== 0) {
            const reason = signal ? `signal=${signal}` : `exitCode=${code}`;
            reject(new Error(`Gemini CLI failed (${reason}): ${stderr.trim() || stdout.trim() || '(no output)'}`));
            return;
          }
          resolve(stdout);
        });
      });

      try {
        child.stdin.end(flattenedPrompt);
      } catch (err) {
        finish(() => reject(err instanceof Error ? err : new Error(String(err))));
      }
    });

    let parsed: GeminiCliJsonResult;
    try {
      parsed = JSON.parse(stdoutText) as GeminiCliJsonResult;
    } catch (err) {
      // CLI sometimes prints non-JSON banner lines before the JSON object on
      // first run; try to recover by extracting the last balanced JSON object.
      const lastBrace = stdoutText.lastIndexOf('}');
      const firstBrace = stdoutText.indexOf('{');
      if (firstBrace >= 0 && lastBrace > firstBrace) {
        try {
          parsed = JSON.parse(stdoutText.slice(firstBrace, lastBrace + 1)) as GeminiCliJsonResult;
        } catch {
          throw new Error(`Gemini CLI produced unparseable output: ${stdoutText.slice(0, 500)}`);
        }
      } else {
        throw new Error(`Gemini CLI produced no JSON output: ${stdoutText.slice(0, 500)}`);
      }
    }

    if (parsed.error) {
      const msg = typeof parsed.error === 'string'
        ? parsed.error
        : parsed.error.message ?? JSON.stringify(parsed.error);
      throw new Error(`Gemini CLI returned error: ${msg}`);
    }

    const content = parsed.response ?? '';
    const tokensUsed = extractTotalTokens(parsed.stats);
    return { content, tokensUsed };
  }

  // Resolve the `gemini` binary. Mirrors findClaudeExecutable in ClaudeProvider:
  // honors an explicit settings override, falls back to PATH lookup, throws a
  // helpful error if absent.
  private findGeminiExecutable(): string {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    if (settings.CLAUDE_MEM_GEMINI_PATH) {
      if (!existsSync(settings.CLAUDE_MEM_GEMINI_PATH)) {
        throw new Error(`CLAUDE_MEM_GEMINI_PATH is set to "${settings.CLAUDE_MEM_GEMINI_PATH}" but the file does not exist.`);
      }
      return settings.CLAUDE_MEM_GEMINI_PATH;
    }

    try {
      const found = execSync(
        process.platform === 'win32' ? 'where gemini' : 'which gemini',
        { encoding: 'utf8', windowsHide: true, stdio: ['ignore', 'pipe', 'ignore'] }
      ).trim().split('\n')[0].trim();
      if (found) return found;
    } catch (error) {
      logger.debug('SDK', 'Gemini executable auto-detection failed', {}, error instanceof Error ? error : new Error(String(error)));
    }

    throw new Error('Gemini executable not found. Either:\n1. Install Gemini CLI and ensure `gemini` is on PATH, or\n2. Set CLAUDE_MEM_GEMINI_PATH in ~/.claude-mem/settings.json, or\n3. Switch back to CLAUDE_MEM_GEMINI_AUTH_METHOD=api with a CLAUDE_MEM_GEMINI_API_KEY.');
  }

  private getGeminiConfig(): GeminiConfig {
    const settingsPath = path.join(homedir(), '.claude-mem', 'settings.json');
    const settings = SettingsDefaultsManager.loadFromFile(settingsPath);

    const rawAuth = (settings.CLAUDE_MEM_GEMINI_AUTH_METHOD || 'api').toLowerCase();
    const authMethod: GeminiAuthMethod = rawAuth === 'cli' ? 'cli' : 'api';

    const apiKey = settings.CLAUDE_MEM_GEMINI_API_KEY || getCredential('GEMINI_API_KEY') || '';
    const geminiPath = settings.CLAUDE_MEM_GEMINI_PATH || '';
    const cliTimeoutMs = parseInt(settings.CLAUDE_MEM_GEMINI_CLI_TIMEOUT_MS, 10) || 120_000;

    const defaultModel: GeminiModel = 'gemini-2.5-flash';
    const configuredModel = settings.CLAUDE_MEM_GEMINI_MODEL || defaultModel;

    let model: string;
    if (authMethod === 'cli') {
      // CLI accepts any model name the local Gemini binary recognises,
      // including preview variants the REST whitelist doesn't list. Trust the
      // user's choice; surface errors at invocation time instead.
      model = configuredModel;
    } else {
      const validModels: GeminiModel[] = [
        'gemini-2.5-flash-lite',
        'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gemini-2.0-flash',
        'gemini-2.0-flash-lite',
        'gemini-3-flash',
        'gemini-3-flash-preview',
      ];
      if (validModels.includes(configuredModel as GeminiModel)) {
        model = configuredModel;
      } else {
        logger.warn('SDK', `Invalid Gemini model "${configuredModel}", falling back to ${defaultModel}`, {
          configured: configuredModel,
          validModels,
        });
        model = defaultModel;
      }
    }

    const rateLimitingEnabled = settings.CLAUDE_MEM_GEMINI_RATE_LIMITING_ENABLED !== 'false';

    return { authMethod, apiKey, geminiPath, model, rateLimitingEnabled, cliTimeoutMs };
  }
}

export function isGeminiAvailable(): boolean {
  const settingsPath = path.join(homedir(), '.claude-mem', 'settings.json');
  const settings = SettingsDefaultsManager.loadFromFile(settingsPath);
  // CLI auth needs no API key — it inherits OAuth from the local Gemini CLI.
  if ((settings.CLAUDE_MEM_GEMINI_AUTH_METHOD || 'api').toLowerCase() === 'cli') {
    return true;
  }
  return !!(settings.CLAUDE_MEM_GEMINI_API_KEY || getCredential('GEMINI_API_KEY'));
}

export function isGeminiSelected(): boolean {
  const settingsPath = path.join(homedir(), '.claude-mem', 'settings.json');
  const settings = SettingsDefaultsManager.loadFromFile(settingsPath);
  return settings.CLAUDE_MEM_PROVIDER === 'gemini';
}
