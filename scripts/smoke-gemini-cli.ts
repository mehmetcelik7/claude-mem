// Smoke test for CLAUDE_MEM_GEMINI_AUTH_METHOD=cli
// Bypasses the worker/session machinery and exercises the queryGeminiCli code
// path directly so we can validate (a) subprocess invocation, (b) JSON parsing,
// (c) recovery from claude-mem hook noise that gets prepended to stdout.

import { GeminiProvider } from '../src/services/worker/GeminiProvider.js';
import { SettingsDefaultsManager } from '../src/shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../src/shared/paths.js';
import { writeFileSync, readFileSync, existsSync } from 'fs';

async function main() {
  // Snapshot existing settings, swap to cli auth + a known model, restore on
  // exit. Avoids permanently mutating ~/.claude-mem/settings.json on a smoke
  // test that may be re-run.
  const existing = existsSync(USER_SETTINGS_PATH)
    ? readFileSync(USER_SETTINGS_PATH, 'utf-8')
    : null;

  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  settings.CLAUDE_MEM_GEMINI_AUTH_METHOD = 'cli';
  settings.CLAUDE_MEM_GEMINI_MODEL = 'gemini-2.5-flash';
  writeFileSync(USER_SETTINGS_PATH, JSON.stringify(settings, null, 2), 'utf-8');

  try {
    const provider = new GeminiProvider({} as any, {} as any);
    // Reach into the private queryGemini dispatcher via any-cast — this is a
    // smoke test, not a public API contract.
    const cfg = (provider as any).getGeminiConfig();
    console.log('Resolved config:', { authMethod: cfg.authMethod, model: cfg.model, hasApiKey: !!cfg.apiKey, geminiPath: cfg.geminiPath || '(auto-detect)', cliTimeoutMs: cfg.cliTimeoutMs });

    const history = [
      { role: 'user' as const, content: 'You are a terse assistant.' },
      { role: 'assistant' as const, content: 'Understood.' },
      { role: 'user' as const, content: 'Reply with exactly: SMOKE-TEST-CLI-OK' },
    ];
    const start = Date.now();
    const result = await (provider as any).queryGemini(history, cfg);
    const dur = Date.now() - start;
    console.log('Result:', { content: result.content, tokensUsed: result.tokensUsed, durationMs: dur });

    if (typeof result.content === 'string' && result.content.includes('SMOKE-TEST-CLI-OK')) {
      console.log('PASS');
      process.exit(0);
    } else {
      console.error('FAIL: expected response to contain SMOKE-TEST-CLI-OK');
      process.exit(2);
    }
  } finally {
    if (existing !== null) {
      writeFileSync(USER_SETTINGS_PATH, existing, 'utf-8');
    }
  }
}

main().catch((err) => {
  console.error('SMOKE TEST FAILED:', err instanceof Error ? err.message : err);
  process.exit(1);
});
