// server.js — PRS Co-Pilot API (v2.0.0)
// Local Express server. Live endpoints share logic with the Vercel function
// via lib/orchestrator.js. Offline demo endpoints need no API key.

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';
import { demoCase, cases } from './demo/cases.js';
import {
  plannerPrompt, reviewerPrompt, synthPrompt,
  runLLM, runLLMRetry, generatePlan,
} from './lib/orchestrator.js';

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.use(express.static('public'));

// Boot fully offline with a placeholder key; live endpoints still require a
// real OPENAI_API_KEY at call time.
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY || 'sk-offline-demo-no-key' });
const MODEL_DEFAULT = (process.env.MODEL_DEFAULT || 'gpt-4o').trim();
const VERSION = '2.0.0';

const llmOpts = (body = {}) => ({
  client,
  model: body.model,
  reasoningEffort: body.reasoningEffort,
  modelDefault: MODEL_DEFAULT,
});

/* ---------------- Atomic endpoints ---------------- */
app.post('/plan', async (req, res) => {
  try {
    const { caseText, verbosity } = req.body || {};
    if (!caseText || !caseText.trim()) return res.status(400).json({ error: 'Missing caseText' });
    const content = await runLLM({
      system: 'Return ONLY the <SurgicalPlan>…</SurgicalPlan>.',
      user: plannerPrompt(caseText, verbosity)
    }, llmOpts(req.body));
    const m = content.match(/<SurgicalPlan[\s\S]*?<\/SurgicalPlan>/i);
    res.json({ xml: m ? m[0] : content });
  } catch (e) { console.error(e); res.status(500).json({ error: 'planner_failed' }); }
});

app.post('/review', async (req, res) => {
  try {
    const { plannerXml } = req.body || {};
    if (!plannerXml) return res.status(400).json({ error: 'Missing plannerXml' });
    const content = await runLLM({
      system: 'Return only the verify & feedback tags on one line.',
      user: reviewerPrompt(plannerXml)
    }, llmOpts(req.body));
    const v = (content.match(/<SurgicalBoard_Verify>(.*?)<\/SurgicalBoard_Verify>/i) || [, 'reject'])[1];
    const c = (content.match(/<Feedback_Comment>([\s\S]*?)<\/Feedback_Comment>/i) || [, ''])[1].trim();
    res.json({ verdict: /accept/i.test(v || '') ? 'accept' : 'reject', comment: c });
  } catch (e) { console.error(e); res.status(500).json({ error: 'review_failed' }); }
});

app.post('/synthesize', async (req, res) => {
  try {
    const { approvedPlannerXml, verbosity } = req.body || {};
    if (!approvedPlannerXml) return res.status(400).json({ error: 'Missing approvedPlannerXml' });
    const content = await runLLM({
      system: 'You write clean, professional operative plans in Markdown.',
      user: synthPrompt(approvedPlannerXml, verbosity)
    }, llmOpts(req.body));
    res.json({ markdown: content });
  } catch (e) { console.error(e); res.status(500).json({ error: 'synth_failed' }); }
});

/* ---------------- Orchestrated /generate (+ /api/generate alias) ---------------- */
async function handleGenerate(req, res) {
  try {
    const { caseText, model, reasoningEffort, verbosity } = req.body || {};
    if (!caseText || !caseText.trim()) return res.status(400).json({ error: 'Missing caseText' });
    if (!process.env.OPENAI_API_KEY) return res.status(503).json({ error: 'live_unavailable' });
    const result = await generatePlan({
      client, caseText, model, reasoningEffort, verbosity, modelDefault: MODEL_DEFAULT,
    });
    res.json(result);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'generate_failed' });
  }
}
app.post('/generate', handleGenerate);
app.post('/api/generate', handleGenerate);

/* ---------------- Orchestrated /generate-stream (NDJSON events) ---------------- */
// Same pipeline as /generate, but streams each real pipeline event as it
// happens, then a final { type: 'final', result } line with the same result
// shape /generate returns. On a mid-pipeline error it writes a final
// { type: 'error', error: 'generate_failed' } line and ends. Guard failures
// (missing caseText, no key) respond as plain JSON, matching /generate.
async function handleGenerateStream(req, res) {
  try {
    const { caseText, model, reasoningEffort, verbosity } = req.body || {};
    if (!caseText || !caseText.trim()) return res.status(400).json({ error: 'Missing caseText' });
    if (!process.env.OPENAI_API_KEY) return res.status(503).json({ error: 'live_unavailable' });

    res.status(200);
    res.setHeader('Content-Type', 'application/x-ndjson');
    res.setHeader('Cache-Control', 'no-cache');
    if (typeof res.flushHeaders === 'function') res.flushHeaders();

    const writeLine = (obj) => {
      try {
        res.write(JSON.stringify(obj) + '\n');
        if (typeof res.flush === 'function') res.flush();
      } catch {
        // Best-effort; if the connection dropped there is nothing more to do.
      }
    };

    try {
      const result = await generatePlan({
        client, caseText, model, reasoningEffort, verbosity, modelDefault: MODEL_DEFAULT,
        onEvent: writeLine,
      });
      writeLine({ type: 'final', result });
      res.end();
    } catch (e) {
      console.error(e);
      writeLine({ type: 'error', error: 'generate_failed' });
      res.end();
    }
  } catch (e) {
    console.error(e);
    try { res.status(500).json({ error: 'generate_failed' }); } catch {}
  }
}
app.post('/generate-stream', handleGenerateStream);
app.post('/api/generate-stream', handleGenerateStream);

/* ---------------- Bundled offline demo (synthetic case library) ---------------- */
app.get('/api/cases', (_req, res) => {
  res.json((cases || [demoCase]).map((c) => ({ id: c.id, title: c.title })));
});

app.post('/api/demo-generate', (req, res) => {
  const lib = (cases && cases.length) ? cases : [demoCase];
  const id = (req.body && req.body.id) || (req.query && req.query.id) || null;
  let picked = null;
  if (id) picked = lib.find((c) => c.id === id) || null;
  if (!picked) picked = lib[Math.floor(Math.random() * lib.length)];
  res.json(picked);
});

/* ---------------- Health ---------------- */
app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    service: 'PRS Co-Pilot API',
    version: VERSION,
    liveConfigured: !!process.env.OPENAI_API_KEY,
    features: { generate: true, managerOverride: true, reason: true, structureExample: true, responsesFallback: true, demoGenerate: true }
  });
});

const port = process.env.PORT || 8787;
app.listen(port, () => console.log(`PRS Co-Pilot API v${VERSION} listening on :${port}`));
