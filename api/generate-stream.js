// api/generate-stream.js — Vercel serverless function for live generation with
// a real-time NDJSON event stream. Mirrors the guards in api/generate.js
// (method, key configured, passcode, case length, rate limit) but streams each
// real pipeline event as it happens, ending with a { type: 'final', result }
// line carrying the exact same result shape /api/generate returns.
//
// This file does not change or replace /api/generate; it is an additional,
// optional path. Any guard failure or mid-stream error responds in a way the
// client can detect and fall back from, so the existing /api/generate path
// remains the safety net. api/generate.js itself is untouched.

import OpenAI from 'openai';
import { generatePlan } from '../lib/orchestrator.js';

const MODEL_DEFAULT = (process.env.MODEL_DEFAULT || 'gpt-4o').trim();
const DEMO_PASSCODE = (process.env.DEMO_PASSCODE || '').trim();
const MAX_CASE_CHARS = 8000;

// Best-effort in-memory rate limit (per warm instance), independent from the
// one in api/generate.js since serverless instances/modules are isolated.
const RL_WINDOW_MS = 60 * 1000;
const RL_MAX = 5; // requests per IP per window
const hits = new Map();

function rateLimited(ip) {
  const now = Date.now();
  const arr = (hits.get(ip) || []).filter((t) => now - t < RL_WINDOW_MS);
  arr.push(now);
  hits.set(ip, arr);
  return arr.length > RL_MAX;
}

function safeParse(s) {
  try { return JSON.parse(s); } catch { return {}; }
}

function writeLine(res, obj) {
  try {
    res.write(JSON.stringify(obj) + '\n');
    if (typeof res.flush === 'function') res.flush();
  } catch {
    // Best-effort; if the connection dropped there is nothing more to do.
  }
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'method_not_allowed' });
  }

  if (!process.env.OPENAI_API_KEY) {
    return res.status(503).json({ error: 'live_unavailable' });
  }

  const body = typeof req.body === 'string' ? safeParse(req.body) : (req.body || {});
  const { caseText, model, reasoningEffort, verbosity, passcode } = body;

  if (DEMO_PASSCODE && (passcode || '').trim() !== DEMO_PASSCODE) {
    return res.status(401).json({ error: 'bad_passcode' });
  }

  if (!caseText || !String(caseText).trim()) {
    return res.status(400).json({ error: 'missing_caseText' });
  }
  if (String(caseText).length > MAX_CASE_CHARS) {
    return res.status(413).json({ error: 'case_too_long' });
  }

  const ip = (req.headers['x-forwarded-for'] || '').split(',')[0].trim() || 'unknown';
  if (rateLimited(ip)) {
    return res.status(429).json({ error: 'rate_limited' });
  }

  // All guards passed: from here on, respond as a streamed NDJSON body. Any
  // failure past this point must still end the stream (not throw an
  // unhandled response), so the client can detect it and fall back.
  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/x-ndjson');
  res.setHeader('Cache-Control', 'no-cache');
  if (typeof res.flushHeaders === 'function') {
    try { res.flushHeaders(); } catch {}
  }

  try {
    const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    const result = await generatePlan({
      client,
      caseText: String(caseText),
      model,
      reasoningEffort,
      verbosity,
      modelDefault: MODEL_DEFAULT,
      onEvent: (ev) => writeLine(res, ev),
    });
    writeLine(res, { type: 'final', result });
    res.end();
  } catch (e) {
    console.error('generate_stream_failed', e?.message || e);
    writeLine(res, { type: 'error', error: 'generate_failed' });
    res.end();
  }
}

// Allow a longer run for the multi-step pipeline, and ask Vercel's Node.js
// runtime to stream the response as it is written rather than buffering it.
export const config = { maxDuration: 60, supportsResponseStreaming: true };
