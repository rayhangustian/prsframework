// api/generate.js — Vercel serverless function for live generation.
// Shares orchestration logic with the local server via lib/orchestrator.js.
// Protected by input caps, a best-effort rate limit, and an optional passcode.

import OpenAI from 'openai';
import { generatePlan } from '../lib/orchestrator.js';

const MODEL_DEFAULT = (process.env.MODEL_DEFAULT || 'gpt-4o').trim();
const DEMO_PASSCODE = (process.env.DEMO_PASSCODE || '').trim();
const MAX_CASE_CHARS = 8000;

// Best-effort in-memory rate limit (per warm instance). Not bulletproof, but
// deters casual abuse. Pair with an OpenAI monthly budget cap for real safety.
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

  try {
    const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    const result = await generatePlan({
      client,
      caseText: String(caseText),
      model,
      reasoningEffort,
      verbosity,
      modelDefault: MODEL_DEFAULT,
    });
    return res.status(200).json(result);
  } catch (e) {
    console.error('generate_failed', e?.message || e);
    return res.status(500).json({ error: 'generate_failed' });
  }
}

function safeParse(s) {
  try { return JSON.parse(s); } catch { return {}; }
}

// Allow a longer run for the multi-step pipeline.
export const config = { maxDuration: 60 };
