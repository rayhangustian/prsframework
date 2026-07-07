# PRS Co-Pilot — Plan-Review-Synthesize Framework

An AI surgical-planning prototype with an integrated review loop, built as a companion to the undergraduate thesis *"Assessing the Clinical Judgement of ChatGPT in the Planning of Complex Head and Neck Microsurgical Reconstructive Procedure."*

The system explores a single question: not whether an AI can draft a surgical plan, but whether that plan can be made safe and trustworthy enough to be useful in clinical practice. It does this by putting a drafted plan through an automated review before it is finalized.

## How it works

The pipeline mirrors a surgical team:

1. **Planner (operating surgeon)** drafts a structured, conditional surgical plan from a patient case.
2. **Surgical Review Board** evaluates the plan for oncologic soundness, reconstructive soundness, contingency planning, and clarity, then accepts it or returns concerns for revision.
3. **Manager override** distinguishes safety-critical rejections from minor formatting issues, so the loop is strict on substance but not on style.
4. **Synthesizer (chief resident)** converts the approved plan into a formal operative note.

## Live generation with an offline fallback

"Generate Surgical Plan" runs the full pipeline live on the case in the box. If
live generation is unavailable for any reason (no key configured, rate limit,
network error), it falls back automatically to a bundled result so the demo
never breaks. "Run offline demo" always plays a bundled synthetic case with no
network call at all.

The bundled library is a set of synthetic, de-identified head and neck cases, so
the offline path requires no API key and cannot fail mid-demonstration.

## Deploying (Vercel)

The app deploys on Vercel with no credit card required:

1. Import this repository at vercel.com.
2. To enable live generation, add these environment variables in the project settings:
   - `OPENAI_API_KEY` — your OpenAI API key (required for live generation).
   - `MODEL_DEFAULT` — optional, defaults to `gpt-4o`.
   - `DEMO_PASSCODE` — optional. If set, live generation requires this code, which
     protects the key from public abuse. The offline demo stays open.
3. Deploy. Without `OPENAI_API_KEY` the site still works as the offline demo.

Static files are served from `public/`; live generation runs as the serverless
function in `api/generate.js`. Both share the pipeline logic in `lib/orchestrator.js`.

## Running locally

```bash
npm install
node scripts/build-cases.mjs   # generates public/cases.json for the browser
npm start                      # http://localhost:8787
```

To enable live generation locally, add a `.env` file with `OPENAI_API_KEY=your-key`.

## Status and scope

This is a research prototype for academic evaluation. All bundled cases are synthetic and illustrative. **Not for clinical use.**

Author: Muhammad Rayhan Gustian Habibie, Faculty of Medicine, Universitas Indonesia.
