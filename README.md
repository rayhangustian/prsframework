# PRS Co-Pilot — Plan-Review-Synthesize Framework

An AI surgical-planning prototype with an integrated review loop, built as a companion to the undergraduate thesis *"Assessing the Clinical Judgement of ChatGPT in the Planning of Complex Head and Neck Microsurgical Reconstructive Procedure."*

The system explores a single question: not whether an AI can draft a surgical plan, but whether that plan can be made safe and trustworthy enough to be useful in clinical practice. It does this by putting a drafted plan through an automated review before it is finalized.

## How it works

The pipeline mirrors a surgical team:

1. **Planner (operating surgeon)** drafts a structured, conditional surgical plan from a patient case.
2. **Surgical Review Board** evaluates the plan for oncologic soundness, reconstructive soundness, contingency planning, and clarity, then accepts it or returns concerns for revision.
3. **Manager override** distinguishes safety-critical rejections from minor formatting issues, so the loop is strict on substance but not on style.
4. **Synthesizer (chief resident)** converts the approved plan into a formal operative note.

## Live demo

The deployed demo runs fully offline on a library of synthetic, de-identified head and neck cases, so it requires no API key and cannot fail mid-demonstration. Open the site, load an example case, and generate a plan to walk through the full loop.

The live generation path (`/generate`, `/plan`, `/review`, `/synthesize`), which drafts plans for new cases in real time, uses the OpenAI API and requires an `OPENAI_API_KEY` at runtime. It is optional and not needed for the demo.

## Running locally

```bash
npm install
npm start
# opens on http://localhost:8787
```

To enable live generation, add a `.env` file with `OPENAI_API_KEY=your-key`.

## Status and scope

This is a research prototype for academic evaluation. All bundled cases are synthetic and illustrative. **Not for clinical use.**

Author: Muhammad Rayhan Gustian Habibie, Faculty of Medicine, Universitas Indonesia.
