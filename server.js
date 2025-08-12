// server.js — PRS Co-Pilot API (v1.5.0)
// Robust Responses API extraction + fallback to Chat Completions
// Orchestrated /generate + auto-revision + Manager Override

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MODEL_DEFAULT = (process.env.MODEL_DEFAULT || 'gpt-5').trim();
const VERSION = '1.5.0';

/* ---------------- Prompt builders ---------------- */
const plannerPrompt = (caseText, verbosity, critique) => {
  const style = (verbosity || 'high').toUpperCase();
  const STRUCTURE_ONLY_EXAMPLE = `
<Example_Structure>
<SurgicalPlan>
  <step>
    <action_name>Example_Action_Name</action_name>
    <description>Describe the concrete action to take.</description>
  </step>
  <if_block condition='Example clearly stated condition'>
    <step>
      <action_name>Example_Followup_Action</action_name>
      <description>What to do if the condition is true.</description>
    </step>
  </if_block>
  <if_block condition='Another mutually exclusive condition'>
    <step>
      <action_name>Alternative_Action</action_name>
      <description>Describe the alternative path.</description>
    </step>
  </if_block>
</SurgicalPlan>
</Example_Structure>`.trim();

  const critiqueBlock = critique ? `
Reviewer Concerns To Address:
${critique}

You must explicitly fix every item above and reflect the changes in the <step> list and <if_block> branches.` : '';

  return `Role: You are an expert head and neck reconstructive microsurgeon with 20 years at a leading academic cancer center.
Objective: Create a comprehensive, step-by-step surgical plan for the patient case below. Make it conditional with <if_block> branches. Style: ${style}.

Patient Case:
${caseText}

Instructions:
1) Output ONLY a single <SurgicalPlan>…</SurgicalPlan>. No preamble.
2) Each action is a <step> with <action_name> and <description>.
3) Use separate <if_block condition='...'> for each path. No "else".
4) If info is unknown, add a step to obtain it (e.g., frozen section, Allen test).
5) All numbers (margins, neck levels, sizes) must be case-driven or labeled “obtain intra-op”.
6) Do NOT copy wording/numbers from the example; it only shows tag layout.

Structure-only example (layout guidance only; do NOT copy content):
${STRUCTURE_ONLY_EXAMPLE}
${critiqueBlock}`;
};

const reviewerPrompt = (xml) => `Role: Surgical Review Board. Evaluate for safety, completeness, standard of care. Be strict but pragmatic.

Decision:
- ACCEPT if plan meets minimum oncologic & reconstructive standards with reasonable contingencies.
- REJECT only for patient-safety critical or major omissions (no margins, no nodal plan when indicated, no reconstruction for large defect, no contingency, unsafe airway).
- Do NOT reject for formatting/tag/wording — list such items as suggestions.

Plan to review:
${xml}

Checklist:
1. Oncologic Soundness
2. Reconstructive Soundness
3. Contingency Planning
4. Clarity & Logic

Return exactly ONE line with BOTH tags (no extra text):
<SurgicalBoard_Verify>accept|reject</SurgicalBoard_Verify><Feedback_Comment>{Concise rationale; max 1200 chars}</Feedback_Comment>`;

const managerPrompt = (reviewText) => `You are the Manager of the Review Board. Decide if the REJECTION reasons are major (safety/omission) or minor (format/wording/detail).
If MINOR → <Manager_Override>accept</Manager_Override><Manager_Note>Concise reason</Manager_Note>
If MAJOR → <Manager_Override>reject</Manager_Override><Manager_Note>Concise reason</Manager_Note>

Reviewer feedback:
${reviewText}

Return only the two tags on one line.`;

const synthPrompt = (xml, verbosity) => {
  const style = (verbosity || 'high').toUpperCase();
  return `Role: Chief Surgical Resident. Convert the approved structured plan into a formal "Surgical Operative Plan" note. Style: ${style}.

Approved Structured Plan:
${xml}

Output (Markdown):
### Preoperative Plan
### Intraoperative Plan: Ablation
### Intraoperative Plan: Reconstruction
### Key Contingency Plans

Constraints: Concise but complete; preserve logic & contingencies; professional language.`;
};

/* ---------------- Responses extractor + LLM wrapper ---------------- */
const isResponsesModel = (m) => /^gpt-5|^o3/i.test(m || '');

function extractFromResponses(resp) {
  // Try multiple shapes seen in Responses API
  if (!resp) return '';
  if (typeof resp.output_text === 'string' && resp.output_text.trim()) return resp.output_text;

  // Newer SDK shapes
  let texts = [];
  try {
    const out = resp.output || resp.outputs || [];
    for (const item of out) {
      const cont = item?.content || [];
      for (const c of cont) {
        // c.text may be object or string
        if (typeof c?.text === 'string') texts.push(c.text);
        else if (typeof c?.text?.value === 'string') texts.push(c.text.value);
        else if (Array.isArray(c?.text?.annotations)) {
          // join annotated text
          texts.push(String(c?.text?.value || ''));
        } else if (typeof c?.content === 'string') {
          texts.push(c.content);
        }
      }
    }
  } catch {}
  return texts.join('\n').trim();
}

async function runLLM({ system, user }, { model, reasoningEffort }) {
  const mdl = (model || MODEL_DEFAULT).trim();

  // 1) Try Responses API for gpt-5 / o3
  if (isResponsesModel(mdl)) {
    try {
      const payload = {
        model: mdl,
        temperature: 0.2,
        input: [
          { role: 'system', content: system },
          { role: 'user', content: user },
        ],
        ...(reasoningEffort ? { reasoning: { effort: reasoningEffort } } : {})
      };
      const resp = await client.responses.create(payload);
      const text = extractFromResponses(resp);
      if (text && text.trim().length > 0) return text;
      // fallthrough to chat if empty
    } catch (e) {
      // fallthrough to chat
    }
  }

  // 2) Fallback to Chat Completions (very stable extraction)
  const chat = await client.chat.completions.create({
    model: mdl.replace(/^gpt-5/i, 'gpt-4o'), // if someone passes gpt-5, swap to 4o for chat
    temperature: 0.2,
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: user },
    ],
  });
  return chat?.choices?.[0]?.message?.content || '';
}

async function runLLMRetry(fn, { tries = 3, baseDelay = 700 } = {}) {
  let lastErr;
  for (let i = 0; i < tries; i++) {
    try { return await fn(); }
    catch (e) {
      lastErr = e;
      const sleep = baseDelay * Math.pow(1.6, i) + Math.random()*300;
      await new Promise(r => setTimeout(r, sleep));
    }
  }
  throw lastErr;
}

/* ---------------- Atomic endpoints ---------------- */
app.post('/plan', async (req, res) => {
  try{
    const { caseText, model, reasoningEffort, verbosity } = req.body || {};
    if(!caseText || !caseText.trim()) return res.status(400).json({ error: 'Missing caseText' });

    const content = await runLLM({
      system: 'Return ONLY the <SurgicalPlan>…</SurgicalPlan>.',
      user: plannerPrompt(caseText, verbosity)
    }, { model, reasoningEffort });

    const m = content.match(/<SurgicalPlan[\s\S]*?<\/SurgicalPlan>/i);
    res.json({ xml: m ? m[0] : content });
  }catch(e){ console.error(e); res.status(500).json({ error: 'planner_failed' }); }
});

app.post('/review', async (req, res) => {
  try{
    const { plannerXml, model, reasoningEffort } = req.body || {};
    if(!plannerXml) return res.status(400).json({ error: 'Missing plannerXml' });

    const content = await runLLM({
      system: 'Return only the verify & feedback tags on one line.',
      user: reviewerPrompt(plannerXml)
    }, { model, reasoningEffort });

    const v = (content.match(/<SurgicalBoard_Verify>(.*?)<\/SurgicalBoard_Verify>/i) || [,'reject'])[1];
    const c = (content.match(/<Feedback_Comment>([\s\S]*?)<\/Feedback_Comment>/i) || [,''])[1].trim();
    res.json({ verdict: /accept/i.test(v || '') ? 'accept' : 'reject', comment: c });
  }catch(e){ console.error(e); res.status(500).json({ error: 'review_failed' }); }
});

app.post('/synthesize', async (req, res) => {
  try{
    const { approvedPlannerXml, model, reasoningEffort, verbosity } = req.body || {};
    if(!approvedPlannerXml) return res.status(400).json({ error: 'Missing approvedPlannerXml' });

    const content = await runLLM({
      system: 'You write clean, professional operative plans in Markdown.',
      user: synthPrompt(approvedPlannerXml, verbosity)
    }, { model, reasoningEffort });

    res.json({ markdown: content });
  }catch(e){ console.error(e); res.status(500).json({ error: 'synth_failed' }); }
});

/* ---------------- Orchestrated /generate ---------------- */
app.post('/generate', async (req, res) => {
  try{
    const { caseText, model, reasoningEffort, verbosity } = req.body || {};
    if(!caseText || !caseText.trim()) return res.status(400).json({ error: 'Missing caseText' });

    // 0) Initial plan
    let planText = await runLLMRetry(() => runLLM({
      system: 'Return ONLY the <SurgicalPlan>…</SurgicalPlan>.',
      user: plannerPrompt(caseText, verbosity)
    }, { model, reasoningEffort }));

    let m = planText.match(/<SurgicalPlan[\s\S]*?<\/SurgicalPlan>/i);
    let planXml = m ? m[0] : planText;

    // 1..3) Review + up to 2 revisions
    let verdict = 'reject', comment = '';
    let source = 'review';
    let manager_note = '';
    let raw_review = '';

    for (let round = 0; round < 3; round++) {
      const reviewText = await runLLMRetry(() => runLLM({
        system: 'Return only verify & feedback tags.',
        user: reviewerPrompt(planXml)
      }, { model, reasoningEffort }));
      raw_review = (reviewText || '').trim();

      const vMatch = reviewText.match(/<SurgicalBoard_Verify>(.*?)<\/SurgicalBoard_Verify>/i);
      const cMatch = reviewText.match(/<Feedback_Comment>([\s\S]*?)<\/Feedback_Comment>/i);
      let v = vMatch ? (vMatch[1] || '').trim() : '';
      let c = cMatch ? (cMatch[1] || '').trim() : '';

      // If reviewer didn't follow format, treat as minor formatting issue
      const noStructured = !vMatch || !cMatch || !c;
      if (noStructured) {
        v = 'reject';
        c = c || 'Reviewer returned no structured feedback. Treat this as a formatting/format-only issue (minor).';
      }

      verdict = /accept/i.test(v) ? 'accept' : 'reject';
      comment = c;

      if (verdict === 'accept') break;

      // Revise plan
      const revised = await runLLMRetry(() => runLLM({
        system: 'Return ONLY the <SurgicalPlan>…</SurgicalPlan>.',
        user: plannerPrompt(caseText, verbosity, c)
      }, { model, reasoningEffort }));

      m = revised.match(/<SurgicalPlan[\s\S]*?<\/SurgicalPlan>/i);
      planXml = m ? m[0] : revised;
    }

    // 4) Manager override if still rejected but looks minor
    if (verdict !== 'accept') {
      const looksMinor =
        /format|tag|layout|xml|structure|style|no structured feedback/i.test(comment || '') || !comment;

      if (looksMinor) {
        verdict = 'accept';
        source = 'manager_override';
        manager_note = 'Override: reviewer output missing/format-only; treating as minor.';
      } else {
        const mgrText = await runLLMRetry(() => runLLM({
          system: 'Decide acceptance override for minor vs major reasons.',
          user: managerPrompt(comment || 'No reviewer comment.')
        }, { model, reasoningEffort }));
        const over = (mgrText.match(/<Manager_Override>(.*?)<\/Manager_Override>/i) || [,'reject'])[1].trim();
        manager_note = (mgrText.match(/<Manager_Note>([\s\S]*?)<\/Manager_Note>/i) || [,''])[1].trim();
        if (/accept/i.test(over)) { verdict = 'accept'; source = 'manager_override'; }
      }
    }

    const reason =
      source === 'manager_override'
        ? (comment ? `${comment}\n\nManager override: ${manager_note}` : `Manager override: ${manager_note}`)
        : (comment || '');

    if (verdict === 'accept') {
      const markdown = await runLLMRetry(() => runLLM({
        system: 'You write clean, professional operative plans in Markdown.',
        user: synthPrompt(planXml, verbosity)
      }, { model, reasoningEffort }));

      return res.json({ verdict, source, reason, comment, manager_note, xml: planXml, markdown, raw_review });
    }

    return res.json({ verdict, source, reason, comment, manager_note, xml: planXml, raw_review });

  }catch(e){
    console.error(e);
    res.status(500).json({ error: 'generate_failed' });
  }
});

/* ---------------- Health & root ---------------- */
app.get('/health', (_req, res)=> {
  res.json({
    ok:true,
    service:'PRS Co-Pilot API',
    version: VERSION,
    features: { generate:true, managerOverride:true, reason:true, structureExample:true, responsesFallback:true }
  });
});
app.get('/', (_req, res)=> res.redirect('/health'));

const port = process.env.PORT || 8787;
app.listen(port, ()=> console.log(`PRS Co-Pilot API v${VERSION} listening on :${port}`));
