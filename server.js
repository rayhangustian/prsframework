// server.js — PRS Co-Pilot API
// Orchestrated /generate with auto-revision + Manager Override,
// Reviewer tuned to accept unless safety-critical omissions exist.
// Supports OpenAI Responses (gpt-5 / o3*) and Chat (gpt-4o / 4o-mini).

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MODEL_DEFAULT = (process.env.MODEL_DEFAULT || 'gpt-5').trim();

/* -------------------- Prompt builders -------------------- */
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
      <description>Describe what to do if the condition is true.</description>
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

  return `Role: You are an expert head and neck reconstructive microsurgeon with 20 years of experience at a leading academic cancer center. Your specialty is complex cases requiring free tissue transfer.

Objective: Create a comprehensive, step-by-step surgical plan for the patient case provided below. The plan must be conditional, anticipating intraoperative findings and outlining corresponding actions. Write style: ${style} VERBOSITY.

Patient Case:
${caseText}

Instructions for Your Plan:
1) Output ONLY a single <SurgicalPlan>…</SurgicalPlan>. Do NOT output <Example_Structure> or any preamble.
2) Each action is a <step> with <action_name> and <description>.
3) Use separate <if_block condition='...'> branches for each potential path. Do NOT use "else".
4) If information is unknown, add a step to obtain it (e.g., frozen section, Allen test, imaging, labs).
5) All numeric decisions (margins, neck levels, flap size, doses) must be derived from THIS case or explicitly labeled as “obtain intra-op”.
6) Do not copy exact wording or values from the example below—use it only to mimic tag structure.

Structure-only example (for tag/layout guidance; DO NOT copy its content):
${STRUCTURE_ONLY_EXAMPLE}
${critiqueBlock}`;
};

const reviewerPrompt = (xml, verbosity) => {
  const style = (verbosity || 'medium').toUpperCase();
  return `Role: You are a Surgical Review Board at a major teaching hospital. Your purpose is to evaluate a proposed surgical plan for safety, completeness, and adherence to standard of care. You are strict yet pragmatic. Write style: ${style} VERBOSITY.

Decision rule:
- ACCEPT if the plan meets minimum oncologic and reconstructive standards and includes reasonable contingencies.
- REJECT only for *patient-safety critical* or *major omissions*, for example:
  • No margin strategy for ablation
  • No nodal management when indicated
  • No reconstruction plan for a large defect
  • No contingency planning for common intra-op problems
  • Unsafe airway strategy
- Do NOT reject for formatting/tag naming/style/wording issues—list those as suggestions in feedback.

Proposed Surgical Plan to Review:
${xml}

Evaluation Checklist:
1. Oncologic Soundness (ablation strategy, margins, nodal management).
2. Reconstructive Soundness (flap choice & recipient vessels appropriate).
3. Contingency Planning (common pitfalls anticipated with actions).
4. Clarity and Logic (stepwise and unambiguous).

Return ONLY the tags below on ONE line (no extra text/spaces around):
<SurgicalBoard_Verify>accept|reject</SurgicalBoard_Verify><Feedback_Comment>{Concise rationale; max 1200 chars}</Feedback_Comment>`;
};

const managerPrompt = (reviewText) => {
  return `You are the Manager of the Surgical Review Board. Decide whether the REJECTION reasons below are safety-critical/major or minor.
If the reasons are MINOR (formatting/wording/level-of-detail), output:
<Manager_Override>accept</Manager_Override><Manager_Note>Concise reason</Manager_Note>
If the reasons are MAJOR (patient safety or major omission), output:
<Manager_Override>reject</Manager_Override><Manager_Note>Concise reason</Manager_Note>

Reviewer feedback to evaluate:
${reviewText}

Return only the two tags on one line.`;
};

const synthPrompt = (xml, verbosity) => {
  const style = (verbosity || 'high').toUpperCase();
  return `Role: You are the Chief Surgical Resident. The attending has approved the plan. Convert the structured plan into a formal "Surgical Operative Plan" note for the medical record and team briefing. Write style: ${style} VERBOSITY.

Approved Structured Plan:
${xml}

Output Format (Markdown):
### Preoperative Plan
### Intraoperative Plan: Ablation
### Intraoperative Plan: Reconstruction
### Key Contingency Plans

Constraints: Be concise but complete; preserve logic and contingencies; use professional clinical language.`;
};

/* -------------------- LLM wrapper -------------------- */
const isResponsesModel = (m) => /^gpt-5|^o3/i.test(m || '');
function extractText(resp){
  return resp?.output_text
      || resp?.content?.[0]?.text
      || resp?.choices?.[0]?.message?.content
      || '';
}

async function runLLM({ system, user }, { model, reasoningEffort }) {
  const mdl = (model || MODEL_DEFAULT).trim();

  if (isResponsesModel(mdl)) {
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
    return extractText(resp);
  }

  const chat = await client.chat.completions.create({
    model: mdl,
    temperature: 0.2,
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: user },
    ],
  });
  return extractText(chat);
}

async function runLLMRetry(fn, { tries = 3, baseDelay = 600 } = {}) {
  let lastErr;
  for (let i = 0; i < tries; i++) {
    try { return await fn(); }
    catch (e) {
      lastErr = e;
      const sleep = baseDelay * Math.pow(1.6, i) + Math.random() * 300;
      await new Promise(r => setTimeout(r, sleep));
    }
  }
  throw lastErr;
}

/* -------------------- Atomic step endpoints -------------------- */
app.post('/plan', async (req, res) => {
  try{
    const { caseText, model, reasoningEffort, verbosity } = req.body || {};
    if(!caseText || !caseText.trim()) return res.status(400).json({ error: 'Missing caseText' });

    const content = await runLLM({
      system: 'You create structured, safe, conservative surgical plans and return ONLY the <SurgicalPlan> XML.',
      user: plannerPrompt(caseText, verbosity)
    }, { model, reasoningEffort });

    const m = content.match(/<SurgicalPlan[\s\S]*?<\/SurgicalPlan>/i);
    res.json({ xml: m ? m[0] : content });
  }catch(e){ console.error(e); res.status(500).json({ error: 'planner_failed' }); }
});

app.post('/review', async (req, res) => {
  try{
    const { plannerXml, model, reasoningEffort, verbosity } = req.body || {};
    if(!plannerXml) return res.status(400).json({ error: 'Missing plannerXml' });

    const content = await runLLM({
      system: 'Return only a verify tag and a feedback tag for the safety review.',
      user: reviewerPrompt(plannerXml, verbosity)
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

/* -------------------- Orchestrated flow (+ Manager Override) -------------------- */
app.post('/generate', async (req, res) => {
  try{
    const { caseText, model, reasoningEffort, verbosity } = req.body || {};
    if(!caseText || !caseText.trim()) return res.status(400).json({ error: 'Missing caseText' });

    // 0) Initial plan
    let planText = await runLLMRetry(() => runLLM({
      system: 'You create structured, safe, conservative surgical plans and return ONLY the <SurgicalPlan> XML.',
      user: plannerPrompt(caseText, verbosity)
    }, { model, reasoningEffort }));

    let m = planText.match(/<SurgicalPlan[\s\S]*?<\/SurgicalPlan>/i);
    let planXml = m ? m[0] : planText;

    // 1..3) Review + up to 2 revisions
    let verdict = 'reject', comment = '';
    for (let round = 0; round < 3; round++) {
      const reviewText = await runLLMRetry(() => runLLM({
        system: 'Return only a verify tag and a feedback tag for the safety review.',
        user: reviewerPrompt(planXml, 'medium')
      }, { model, reasoningEffort }));

      const v = (reviewText.match(/<SurgicalBoard_Verify>(.*?)<\/SurgicalBoard_Verify>/i) || [,'reject'])[1];
      const c = (reviewText.match(/<Feedback_Comment>([\s\S]*?)<\/Feedback_Comment>/i) || [,''])[1].trim();

      verdict = /accept/i.test(v || '') ? 'accept' : 'reject';
      comment = c;

      if (verdict === 'accept') break;

      // Revise plan based on reviewer concerns
      const revised = await runLLMRetry(() => runLLM({
        system: 'You create structured, safe, conservative surgical plans and return ONLY the <SurgicalPlan> XML.',
        user: plannerPrompt(caseText, verbosity, c)
      }, { model, reasoningEffort }));

      m = revised.match(/<SurgicalPlan[\s\S]*?<\/SurgicalPlan>/i);
      planXml = m ? m[0] : revised;
    }

    // 4) Manager override if still rejected but reasons are minor
    if (verdict !== 'accept') {
      const mgrText = await runLLMRetry(() => runLLM({
        system: 'Decide acceptance override for minor vs major reasons.',
        user: managerPrompt(comment || 'No comment text from reviewer.')
      }, { model, reasoningEffort }));

      const over = (mgrText.match(/<Manager_Override>(.*?)<\/Manager_Override>/i) || [,'reject'])[1].trim();
      const note = (mgrText.match(/<Manager_Note>([\s\S]*?)<\/Manager_Note>/i) || [,''])[1].trim();

      if (/accept/i.test(over)) {
        verdict = 'accept';
        comment = (comment ? comment + ' ' : '') + `(Manager override: ${note})`;
      } else {
        return res.json({ verdict, comment, xml: planXml, manager_note: note });
      }
    }

    // 5) Synthesize final note
    const markdown = await runLLMRetry(() => runLLM({
      system: 'You write clean, professional operative plans in Markdown.',
      user: synthPrompt(planXml, verbosity)
    }, { model, reasoningEffort }));

    return res.json({ verdict, comment, xml: planXml, markdown });
  }catch(e){
    console.error(e);
    res.status(500).json({ error: 'generate_failed' });
  }
});

/* -------------------- Health root -------------------- */
app.get('/', (_req, res)=> res.json({ ok:true, service:'PRS Co-Pilot API (Responses/Chat Hybrid + Manager Override)' }));

const port = process.env.PORT || 8787;
app.listen(port, ()=> console.log(`PRS Co-Pilot API listening on :${port}`));
