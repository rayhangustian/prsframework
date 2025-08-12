// server.js
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MODEL_DEFAULT = process.env.MODEL_DEFAULT || 'gpt-5';

// ---------- Prompt builders ----------
const plannerPrompt = (caseText, verbosity) => {
  const style = (verbosity || 'high').toUpperCase();
  return (
`Role: You are an expert head and neck reconstructive microsurgeon with 20 years of experience at a leading academic cancer center. Your specialty is complex cases requiring free tissue transfer.

Objective: Create a comprehensive, step-by-step surgical plan for the patient case provided below. The plan must be conditional, anticipating common intraoperative findings and outlining corresponding actions. Write style: ${style} VERBOSITY.

Patient Case:
${caseText}

Instructions for Your Plan:
1. Enclose the entire plan in <SurgicalPlan> … </SurgicalPlan> tags.
2. Each distinct action must be a <step> with <action_name> and <description>.
3. Use <if_block condition='...'> to plan for different outcomes. Do NOT use "else." Create a separate <if_block> per path.
4. Do not assume unknowns. If information is missing, add a step to obtain it.
5. Include specific doses, suture sizes, drain types, vessel choices, flap dimensions, monitoring plan, and airway strategy wherever applicable.`
  );
};

const reviewerPrompt = (xml, verbosity) => {
  const style = (verbosity || 'medium').toUpperCase();
  return (
`Role: You are a Surgical Review Board at a major teaching hospital. Your sole purpose is to evaluate a proposed surgical plan for safety, completeness, and adherence to standard of care. You are strict, detail-oriented, and must justify all decisions. Write style: ${style} VERBOSITY.

Task: Analyze the plan below and decide whether to "accept" or "reject" it, providing brief justification based on the checklist.

Proposed Surgical Plan to Review:
${xml}

Evaluation Checklist:
1. Oncologic Soundness: Are ablation strategy and margins appropriate? Is nodal management addressed?
2. Reconstructive Soundness: Is the reconstructive choice appropriate for this defect/patient?
3. Contingency Planning: Are common complications/alternative findings anticipated with actions?
4. Clarity and Logic: Is the plan logical, unambiguous, and stepwise?

Response Format (both tags required):
<SurgicalBoard_Verify>accept|reject</SurgicalBoard_Verify><Feedback_Comment>{RATIONALE_MAX_1200_CHARS}</Feedback_Comment>`
  );
};

const synthPrompt = (xml, verbosity) => {
  const style = (verbosity || 'high').toUpperCase();
  return (
`Role: You are the Chief Surgical Resident. The attending has approved the plan. Convert the structured plan into a formal "Surgical Operative Plan" note for the medical record and team briefing. Write style: ${style} VERBOSITY.

Approved Structured Plan:
${xml}

Output Format (Markdown):
### Preoperative Plan
### Intraoperative Plan: Ablation
### Intraoperative Plan: Reconstruction
### Key Contingency Plans

Constraints: Be concise but complete; preserve logic and contingencies; use professional clinical language.`
  );
};

// ---------- Responses API wrapper ----------
function extractText(resp){
  // Works for Responses API; keeps fallback fields for safety
  return resp.output_text
      || resp.content?.[0]?.text
      || resp.choices?.[0]?.message?.content
      || '';
}

async function runLLM({ system, user }, { model, reasoningEffort, verbosity }) {
  const payload = {
    model: model || MODEL_DEFAULT,
    temperature: 0.2,
    input: [
      { role: 'system', content: system },
      { role: 'user', content: user }
    ]
  };
  // Reasoning control (supported on reasoning-capable models)
  if (reasoningEffort) payload.reasoning = { effort: reasoningEffort };
  // "verbosity" is handled inside the prompt text; not passed as a param to avoid 400s
  const resp = await client.responses.create(payload);
  return extractText(resp);
}

// ---------- Routes ----------
app.post('/plan', async (req, res) => {
  try{
    const { caseText, model, reasoningEffort, verbosity } = req.body || {};
    if(!caseText || !caseText.trim()) return res.status(400).json({ error: 'Missing caseText' });
    const content = await runLLM({
      system: 'You create structured, safe, conservative surgical plans and return ONLY the <SurgicalPlan> XML.',
      user: plannerPrompt(caseText, verbosity)
    }, { model, reasoningEffort, verbosity });

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
    }, { model, reasoningEffort, verbosity: 'medium' });

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
    }, { model, reasoningEffort, verbosity });

    res.json({ markdown: content });
  }catch(e){ console.error(e); res.status(500).json({ error: 'synth_failed' }); }
});

app.get('/', (_, res)=> res.json({ ok:true, service:'PRS Co-Pilot API (Responses API)' }));

const port = process.env.PORT || 8787;
app.listen(port, ()=> console.log(`PRS Co-Pilot API listening on :${port}`));
