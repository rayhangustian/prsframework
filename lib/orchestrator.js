// lib/orchestrator.js — shared Plan-Review-Synthesize logic.
// Used by both the local Express server (server.js) and the Vercel
// serverless function (api/generate.js). Pure logic: pass in an OpenAI client.

/* ---------------- Prompt builders ---------------- */
export const plannerPrompt = (caseText, verbosity, critique) => {
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

// Adapted from the thesis review protocol (Prompt 2, v2.0): same role framing,
// audit criteria, five scoring domains, and acceptance standard, converted to a
// single-line tagged output the pipeline can parse reliably.
export const reviewerPrompt = (xml) => `You are a surgical review board of senior reconstructive microsurgeons reviewing the proposed plan for this case.

Your task is to audit the planner output for appropriateness, completeness, safety, and clinical applicability.

Plan to review:
${xml}

Critical rules:
- Evaluate the plan only against the case information it reflects. Do not invent missing case facts.
- Do not reject a plan solely because another reasonable option exists.
- Distinguish clearly between critical safety issues, major flaws, and minor omissions.
- Do NOT reject for formatting/tag/wording issues; treat those as minor omissions.
- Do not rewrite the full plan.

Domain scores. Score each from 1 (Strongly Disagree) to 5 (Strongly Agree):
1. The plan correctly understood and captured the clinical issue.
2. The plan stated the most appropriate reconstructive procedure.
3. The plan identified appropriate alternative treatment options.
4. The plan provided comprehensive information beyond treatment options.
5. The plan did not mention therapeutic options that do not exist.

Acceptance standard:
- ACCEPT if the plan is clinically acceptable overall, even if minor omissions remain.
- REJECT only for a critical safety issue, a major domain mismatch, a major omission, or clearly unsupported certainty.

Return exactly ONE line with ALL THREE tags (no extra text):
<SurgicalBoard_Verify>accept|reject</SurgicalBoard_Verify><Domain_Scores>n,n,n,n,n</Domain_Scores><Feedback_Comment>{Concise rationale; max 1200 chars}</Feedback_Comment>`;

export const managerPrompt = (reviewText) => `You are the Manager of the Review Board. Decide if the REJECTION reasons are major (safety/omission) or minor (format/wording/detail).
If MINOR → <Manager_Override>accept</Manager_Override><Manager_Note>Concise reason</Manager_Note>
If MAJOR → <Manager_Override>reject</Manager_Override><Manager_Note>Concise reason</Manager_Note>

Reviewer feedback:
${reviewText}

Return only the two tags on one line.`;

export const synthPrompt = (xml, verbosity) => {
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
  if (!resp) return '';
  if (typeof resp.output_text === 'string' && resp.output_text.trim()) return resp.output_text;
  let texts = [];
  try {
    const out = resp.output || resp.outputs || [];
    for (const item of out) {
      const cont = item?.content || [];
      for (const c of cont) {
        if (typeof c?.text === 'string') texts.push(c.text);
        else if (typeof c?.text?.value === 'string') texts.push(c.text.value);
        else if (Array.isArray(c?.text?.annotations)) texts.push(String(c?.text?.value || ''));
        else if (typeof c?.content === 'string') texts.push(c.content);
      }
    }
  } catch {}
  return texts.join('\n').trim();
}

export async function runLLM({ system, user }, { client, model, reasoningEffort, modelDefault }) {
  const mdl = (model || modelDefault || 'gpt-4o').trim();

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
    } catch (e) {
      // fallthrough to chat
    }
  }

  const chat = await client.chat.completions.create({
    model: mdl.replace(/^gpt-5/i, 'gpt-4o'),
    temperature: 0.2,
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: user },
    ],
  });
  return chat?.choices?.[0]?.message?.content || '';
}

export async function runLLMRetry(fn, { tries = 3, baseDelay = 700 } = {}) {
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

/* ---------------- Orchestrated generate ---------------- */
// Returns { verdict, source, reason, comment, manager_note, xml, markdown?, raw_review }.
//
// Optional `onEvent(ev)` callback reports real pipeline progress as it happens,
// so a caller (e.g. a streaming API route) can drive an honest UI instead of a
// scripted animation. It is entirely optional: when absent, behavior is exactly
// unchanged. Guarded so a throwing/misbehaving listener can never break the
// pipeline. Never include API keys or raw request internals in emitted events.
//
// Event shapes emitted (rounds are 1-indexed for display):
//   { type: 'planner_start' } / { type: 'planner_done' }
//   { type: 'review_start', round } / { type: 'review_done', round, verdict, comment }
//   { type: 'revision_start', round } / { type: 'revision_done', round }
//   { type: 'manager_start' } / { type: 'manager_done', override, note }
//   { type: 'synth_start' } / { type: 'synth_done' }
export async function generatePlan({ client, caseText, model, reasoningEffort, verbosity, modelDefault, onEvent }) {
  const opts = { client, model, reasoningEffort, modelDefault };
  const emit = (ev) => { if (typeof onEvent === 'function') { try { onEvent(ev); } catch {} } };

  // 0) Initial plan
  emit({ type: 'planner_start' });
  let planText = await runLLMRetry(() => runLLM({
    system: 'Return ONLY the <SurgicalPlan>…</SurgicalPlan>.',
    user: plannerPrompt(caseText, verbosity)
  }, opts));
  emit({ type: 'planner_done' });

  let m = planText.match(/<SurgicalPlan[\s\S]*?<\/SurgicalPlan>/i);
  let planXml = m ? m[0] : planText;

  // 1..3) Review + up to 2 revisions
  let verdict = 'reject', comment = '';
  let source = 'review';
  let manager_note = '';
  let raw_review = '';
  let scores = null;

  for (let round = 0; round < 3; round++) {
    const roundNum = round + 1;
    emit({ type: 'review_start', round: roundNum });
    const reviewText = await runLLMRetry(() => runLLM({
      system: 'Return only the single line of verify, domain-score, and feedback tags.',
      user: reviewerPrompt(planXml)
    }, opts));
    raw_review = (reviewText || '').trim();

    const vMatch = reviewText.match(/<SurgicalBoard_Verify>(.*?)<\/SurgicalBoard_Verify>/i);
    const cMatch = reviewText.match(/<Feedback_Comment>([\s\S]*?)<\/Feedback_Comment>/i);
    let v = vMatch ? (vMatch[1] || '').trim() : '';
    let c = cMatch ? (cMatch[1] || '').trim() : '';

    // Optional five-domain Likert scores (thesis instrument). Missing or
    // malformed scores never affect the verdict; they are display-only.
    const sMatch = reviewText.match(/<Domain_Scores>([^<]*)<\/Domain_Scores>/i);
    scores = null;
    if (sMatch) {
      const nums = sMatch[1].split(/[,\s]+/)
        .map((n) => parseInt(n, 10))
        .filter((n) => Number.isInteger(n) && n >= 1 && n <= 5);
      if (nums.length === 5) scores = nums;
    }

    const noStructured = !vMatch || !cMatch || !c;
    if (noStructured) {
      v = 'reject';
      c = c || 'Reviewer returned no structured feedback. Treat this as a formatting/format-only issue (minor).';
    }

    verdict = /accept/i.test(v) ? 'accept' : 'reject';
    comment = c;
    emit({ type: 'review_done', round: roundNum, verdict, comment, scores });

    if (verdict === 'accept') break;

    emit({ type: 'revision_start', round: roundNum });
    const revised = await runLLMRetry(() => runLLM({
      system: 'Return ONLY the <SurgicalPlan>…</SurgicalPlan>.',
      user: plannerPrompt(caseText, verbosity, c)
    }, opts));

    m = revised.match(/<SurgicalPlan[\s\S]*?<\/SurgicalPlan>/i);
    planXml = m ? m[0] : revised;
    emit({ type: 'revision_done', round: roundNum });
  }

  // 4) Manager override if still rejected but looks minor
  if (verdict !== 'accept') {
    emit({ type: 'manager_start' });
    const looksMinor =
      /format|tag|layout|xml|structure|style|no structured feedback/i.test(comment || '') || !comment;

    if (looksMinor) {
      verdict = 'accept';
      source = 'manager_override';
      manager_note = 'Override: reviewer output missing/format-only; treating as minor.';
      emit({ type: 'manager_done', override: true, note: manager_note });
    } else {
      const mgrText = await runLLMRetry(() => runLLM({
        system: 'Decide acceptance override for minor vs major reasons.',
        user: managerPrompt(comment || 'No reviewer comment.')
      }, opts));
      const over = (mgrText.match(/<Manager_Override>(.*?)<\/Manager_Override>/i) || [, 'reject'])[1].trim();
      manager_note = (mgrText.match(/<Manager_Note>([\s\S]*?)<\/Manager_Note>/i) || [, ''])[1].trim();
      const overrideAccepted = /accept/i.test(over);
      if (overrideAccepted) { verdict = 'accept'; source = 'manager_override'; }
      emit({ type: 'manager_done', override: overrideAccepted, note: manager_note });
    }
  }

  const reason =
    source === 'manager_override'
      ? (comment ? `${comment}\n\nManager override: ${manager_note}` : `Manager override: ${manager_note}`)
      : (comment || '');

  const result = { verdict, source, reason, comment, manager_note, xml: planXml, raw_review, scores };

  if (verdict === 'accept') {
    emit({ type: 'synth_start' });
    result.markdown = await runLLMRetry(() => runLLM({
      system: 'You write clean, professional operative plans in Markdown.',
      user: synthPrompt(planXml, verbosity)
    }, opts));
    emit({ type: 'synth_done' });
  }

  return result;
}
