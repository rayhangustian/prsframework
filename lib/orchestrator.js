// lib/orchestrator.js — shared Plan-Review-Synthesize logic.
// Used by both the local Express server (server.js) and the Vercel
// serverless function (api/generate.js). Pure logic: pass in an OpenAI client.

/* ---------------- Prompt builders ---------------- */
// Carried over from the thesis planning protocol (Prompt 1, v2.0). On revision
// rounds a reviewer-critique block is appended; the thesis workflow likewise
// re-runs Prompt 1 when the board rejects. `verbosity` is kept for signature
// compatibility but unused: v2.0 fixes its own style ("concise,
// information-dense, and clinically executable").
export const plannerPrompt = (caseText, verbosity, critique) => {
  const critiqueBlock = critique ? `

<Reviewer_Concerns_To_Address>
${critique}

You must explicitly fix every item above and reflect the changes in the plan sections.
</Reviewer_Concerns_To_Address>` : '';

  return `<Task>
You are assisting with preoperative and intraoperative planning for a complex head and neck microsurgical reconstruction case.

Using only the information provided in the case summary, generate a case-specific surgical plan focused on clinical applicability, completeness, and safety.

<Case_Input>
${caseText}
</Case_Input>

<Critical_Rules>
- Use only the information explicitly provided in the case.
- Do not invent imaging findings, laboratory values, vessel status, pathology details, prior treatment details, dentition, or operative findings.
- If important information is missing, do not guess. State it under "Unknowns / Clarifications Needed" and explain why it matters.
- Do not ask follow-up questions. Proceed using only the given case data.
- Keep the response case-specific. Do not provide generic textbook discussion.
- Include only sections relevant to the case.
- Be concise, information-dense, and clinically executable.
</Critical_Rules>

<Decision_Requirements>
When relevant to the case, explicitly address:
- defect extent and involved tissues
- anatomic, functional, and aesthetic objectives
- the primary reconstructive strategy
- why the primary strategy fits this case
- reasonable alternatives and why they were not selected
- airway implications
- recipient vessel issues
- donor-site considerations
- bone, lining, skin, mucosa, nerve, and soft-tissue requirements
- fixation or skeletal support issues
- contamination, infection, prior radiation, or vessel-depleted neck considerations
- staged versus definitive reconstruction
- case-relevant intraoperative contingencies
</Decision_Requirements>

<Output_Contract>
Return exactly these sections, in this order:

# Patient Summary
- One short paragraph summarizing the clinical problem.

# Defect / Problem Definition
- Site and extent
- Tissues involved
- Important modifiers affecting reconstruction

# Reconstructive Objectives
- Anatomic objectives
- Functional objectives
- Aesthetic objectives, if relevant

# Primary Reconstructive Plan
- Recommended strategy
- Brief justification for why this is the best-fit option

# Alternatives Considered
For each reasonable alternative:
- Option
- Why it was considered
- Why it was not chosen as the primary plan

# Operative Plan
Provide numbered, executable steps.
Include only case-relevant steps.

# Key Contingencies
List only case-relevant contingencies in this format:
- If [specific problem], then [specific response].

# Unknowns / Clarifications Needed
For each item:
- What is unknown
- Why it matters
- How it could change the plan

# Assumptions and Confidence
- Explicit assumptions made
- Which parts of the plan are high-confidence
- Which parts are conditional
</Output_Contract>

<Definition_of_Done>
The response is complete only if:
- a primary plan is clearly stated,
- reasonable alternatives are discussed,
- major uncertainties are declared,
- and contingencies are included when clinically relevant.
</Definition_of_Done>${critiqueBlock}`;
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

// Carried over from the thesis synthesis protocol (Prompt 3, v2.0).
// `verbosity` is kept for signature compatibility but unused; `caseText` is
// optional (the thesis form includes the case for context when available).
export const synthPrompt = (acceptedPlan, verbosity, caseText) => {
  const caseBlock = caseText ? `

<Case_Input>
${caseText}
</Case_Input>` : '';

  return `<Task>
You are preparing a final preoperative surgical planning note for briefing and documentation purposes.

Transform the accepted plan into a concise, clinically natural note.${caseBlock}

<Accepted_Plan>
${acceptedPlan}
</Accepted_Plan>

<Critical_Rules>
- This is a formatting and synthesis task only.
- Do not introduce new clinical content.
- Do not add new rationale, new assumptions, new contingencies, or new recommendations not already present in the accepted plan.
- Preserve uncertainty if it exists.
- Do not infer missing details.
- Use natural clinical language.
- Omit sections not relevant to the case.
</Critical_Rules>

<Output_Contract>
Return exactly these sections, in this order:

# Preoperative Surgical Plan

## Patient Summary
- Brief clinical summary and operative problem

## Defect / Problem and Objectives
- Defect summary
- Anatomic objectives
- Functional objectives
- Aesthetic objectives, if relevant

## Planned Reconstruction
- Primary reconstructive strategy
- Brief rationale carried over from the accepted plan
- Backup strategy, if specified

## Intraoperative Plan
- Numbered operative steps only

## Key Contingency Plans
- Scenario: planned response

## Unknowns / Clarifications Needed
- List unresolved issues exactly as reflected in the accepted plan

## Assumptions / Conditional Elements
- List assumptions or conditional decisions exactly as reflected in the accepted plan
</Output_Contract>

<Definition_of_Done>
The note is complete only if:
- the accepted plan has been reformatted clearly,
- no new clinical content has been added,
- and all unresolved uncertainty has been preserved.
</Definition_of_Done>`;
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
// Returns { verdict, source, reason, comment, manager_note, plan_markdown, markdown?, raw_review, scores }.
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

  // Strip stray code fences some models wrap Markdown output in.
  const stripFences = (s) => String(s || '')
    .replace(/^\s*```(?:markdown|md)?\s*\n?/i, '')
    .replace(/\n?\s*```\s*$/, '')
    .trim();

  // 0) Initial plan (thesis Prompt 1 v2.0: Markdown sections, no XML)
  emit({ type: 'planner_start' });
  let planText = await runLLMRetry(() => runLLM({
    system: 'Return only the surgical plan as Markdown, following the Output_Contract sections exactly. No preamble.',
    user: plannerPrompt(caseText, verbosity)
  }, opts));
  emit({ type: 'planner_done' });

  let planMd = stripFences(planText);

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
      user: reviewerPrompt(planMd)
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
      system: 'Return only the surgical plan as Markdown, following the Output_Contract sections exactly. No preamble.',
      user: plannerPrompt(caseText, verbosity, c)
    }, opts));

    planMd = stripFences(revised);
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

  const result = { verdict, source, reason, comment, manager_note, plan_markdown: planMd, raw_review, scores };

  if (verdict === 'accept') {
    emit({ type: 'synth_start' });
    result.markdown = stripFences(await runLLMRetry(() => runLLM({
      system: 'You write clean, professional preoperative planning notes in Markdown.',
      user: synthPrompt(planMd, verbosity, caseText)
    }, opts)));
    emit({ type: 'synth_done' });
  }

  return result;
}
