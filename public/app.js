/* Plan–Review–Synthesize Framework — Round 2 front-end (plain JS, no build step).
 * Flow: Load Example Case -> Generate -> SLOW staged review/revision stepper ->
 *       typeset results (rendering unchanged from Phase 1).
 * Never-fail: any error falls back to the bundled result silently. */

(function () {
  'use strict';

  // Editable academic context constant (per spec, kept in one obvious place).
  var THESIS_TITLE =
    'Assessing the Clinical Judgement of ChatGPT in the Planning of Complex Head and Neck Microsurgical Reconstructive Procedure';

  var els = {
    randomBtn: document.getElementById('random-case'),
    generateBtn: document.getElementById('generate'),
    caseText: document.getElementById('case-text'),
    caseHint: document.getElementById('case-hint'),
    thesisTitle: document.getElementById('thesis-title'),
    stepperCard: document.getElementById('stepper-card'),
    stepperTitle: document.getElementById('stepper-title'),
    workingPulse: document.getElementById('working-pulse'),
    stepper: document.getElementById('stepper'),
    results: document.getElementById('results'),
    planList: document.getElementById('plan-list'),
    verdictBadge: document.getElementById('verdict-badge'),
    reviewComment: document.getElementById('review-comment'),
    opnote: document.getElementById('opnote')
  };

  // ---- Local fallback so the demo works even if the network/route fails ----
  var FALLBACK = {
    caseText:
      '62-year-old man with biopsy-proven squamous cell carcinoma of the right lateral oral tongue. ' +
      'Primary tumor 4.2 cm, MRI depth of invasion 15 mm, single ipsilateral level II node (2.1 cm). ' +
      'Clinical stage cT3 N1 M0. Cleared for prolonged free-flap surgery; Allen test patent.',
    plan: [
      { type: 'step', name: 'Airway Management',
        description: 'Elective tracheostomy before resection, anticipating tongue and floor-of-mouth edema.' },
      { type: 'step', name: 'Right Hemiglossectomy with Floor-of-Mouth Resection',
        description: 'Resect the right hemitongue in continuity with the involved floor of mouth at 1 cm margins.' },
      { type: 'branch', condition: 'Frozen section shows a positive or close margin',
        steps: [{ name: 'Re-resect Involved Margin', description: 'Re-excise and re-send until margins are clear before reconstruction.' }] },
      { type: 'step', name: 'Radial Forearm Free Flap Reconstruction',
        description: 'Harvest and inset a fasciocutaneous radial forearm free flap to resurface the defect.' }
    ],
    verdict: 'accept',
    comment: 'The plan meets oncologic and reconstructive standards, with secured airway, case-driven margins, and explicit contingencies at each high-risk step.',
    markdown:
      '### Preoperative Plan\n' +
      '- Confirm cT3 N1 M0 oral tongue SCC; verify anesthesia clearance and patent Allen test.\n\n' +
      '### Intraoperative Plan: Ablation\n' +
      '1. **Airway:** elective tracheostomy.\n' +
      '2. **Resection:** right hemiglossectomy with floor-of-mouth resection at 1 cm margins; frozen-section control.\n\n' +
      '### Intraoperative Plan: Reconstruction\n' +
      '1. **Flap:** radial forearm free flap, microvascular anastomosis to facial vessels, watertight inset.\n\n' +
      '### Key Contingency Plans\n' +
      '- **Positive margin:** re-resect until clear.\n' +
      '- **Flap congestion:** re-explore and revise the anastomosis.',
    process: {
      draftSteps: ['Analyzing case & staging…', 'Structuring resection and reconstruction…', 'Adding contingencies…'],
      reviewChecks: ['Oncologic soundness', 'Reconstructive soundness', 'Contingency planning', 'Airway safety'],
      round1: { verdict: 'flagged', concern: 'Single venous outflow: no backup recipient vein is documented, risking flap congestion if the primary vein is unsuitable.' },
      round2: { fix: 'Added a backup recipient vein (external jugular) and an interposition vein-graft option before committing the anastomosis.', verdict: 'accept' }
    }
  };

  // ---- Utilities ----
  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  // Tiny, safe markdown -> HTML (headings, bold, ol/ul, paragraphs). Escaped first.
  function renderMarkdown(md) {
    var lines = String(md || '').replace(/\r\n/g, '\n').split('\n');
    var html = '';
    var listType = null;
    function closeList() { if (listType) { html += '</' + listType + '>'; listType = null; } }
    function inline(text) { return escapeHtml(text).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>'); }
    for (var i = 0; i < lines.length; i++) {
      var trimmed = lines[i].trim();
      if (trimmed === '') { closeList(); continue; }
      var h = /^(#{1,6})\s+(.*)$/.exec(trimmed);
      if (h) { closeList(); html += '<h3>' + inline(h[2]) + '</h3>'; continue; }
      var ol = /^\d+\.\s+(.*)$/.exec(trimmed);
      if (ol) { if (listType !== 'ol') { closeList(); html += '<ol>'; listType = 'ol'; } html += '<li>' + inline(ol[1]) + '</li>'; continue; }
      var ul = /^[-*]\s+(.*)$/.exec(trimmed);
      if (ul) { if (listType !== 'ul') { closeList(); html += '<ul>'; listType = 'ul'; } html += '<li>' + inline(ul[1]) + '</li>'; continue; }
      closeList();
      html += '<p>' + inline(trimmed) + '</p>';
    }
    closeList();
    return html;
  }

  // ---- Results rendering (unchanged from Phase 1) ----
  function loadCase(data) {
    els.caseText.value = data.caseText || '';
    els.caseHint.textContent = 'Example case loaded. Review, then generate a surgical plan.';
    els.generateBtn.disabled = false;
  }

  function renderPlan(plan) {
    var out = '';
    var arr = Array.isArray(plan) ? plan : [];
    for (var i = 0; i < arr.length; i++) {
      var item = arr[i] || {};
      if (item.type === 'branch') {
        var stepsHtml = '';
        var bsteps = Array.isArray(item.steps) ? item.steps : [];
        for (var j = 0; j < bsteps.length; j++) {
          var bs = bsteps[j] || {};
          stepsHtml +=
            '<div class="branch-step">' +
            '<p class="step-title">' + escapeHtml(bs.name) + '</p>' +
            '<p class="step-desc">' + escapeHtml(bs.description) + '</p></div>';
        }
        out +=
          '<div class="plan-branch"><div class="branch-cond"><span class="branch-tag">IF</span>' +
          '<span class="cond-text">' + escapeHtml(item.condition) + '</span></div>' + stepsHtml + '</div>';
      } else {
        out +=
          '<li class="plan-step"><p class="step-title">' + escapeHtml(item.name) + '</p>' +
          '<p class="step-desc">' + escapeHtml(item.description) + '</p></li>';
      }
    }
    els.planList.innerHTML = out;
  }

  function renderResults(data) {
    renderPlan(data.plan);
    els.verdictBadge.textContent = 'APPROVED ✓';
    els.verdictBadge.className = 'verdict-badge verdict-approved';
    els.reviewComment.textContent = data.comment || data.reason || '';
    els.opnote.innerHTML = renderMarkdown(data.markdown || '');
    els.results.hidden = false;
  }

  // ---- Stepper construction helpers ----
  function createStage(role, roundLabel) {
    var li = document.createElement('li');
    li.className = 'stage';
    var head = document.createElement('div');
    head.className = 'stage-head';
    var dot = document.createElement('span');
    dot.className = 'stage-dot';
    var roleEl = document.createElement('span');
    roleEl.className = 'stage-role';
    roleEl.textContent = role;
    head.appendChild(dot);
    head.appendChild(roleEl);
    if (roundLabel) {
      var rnd = document.createElement('span');
      rnd.className = 'stage-round';
      rnd.textContent = roundLabel;
      head.appendChild(rnd);
    }
    li.appendChild(head);
    var body = document.createElement('div');
    body.className = 'stage-body';
    var sub = document.createElement('ul');
    sub.className = 'substatus';
    body.appendChild(sub);
    li.appendChild(body);
    els.stepper.appendChild(li);
    return { li: li, body: body, sub: sub };
  }

  function setStageState(stage, state) {
    stage.li.classList.remove('is-active', 'is-done', 'is-flagged');
    if (state) stage.li.classList.add('is-' + state);
  }

  function addSubline(stage, text, kind) {
    var li = document.createElement('li');
    li.className = 'subline';
    var tick = document.createElement('span');
    tick.className = 'tick';
    tick.textContent = kind === 'check' ? '✓' : (kind === 'work' ? '›' : '•');
    var span = document.createElement('span');
    span.textContent = text;
    li.appendChild(tick);
    li.appendChild(span);
    stage.sub.appendChild(li);
  }

  function addVerdict(stage, type, label) {
    var d = document.createElement('div');
    d.className = 'stage-verdict ' + type;
    var l = document.createElement('span');
    l.className = 'vlabel';
    l.textContent = label;
    d.appendChild(l);
    stage.body.appendChild(d);
  }

  function addBlock(stage, cls, labelText, bodyText) {
    var d = document.createElement('div');
    d.className = cls;
    var lab = document.createElement('span');
    lab.className = cls === 'fix-text' ? 'fix-label' : 'concern-label';
    lab.textContent = labelText;
    var body = document.createElement('span');
    body.textContent = bodyText;
    d.appendChild(lab);
    d.appendChild(body);
    stage.body.appendChild(d);
  }

  function scrollStepper() {
    try { els.stepperCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); } catch (e) {}
  }

  // ---- The paced multi-agent stepper (~10-15s) ----
  async function runStepper(p) {
    els.stepperCard.hidden = false;
    els.stepper.innerHTML = '';
    els.workingPulse.classList.remove('is-idle');
    scrollStepper();

    var draftSteps = (p && p.draftSteps) || FALLBACK.process.draftSteps;
    var reviewChecks = (p && p.reviewChecks) || FALLBACK.process.reviewChecks;
    var round1 = (p && p.round1) || FALLBACK.process.round1;
    var round2 = (p && p.round2) || FALLBACK.process.round2;

    // Stage 1 — Operating Surgeon drafts
    els.stepperTitle.textContent = 'Operating surgeon — drafting the plan';
    var s1 = createStage('Operating Surgeon', null);
    setStageState(s1, 'active');
    for (var i = 0; i < draftSteps.length; i++) {
      await sleep(700);
      addSubline(s1, draftSteps[i], 'work');
      scrollStepper();
    }
    await sleep(550);
    setStageState(s1, 'done');

    // Stage 2 — Surgical Review Board, Round 1 -> FLAGGED
    els.stepperTitle.textContent = 'Surgical Review Board — reviewing (Round 1)';
    var s2 = createStage('Surgical Review Board', 'Round 1');
    setStageState(s2, 'active');
    for (var j = 0; j < reviewChecks.length; j++) {
      await sleep(600);
      addSubline(s2, reviewChecks[j], 'check');
      scrollStepper();
    }
    await sleep(650);
    addVerdict(s2, 'flagged', 'Flagged — concern raised');
    await sleep(450);
    addBlock(s2, 'concern-text', 'Board concern', round1.concern || 'A safety concern was raised.');
    setStageState(s2, 'flagged');
    scrollStepper();
    await sleep(1600); // let the audience read the concern

    // Stage 3 — Revision, Round 2 -> APPROVED
    els.stepperTitle.textContent = 'Revision — Round 2 (addressing the board)';
    var s3 = createStage('Revision', 'Round 2');
    setStageState(s3, 'active');
    await sleep(700);
    addSubline(s3, 'Operating surgeon addressing the board concern…', 'work');
    scrollStepper();
    await sleep(750);
    addBlock(s3, 'fix-text', 'Revision applied', round2.fix || 'The concern was addressed.');
    scrollStepper();
    await sleep(850);
    addSubline(s3, 'Re-submitting to Surgical Review Board…', 'work');
    scrollStepper();
    await sleep(950);
    addSubline(s3, 'Review Board re-evaluating the revised plan…', 'work');
    scrollStepper();
    await sleep(800);
    addVerdict(s3, 'approved', 'Approved ✓');
    setStageState(s3, 'done');
    scrollStepper();
    await sleep(1000);

    // Stage 4 — Chief Resident writes the note
    els.stepperTitle.textContent = 'Chief resident — writing the operative note';
    var s4 = createStage('Chief Resident', null);
    setStageState(s4, 'active');
    await sleep(700);
    addSubline(s4, 'Composing the formal operative note…', 'work');
    scrollStepper();
    await sleep(1000);
    setStageState(s4, 'done');

    els.stepperTitle.textContent = 'Review complete — plan approved';
    els.workingPulse.classList.add('is-idle');
  }

  // ---- State ----
  var currentCase = null;   // full case object currently loaded in the intake area
  var caseIds = [];         // known library ids, used to avoid immediate repeats

  // ---- Handlers ----
  // Fetch a case by id (or random when id is null). Validates shape; never throws.
  async function fetchCase(id) {
    try {
      var res = await fetch('/api/demo-generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(id ? { id: id } : {})
      });
      if (res.ok) {
        var payload = await res.json();
        if (payload && Array.isArray(payload.plan) && payload.markdown) return payload;
      }
    } catch (e) { /* fall through */ }
    return FALLBACK; // never-fail
  }

  // Load the lightweight case-id list once (best effort; never blocks the demo).
  async function loadCaseIds() {
    try {
      var res = await fetch('/api/cases');
      if (res.ok) {
        var list = await res.json();
        if (Array.isArray(list) && list.length) caseIds = list.map(function (c) { return c.id; });
      }
    } catch (e) { /* ignore; random fetch still works */ }
  }

  // Pick a random id different from the currently loaded one, when possible.
  function pickDifferentId() {
    if (!caseIds.length) return null; // server will pick random
    var currId = currentCase && currentCase.id;
    var choices = caseIds.filter(function (x) { return x !== currId; });
    var pool = choices.length ? choices : caseIds;
    return pool[Math.floor(Math.random() * pool.length)];
  }

  async function onRandomCase() {
    var data = await fetchCase(pickDifferentId());
    currentCase = data;
    loadCase(data);
  }

  async function onGenerate() {
    els.generateBtn.disabled = true;
    els.results.hidden = true;

    // Run the SAME case that is loaded so the stepper narrative matches it.
    var data = currentCase || await fetchCase(null);
    currentCase = data;

    try {
      await runStepper(data.process);
    } catch (e) { /* animation must never block results */ }

    try {
      renderResults(data);
      try { els.results.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch (e2) {}
    } catch (e) {
      try { renderResults(FALLBACK); } catch (e3) {}
    }

    els.generateBtn.disabled = false;
  }

  // Guard against any uncaught rejection/error surfacing to the user.
  window.addEventListener('unhandledrejection', function (ev) { ev.preventDefault(); });
  window.addEventListener('error', function () { /* swallow */ });

  els.thesisTitle.textContent = THESIS_TITLE;
  els.randomBtn.addEventListener('click', onRandomCase);
  els.generateBtn.addEventListener('click', onGenerate);
  loadCaseIds(); // best-effort; enables repeat-avoidance on "Load Example Case"
})();
