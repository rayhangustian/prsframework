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
    offlineBtn: document.getElementById('offline-demo'),
    passcode: document.getElementById('passcode'),
    liveNote: document.getElementById('live-note'),
    caseText: document.getElementById('case-text'),
    caseHint: document.getElementById('case-hint'),
    thesisTitle: document.getElementById('thesis-title'),
    reviewScores: document.getElementById('review-scores'),
    stepperCard: document.getElementById('stepper-card'),
    stepperTitle: document.getElementById('stepper-title'),
    liveBadge: document.getElementById('live-badge'),
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

  // Parse a <SurgicalPlan> XML string (from live generation) into the same
  // plan-array shape the renderer uses for bundled cases. Best-effort; on any
  // trouble it returns whatever it managed, and the caller falls back if empty.
  function xmlToPlan(xml) {
    var out = [];
    if (!xml) return out;
    var s = String(xml);
    var re = /<if_block\b[^>]*condition=(?:'([^']*)'|"([^"]*)")[^>]*>([\s\S]*?)<\/if_block>|<step\b[^>]*>([\s\S]*?)<\/step>/gi;
    function parseSteps(block) {
      var steps = [];
      var sre = /<step\b[^>]*>([\s\S]*?)<\/step>/gi, sm;
      while ((sm = sre.exec(block))) {
        steps.push({
          name: field(sm[1], 'action_name'),
          description: field(sm[1], 'description')
        });
      }
      return steps;
    }
    function field(chunk, tag) {
      var m = new RegExp('<' + tag + '\\b[^>]*>([\\s\\S]*?)<\\/' + tag + '>', 'i').exec(chunk || '');
      return m ? m[1].replace(/\s+/g, ' ').replace(/_/g, ' ').trim() : '';
    }
    var m;
    while ((m = re.exec(s))) {
      if (m[3] !== undefined && (m[1] !== undefined || m[2] !== undefined)) {
        out.push({ type: 'branch', condition: (m[1] || m[2] || '').trim(), steps: parseSteps(m[3]) });
      } else if (m[4] !== undefined) {
        out.push({ type: 'step', name: field(m[4], 'action_name'), description: field(m[4], 'description') });
      }
    }
    return out;
  }

  function setNote(text, kind) {
    if (!els.liveNote) return;
    els.liveNote.textContent = text || '';
    els.liveNote.className = 'live-note' + (kind ? ' note-' + kind : '');
  }

  // ---- Results rendering ----
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

  // ---- v2.0 plan rendering (thesis Prompt 1 output: Markdown sections) ----
  // Splits a v2.0 plan into its "# Title" sections. Returns [] when the text
  // has no recognizable sections, so the caller can fall back to plain
  // markdown rendering.
  function parsePlanSections(md) {
    var lines = String(md || '').replace(/\r\n/g, '\n').split('\n');
    var sections = [];
    var cur = null;
    for (var i = 0; i < lines.length; i++) {
      var t = lines[i].trim();
      var h = /^#\s+(.+)$/.exec(t); // exactly one '#': section heading per the contract
      if (h) {
        cur = { title: h[1].trim(), body: [] };
        sections.push(cur);
      } else if (cur) {
        cur.body.push(lines[i]);
      }
    }
    return sections;
  }

  // Renders "- If X, then Y." contingency bullets as the IF-branch boxes the
  // bundled cases use; lines that do not match render as plain markdown.
  function renderContingencies(text) {
    var lines = String(text || '').split('\n');
    var boxes = '';
    var rest = [];
    for (var i = 0; i < lines.length; i++) {
      var t = lines[i].trim();
      var m = /^[-*]\s*If\s+(.+?),\s*then\s+(.+?)\.?\s*$/i.exec(t);
      if (m) {
        boxes +=
          '<div class="plan-branch"><div class="branch-cond"><span class="branch-tag">IF</span>' +
          '<span class="cond-text">' + escapeHtml(m[1]) + '</span></div>' +
          '<div class="branch-step"><p class="step-desc">' + escapeHtml(m[2]) + '</p></div></div>';
      } else if (t) {
        rest.push(lines[i]);
      }
    }
    if (!boxes) return renderMarkdown(text);
    var restMd = rest.join('\n').trim();
    return boxes + (restMd ? renderMarkdown(restMd) : '');
  }

  function renderPlanV2(md) {
    var sections = parsePlanSections(md);
    var html = '';
    if (!sections.length) {
      html = renderMarkdown(md);
    } else {
      for (var i = 0; i < sections.length; i++) {
        var s = sections[i];
        var body = s.body.join('\n');
        html += '<h3 class="plan-sec-title">' + escapeHtml(s.title) + '</h3>';
        html += /^key contingenc/i.test(s.title) ? renderContingencies(body) : renderMarkdown(body);
      }
    }
    els.planList.innerHTML = '<li class="plan-md">' + html + '</li>';
  }

  // Short labels for the five thesis scoring domains, in prompt order.
  var DOMAIN_LABELS = [
    'Clinical issue captured',
    'Procedure choice',
    'Alternatives',
    'Comprehensiveness',
    'No fabricated options'
  ];

  // Builds the row of five domain-score chips. Returns null unless scores is
  // a well-formed array of five 1-5 integers (bundled cases have none).
  function buildScoreChips(scores) {
    if (!Array.isArray(scores) || scores.length !== DOMAIN_LABELS.length) return null;
    var frag = document.createDocumentFragment();
    for (var i = 0; i < DOMAIN_LABELS.length; i++) {
      var n = scores[i];
      if (!(n >= 1 && n <= 5)) return null;
      var chip = document.createElement('span');
      chip.className = 'score-chip' + (n <= 2 ? ' score-low' : '');
      var lab = document.createElement('span');
      lab.textContent = DOMAIN_LABELS[i];
      var val = document.createElement('span');
      val.className = 'score-val';
      val.textContent = n + '/5';
      chip.appendChild(lab);
      chip.appendChild(val);
      frag.appendChild(chip);
    }
    return frag;
  }

  function renderResults(data) {
    if (data.plan_markdown) {
      // Live v2.0 plan: Markdown sections from the thesis planning protocol.
      renderPlanV2(data.plan_markdown);
    } else {
      // Bundled cases (and any legacy result) carry a structured plan array.
      var plan = (Array.isArray(data.plan) && data.plan.length) ? data.plan : xmlToPlan(data.xml);
      renderPlan(plan);
    }
    var accepted = !data.verdict || /accept/i.test(data.verdict);
    els.verdictBadge.textContent = accepted ? 'APPROVED ✓' : 'NEEDS REVISION';
    els.verdictBadge.className = 'verdict-badge ' + (accepted ? 'verdict-approved' : 'verdict-flagged');
    if (els.reviewScores) {
      els.reviewScores.innerHTML = '';
      var chips = buildScoreChips(data.scores);
      if (chips) { els.reviewScores.appendChild(chips); els.reviewScores.hidden = false; }
      else { els.reviewScores.hidden = true; }
    }
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
    lab.className = cls === 'fix-text' ? 'fix-label' : (cls === 'manager-text' ? 'manager-label' : 'concern-label');
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
  // `opts.completionTitle` lets callers (e.g. the offline replay) relabel the
  // final headline honestly without duplicating this whole animation.
  async function runStepper(p, opts) {
    els.stepperCard.hidden = false;
    els.stepper.innerHTML = '';
    if (els.liveBadge) els.liveBadge.hidden = true; // this is a scripted/replayed animation, not a real trace
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

    els.stepperTitle.textContent = (opts && opts.completionTitle) || 'Review complete — plan approved';
    els.workingPulse.classList.add('is-idle');
  }

  // ---- State ----
  var currentCase = null;   // full case object currently loaded in the intake area
  var CASE_LIBRARY = [];    // bundled synthetic cases, loaded once from cases.json

  // Load the bundled case library once from a static file. No server needed,
  // so this works on any static host and powers the offline fallback.
  var libraryReady = (async function () {
    try {
      var res = await fetch('cases.json', { cache: 'force-cache' });
      if (res.ok) {
        var data = await res.json();
        var lib = Array.isArray(data) ? data : (data && data.cases) || [];
        CASE_LIBRARY = lib.filter(function (c) { return c && Array.isArray(c.plan) && c.markdown; });
      }
    } catch (e) { /* fall back to the built-in FALLBACK case */ }
  })();

  // ---- Handlers ----
  // Return a bundled case by id, a random one (different from current), or the
  // built-in FALLBACK. Never throws.
  async function fetchCase(id) {
    try {
      await libraryReady;
      if (CASE_LIBRARY.length) {
        var picked = null;
        if (id) picked = CASE_LIBRARY.filter(function (c) { return c.id === id; })[0] || null;
        if (!picked) {
          var currId = currentCase && currentCase.id;
          var pool = CASE_LIBRARY.filter(function (c) { return c.id !== currId; });
          if (!pool.length) pool = CASE_LIBRARY;
          picked = pool[Math.floor(Math.random() * pool.length)];
        }
        if (picked) return picked;
      }
    } catch (e) { /* fall through */ }
    return FALLBACK; // never-fail
  }

  async function onRandomCase() {
    var data = await fetchCase(null);
    currentCase = data;
    loadCase(data);
    setNote('');
  }

  // Generic stepper narrative for a live run (the live API returns no process).
  function liveProcess() {
    return {
      draftSteps: ['Reading the patient case…', 'Drafting a case-specific plan…', 'Declaring unknowns and contingencies…'],
      reviewChecks: ['Appropriateness', 'Completeness', 'Safety', 'Clinical applicability'],
      round1: { verdict: 'flagged', concern: 'The review board is checking the draft against safety and completeness standards.' },
      round2: { fix: 'The plan is revised to address the board’s concerns before final synthesis.', verdict: 'accept' }
    };
  }

  // Call the live generation endpoint. Never throws; returns {ok, data|reason}.
  async function tryLiveGenerate(caseText) {
    try {
      var res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ caseText: caseText, passcode: (els.passcode && els.passcode.value) || '' })
      });
      if (res.ok) {
        var data = await res.json();
        if (data && (data.xml || (Array.isArray(data.plan) && data.plan.length))) return { ok: true, data: data };
        return { ok: false, reason: 'empty' };
      }
      var err = {};
      try { err = await res.json(); } catch (e) {}
      return { ok: false, reason: (err && err.error) || ('http_' + res.status) };
    } catch (e) {
      return { ok: false, reason: 'network' };
    }
  }

  // ---- Real-event live stepper (honest trace of the actual pipeline run) ----
  // Neutral, checklist-style working sublines shown while waiting for the next
  // real event within a stage. These never assert a verdict or fabricate a
  // finding; they only describe what the stage is generically doing.
  var NEUTRAL_LINES = {
    planner: ['Reading the patient case…', 'Defining the defect and objectives…', 'Weighing alternatives against the primary strategy…', 'Declaring unknowns and assumptions…'],
    review: ['Auditing appropriateness…', 'Auditing completeness…', 'Auditing safety…', 'Auditing clinical applicability…', 'Scoring the five domains…'],
    revision: ['Operating surgeon addressing the board’s concerns…'],
    manager: ['Weighing whether the concern is safety-critical or a formatting note…'],
    synth: ['Composing the formal operative note…']
  };
  var MIN_DWELL_MS = 650;
  var MAX_DWELL_MS = 900;
  function dwellMs() { return MIN_DWELL_MS + Math.random() * (MAX_DWELL_MS - MIN_DWELL_MS); }

  // A tiny async queue: the network reader pushes real pipeline events onto it
  // as they arrive (which may be all at once, if a host buffers the stream);
  // the stepper drains it on its own paced schedule so the animation always
  // plays naturally regardless of arrival timing.
  function createEventQueue() {
    var items = [];
    var waiter = null;
    var closed = false;
    return {
      push: function (ev) {
        items.push(ev);
        if (waiter) { var w = waiter; waiter = null; w(); }
      },
      close: function () {
        closed = true;
        if (waiter) { var w = waiter; waiter = null; w(); }
      },
      next: async function () {
        while (!items.length && !closed) {
          await new Promise(function (resolve) { waiter = resolve; });
        }
        return items.length ? items.shift() : null; // null = closed and drained
      }
    };
  }

  // Reads the NDJSON stream from /api/generate-stream, pushing each real
  // pipeline event onto `queue` as it is parsed. Tolerates partial lines
  // across chunk boundaries. Returns {ok:true, data:result} on a clean
  // 'final' line, or {ok:false, reason} on any guard failure, mid-stream
  // 'error' line, missing 'final' line, or network/parse trouble. Always
  // closes the queue before returning (even on throw), so the stepper loop
  // never hangs waiting for more events.
  async function streamLiveGenerate(caseText, queue) {
    try {
      var res;
      try {
        res = await fetch('/api/generate-stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ caseText: caseText, passcode: (els.passcode && els.passcode.value) || '' })
        });
      } catch (e) {
        return { ok: false, reason: 'network' };
      }

      if (!res.ok) {
        var err = {};
        try { err = await res.json(); } catch (e) {}
        return { ok: false, reason: (err && err.error) || ('http_' + res.status) };
      }

      if (!res.body || typeof res.body.getReader !== 'function') {
        return { ok: false, reason: 'no_stream' };
      }

      var reader = res.body.getReader();
      var decoder = new TextDecoder();
      var buffer = '';
      var finalResult = null;
      var streamError = false;

      function handleLine(line) {
        line = (line || '').trim();
        if (!line) return;
        var obj;
        try { obj = JSON.parse(line); } catch (e) { return; } // tolerate a stray partial/bad line
        if (!obj || typeof obj !== 'object') return;
        if (obj.type === 'final') finalResult = obj.result || null;
        else if (obj.type === 'error') streamError = true;
        else queue.push(obj);
      }

      try {
        while (true) {
          var chunk = await reader.read();
          if (chunk.done) break;
          buffer += decoder.decode(chunk.value, { stream: true });
          var parts = buffer.split('\n');
          buffer = parts.pop();
          for (var i = 0; i < parts.length; i++) handleLine(parts[i]);
        }
        if (buffer) handleLine(buffer);
      } catch (e) {
        return { ok: false, reason: 'network' };
      }

      if (streamError) return { ok: false, reason: 'generate_failed' };
      if (!finalResult) return { ok: false, reason: 'stream_incomplete' };
      return { ok: true, data: finalResult };
    } finally {
      queue.close();
    }
  }

  // Drains the queue and renders one stepper stage/subline per real pipeline
  // event, honestly. Never fabricates a flag: a round-1 accept is shown as
  // exactly that. Applies a minimum per-event dwell for pacing, and shows
  // neutral working sublines while waiting on a stage for the next event.
  async function runLiveStepper(queue) {
    els.stepperCard.hidden = false;
    els.stepper.innerHTML = '';
    if (els.liveBadge) els.liveBadge.hidden = false;
    els.workingPulse.classList.remove('is-idle');
    els.stepperTitle.textContent = 'Operating surgeon — reading the case';
    scrollStepper();

    var reviewStages = {};
    var revisionStages = {};
    var managerStage = null;
    var neutralTimer = null;

    function stopNeutral() {
      if (neutralTimer) { clearInterval(neutralTimer); neutralTimer = null; }
    }
    function startNeutral(stage, lines) {
      stopNeutral();
      if (!stage || !lines || !lines.length) return;
      var idx = 0;
      neutralTimer = setInterval(function () {
        if (idx >= lines.length) { stopNeutral(); return; } // show each line once; never repeat
        addSubline(stage, lines[idx], 'work');
        idx++;
        scrollStepper();
      }, 900);
    }

    try {
      while (true) {
        var ev = await queue.next();
        if (ev === null) break; // stream closed, nothing more to show

        var startedAt = Date.now();

        switch (ev.type) {
          case 'planner_start': {
            els.stepperTitle.textContent = 'Operating surgeon — drafting the plan';
            var s1 = createStage('Operating Surgeon', null);
            setStageState(s1, 'active');
            reviewStages.__planner = s1;
            startNeutral(s1, NEUTRAL_LINES.planner);
            break;
          }
          case 'planner_done': {
            stopNeutral();
            if (reviewStages.__planner) setStageState(reviewStages.__planner, 'done');
            break;
          }
          case 'review_start': {
            stopNeutral();
            els.stepperTitle.textContent = 'Surgical Review Board — reviewing (Round ' + ev.round + ')';
            var rs = createStage('Surgical Review Board', 'Round ' + ev.round);
            setStageState(rs, 'active');
            reviewStages[ev.round] = rs;
            startNeutral(rs, NEUTRAL_LINES.review);
            break;
          }
          case 'review_done': {
            stopNeutral();
            var rstage = reviewStages[ev.round];
            if (rstage) {
              var accepted = /accept/i.test(ev.verdict || '');
              var chips = buildScoreChips(ev.scores);
              if (chips) {
                var strip = document.createElement('div');
                strip.className = 'score-strip';
                strip.appendChild(chips);
                rstage.body.appendChild(strip);
              }
              if (accepted) {
                addVerdict(rstage, 'approved', ev.round === 1 ? 'Approved on first review' : 'Approved ✓');
                setStageState(rstage, 'done');
              } else {
                addVerdict(rstage, 'flagged', 'Flagged — concern raised');
                addBlock(rstage, 'concern-text', 'Board concern', ev.comment || 'A concern was raised.');
                setStageState(rstage, 'flagged');
              }
            }
            break;
          }
          case 'revision_start': {
            els.stepperTitle.textContent = 'Revision — Round ' + ev.round + ' (addressing the board)';
            var rvs = createStage('Revision', 'Round ' + ev.round);
            setStageState(rvs, 'active');
            revisionStages[ev.round] = rvs;
            startNeutral(rvs, NEUTRAL_LINES.revision);
            break;
          }
          case 'revision_done': {
            stopNeutral();
            var rvstage = revisionStages[ev.round];
            if (rvstage) setStageState(rvstage, 'done');
            break;
          }
          case 'manager_start': {
            els.stepperTitle.textContent = 'Manager — reviewing the board’s rejection';
            managerStage = createStage('Manager', null);
            setStageState(managerStage, 'active');
            startNeutral(managerStage, NEUTRAL_LINES.manager);
            break;
          }
          case 'manager_done': {
            stopNeutral();
            if (managerStage) {
              var overrideAccepted = !!ev.override;
              if (overrideAccepted) {
                addVerdict(managerStage, 'approved', 'Manager override: accepted');
                addBlock(managerStage, 'manager-text', 'Manager note', ev.note || 'Treated as a minor, non-safety concern.');
                setStageState(managerStage, 'done');
              } else {
                addVerdict(managerStage, 'flagged', 'Manager: rejection upheld');
                addBlock(managerStage, 'manager-text', 'Manager note', ev.note || 'The concern is safety-critical and stands.');
                setStageState(managerStage, 'flagged');
                els.stepperTitle.textContent = 'Review complete — plan requires revision';
                els.workingPulse.classList.add('is-idle');
              }
            }
            break;
          }
          case 'synth_start': {
            els.stepperTitle.textContent = 'Chief resident — writing the operative note';
            var s4 = createStage('Chief Resident', null);
            setStageState(s4, 'active');
            reviewStages.__synth = s4;
            startNeutral(s4, NEUTRAL_LINES.synth);
            break;
          }
          case 'synth_done': {
            stopNeutral();
            if (reviewStages.__synth) setStageState(reviewStages.__synth, 'done');
            els.stepperTitle.textContent = 'Review complete — plan approved';
            els.workingPulse.classList.add('is-idle');
            break;
          }
          default:
            break;
        }

        scrollStepper();
        var elapsed = Date.now() - startedAt;
        var remain = dwellMs() - elapsed;
        if (remain > 0) await sleep(remain);
      }
    } finally {
      stopNeutral();
    }
  }

  // Fallback path when the real-event stream is unavailable or fails for any
  // reason: replays the existing scripted animation while the buffered
  // /api/generate call runs, exactly as before this feature existed. If that
  // also fails, it falls back further to the bundled offline result.
  async function runBufferedFallback(caseText) {
    var livePromise = tryLiveGenerate(caseText);
    try { await runStepper(liveProcess()); } catch (e) {}
    var live = await livePromise;

    if (live && live.ok) {
      try {
        var d = live.data;
        d.plan = (Array.isArray(d.plan) && d.plan.length) ? d.plan : xmlToPlan(d.xml);
        renderResults(d);
        setNote('Generated live by the model.', 'live');
      } catch (e) {
        renderOfflineFallback('render');
      }
    } else {
      renderOfflineFallback(live && live.reason);
    }
  }

  function renderOfflineFallback(reason) {
    var data = (currentCase && Array.isArray(currentCase.plan) && currentCase.plan.length) ? currentCase : FALLBACK;
    try { renderResults(data); } catch (e) { try { renderResults(FALLBACK); } catch (e2) {} }
    var msg = 'Live generation was unavailable, so this shows a bundled example result.';
    if (reason === 'bad_passcode') msg = 'A valid access code is required for live generation. Showing a bundled example instead.';
    else if (reason === 'rate_limited') msg = 'Live rate limit reached. Showing a bundled example result.';
    else if (reason === 'live_unavailable') msg = 'Live generation is not configured on this deployment. Showing a bundled example result.';
    else if (reason === 'case_too_long') msg = 'That case text is too long for live generation. Showing a bundled example result.';
    setNote(msg, 'offline');
  }

  // Primary action: try the real-event live stream first, so the stepper
  // shows what the pipeline actually did. If the stream is unavailable or
  // fails for any reason, fall back to the buffered live path with the
  // scripted animation, and from there to a bundled result, so the demo never
  // fails and never shows a broken state.
  async function onGenerate() {
    var caseText = (els.caseText.value || '').trim();
    if (!caseText) { await onOfflineDemo(); return; }

    els.generateBtn.disabled = true;
    els.offlineBtn.disabled = true;
    els.results.hidden = true;
    setNote('Contacting the live model…', 'live');

    var usedRealStream = false;
    var streamSupported = typeof window.fetch === 'function' && typeof window.ReadableStream === 'function';

    if (streamSupported) {
      var queue = createEventQueue();
      var streamPromise = streamLiveGenerate(caseText, queue);
      var stepperPromise = runLiveStepper(queue);
      var stream = await streamPromise;
      try { await stepperPromise; } catch (e) {}

      if (stream && stream.ok) {
        try {
          var d = stream.data;
          d.plan = (Array.isArray(d.plan) && d.plan.length) ? d.plan : xmlToPlan(d.xml);
          renderResults(d);
          setNote('Generated live. The review trace above is the actual pipeline output.', 'live');
        } catch (e) {
          // Rendering the live result failed; do not rerun the whole pipeline,
          // fall straight back to a bundled result instead.
          renderOfflineFallback('render');
        }
        usedRealStream = true;
      }
    }

    if (!usedRealStream) {
      await runBufferedFallback(caseText);
    }

    try { els.results.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch (e) {}
    els.generateBtn.disabled = false;
    els.offlineBtn.disabled = false;
  }

  // Explicit offline demo: always renders a bundled synthetic case, never calls
  // the network. Guaranteed to work anywhere.
  async function onOfflineDemo() {
    els.generateBtn.disabled = true;
    els.offlineBtn.disabled = true;
    els.results.hidden = true;

    var data = (currentCase && Array.isArray(currentCase.plan) && currentCase.plan.length) ? currentCase : await fetchCase(null);
    currentCase = data;
    loadCase(data);
    setNote('Offline demo: a pre-prepared review sequence on a bundled synthetic case.', 'offline');

    try { await runStepper(data.process, { completionTitle: 'Walkthrough complete — bundled synthetic case' }); } catch (e) {}
    try { renderResults(data); } catch (e) { try { renderResults(FALLBACK); } catch (e2) {} }
    try { els.results.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch (e) {}

    els.generateBtn.disabled = false;
    els.offlineBtn.disabled = false;
  }

  // Guard against any uncaught rejection/error surfacing to the user.
  window.addEventListener('unhandledrejection', function (ev) { ev.preventDefault(); });
  window.addEventListener('error', function () { /* swallow */ });

  els.thesisTitle.textContent = THESIS_TITLE;
  els.randomBtn.addEventListener('click', onRandomCase);
  els.generateBtn.addEventListener('click', onGenerate);
  els.offlineBtn.addEventListener('click', onOfflineDemo);
  // Enable "Generate" as soon as there is case text (typed or loaded).
  els.caseText.addEventListener('input', function () {
    els.generateBtn.disabled = !els.caseText.value.trim();
  });
  libraryReady; // kick off the bundled-case load (best effort)
})();
