// demo/cases.js — Bundled synthetic case library (fully offline).
// All cases are SYNTHETIC and illustrative. No real patient data, no identifiers.
// Every case matches one shape (see buildCase). The front-end stepper narrative
// is driven by each case's `process` object; results render from plan + markdown.

// XML mirror of a structured plan, kept for shape-parity with the live /generate response.
function planToXml(items) {
  const stepXml = (s, indent) =>
    `${indent}<step>\n${indent}  <action_name>${s.name.replace(/\s+/g, '_')}</action_name>\n${indent}  <description>${s.description}</description>\n${indent}</step>`;
  const parts = items.map((item) => {
    if (item.type === 'branch') {
      const inner = item.steps.map((s) => stepXml(s, '    ')).join('\n');
      return `  <if_block condition='${item.condition}'>\n${inner}\n  </if_block>`;
    }
    return stepXml(item, '  ');
  });
  return `<SurgicalPlan>\n${parts.join('\n')}\n</SurgicalPlan>`;
}

// Assemble the full case object (derives xml, verdict, raw_review, etc.) from raw content.
function buildCase(c) {
  const comment = c.comment;
  return {
    id: c.id,
    title: c.title,
    caseText: c.caseText,
    plan: c.plan,
    xml: planToXml(c.plan),
    verdict: 'accept',
    source: 'review',
    comment,
    reason: comment,
    manager_note: '',
    raw_review: `<SurgicalBoard_Verify>accept</SurgicalBoard_Verify><Feedback_Comment>${comment}</Feedback_Comment>`,
    markdown: c.operativeNote,
    process: c.process
  };
}

/* =========================================================================
 * CASE 1 — Oral tongue SCC (cT3 N1 M0) — radial forearm free flap
 * (Existing case, kept verbatim.)
 * ====================================================================== */
const case1 = {
  id: 'oral-tongue-scc-01',
  title: 'Oral tongue carcinoma (cT3 N1 M0) — composite resection and free-flap reconstruction',
  caseText: `62-year-old man with biopsy-proven squamous cell carcinoma of the right lateral oral tongue.

- Primary tumor: 4.2 cm ulcerated lesion of the right lateral tongue extending onto the anterior floor of mouth. MRI depth of invasion 15 mm. No mandibular cortical invasion on CT.
- Neck: single ipsilateral level II lymph node, 2.1 cm, no radiologic extranodal extension. Clinical stage cT3 N1 M0.
- History: 30 pack-year smoker (quit 2 years ago); hypertension, well controlled. No prior head and neck surgery or radiotherapy.
- Function: normal mouth opening; fair dentition; speech and swallowing currently intact.
- Workup: chest CT negative for distant disease; ECOG performance status 1; cleared by anesthesia for prolonged free-flap surgery. Allen test of the non-dominant forearm is patent.`,
  plan: [
    { type: 'step', name: 'Airway Management',
      description: 'Perform elective tracheostomy under general anesthesia before resection, anticipating significant tongue and floor-of-mouth edema and flap bulk that would compromise the native airway.' },
    { type: 'step', name: 'Exposure and Margin Marking',
      description: 'Assess transoral access; if adequate, proceed transorally. Mark mucosal resection margins of 1.0-1.5 cm around the visible and palpable tumor, guided by the 15 mm depth of invasion.' },
    { type: 'step', name: 'Right Hemiglossectomy with Floor-of-Mouth Resection',
      description: 'Resect the right hemitongue in continuity with the involved anterior floor of mouth, maintaining at least 1 cm margins in all three dimensions, including the deep muscular margin.' },
    { type: 'step', name: 'Frozen Section Margin Assessment',
      description: 'Send circumferential mucosal and deep margins for intraoperative frozen section before beginning reconstruction.' },
    { type: 'branch', condition: 'Frozen section shows a positive or close (< 5 mm) margin',
      steps: [{ name: 'Re-resect Involved Margin', description: 'Re-excise the involved margin with an additional cuff of tissue, re-orient and re-send for frozen section; repeat until margins are clear before inset.' }] },
    { type: 'step', name: 'Right Selective Neck Dissection, Levels I-IV',
      description: 'Perform a right selective neck dissection of levels I-IV for the cN1 neck, preserving the spinal accessory nerve, internal jugular vein, and sternocleidomastoid muscle unless directly involved by disease.' },
    { type: 'branch', condition: 'Intraoperative finding of gross extranodal extension or disease at level IV',
      steps: [{ name: 'Convert to Comprehensive Neck Dissection', description: 'Extend the dissection to include level V and sacrifice involved non-vital structures as oncologically required; document extent for adjuvant-therapy planning.' }] },
    { type: 'step', name: 'Recipient Vessel Preparation',
      description: 'During neck dissection, identify and prepare the facial artery and common facial vein as primary recipient vessels; verify caliber and flow before flap harvest is completed.' },
    { type: 'step', name: 'Radial Forearm Free Flap Harvest',
      description: 'Harvest a fasciocutaneous radial forearm free flap from the non-dominant arm (patent Allen test documented), sized to the measured hemiglossectomy and floor-of-mouth defect with a long vascular pedicle.' },
    { type: 'branch', condition: 'Defect proves larger than anticipated or requires substantially more bulk',
      steps: [{ name: 'Convert to Anterolateral Thigh Free Flap', description: 'Abandon or repurpose the forearm flap and harvest an anterolateral thigh free flap to provide adequate volume for the composite defect.' }] },
    { type: 'step', name: 'Microvascular Anastomosis',
      description: 'Under the operating microscope, perform end-to-end arterial anastomosis to the facial artery and venous anastomosis to the common facial vein (venous coupler if caliber match is suitable); confirm flap perfusion.' },
    { type: 'branch', condition: 'Flap shows venous congestion or loss of arterial signal after anastomosis',
      steps: [{ name: 'Revise Anastomosis', description: 'Immediately re-explore the pedicle, revise the anastomosis, relieve any kinking or compression, and use an interposition vein graft if tension or caliber mismatch persists.' }] },
    { type: 'step', name: 'Flap Inset and Oral Closure',
      description: 'Inset the flap to resurface the hemitongue and floor of mouth, preserving mobility of the residual tongue and achieving a watertight oral seal; place a nasogastric feeding tube under vision.' },
    { type: 'step', name: 'Donor Site Closure',
      description: 'Close the forearm donor site with a split-thickness skin graft and apply a volar splint; confirm perfusion of the hand before leaving the operating room.' },
    { type: 'step', name: 'Postoperative Flap Monitoring',
      description: 'Admit to a monitored unit with hourly clinical and handheld Doppler flap checks for the first 72 hours, standard tracheostomy care, and a low threshold for return to theatre for any sign of pedicle compromise.' }
  ],
  comment:
    'The plan meets oncologic and reconstructive standards. Resection margins are appropriately case-driven for a depth of invasion of 15 mm, the nodal strategy (selective levels I-IV with clear escalation criteria) is consistent with standard of care for cT3 N1 oral tongue carcinoma, the airway is secured up front, and every high-risk step carries an explicit contingency: margin re-resection, nodal escalation, alternative flap choice, and anastomotic revision. Suggestions only: state tracheostomy decannulation criteria and document baseline donor-limb perfusion in the operative record.',
  operativeNote: `### Preoperative Plan

- Confirm biopsy-proven squamous cell carcinoma of the right lateral oral tongue, cT3 N1 M0, with MRI depth of invasion 15 mm and no mandibular cortical invasion on CT.
- Verify negative metastatic workup (chest CT), ECOG 1, anesthesia clearance for prolonged microsurgery, and a patent Allen test of the non-dominant forearm.
- Consent for tracheostomy, right hemiglossectomy with floor-of-mouth resection, right selective neck dissection (levels I-IV, with possible extension), radial forearm free flap reconstruction (possible conversion to anterolateral thigh), and split-thickness skin grafting of the donor site.

### Intraoperative Plan: Ablation

1. **Airway:** elective tracheostomy under general anesthesia prior to resection.
2. **Resection:** right hemiglossectomy in continuity with the involved anterior floor of mouth, maintaining at least 1 cm margins in all dimensions including the deep muscular margin; mucosal margins marked at 1.0-1.5 cm.
3. **Margin control:** circumferential and deep margins to intraoperative frozen section; any positive or close (< 5 mm) margin is re-resected and re-sent until clear before reconstruction begins.
4. **Neck:** right selective neck dissection of levels I-IV, preserving the spinal accessory nerve, internal jugular vein, and sternocleidomastoid unless directly involved. Gross extranodal extension or level IV disease converts the procedure to a comprehensive dissection including level V.
5. **Recipient vessels:** facial artery and common facial vein identified and prepared during the neck dissection.

### Intraoperative Plan: Reconstruction

1. **Flap harvest:** fasciocutaneous radial forearm free flap from the non-dominant arm, sized to the measured defect with a long vascular pedicle.
2. **Anastomosis:** end-to-end arterial anastomosis to the facial artery and venous anastomosis to the common facial vein under the operating microscope, using a venous coupler where caliber allows; flap perfusion confirmed before inset.
3. **Inset:** flap inset to resurface the hemitongue and floor of mouth, preserving residual tongue mobility and achieving a watertight oral seal; nasogastric feeding tube placed under vision.
4. **Donor site:** split-thickness skin graft closure of the forearm with a volar splint; hand perfusion confirmed before leaving the operating room.

### Key Contingency Plans

- **Positive or close frozen-section margin:** re-resect the involved margin with an additional cuff of tissue and repeat frozen section until clear.
- **Gross extranodal extension or level IV disease:** extend to a comprehensive neck dissection including level V; document extent for adjuvant-therapy planning.
- **Defect larger or bulkier than anticipated:** convert to an anterolateral thigh free flap for adequate volume.
- **Venous congestion or arterial signal loss:** immediate pedicle re-exploration and anastomotic revision, with interposition vein grafting for persistent tension or caliber mismatch.
- **Postoperative pedicle compromise:** hourly clinical and Doppler monitoring for 72 hours with a low threshold for urgent return to theatre.`,
  process: {
    draftSteps: [
      'Analyzing case & staging (cT3 N1 M0 oral tongue SCC)…',
      'Structuring resection and reconstruction…',
      'Adding intraoperative contingencies…'
    ],
    reviewChecks: ['Oncologic soundness', 'Reconstructive soundness', 'Contingency planning', 'Airway safety'],
    round1: {
      verdict: 'flagged',
      concern:
        'Single venous outflow: the draft prepares only the common facial vein as recipient, with no documented backup vein. A single venous anastomosis risks flap congestion and loss if that vein proves unsuitable intraoperatively.'
    },
    round2: {
      fix:
        'Added the external jugular vein as a backup recipient vein and specified an interposition vein-graft option, guaranteeing a second venous outflow is available before the anastomosis is committed.',
      verdict: 'accept'
    }
  }
};

/* =========================================================================
 * CASE 2 — Floor-of-mouth SCC with mandibular invasion (cT4a) — fibula flap
 * Round-1 concern: no lower-limb vascular assessment before fibula harvest.
 * ====================================================================== */
const case2 = {
  id: 'fom-mandible-t4a-fibula-02',
  title: 'Floor-of-mouth SCC with mandibular invasion (cT4a N0) — segmental mandibulectomy and fibula free flap',
  caseText: `58-year-old man with biopsy-proven squamous cell carcinoma of the anterior floor of mouth.

- Primary tumor: 3.6 cm lesion abutting the mandible; CT shows cortical erosion of the anterior mandibular body with early medullary invasion. Clinical stage cT4a N0 M0.
- Neck: no palpable or radiologic nodal disease (cN0); elective neck management indicated by primary depth and site.
- History: 40 pack-year smoker; type 2 diabetes on metformin; peripheral pulses clinically palpable but not formally assessed.
- Function: adequate mouth opening; partial dentition.
- Workup: chest CT clear; ECOG 1; anesthesia cleared for prolonged osseous free-flap surgery.`,
  plan: [
    { type: 'step', name: 'Airway Management',
      description: 'Elective tracheostomy under general anesthesia before resection, anticipating floor-of-mouth swelling and the need for prolonged airway protection.' },
    { type: 'step', name: 'Lower-Limb Vascular Assessment',
      description: 'Confirm CT angiography of both lower limbs demonstrating a patent three-vessel runoff on the intended donor leg before committing to fibula harvest; document peroneal, anterior and posterior tibial patency.' },
    { type: 'step', name: 'Exposure and Segmental Mandibulectomy',
      description: 'Resect the involved anterior mandibular segment with the floor-of-mouth primary in continuity, maintaining 1 cm mucosal and bony margins; template the bony defect for reconstruction.' },
    { type: 'step', name: 'Frozen Section Margin Assessment',
      description: 'Send mucosal, soft-tissue, and proximal/distal bone-marrow margins for intraoperative frozen section before reconstruction.' },
    { type: 'branch', condition: 'Frozen section shows a positive mucosal or bony marrow margin',
      steps: [{ name: 'Re-resect Involved Margin', description: 'Extend the resection (soft tissue or additional bone) and re-send until margins are clear before plating and inset.' }] },
    { type: 'step', name: 'Bilateral Selective Neck Dissection, Levels I-III',
      description: 'Perform elective bilateral selective neck dissection (levels I-III) for the cN0 anterior floor-of-mouth primary, and prepare recipient vessels in the same field.' },
    { type: 'step', name: 'Recipient Vessel Preparation',
      description: 'Identify and prepare the facial artery and common facial vein (and superior thyroid artery as backup) as recipient vessels; verify caliber and flow.' },
    { type: 'branch', condition: 'CT angiography shows inadequate or single-vessel runoff on the planned donor leg',
      steps: [{ name: 'Switch Donor Leg or Flap', description: 'Harvest the contralateral fibula if its runoff is adequate, or convert to a scapular osteocutaneous free flap to avoid donor-limb ischemia.' }] },
    { type: 'step', name: 'Fibula Osteocutaneous Free Flap Harvest',
      description: 'Harvest a fibula osteocutaneous free flap from the assessed leg, preserving proximal and distal fibula for ankle and knee stability, with a skin paddle on a reliable peroneal perforator.' },
    { type: 'step', name: 'Osteotomies and Neomandible Contouring',
      description: 'Perform closing-wedge osteotomies to recreate the anterior mandibular contour and fix with a reconstruction plate or miniplates; confirm occlusion and symmetry before dividing the pedicle.' },
    { type: 'branch', condition: 'Skin paddle perfusion is unreliable or soft-tissue bulk is insufficient',
      steps: [{ name: 'Add Soft-Tissue Coverage', description: 'Raise a second (chimeric or separate) soft-tissue flap for intraoral lining or external skin rather than relying on a marginal paddle.' }] },
    { type: 'step', name: 'Microvascular Anastomosis',
      description: 'Divide the pedicle after contouring and perform microvascular arterial and venous anastomoses to the prepared neck vessels under the operating microscope; confirm bone and skin-paddle perfusion.' },
    { type: 'branch', condition: 'Flap shows venous congestion or loss of arterial signal',
      steps: [{ name: 'Revise Anastomosis', description: 'Re-explore the pedicle, revise the anastomosis, relieve kinking, and interpose a vein graft for tension or caliber mismatch.' }] },
    { type: 'step', name: 'Inset and Closure',
      description: 'Inset the bony and soft-tissue components, restore intraoral lining, and close in layers; place a nasogastric feeding tube under vision.' },
    { type: 'step', name: 'Donor Site Closure and Limb Care',
      description: 'Close the leg donor site (primary or with split-thickness skin graft), apply a below-knee splint, and confirm distal foot perfusion and compartment softness before leaving theatre.' },
    { type: 'step', name: 'Postoperative Flap and Limb Monitoring',
      description: 'Admit to a monitored unit with hourly flap checks for 72 hours, plus scheduled donor-limb neurovascular and compartment observation.' }
  ],
  comment:
    'The revised plan is oncologically and reconstructively sound: segmental mandibulectomy with case-driven bony margins, elective bilateral neck treatment for a cN0 anterior floor-of-mouth primary, and a fibula osteocutaneous flap appropriate for the composite mandibular defect. Critically, donor-limb safety is now addressed with preoperative CT angiography confirming three-vessel runoff and an explicit switch to the contralateral leg or a scapular flap if runoff is inadequate. Contingencies for margins, skin-paddle reliability, and anastomotic failure are in place.',
  operativeNote: `### Preoperative Plan

- Confirm cT4a N0 M0 floor-of-mouth SCC with anterior mandibular cortical and early medullary invasion; negative metastatic workup; ECOG 1.
- Obtain CT angiography of both lower limbs and confirm a patent three-vessel runoff on the intended donor leg before committing to fibula harvest.
- Consent for tracheostomy, segmental mandibulectomy with floor-of-mouth resection, bilateral selective neck dissection (I-III), fibula osteocutaneous free flap (with contralateral-leg or scapular alternatives), and donor-site skin grafting.

### Intraoperative Plan: Ablation

1. **Airway:** elective tracheostomy prior to resection.
2. **Resection:** segmental anterior mandibulectomy in continuity with the floor-of-mouth primary at 1 cm mucosal and bony margins; defect templated for reconstruction.
3. **Margin control:** mucosal, soft-tissue, and bone-marrow margins to frozen section; positive margins re-resected until clear.
4. **Neck:** elective bilateral selective neck dissection (levels I-III); recipient vessels prepared in the same field.

### Intraoperative Plan: Reconstruction

1. **Donor confirmation:** proceed with the CTA-cleared leg; switch to the contralateral fibula or a scapular flap if runoff is inadequate.
2. **Flap harvest:** fibula osteocutaneous free flap on a peroneal perforator, preserving proximal and distal fibular stumps.
3. **Osseous reconstruction:** closing-wedge osteotomies to restore anterior mandibular contour, fixed with a reconstruction plate; occlusion confirmed before pedicle division.
4. **Anastomosis:** microvascular arterial and venous anastomoses to facial or superior thyroid vessels; bone and skin-paddle perfusion confirmed before inset.

### Key Contingency Plans

- **Inadequate donor-leg runoff on CTA:** harvest the contralateral fibula or convert to a scapular osteocutaneous free flap.
- **Positive frozen-section margin:** extend soft-tissue or bony resection and re-send until clear.
- **Unreliable skin paddle:** add a chimeric or separate soft-tissue flap for lining or external cover.
- **Venous congestion or arterial signal loss:** re-explore and revise the anastomosis, with vein grafting for tension or caliber mismatch.
- **Donor limb:** below-knee splint with scheduled distal-perfusion and compartment observation.`,
  process: {
    draftSteps: [
      'Analyzing case & staging (cT4a N0 floor-of-mouth SCC, mandibular invasion)…',
      'Planning segmental mandibulectomy and osseous reconstruction…',
      'Selecting fibula osteocutaneous free flap…'
    ],
    reviewChecks: ['Oncologic soundness', 'Osseous reconstruction', 'Donor-site safety', 'Contingency planning'],
    round1: {
      verdict: 'flagged',
      concern:
        'No lower-limb vascular assessment before fibula harvest. Proceeding without CT angiography and confirmation of three-vessel runoff risks critical donor-limb ischemia if the peroneal artery is dominant or the runoff is inadequate.'
    },
    round2: {
      fix:
        'Added preoperative CT angiography of both legs with documented three-vessel runoff, and an explicit intraoperative switch to the contralateral fibula or a scapular osteocutaneous flap if runoff is inadequate.',
      verdict: 'accept'
    }
  }
};

/* =========================================================================
 * CASE 3 — Maxillary sinus SCC (cT4a) with orbital floor involvement — scapular flap
 * Round-1 concern: no orbital floor support planned → globe malposition/diplopia.
 * ====================================================================== */
const case3 = {
  id: 'maxilla-orbital-t4a-scapula-03',
  title: 'Maxillary sinus SCC (cT4a N0) — maxillectomy with orbital floor resection and subscapular-system free flap',
  caseText: `64-year-old woman with squamous cell carcinoma of the right maxillary sinus.

- Primary tumor: fills the maxillary antrum with erosion of the medial and inferior orbital walls; the orbital floor is involved but orbital contents (periorbita, globe) are preserved. Clinical stage cT4a N0 M0.
- Neck: clinically and radiologically node-negative.
- History: former smoker; no significant comorbidity; normal vision with intact extraocular movements preoperatively.
- Workup: MRI defines orbital-floor involvement without frank intraorbital fat invasion; ophthalmology confirms a salvageable globe; chest CT clear; ECOG 0.`,
  plan: [
    { type: 'step', name: 'Multidisciplinary and Ophthalmology Confirmation',
      description: 'Confirm with ophthalmology that the globe is salvageable (periorbita as the deep margin) and plan orbital-floor reconstruction as an integral part of the resection.' },
    { type: 'step', name: 'Maxillectomy with Orbital Floor Resection',
      description: 'Perform a subtotal/total maxillectomy resecting the involved orbital floor and medial wall en bloc with the antral tumor, preserving the periorbita and globe where oncologically safe.' },
    { type: 'branch', condition: 'Periorbita or intraconal fat is grossly involved by tumor',
      steps: [{ name: 'Orbital Exenteration', description: 'Proceed to orbital exenteration for oncologic clearance and plan obliteration of the orbital cavity rather than globe-sparing floor reconstruction.' }] },
    { type: 'step', name: 'Frozen Section Margin Assessment',
      description: 'Send mucosal, bony, and periorbital margins for intraoperative frozen section before reconstruction.' },
    { type: 'branch', condition: 'Frozen section shows a positive margin',
      steps: [{ name: 'Re-resect Involved Margin', description: 'Extend the resection at the involved margin and re-send until clear before reconstruction.' }] },
    { type: 'step', name: 'Recipient Vessel Preparation and Neck',
      description: 'Prepare facial or superior thyroid recipient vessels via a limited upper-neck approach; address the cN0 neck per institutional policy for the subsite.' },
    { type: 'step', name: 'Orbital Floor Reconstruction and Globe Support',
      description: 'Reconstruct the orbital floor to restore globe position and volume, using vascularized scapular bone (or a titanium mesh cradle lined by soft tissue) contoured to the native floor level; confirm globe height against the contralateral side.' },
    { type: 'step', name: 'Subscapular-System Free Flap Harvest',
      description: 'Harvest a subscapular-system free flap (scapular/parascapular skin with lateral scapular border bone, chimeric with latissimus or serratus if more bulk is needed) to reconstruct the orbital floor, midface buttress, and palatal defect.' },
    { type: 'step', name: 'Midface Buttress and Palatal Reconstruction',
      description: 'Restore the zygomaticomaxillary buttress and separate the orbit, sinus, and oral cavity, providing a competent palatal seal for speech and swallowing.' },
    { type: 'step', name: 'Microvascular Anastomosis',
      description: 'Perform microvascular arterial and venous anastomoses to the prepared neck vessels under the operating microscope; confirm perfusion of all chimeric components.' },
    { type: 'branch', condition: 'Flap shows venous congestion or arterial compromise',
      steps: [{ name: 'Revise Anastomosis', description: 'Re-explore and revise the anastomosis, with vein grafting if tension or caliber mismatch persists.' }] },
    { type: 'step', name: 'Inset and Closure',
      description: 'Inset the composite flap, secure orbital-floor support, and close the midface; verify symmetric globe position and free extraocular excursion at the table.' },
    { type: 'step', name: 'Donor Site Closure',
      description: 'Close the scapular/subscapular donor site in layers and manage the shoulder with early guided mobilization.' },
    { type: 'step', name: 'Postoperative Monitoring',
      description: 'Hourly flap monitoring for 72 hours plus scheduled ophthalmologic assessment of globe position, diplopia, and visual acuity.' }
  ],
  comment:
    'The revised plan achieves oncologic clearance of a cT4a maxillary tumor with a globe-sparing approach where the periorbita is uninvolved, and correctly escalates to exenteration when it is not. The key correction is present: the orbital floor is now formally reconstructed with vascularized scapular bone (or a soft-tissue-lined titanium cradle) to support the globe and prevent malposition and diplopia, with intraoperative confirmation of globe height. Midface buttress and palatal separation restore function, and contingencies for margins and anastomotic failure are defined.',
  operativeNote: `### Preoperative Plan

- Confirm cT4a N0 M0 maxillary sinus SCC with orbital-floor involvement but a salvageable globe (ophthalmology and MRI review); negative metastatic workup; ECOG 0.
- Plan orbital-floor reconstruction as an integral step; consent for globe-sparing maxillectomy with a defined conversion to orbital exenteration if the periorbita is involved.
- Consent for subscapular-system free flap reconstruction of the orbital floor, midface buttress, and palate, with donor-site and neck-vessel preparation.

### Intraoperative Plan: Ablation

1. **Resection:** subtotal/total maxillectomy resecting the involved orbital floor and medial wall en bloc with the antral tumor, preserving periorbita and globe where safe.
2. **Escalation:** orbital exenteration if the periorbita or intraconal fat is grossly involved.
3. **Margin control:** mucosal, bony, and periorbital margins to frozen section; positive margins re-resected until clear.
4. **Access:** facial or superior thyroid recipient vessels prepared via a limited upper-neck approach.

### Intraoperative Plan: Reconstruction

1. **Orbital floor:** reconstructed with vascularized scapular bone (or a soft-tissue-lined titanium cradle) at the native floor level; globe height matched to the contralateral side.
2. **Flap harvest:** subscapular-system free flap (scapular/parascapular skin with lateral border bone, chimeric with latissimus/serratus for bulk).
3. **Midface:** zygomaticomaxillary buttress restored and orbit/sinus/oral cavity separated with a competent palatal seal.
4. **Anastomosis:** microvascular anastomoses to neck vessels; perfusion of all components confirmed before inset.

### Key Contingency Plans

- **Periorbital or intraconal involvement:** convert to orbital exenteration with cavity obliteration.
- **Positive frozen-section margin:** extend resection and re-send until clear.
- **Globe malposition risk:** intraoperative confirmation of globe height and extraocular excursion before closure.
- **Venous congestion or arterial compromise:** re-explore and revise the anastomosis, with vein grafting as needed.
- **Postoperative:** scheduled ophthalmologic review of globe position, diplopia, and acuity alongside flap monitoring.`,
  process: {
    draftSteps: [
      'Analyzing case & staging (cT4a N0 maxillary sinus SCC, orbital-floor involvement)…',
      'Planning maxillectomy and midface reconstruction…',
      'Selecting subscapular-system free flap…'
    ],
    reviewChecks: ['Oncologic soundness', 'Midface reconstruction', 'Orbital / globe support', 'Contingency planning'],
    round1: {
      verdict: 'flagged',
      concern:
        'No orbital floor support is planned after resecting the involved floor. Leaving the orbital floor unreconstructed will allow the globe to drop, producing globe malposition, enophthalmos, and diplopia.'
    },
    round2: {
      fix:
        'Added formal orbital-floor reconstruction with vascularized scapular bone (or a soft-tissue-lined titanium cradle) restoring floor level and globe support, with intraoperative confirmation of globe height and extraocular movement.',
      verdict: 'accept'
    }
  }
};

/* =========================================================================
 * CASE 4 — Hypopharyngeal SCC, circumferential — tubed ALT free flap
 * Round-1 concern: no fistula/leak management or salivary diversion.
 * ====================================================================== */
const case4 = {
  id: 'hypopharynx-circumferential-alt-04',
  title: 'Circumferential hypopharyngeal SCC (cT4a N2) — total laryngopharyngectomy and tubed ALT free flap',
  caseText: `61-year-old man with circumferential squamous cell carcinoma of the hypopharynx (pyriform sinus with post-cricoid extension).

- Primary tumor: circumferential involvement of the pyriform sinus and post-cricoid region with laryngeal fixation. Clinical stage cT4a N2b M0.
- Neck: bilateral level II-III nodal disease.
- History: heavy smoker and alcohol use; malnourished (recent weight loss), albumin low-normal.
- Function: aspiration on swallow study; hoarse, with a marginal airway.
- Workup: chest CT clear of metastasis; ECOG 1; gastroenterology and dietetics engaged for perioperative nutrition.`,
  plan: [
    { type: 'step', name: 'Airway and Nutritional Optimization',
      description: 'Secure the airway for total laryngopharyngectomy (end tracheostoma) and optimize perioperative nutrition; a circumferential pharyngoesophageal reconstruction is planned.' },
    { type: 'step', name: 'Total Laryngopharyngectomy',
      description: 'Resect the larynx and circumferential hypopharynx with adequate mucosal margins proximally (oropharynx) and distally (cervical esophagus); create a permanent end tracheostoma.' },
    { type: 'step', name: 'Frozen Section Margin Assessment',
      description: 'Send proximal pharyngeal and distal esophageal mucosal margins for frozen section before reconstruction.' },
    { type: 'branch', condition: 'Frozen section shows a positive proximal or distal mucosal margin',
      steps: [{ name: 'Re-resect Involved Margin', description: 'Extend the mucosal resection and re-send until margins are clear before tubing the flap.' }] },
    { type: 'step', name: 'Bilateral Neck Dissection and Recipient Vessels',
      description: 'Perform bilateral selective neck dissection (levels II-IV) for N2b disease and prepare recipient vessels (transverse cervical or superior thyroid artery and a suitable vein).' },
    { type: 'step', name: 'Tubed ALT Free Flap Harvest',
      description: 'Harvest an anterolateral thigh free flap sized to bridge the pharyngoesophageal gap; tube the skin paddle around a stent to recreate a neopharyngeal conduit.' },
    { type: 'step', name: 'Salivary Bypass Tube Placement',
      description: 'Place a salivary bypass tube through the neopharynx across both suture lines to divert saliva away from the anastomoses during healing.' },
    { type: 'step', name: 'Pharyngoesophageal Anastomoses',
      description: 'Perform watertight proximal (oropharyngeal) and distal (cervical esophageal) anastomoses of the tubed flap over the bypass tube; test for leaks.' },
    { type: 'branch', condition: 'Anastomotic tension or a mucosal gap is present at inset',
      steps: [{ name: 'Adjust Conduit Length or Reinforce', description: 'Re-tailor the flap length, reinforce the suture line, and buttress with adjacent vascularized tissue to reduce leak risk.' }] },
    { type: 'step', name: 'Microvascular Anastomosis',
      description: 'Perform microvascular arterial and venous anastomoses to the prepared neck vessels under the operating microscope; confirm conduit perfusion.' },
    { type: 'branch', condition: 'Flap shows venous congestion or arterial compromise',
      steps: [{ name: 'Revise Anastomosis', description: 'Re-explore and revise the anastomosis, with vein grafting for tension or caliber mismatch.' }] },
    { type: 'step', name: 'Inset, Drainage, and Closure',
      description: 'Inset the conduit, buttress suture lines with vascularized soft tissue, place closed-suction drains away from the anastomoses, and close the neck in layers over the tracheostoma.' },
    { type: 'branch', condition: 'A pharyngocutaneous fistula or salivary leak develops postoperatively',
      steps: [{ name: 'Manage Fistula Conservatively then Surgically', description: 'Maintain the salivary bypass tube and drainage, withhold oral intake, treat infection, and return to theatre for washout or a pectoralis major flap patch if the leak fails to settle.' }] },
    { type: 'step', name: 'Staged Swallow Assessment and Monitoring',
      description: 'Hourly flap monitoring for 72 hours; obtain a contrast swallow study before starting oral intake to confirm the absence of a leak, then progress swallow rehabilitation.' }
  ],
  comment:
    'The revised plan is appropriate for a circumferential hypopharyngeal cancer requiring total laryngopharyngectomy and a tubed ALT neopharynx, with bilateral nodal treatment for N2b disease. The essential corrections are in place: a salivary bypass tube diverts saliva from the suture lines, suture-line buttressing and drainage reduce fistula risk, and an explicit pharyngocutaneous-fistula pathway (conservative measures, then pectoralis major patch) is defined. A contrast swallow before oral intake gives an objective leak check. Nutritional optimization is addressed given the patient factors.',
  operativeNote: `### Preoperative Plan

- Confirm cT4a N2b M0 circumferential hypopharyngeal SCC with laryngeal fixation; negative metastatic workup; ECOG 1.
- Optimize perioperative nutrition (dietetics/gastroenterology); consent for total laryngopharyngectomy with permanent tracheostoma and tubed ALT neopharyngeal reconstruction.
- Plan salivary diversion and a defined fistula-management pathway as integral parts of the reconstruction.

### Intraoperative Plan: Ablation

1. **Airway:** total laryngopharyngectomy with creation of a permanent end tracheostoma.
2. **Resection:** larynx and circumferential hypopharynx removed with adequate proximal (oropharyngeal) and distal (cervical esophageal) mucosal margins.
3. **Margin control:** proximal and distal mucosal margins to frozen section; positive margins re-resected until clear.
4. **Neck:** bilateral selective neck dissection (levels II-IV); recipient vessels prepared.

### Intraoperative Plan: Reconstruction

1. **Conduit:** anterolateral thigh flap tubed around a stent to recreate the neopharynx.
2. **Salivary diversion:** salivary bypass tube placed across both suture lines before anastomosis.
3. **Anastomoses:** watertight proximal and distal pharyngoesophageal anastomoses over the bypass tube, buttressed with vascularized tissue and leak-tested.
4. **Microvascular:** arterial and venous anastomoses to transverse cervical or superior thyroid vessels; conduit perfusion confirmed.

### Key Contingency Plans

- **Positive frozen-section margin:** extend mucosal resection and re-send until clear.
- **Anastomotic tension or mucosal gap:** re-tailor conduit length and reinforce/buttress the suture line.
- **Pharyngocutaneous fistula or salivary leak:** maintain the bypass tube and drainage, withhold oral intake, treat infection, and return to theatre for washout or a pectoralis major flap patch if it fails to settle.
- **Venous congestion or arterial compromise:** re-explore and revise the anastomosis, with vein grafting as needed.
- **Swallow:** contrast swallow study before oral intake to confirm no leak, then staged swallow rehabilitation.`,
  process: {
    draftSteps: [
      'Analyzing case & staging (cT4a N2b circumferential hypopharyngeal SCC)…',
      'Planning total laryngopharyngectomy and neopharyngeal reconstruction…',
      'Configuring tubed ALT conduit…'
    ],
    reviewChecks: ['Oncologic soundness', 'Neopharyngeal reconstruction', 'Fistula / leak safety', 'Contingency planning'],
    round1: {
      verdict: 'flagged',
      concern:
        'No plan to manage a pharyngocutaneous fistula or anastomotic leak, and no salivary diversion. Tubed pharyngeal reconstructions have a high leak rate; without a salivary bypass tube and a defined fistula pathway, a leak could become a life-threatening neck sepsis.'
    },
    round2: {
      fix:
        'Added a salivary bypass tube across both suture lines, suture-line buttressing with drainage, an explicit pharyngocutaneous-fistula pathway (conservative measures then a pectoralis major patch), and a contrast swallow study before oral intake.',
      verdict: 'accept'
    }
  }
};

/* =========================================================================
 * CASE 5 — Advanced tongue SCC, total glossectomy — bulky ALT free flap
 * Round-1 concern: aspiration risk / airway not secured / no laryngeal suspension.
 * ====================================================================== */
const case5 = {
  id: 'total-glossectomy-alt-05',
  title: 'Advanced tongue SCC (cT4a N2) — total glossectomy and bulky ALT free flap',
  caseText: `55-year-old man with extensive squamous cell carcinoma involving both sides of the oral tongue and tongue base.

- Primary tumor: bulky tumor crossing the midline and involving the tongue base, requiring total glossectomy; the larynx is not directly invaded and is potentially preservable. Clinical stage cT4a N2c M0.
- Neck: bilateral nodal disease.
- History: smoker; otherwise reasonable performance status.
- Function: baseline swallowing impaired; high anticipated aspiration risk after total glossectomy.
- Workup: chest CT clear; ECOG 1; cleared for prolonged free-flap surgery; speech-and-language therapy engaged.`,
  plan: [
    { type: 'step', name: 'Airway Management with Tracheostomy',
      description: 'Perform tracheostomy at the outset to secure the airway; total glossectomy with a bulky flap and high aspiration risk mandates controlled airway management.' },
    { type: 'step', name: 'Total Glossectomy',
      description: 'Resect the entire oral tongue and tongue base with adequate margins; assess and preserve the larynx if it is oncologically uninvolved.' },
    { type: 'branch', condition: 'Tumor directly invades the larynx or pre-epiglottic space',
      steps: [{ name: 'Include Laryngectomy', description: 'Extend to total laryngectomy for oncologic clearance and plan the airway and reconstruction accordingly.' }] },
    { type: 'step', name: 'Frozen Section Margin Assessment',
      description: 'Send circumferential and deep margins for intraoperative frozen section before reconstruction.' },
    { type: 'branch', condition: 'Frozen section shows a positive or close margin',
      steps: [{ name: 'Re-resect Involved Margin', description: 'Extend the resection and re-send until margins are clear before inset.' }] },
    { type: 'step', name: 'Bilateral Neck Dissection and Recipient Vessels',
      description: 'Perform bilateral selective neck dissection (levels I-IV) for N2c disease and prepare recipient vessels.' },
    { type: 'step', name: 'Laryngeal Suspension and Aspiration Prevention',
      description: 'Suspend the larynx to the mandible/hyoid to elevate and protect the laryngeal inlet, reducing postoperative aspiration after loss of tongue function.' },
    { type: 'step', name: 'Bulky ALT Free Flap Harvest',
      description: 'Harvest a bulky anterolateral thigh free flap (with vastus lateralis bulk as needed) to create a convex neotongue mound that reaches the palate for speech and swallow.' },
    { type: 'step', name: 'Neotongue Shaping and Inset',
      description: 'Shape and inset the flap as a mounded neotongue in contact with the palate, restoring oral competence and directing the bolus away from the airway.' },
    { type: 'step', name: 'Microvascular Anastomosis',
      description: 'Perform microvascular arterial and venous anastomoses to the prepared neck vessels under the operating microscope; confirm perfusion.' },
    { type: 'branch', condition: 'Flap shows venous congestion or arterial compromise',
      steps: [{ name: 'Revise Anastomosis', description: 'Re-explore and revise the anastomosis, with vein grafting for tension or caliber mismatch.' }] },
    { type: 'branch', condition: 'Intractable aspiration despite suspension and flap bulk',
      steps: [{ name: 'Plan Airway Separation', description: 'Consider a staged narrow-field laryngectomy or laryngeal closure to definitively separate the airway from the alimentary tract.' }] },
    { type: 'step', name: 'Feeding Access and Donor Closure',
      description: 'Place enteral feeding access (nasogastric or gastrostomy) under vision and close the thigh donor site primarily or with a skin graft.' },
    { type: 'step', name: 'Postoperative Monitoring and Swallow Rehabilitation',
      description: 'Hourly flap monitoring for 72 hours; delayed decannulation guided by airway protection, with a structured speech-and-language swallow rehabilitation pathway.' }
  ],
  comment:
    'The revised plan is sound for a total glossectomy defect with a bulky ALT neotongue and bilateral nodal treatment. The critical safety corrections are present: the airway is secured with an up-front tracheostomy, and laryngeal suspension is added to protect the laryngeal inlet and reduce aspiration after loss of tongue function. A structured swallow-rehabilitation pathway with delayed, protection-guided decannulation and enteral feeding access is defined, plus a clear escalation to airway separation for intractable aspiration. Margin and anastomotic contingencies are in place.',
  operativeNote: `### Preoperative Plan

- Confirm cT4a N2c M0 tongue SCC requiring total glossectomy with a potentially preservable larynx; negative metastatic workup; ECOG 1.
- Engage speech-and-language therapy; consent for tracheostomy, total glossectomy (with possible laryngectomy), bilateral neck dissection, laryngeal suspension, and bulky ALT reconstruction.
- Plan airway protection and aspiration management as integral parts of the operation.

### Intraoperative Plan: Ablation

1. **Airway:** tracheostomy at the outset to secure the airway.
2. **Resection:** total glossectomy of the oral tongue and tongue base with adequate margins; larynx preserved if uninvolved, otherwise laryngectomy.
3. **Margin control:** circumferential and deep margins to frozen section; positive margins re-resected until clear.
4. **Neck:** bilateral selective neck dissection (levels I-IV); recipient vessels prepared.

### Intraoperative Plan: Reconstruction

1. **Aspiration prevention:** laryngeal suspension to the mandible/hyoid to elevate and protect the laryngeal inlet.
2. **Flap harvest:** bulky anterolateral thigh free flap (with vastus lateralis bulk) to create a palate-reaching neotongue mound.
3. **Inset:** neotongue shaped for palatal contact to restore oral competence and direct the bolus away from the airway.
4. **Anastomosis:** microvascular anastomoses to neck vessels; perfusion confirmed before inset completion.

### Key Contingency Plans

- **Laryngeal invasion:** extend to total laryngectomy with appropriate airway and reconstruction planning.
- **Positive frozen-section margin:** extend resection and re-send until clear.
- **Venous congestion or arterial compromise:** re-explore and revise the anastomosis, with vein grafting as needed.
- **Intractable aspiration despite suspension:** staged narrow-field laryngectomy or laryngeal closure to separate the airway.
- **Rehabilitation:** enteral feeding access, delayed protection-guided decannulation, and structured swallow rehabilitation.`,
  process: {
    draftSteps: [
      'Analyzing case & staging (cT4a N2c tongue SCC, total glossectomy)…',
      'Planning total glossectomy and neotongue reconstruction…',
      'Selecting bulky ALT free flap…'
    ],
    reviewChecks: ['Oncologic soundness', 'Reconstructive soundness', 'Airway / aspiration safety', 'Contingency planning'],
    round1: {
      verdict: 'flagged',
      concern:
        'Aspiration risk is not addressed: the draft does not secure the airway with a tracheostomy and includes no laryngeal suspension. After total glossectomy, loss of tongue function causes severe aspiration, and an unprotected airway is dangerous.'
    },
    round2: {
      fix:
        'Added an up-front tracheostomy, laryngeal suspension to protect the laryngeal inlet, a structured swallow-rehabilitation pathway with protection-guided decannulation, and an escalation to airway separation for intractable aspiration.',
      verdict: 'accept'
    }
  }
};

/* =========================================================================
 * CASE 6 — Recurrent irradiated scalp/calvarial malignancy — latissimus dorsi flap
 * Round-1 concern: vessel-depleted/irradiated neck, no recipient vessels / vein-graft plan.
 * ====================================================================== */
const case6 = {
  id: 'scalp-calvarial-irradiated-latissimus-06',
  title: 'Recurrent scalp/calvarial malignancy in an irradiated field — wide excision and latissimus dorsi free flap',
  caseText: `70-year-old man with recurrent cutaneous malignancy of the vertex scalp in a previously irradiated field.

- Primary problem: recurrent tumor with a large full-thickness scalp defect anticipated; outer-table calvarial involvement on CT, without intracranial extension.
- Field: prior wide-field scalp radiotherapy; the neck is vessel-depleted from previous surgery and radiation.
- History: multiple prior resections; anticoagulated for atrial fibrillation (bridging planned).
- Workup: CT/MRI shows outer-table involvement without dural invasion; chest CT clear; ECOG 1; anesthesia cleared for prolonged free-flap surgery.`,
  plan: [
    { type: 'step', name: 'Positioning and Preoperative Vessel Mapping',
      description: 'Plan positioning for scalp resection and latissimus harvest, and map candidate recipient vessels (superficial temporal, and transverse cervical in the neck) with imaging or Doppler given the irradiated, vessel-depleted field.' },
    { type: 'step', name: 'Wide Excision of Scalp and Outer-Table Calvarium',
      description: 'Excise the recurrent tumor with wide margins including the involved outer-table calvarium (burr down or resect outer cortex), preserving the inner table if uninvolved.' },
    { type: 'branch', condition: 'Full-thickness calvarial or dural involvement is found',
      steps: [{ name: 'Craniectomy and Dural Repair', description: 'Perform craniectomy with neurosurgery and repair or reconstruct the dura (pericranial or graft) before soft-tissue reconstruction.' }] },
    { type: 'step', name: 'Frozen Section Margin Assessment',
      description: 'Send peripheral and deep margins for intraoperative frozen section before reconstruction.' },
    { type: 'branch', condition: 'Frozen section shows a positive margin',
      steps: [{ name: 'Re-resect Involved Margin', description: 'Extend the resection and re-send until margins are clear before flap inset.' }] },
    { type: 'step', name: 'Recipient Vessel Exploration',
      description: 'Explore the superficial temporal vessels first; if the caliber or quality is inadequate in the irradiated field, expose the transverse cervical vessels in the neck as a reliable recipient.' },
    { type: 'branch', condition: 'No adequate local recipient vessels are available in the irradiated field',
      steps: [{ name: 'Vein Grafts to Neck Vessels', description: 'Bring vein grafts (arteriovenous loop or interposition) from the transverse cervical or other neck vessels up to the flap pedicle to guarantee an out-of-field recipient.' }] },
    { type: 'step', name: 'Latissimus Dorsi Free Flap Harvest',
      description: 'Harvest a latissimus dorsi muscle (or myocutaneous) free flap sized to resurface the scalp/calvarial defect, with a long thoracodorsal pedicle to reach the chosen recipient vessels.' },
    { type: 'step', name: 'Microvascular Anastomosis',
      description: 'Perform microvascular arterial and venous anastomoses to the superficial temporal or vein-grafted neck vessels under the operating microscope; confirm muscle perfusion.' },
    { type: 'branch', condition: 'Flap shows venous congestion or arterial compromise',
      steps: [{ name: 'Revise Anastomosis', description: 'Re-explore and revise the anastomosis, favoring a vein graft to an out-of-field vessel if the local vessel is unreliable.' }] },
    { type: 'step', name: 'Inset and Skin Graft',
      description: 'Inset the muscle over the calvarium/dura and cover it with a split-thickness skin graft; ensure tension-free coverage of any exposed bone or dural repair.' },
    { type: 'step', name: 'Donor Site Closure',
      description: 'Close the back donor site over closed-suction drains and manage the shoulder with early guided mobilization.' },
    { type: 'step', name: 'Postoperative Monitoring',
      description: 'Hourly flap monitoring for 72 hours (buried-muscle protocols as needed), neurologic observation if craniectomy was performed, and careful anticoagulation management.' }
  ],
  comment:
    'The revised plan appropriately treats a recurrent scalp/calvarial malignancy in a hostile irradiated field with wide excision (including outer-table calvarium and a defined craniectomy/dural pathway) and a latissimus dorsi flap with skin graft. The decisive correction addresses the vessel-depleted neck: recipient options are now explicitly mapped, with superficial temporal vessels explored first and a firm plan for vein grafts to transverse cervical (out-of-field) vessels when no adequate local recipient exists. Margin, dural, and anastomotic contingencies are defined, and anticoagulation is accounted for.',
  operativeNote: `### Preoperative Plan

- Confirm recurrent scalp/calvarial malignancy with outer-table involvement and no intracranial extension in a previously irradiated, vessel-depleted field; negative metastatic workup; ECOG 1.
- Map recipient vessels (superficial temporal and transverse cervical) preoperatively and plan for vein grafts; arrange neurosurgical support and anticoagulation bridging.
- Consent for wide excision (possible craniectomy/dural repair), latissimus dorsi free flap with skin graft, and vein grafting if required.

### Intraoperative Plan: Ablation

1. **Resection:** wide excision of the recurrent tumor including involved outer-table calvarium, preserving the inner table if uninvolved.
2. **Escalation:** craniectomy with neurosurgery and dural repair if full-thickness bone or dura is involved.
3. **Margin control:** peripheral and deep margins to frozen section; positive margins re-resected until clear.

### Intraoperative Plan: Reconstruction

1. **Recipient vessels:** superficial temporal vessels explored first; transverse cervical vessels exposed if local vessels are inadequate.
2. **Vessel salvage:** vein grafts (interposition or arteriovenous loop) to out-of-field neck vessels when no adequate local recipient is available.
3. **Flap harvest:** latissimus dorsi muscle/myocutaneous free flap with a long thoracodorsal pedicle.
4. **Coverage:** muscle inset over calvarium/dura with split-thickness skin graft; tension-free cover of exposed bone or dural repair.

### Key Contingency Plans

- **No adequate local recipient vessels:** vein grafts to transverse cervical or other neck vessels to reach an out-of-field recipient.
- **Full-thickness calvarial/dural involvement:** craniectomy and dural repair before reconstruction.
- **Positive frozen-section margin:** extend resection and re-send until clear.
- **Venous congestion or arterial compromise:** re-explore and revise, favoring a vein graft to a reliable out-of-field vessel.
- **Postoperative:** flap monitoring, neurologic observation after craniectomy, and careful anticoagulation management.`,
  process: {
    draftSteps: [
      'Analyzing case (recurrent scalp/calvarial malignancy, irradiated field)…',
      'Planning wide excision and calvarial management…',
      'Selecting latissimus dorsi free flap…'
    ],
    reviewChecks: ['Oncologic soundness', 'Recipient-vessel strategy', 'Reconstructive soundness', 'Contingency planning'],
    round1: {
      verdict: 'flagged',
      concern:
        'The neck is vessel-depleted and irradiated, yet no viable recipient vessels are identified and there is no vein-graft plan. Without a reliable, often out-of-field recipient, the free flap has no dependable blood supply and is at high risk of failure.'
    },
    round2: {
      fix:
        'Added explicit recipient-vessel mapping (superficial temporal first, transverse cervical as backup) and a firm plan for vein grafts / an arteriovenous loop to out-of-field neck vessels when no adequate local recipient is available.',
      verdict: 'accept'
    }
  }
};

/* =========================================================================
 * CASE 7 — Parotid malignancy with facial nerve involvement — ALT free flap
 * Round-1 concern: no facial nerve reconstruction planned after nerve sacrifice.
 * ====================================================================== */
const case7 = {
  id: 'parotid-facial-nerve-alt-07',
  title: 'Parotid malignancy with facial nerve involvement (cT4a N1) — total parotidectomy and ALT free flap',
  caseText: `66-year-old woman with a high-grade parotid malignancy involving the facial nerve.

- Primary tumor: deep-lobe parotid malignancy with preoperative House-Brackmann grade IV weakness, indicating facial nerve involvement; skin and soft-tissue resection anticipated. Clinical stage cT4a N1 M0.
- Neck: single ipsilateral node.
- History: no prior head and neck surgery; otherwise fit.
- Function: partial facial weakness preoperatively; incomplete eye closure on the affected side.
- Workup: MRI shows facial nerve involvement without skull-base extension; chest CT clear; ECOG 0.`,
  plan: [
    { type: 'step', name: 'Total Parotidectomy with Facial Nerve Assessment',
      description: 'Perform total parotidectomy, identifying the facial nerve main trunk and distal branches; sacrifice involved nerve segments for oncologic clearance and tag the proximal stump and distal branch ends for grafting.' },
    { type: 'step', name: 'Ipsilateral Neck Dissection',
      description: 'Perform ipsilateral selective neck dissection (levels I-III) for the cN1 neck and to harvest the great auricular nerve if suitable, while preparing recipient vessels.' },
    { type: 'branch', condition: 'Gross extranodal extension is identified',
      steps: [{ name: 'Extend Neck Dissection', description: 'Extend the dissection as oncologically required and document extent for adjuvant-therapy planning.' }] },
    { type: 'step', name: 'Frozen Section Margin Assessment',
      description: 'Send soft-tissue and facial nerve stump margins for frozen section to confirm clearance before nerve reconstruction.' },
    { type: 'branch', condition: 'The proximal facial nerve stump margin is positive for tumor',
      steps: [{ name: 'Resect to Clear Stump then Reconstruct', description: 'Resect the nerve proximally to a tumor-free stump (towards the stylomastoid foramen) before cable grafting; if no proximal stump is usable, plan a nerve-transfer strategy.' }] },
    { type: 'step', name: 'Facial Nerve Cable Grafting',
      description: 'Reconstruct the facial nerve with cable interposition grafts (great auricular or sural nerve) from the proximal trunk to the distal branches to restore resting tone and future movement.' },
    { type: 'branch', condition: 'No usable proximal facial nerve stump is available',
      steps: [{ name: 'Nerve Transfer', description: 'Perform a masseteric-to-facial nerve transfer (± cross-face nerve graft or hypoglossal-facial transfer) to reinnervate the distal branches.' }] },
    { type: 'step', name: 'Eye Protection Procedures',
      description: 'Address the paralyzed eye at the same sitting with an upper-lid gold/platinum weight and/or lateral tarsorrhaphy to protect the cornea until reinnervation.' },
    { type: 'step', name: 'ALT Free Flap Harvest',
      description: 'Harvest an anterolateral thigh free flap to restore the parotidectomy volume defect and any resected overlying skin, contouring for facial symmetry.' },
    { type: 'step', name: 'Microvascular Anastomosis',
      description: 'Perform microvascular arterial and venous anastomoses to the prepared neck vessels under the operating microscope; confirm flap perfusion.' },
    { type: 'branch', condition: 'Flap shows venous congestion or arterial compromise',
      steps: [{ name: 'Revise Anastomosis', description: 'Re-explore and revise the anastomosis, with vein grafting for tension or caliber mismatch.' }] },
    { type: 'step', name: 'Inset and Donor Closure',
      description: 'Inset and contour the flap for facial symmetry, protect the nerve grafts, and close the thigh donor site primarily or with a skin graft.' },
    { type: 'step', name: 'Postoperative Monitoring and Rehabilitation',
      description: 'Hourly flap monitoring for 72 hours, ophthalmologic care of the affected eye, and referral for facial reanimation physiotherapy.' }
  ],
  comment:
    'The revised plan is appropriate for a high-grade parotid malignancy with facial nerve involvement: total parotidectomy with tumor-driven nerve sacrifice, ipsilateral nodal treatment, and an ALT flap for the volume and skin defect. The essential correction is present: facial nerve continuity is reconstructed with cable interposition grafts (great auricular/sural), with a defined nerve-transfer fallback when no proximal stump is usable, and the paralyzed eye is protected with a lid-loading procedure and/or tarsorrhaphy. Margin control at the nerve stump and anastomotic contingencies are defined, and facial reanimation rehabilitation is arranged.',
  operativeNote: `### Preoperative Plan

- Confirm cT4a N1 M0 high-grade parotid malignancy with facial nerve involvement (House-Brackmann IV) and no skull-base extension; negative metastatic workup; ECOG 0.
- Plan facial nerve reconstruction and eye protection as integral steps; counsel on graft donor sites (great auricular/sural) and expected recovery.
- Consent for total parotidectomy, ipsilateral neck dissection, facial nerve cable grafting (or nerve transfer), eye-protection procedure, and ALT free flap reconstruction.

### Intraoperative Plan: Ablation

1. **Resection:** total parotidectomy with identification of the facial nerve trunk and branches; involved nerve sacrificed for clearance, with stumps tagged for grafting.
2. **Neck:** ipsilateral selective neck dissection (levels I-III); great auricular nerve harvested if suitable; recipient vessels prepared.
3. **Margin control:** soft-tissue and facial nerve stump margins to frozen section; proximal nerve resected to a clear stump before grafting.

### Intraoperative Plan: Reconstruction

1. **Facial nerve:** cable interposition grafts (great auricular or sural) from the proximal trunk to distal branches; masseteric-to-facial (± cross-face or hypoglossal) transfer if no proximal stump is usable.
2. **Eye protection:** upper-lid gold/platinum weight and/or lateral tarsorrhaphy at the same sitting.
3. **Volume/skin:** anterolateral thigh free flap to restore the parotid volume and skin defect, contoured for symmetry.
4. **Anastomosis:** microvascular anastomoses to neck vessels; perfusion confirmed before inset.

### Key Contingency Plans

- **Positive proximal nerve stump:** resect towards the stylomastoid foramen to a clear stump, or switch to a nerve-transfer strategy.
- **No usable proximal stump:** masseteric-to-facial nerve transfer (± cross-face graft or hypoglossal-facial transfer).
- **Gross extranodal extension:** extend the neck dissection and document for adjuvant planning.
- **Venous congestion or arterial compromise:** re-explore and revise the anastomosis, with vein grafting as needed.
- **Postoperative:** corneal protection and care of the affected eye, and referral for facial reanimation physiotherapy.`,
  process: {
    draftSteps: [
      'Analyzing case & staging (cT4a N1 parotid malignancy, facial nerve involvement)…',
      'Planning total parotidectomy and volume reconstruction…',
      'Selecting ALT free flap…'
    ],
    reviewChecks: ['Oncologic soundness', 'Facial nerve strategy', 'Reconstructive soundness', 'Contingency planning'],
    round1: {
      verdict: 'flagged',
      concern:
        'The facial nerve is sacrificed for clearance but no facial nerve reconstruction is planned, and the paralyzed eye is not protected. This leaves permanent facial paralysis and exposes the cornea to sight-threatening injury.'
    },
    round2: {
      fix:
        'Added facial nerve cable grafting (great auricular/sural) from the proximal trunk to distal branches, a masseteric-to-facial nerve-transfer fallback when no proximal stump is usable, and same-sitting eye protection (lid weight and/or tarsorrhaphy).',
      verdict: 'accept'
    }
  }
};

// The library. Case 1 stays first for backward compatibility.
export const cases = [case1, case2, case3, case4, case5, case6, case7].map(buildCase);

// Backward-compatible single-case export (points at case 1).
export const demoCase = cases[0];

export default demoCase;
