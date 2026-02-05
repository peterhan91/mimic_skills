# Complete Example Walkthrough: One Evolution Cycle

This traces **exactly what happens** in one cycle, with real data.

The key insight: the Evolver doesn't just analyze agent failures — it reads the
**actual discharge summaries** to see what real doctors did, then teaches the
agent general clinical reasoning patterns (not disease-specific checklists).

---

## The Setup

We have 10 train admissions per pathology (40 total across appendicitis,
cholecystitis, diverticulitis, pancreatitis) in `data_splits/{pathology}/train.pkl`.
The agent (Llama 3.3 70B) will try to diagnose all of them each cycle.

Each admission also has a full **discharge summary** (in the `Discharge` field
of the pkl, or from `mimic-iv-note/2.2/note/discharge.csv.gz`). The discharge
summary records what the real doctor did — the workup, reasoning, and treatment.
This is our ground truth for how to do it right.

**Why mixed batch**: Evolving on a single disease (e.g., only appendicitis)
causes the skill to implicitly overfit to that disease's patterns — even if
phrased generally. For example, "US first" is correct for appendicitis and
cholecystitis but wrong for diverticulitis (CT is first-line). A mixed batch
forces the skill to learn truly general reasoning.

---

## PHASE 1: Agent Runs on GPU Server (the "Act" phase)

### What the agent sees

For admission `29668508`, the agent receives ONLY this as input:

```
Patient History:
___. female awoke this morning with severe crampy infraumbilical pain,
evolved into bandlike suprapubic distribution. Denies fever, chills.
C/o nausea with non-bloody, bilious emesis x 3. Reports diarrhea.
Denies sick contacts with GI symptoms.

Past Medical History: None
Social History: ___
Family History: Denies family history of gastrointestinal disorders
```

### What the agent does (multi-turn ReAct loop)

The agent thinks and acts in a loop. A **typical trajectory** looks like:

```
Thought: Young female with acute infraumbilical pain migrating to RLQ,
nausea, vomiting. DDx includes appendicitis, ovarian pathology, UTI.
I should start with a physical examination.
Action: Physical Examination
Action Input: Perform physical examination
Observation: T: 98.2 P: 113 BP: 138/86 RR: 18 O2sat: 100% RA
  Abdomen: soft, ND, focal severe RLQ tenderness with involuntary guarding...

Thought: RLQ tenderness with guarding is classic for appendicitis.
Tachycardic at 113. I need to check inflammatory markers.
Action: Laboratory Tests
Action Input: ["Complete Blood Count", "C-Reactive Protein"]
Observation: WBC: 14.5 K/uL (HIGH), CRP: 8.2 mg/dL (HIGH)...

Thought: Elevated WBC and CRP support acute inflammation. RLQ pain
with guarding. Should get imaging to confirm. Guidelines say US first.
Action: Imaging
Action Input: {"region": "Abdomen", "modality": "Ultrasound"}
Observation: Abdomen Ultrasound: appendix measures 9mm, mucosal
  hyperenhancement, periappendiceal fat stranding...

Thought: US shows dilated appendix 9mm with inflammation. This is
consistent with acute appendicitis.
Final Diagnosis: Acute Appendicitis
Treatment: Laparoscopic appendectomy, IV antibiotics, IV fluids, analgesia
```

### But a BAD trajectory might look like:

```
Thought: Patient has abdominal pain. Let me order some tests.
Action: Laboratory Tests                          ← SKIPPED PE!
Action Input: ["Lipase", "Amylase", "Liver Panel"] ← WRONG LABS!
Observation: Lipase: 25 (normal), Amylase: 40 (normal)...

Thought: Lipase normal, probably not pancreatitis. Let me image.
Action: Imaging
Action Input: {"region": "Abdomen", "modality": "CT"} ← CT before US!
Observation: CT Abdomen: dilated appendix, periappendiceal inflammation...

Thought: CT shows appendicitis.
Final Diagnosis: Appendicitis
Treatment: Antibiotics                            ← MISSED SURGERY!
```

### What PathologyEvaluator scores

For each admission, the evaluator produces:

```json
// GOOD trajectory scores:
{
  "Diagnosis": 1,              // Correct diagnosis
  "Physical Examination": 1,    // PE done first
  "Late Physical Examination": 1,// PE done at all
  "Laboratory Tests": 1,        // Ordered inflammation markers (WBC/CRP)
  "Imaging": 2,                 // US first (best) = 2 points
  "Invalid Tools": 0,           // No hallucinated tools
  "Action Parsing": 0,          // Clean format
  "Rounds": 4                   // Efficient (4 turns)
}
// Treatment: Appendectomy ✓, Antibiotics ✓, Support ✓

// BAD trajectory scores:
{
  "Diagnosis": 1,              // Got diagnosis right (from CT)
  "Physical Examination": 0,    // PE NOT done first ← FAILURE
  "Late Physical Examination": 0,// PE never done ← FAILURE
  "Laboratory Tests": 0,        // Lipase/amylase are WRONG labs ← FAILURE
  "Imaging": 1,                 // CT instead of US ← SUBOPTIMAL
  "Invalid Tools": 0,
  "Action Parsing": 0,
  "Rounds": 3
}
// Treatment: Appendectomy ✗ ← MISSED SURGERY
```

### GPU server command

```bash
# On GPU server — run ALL 4 pathologies
cd codes_Hager/MIMIC-Clinical-Decision-Making-Framework

for pathology in appendicitis cholecystitis diverticulitis pancreatitis; do
  python run.py \
    pathology=$pathology \
    model=Llama3.3Instruct70B \
    paths=server \
    base_mimic=/path/to/data_splits/$pathology \
    summarize=True \
    run_descr=_baseline_train10
done

# Output per pathology: results/{run_name}/{run_name}_results.pkl  (trajectories)
#                       results/{run_name}/{run_name}_eval.pkl      (scores)
```

This runs 40 admissions total (10 per pathology). Takes ~40-120 min.

---

## PHASE 2: Analyze Results (Local — Claude Code)

### Pull results from server

```bash
rsync -avz server:/path/to/results/ results/
```

### What the results look like

```python
# results.pkl contains per-admission:
{
  29668508: {
    "input": "Patient History: ___. female awoke...",
    "output": "Final Diagnosis: Appendicitis\nTreatment: Antibiotics",
    "intermediate_steps": [
      (AgentAction(tool="Laboratory Tests", ...), "WBC: 14.5..."),
      (AgentAction(tool="Imaging", ...), "CT Abdomen: ..."),
    ]
  },
  24009162: { ... },
  # ... 10 total
}

# eval.pkl contains per-admission:
{
  29668508: {
    "scores": {"Diagnosis": 1, "Physical Examination": 0, ...},
    "answers": {"Diagnosis": "Appendicitis", "Treatment": "Antibiotics", ...}
  },
  # ... 10 total
}
```

### Aggregate the scores across all 4 pathologies

```
APPENDICITIS (10 patients):
  Diagnosis: 8/10   PE first: 5/10   Labs: 6/10   Imaging: 4/10   Treatment: 4/10

CHOLECYSTITIS (10 patients):
  Diagnosis: 7/10   PE first: 4/10   Labs: 5/10   Imaging: 5/10   Treatment: 3/10

DIVERTICULITIS (10 patients):
  Diagnosis: 6/10   PE first: 5/10   Labs: 4/10   Imaging: 3/10   Treatment: 4/10

PANCREATITIS (10 patients):
  Diagnosis: 7/10   PE first: 6/10   Labs: 3/10   Imaging: 4/10   Treatment: 3/10

AGGREGATE (40 patients):
  Diagnosis accuracy:    28/40 = 70%  ← varies by pathology
  PE done first:         20/40 = 50%  ← PROBLEM across all diseases
  Correct labs:          18/40 = 45%  ← PROBLEM (different labs needed per disease)
  Optimal imaging:       16/40 = 40%  ← PROBLEM (US vs CT depends on disease)
  Full treatment:        14/40 = 35%  ← PROBLEM (surgery vs supportive varies)
```

The mixed batch reveals **cross-disease** failure patterns — e.g., imaging
score is low for different reasons in different diseases (appendicitis: CT
instead of US, diverticulitis: US instead of CT).

---

## PHASE 3: Evolver Generates Improved Config (Local — Claude Code)

### Step 3a: Identify agent failures from trajectories

Claude Code (acting as Evolver) reads all 40 transcripts across 4 diseases
and identifies failure patterns:

| Failure Pattern | Agent Did | Frequency |
|---|---|---|
| Skipped Physical Examination | Jumped to labs or imaging | 20/40 (all diseases) |
| Wrong lab panel | Shotgun-ordered all labs regardless of exam | 22/40 |
| Wrong imaging modality | US when CT needed, or CT when US sufficient | 24/40 |
| Incomplete treatment | Missed surgery or missed supportive care | 26/40 |

The cross-disease view reveals that some "failures" are disease-dependent:
the agent orders CT for appendicitis (should be US) but also orders US for
diverticulitis (should be CT). A single-disease analysis would miss this.

### Step 3b: Read the ACTUAL discharge summaries for failed patients

This is the critical step. Instead of guessing what the agent should have done,
we look at what the **real doctor** did for the same patient.

**Admission 29668508** — agent skipped PE, ordered wrong labs, missed surgery:

```
WHAT THE REAL DOCTOR DID (from discharge summary):

Chief Complaint: Abdominal pain
Service: SURGERY

Physical Exam on admission:
  T: 98.2  P: 113  BP: 138/86  RR: 18  O2sat: 100% RA
  Abdomen: soft, ND, focal severe RLQ tenderness with involuntary
  guarding, no rebound, negative Rovsing's sign

Labs ordered:
  WBC-22.9* (markedly elevated)
  Neuts-87.2* (left shift — acute bacterial process)
  Glucose-89, BUN-15, Creat-0.8 (basic metabolic — normal)
  NOT lipase, NOT amylase, NOT liver panel

Imaging:
  PELVIS US (not abdomen CT):
  "Sonographic findings consistent with early appendicitis.
   Normal uterus and ovaries."
  → US was chosen first, specifically pelvic to also rule out
    ovarian pathology in a young female

Procedure: Laparoscopic appendectomy
Course: Admitted → OR → recovery → discharged home
Medications: Oxycodone-Acetaminophen PRN pain

DOCTOR'S REASONING (implicit from the workup):
  1. Young female + acute abdominal pain → PE first
  2. RLQ tenderness + guarding + tachycardia → suspect appendicitis
  3. CBC to confirm inflammation → WBC 22.9 with neutrophilia
  4. Pelvic US (not CT) → rules out appendicitis AND ovarian pathology
  5. US positive → straight to laparoscopic appendectomy
```

### Step 3c: Gap analysis — Agent vs Doctor

For each failed admission, compare side by side:

```
ADMISSION 29668508:
  Step          Agent                     Doctor
  ──────────    ────────────────────      ─────────────────────────
  First action  Laboratory Tests          Physical Examination
  Labs ordered  Lipase, Amylase, LFTs    CBC (WBC), BMP
  Lab reasoning "Rule out pancreatitis"   "Confirm inflammation"
  Imaging       Abdomen CT                Pelvic US
  Why imaging   Generic "let me image"    Rule out appendicitis + ovarian
  Treatment     "Antibiotics"             Laparoscopic appendectomy
  Outcome       Correct dx, bad workup    Correct dx, correct workup
```

The gap is clear: the agent doesn't understand WHY each step is taken.
It treats diagnosis as a lookup table ("abdominal pain → order tests →
match pattern"). The doctor reasons from first principles ("tenderness
location → targeted labs → least-invasive imaging that answers the question").

### Step 3d: Check the pattern across diseases (crucial for generalization)

Now look at a cholecystitis discharge summary (a different disease
presenting with the same chief complaint — abdominal pain):

```
CHOLECYSTITIS (admission 29897948):

  Physical Exam: TTP RUQ (right UPPER quadrant — different from appendicitis!)
  Labs: WBC-11.5*, Neuts-87.4* (same inflammation pattern, different location)
  Imaging: Abdominal US → distended gallbladder with gallstones
  Procedure: Laparoscopic cholecystectomy

DOCTOR'S REASONING:
  1. Sudden abdominal pain + nausea/vomiting → PE first
  2. RUQ tenderness → suspect biliary pathology (not RLQ → not appendicitis)
  3. CBC to confirm inflammation → WBC 11.5 with neutrophilia
  4. US → gallstones with distended gallbladder
  5. Cholecystectomy
```

**The GENERAL pattern across both diseases:**
1. PE first → tenderness LOCATION narrows the differential
2. CBC/BMP → confirm inflammation (not disease-specific labs)
3. US first → least invasive, answers the question
4. Findings → diagnosis → appropriate procedure

This is the pattern the skill should teach. NOT "if appendicitis → do X."

### Step 3e: Generate a GENERAL clinical reasoning skill

File: `skills/v1/acute_abdominal_pain.md`

```markdown
## Clinical Reasoning Workflow for Acute Abdominal Pain

When a patient presents with acute abdominal pain, follow this systematic
approach. The same workflow applies regardless of the final diagnosis.

### Step 1: Physical Examination FIRST
Before ordering ANY tests, perform a Physical Examination.
The exam narrows your differential based on WHERE the tenderness is:
- Tenderness location tells you which organ system to investigate
- Guarding or rebound suggests peritoneal irritation (surgical urgency)
- Vital signs (tachycardia, fever) indicate severity

Do NOT skip the exam to go straight to labs or imaging. The exam determines
WHICH labs and imaging to order.

### Step 2: Targeted Laboratory Tests Based on Exam Findings
Order labs that confirm or rule out the suspected diagnosis from Step 1:
- For ANY acute abdominal pain: CBC (WBC to assess inflammation), BMP
- WBC with differential: neutrophilia (left shift) suggests acute bacterial process
- Only add organ-specific labs if exam findings point there:
  - Epigastric pain → add Lipase
  - RUQ pain → add Liver Panel, Bilirubin (biliary/hepatic)
  - Flank pain → add Urinalysis (renal)

Do NOT shotgun-order lipase + amylase + liver panel + urinalysis for every
patient. Let the exam guide your lab selection.

### Step 3: Imaging — Guided by Suspected Pathology
Choose imaging based on what question you need answered:
- RLQ or RUQ tenderness → Ultrasound first (less radiation, often diagnostic)
- LLQ tenderness or suspected colonic pathology → CT first (more sensitive)
- Epigastric pain → CT with contrast (best for retroperitoneal assessment)
- CT if US is inconclusive regardless of location
- Match the imaging region to the suspected pathology, not just "abdomen"

### Step 4: Synthesize Findings → Diagnosis → Treatment
- State your diagnosis based on the convergence of exam + labs + imaging
- Treatment must match the severity of the condition:
  - If imaging confirms a surgical condition → recommend the appropriate
    procedure (e.g., surgical intervention)
  - Do NOT recommend "antibiotics" alone for conditions that require surgery
  - Include supportive care: IV fluids, pain management, antibiotics if indicated
```

### Step 3f: Sanitize the skill before injection

**Critical step**: The Evolver may use disease names while generating the skill
(e.g., "for appendicitis, order CBC"). These must be removed before injection,
because Hager's framework sanitizes patient data to remove disease names —
if our skill puts them back, we introduce diagnosis leakage.

```bash
python scripts/sanitize_skill.py skills/v1/acute_abdominal_pain.md --inplace --report
```

The sanitizer replaces disease names and procedure names with `____`
(same mask Hager's framework uses in patient data), consistent with
`CreateDataset.py` sanitize lists. See `scripts/sanitize_skill.py`.

**Why this skill is better than the old one:**

| Old skill (disease-specific) | New skill (reasoning-grounded) |
|---|---|
| "For appendicitis: US first" | Imaging guided by exam location |
| "Order CBC and CRP" | "Order CBC + BMP; add organ-specific labs based on exam" |
| "If appendicitis → appendectomy" | "If surgical condition → appropriate procedure" |
| Only works for appendicitis | Works across all 4 pathologies |
| Teaches WHAT to do | Teaches WHY each step is taken |
| Generated from agent failures alone | Grounded in real doctor workups |
| Contains disease names (leakage!) | Sanitized before injection |

---

## PHASE 4: Re-run with Evolved Skill (GPU Server)

### Inject the skill into the agent

The skill gets inserted into the `{examples}` slot of CHAT_TEMPLATE,
so the agent sees it in its prompt before each patient case:

```
[SYSTEM]
You are a medical artificial intelligence assistant...
[/SYSTEM]

[USER]
## Clinical Reasoning Workflow for Acute Abdominal Pain    ← INJECTED SKILL

### Step 1: Physical Examination FIRST
Before ordering ANY tests, perform a Physical Examination.
The exam narrows your differential based on WHERE the tenderness is...
### Step 2: Targeted Laboratory Tests Based on Exam Findings...
### Step 3: Imaging — Guided by Suspected Pathology...
### Step 4: Synthesize Findings → Diagnosis → Treatment...

Consider the following case and come to a final diagnosis...

Patient History:
___. female awoke this morning with severe crampy infraumbilical pain...
[/USER]

[ASSISTANT]
Thought:
```

### Run on GPU server

```bash
# Same skill file for ALL pathologies — it teaches reasoning, not answers
for pathology in appendicitis cholecystitis diverticulitis pancreatitis; do
  python run.py \
    pathology=$pathology \
    model=Llama3.3Instruct70B \
    paths=server \
    base_mimic=/path/to/data_splits/$pathology \
    summarize=True \
    skill_path=/path/to/skills/v1/acute_abdominal_pain.md \
    run_descr=_v1_train10
done
```

We don't need `appendicitis.md`, `cholecystitis.md`, etc. — one general
reasoning skill covers all 4 pathologies because it teaches the process,
not the answers.

---

## PHASE 5: Compare (Local — Claude Code)

**Note**: This comparison is on the **training set** (the same 40 patients
the Evolver analyzed). It tells us whether the skill is working as intended,
but it is NOT the final evaluation. Phase 6 on the held-out 100-patient
test set is what we report.

### v1 results vs baseline (per-pathology + aggregate)

```
APPENDICITIS (10 patients):
                    Baseline    v1 (with skill)    Change
  Diagnosis:         8/10        9/10              +1
  PE first:          5/10        9/10              +4 ✓✓
  Correct labs:      6/10        8/10              +2 ✓
  Optimal imaging:   4/10        7/10              +3 ✓✓
  Full treatment:    4/10        7/10              +3 ✓✓

CHOLECYSTITIS (10 patients):
  Diagnosis:         7/10        8/10              +1
  PE first:          4/10        9/10              +5 ✓✓
  Correct labs:      5/10        7/10              +2 ✓
  Optimal imaging:   5/10        7/10              +2 ✓
  Full treatment:    3/10        6/10              +3 ✓✓

DIVERTICULITIS (10 patients):
  Diagnosis:         6/10        8/10              +2 ✓
  PE first:          5/10        8/10              +3 ✓✓
  Correct labs:      4/10        6/10              +2 ✓
  Optimal imaging:   3/10        6/10              +3 ✓✓
  Full treatment:    4/10        6/10              +2 ✓

PANCREATITIS (10 patients):
  Diagnosis:         7/10        8/10              +1
  PE first:          6/10        9/10              +3 ✓✓
  Correct labs:      3/10        7/10              +4 ✓✓
  Optimal imaging:   4/10        6/10              +2 ✓
  Full treatment:    3/10        6/10              +3 ✓✓

AGGREGATE (40 patients):
                    Baseline    v1 (with skill)    Change
  Diagnosis:        28/40=70%   33/40=82.5%       +12.5% ✓
  PE first:         20/40=50%   35/40=87.5%       +37.5% ✓✓
  Correct labs:     18/40=45%   28/40=70%         +25% ✓
  Optimal imaging:  16/40=40%   26/40=65%         +25% ✓✓
  Full treatment:   14/40=35%   25/40=62.5%       +27.5% ✓✓
```

The skill helped across all 4 pathologies! The biggest gain is PE ordering
(+37.5%), which makes sense — "PE FIRST" is the most direct instruction.
But still not perfect. Some failures remain.

### Analyze remaining failures for v2

Look at a misdiagnosed admission and its discharge summary:

```
Admission 28795086 (appendicitis — agent WRONG):
  Agent said: "Constipation / functional bowel disorder" ← WRONG
  Correct: Perforated condition with abscess

  What confused the agent:
    - Patient reported "constipation for the last week" and "tried bowel medications"
    - WBC was 6.2 (NORMAL) — no lab evidence of inflammation
    - No fever (T: 98.2)
    - Agent concluded "normal labs + constipation → not surgical"

  What the REAL DOCTOR's discharge summary shows:
    - PE: RLQ tenderness with guarding (surgical sign!)
    - CT: "3.0 x 2.1 cm abscess involving the tip of the appendix...
           focally contained perforation"
    - Doctor recognized: subacute RLQ pain + guarding = surgical
      despite normal WBC and week-long "constipation" presentation
    - Treatment: IV antibiotics + CT-guided drainage of abscess
    - This was a PERFORATED condition with normal WBC — the most
      dangerous atypical presentation
```

### Generate v2 skill

Read more discharge summaries — specifically cases where labs were normal
but imaging confirmed a serious diagnosis. Add to the skill:

```markdown
### Important: Do Not Over-Rely on Lab Values
Lab values are supportive, not definitive. A condition can be present even
when expected lab abnormalities are absent:
- Normal WBC does not rule out an acute surgical condition — even
  perforated ____ can present with WBC in the normal range
- Subacute presentations (days of mild symptoms) may have normalized
  labs despite worsening pathology
- If physical exam shows peritoneal signs (guarding, rebound), treat
  it as surgical until imaging proves otherwise
- When exam findings and lab results conflict, imaging is the tiebreaker
```

(Note: `____` is Hager's mask — the sanitizer already replaced disease names.)

Then sanitize: `python scripts/sanitize_skill.py skills/v2/acute_abdominal_pain.md --inplace`

This refinement is grounded in real discharge summary data (admission 28795086:
WBC 6.2 with perforated condition and abscess), not guesswork.

---

## PHASE 6: Final Evaluation on Test Set (GPU Server)

Once the skill converges (v3 or v4), evaluate on the **held-out 100 admissions**:

```bash
# Swap in test split
python scripts/prepare_split_for_hager.py --pathology appendicitis --split test

# Run baseline
python run.py pathology=appendicitis ... run_descr=_baseline_test100

# Run best skill
python run.py pathology=appendicitis ... skill_path=skills/v3/acute_abdominal_pain.md run_descr=_v3_test100
```

**Test on ALL pathologies with the SAME skill** (because it's general):

```bash
for pathology in appendicitis cholecystitis diverticulitis pancreatitis; do
  python scripts/prepare_split_for_hager.py --pathology $pathology --split test
  python run.py pathology=$pathology ... skill_path=skills/v3/acute_abdominal_pain.md \
    run_descr=_v3_test100
done
```

**This is the number we report in the paper.** One general skill, four diseases.

---

## Summary: The Full Loop

```
┌───────────────────────────────────────────────────────────────────┐
│                     LOCAL (Claude Code)                            │
│                                                                   │
│  1. Write/edit code (agent.py, skills, scripts)                   │
│  2. Push to GPU server                                            │
│                          │                                        │
│                          ▼                                        │
│  ┌──────────────────────────────────────────────────┐             │
│  │            GPU SERVER                             │             │
│  │                                                   │             │
│  │  3. Run: python run.py ... (all 4 pathologies)    │             │
│  │     Llama 3.3 70B diagnoses 40 patients           │             │
│  │     (10 per pathology)                            │             │
│  │     PathologyEvaluator scores each one             │             │
│  │                                                   │             │
│  │  4. Output: results.pkl + eval.pkl per pathology  │             │
│  └───────────────────┬──────────────────────────────┘             │
│                      │                                            │
│                      ▼                                            │
│  5. Pull results back                                             │
│  6. Identify failures from agent trajectories (all 4 diseases)    │
│  7. Read DISCHARGE SUMMARIES for failed patients  ← KEY STEP      │
│     (see what the real doctor did)                                 │
│  8. Gap analysis: agent vs doctor, across diseases                │
│  9. Evolve: generate GENERAL reasoning skill (v1 → v2 → v3)      │
│  10. Sanitize skill (remove disease names) ← REQUIRED             │
│  11. Go to step 2                                                 │
│                                                                   │
│  When converged:                                                  │
│  12. Swap to test split (100 admissions per pathology)            │
│  13. Run baseline + best skill on ALL 4 diseases → REPORT         │
└───────────────────────────────────────────────────────────────────┘
```

### What's different from the old walkthrough

| Before | After |
|---|---|
| Evolver only sees agent failures | Evolver sees failures + real discharge summaries |
| Skill says "For appendicitis: do X" | Skill says "For acute abdominal pain: reason like this" |
| One skill per disease | One skill for all diseases |
| Evolve on 10 patients of one disease | Evolve on 40 patients across all 4 diseases |
| Guesses at correct workup | Grounded in what doctors actually did |
| Teaches WHAT to do | Teaches WHY each step matters |
| Disease names leak into agent prompt | Sanitized before injection (no leakage) |

### Time per cycle

| Step | Time | Where |
|---|---|---|
| Push code | 30 sec | Local → Server |
| Run 40 patients (10 per pathology) | 40-120 min | GPU Server |
| Pull results | 30 sec | Server → Local |
| Analyze failures + read discharge summaries | 10-15 min | Local (Claude Code) |
| Gap analysis across diseases + evolve skill | 10-15 min | Local (Claude Code) |
| Sanitize skill | 1 sec | Local |
| **Total per cycle** | **~60-150 min** | |

Expect 3-10 cycles to converge. That's **1-2 days** to build a skill
that works across all 4 pathologies.

### Later: Automate with EvoTest

Once the manual loop works and we understand the patterns, we automate
steps 6-10 (failure analysis + discharge summary reading + gap analysis +
skill evolution + sanitization) using the EvoTest framework — replacing
human analysis with an Opus/o3 Evolver Agent that:

1. Reads agent trajectories (what happened)
2. Reads discharge summaries (what should have happened)
3. Identifies the reasoning gap
4. Produces an improved general skill

The manual loop above is proof-of-concept; EvoTest makes it run 50 episodes
unattended.
