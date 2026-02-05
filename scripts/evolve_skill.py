"""
Evolver: Analyze agent trajectories + discharge summaries to generate an
improved clinical reasoning skill using the Anthropic API (Claude).

Identifies failure patterns, compares agent behavior to real doctor workups
from discharge summaries, and generates a general clinical reasoning skill.

Usage (single pathology):
    python scripts/evolve_skill.py \
        --trajectories trajectories/baseline_cholecystitis_train10.json \
        --model claude-sonnet-4-20250514 \
        --output skills/v1/acute_abdominal_pain.md

Usage (multiple pathologies — recommended):
    python scripts/evolve_skill.py \
        --trajectories trajectories/baseline_appendicitis_train10.json \
                       trajectories/baseline_cholecystitis_train10.json \
                       trajectories/baseline_diverticulitis_train10.json \
                       trajectories/baseline_pancreatitis_train10.json \
        --model claude-sonnet-4-20250514 \
        --output skills/v2/acute_abdominal_pain.md

Optionally feed a previous skill for iterative refinement:
    python scripts/evolve_skill.py \
        --trajectories ... \
        --prev-skill skills/v1/acute_abdominal_pain.md \
        --output skills/v2/acute_abdominal_pain.md

Requires: ANTHROPIC_API_KEY environment variable.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_trajectories(path):
    """Load trajectory JSON from extract_trajectories.py."""
    with open(path, "r") as f:
        return json.load(f)


def identify_failures(data):
    """Identify patients where the agent failed on key metrics."""
    failures = []
    for admission in data["admissions"]:
        reasons = []
        scores = admission["scores"]
        if scores.get("Diagnosis", 0) == 0:
            reasons.append("wrong_diagnosis")
        if scores.get("Physical Examination", 0) == 0:
            reasons.append("pe_not_first")
        if scores.get("Late Physical Examination", 0) == 0:
            reasons.append("pe_never_done")
        if scores.get("Laboratory Tests", 0) == 0:
            reasons.append("wrong_labs")
        if scores.get("Imaging", 0) == 0:
            reasons.append("wrong_imaging")
        if scores.get("Invalid Tools", 0) > 0:
            reasons.append("hallucinated_tools")

        if reasons:
            failures.append({
                "admission": admission,
                "reasons": reasons,
                "pathology": data["pathology"],
            })
    return failures


def format_trajectory_summary(admission, pathology=None):
    """Format a single admission trajectory for the Evolver prompt."""
    lines = []
    path_tag = f" [{pathology}]" if pathology else ""
    lines.append(f"### Patient hadm_id={admission['hadm_id']}{path_tag}")
    lines.append(f"**Input**: {admission['input'][:500]}...")
    lines.append("")
    lines.append("**Agent Trajectory**:")
    for i, step in enumerate(admission["trajectory"]):
        thought_lines = step.get("log", "").strip()
        # Show just first 300 chars of log
        if len(thought_lines) > 300:
            thought_lines = thought_lines[:300] + "..."
        lines.append(f"  Step {i+1}: {step['tool']}")
        lines.append(f"    Log: {thought_lines}")
        obs = step.get("observation", "")[:200]
        lines.append(f"    Observation: {obs}...")
    lines.append("")
    lines.append(f"**Agent Output**: {admission['output'][:300]}")
    lines.append("")

    # Scores
    scores = admission["scores"]
    lines.append(f"**Scores**: Dx={scores.get('Diagnosis', '?')} "
                 f"PE_first={scores.get('Physical Examination', '?')} "
                 f"Labs={scores.get('Laboratory Tests', '?')} "
                 f"Imaging={scores.get('Imaging', '?')} "
                 f"Rounds={scores.get('Rounds', '?')}")
    return "\n".join(lines)


def format_discharge_summary(admission):
    """Format the real doctor's discharge summary for comparison."""
    ds = admission.get("discharge_summary", "")
    if not ds:
        return "(No discharge summary available)"
    # Truncate to reasonable length
    if len(ds) > 2000:
        ds = ds[:2000] + "\n... [truncated]"
    return ds


def build_aggregate_table(all_data):
    """Build aggregate scores table across all pathologies."""
    lines = [
        "| Pathology | N | Diagnosis | PE First | Labs (avg) | Imaging (avg) | Invalid Tools |",
        "|---|---|---|---|---|---|---|",
    ]
    for data in all_data:
        agg = data["aggregate"]
        n = data["n_patients"]
        pathology = data["pathology"]

        n_dx = sum(1 for a in data["admissions"] if a["scores"].get("Diagnosis", 0) > 0)
        n_pe = sum(1 for a in data["admissions"] if a["scores"].get("Physical Examination", 0) > 0)
        labs_avg = agg.get("Laboratory Tests", 0)
        img_avg = agg.get("Imaging", 0)
        inv_total = sum(a["scores"].get("Invalid Tools", 0) for a in data["admissions"])

        lines.append(
            f"| {pathology} | {n} | {n_dx}/{n} ({100*n_dx/n:.0f}%) | "
            f"{n_pe}/{n} ({100*n_pe/n:.0f}%) | {labs_avg:.2f} | "
            f"{img_avg:.2f} | {inv_total} |"
        )

    # Totals row
    total_n = sum(d["n_patients"] for d in all_data)
    total_dx = sum(
        sum(1 for a in d["admissions"] if a["scores"].get("Diagnosis", 0) > 0)
        for d in all_data
    )
    total_pe = sum(
        sum(1 for a in d["admissions"] if a["scores"].get("Physical Examination", 0) > 0)
        for d in all_data
    )
    total_inv = sum(
        sum(a["scores"].get("Invalid Tools", 0) for a in d["admissions"])
        for d in all_data
    )
    lines.append(
        f"| **TOTAL** | {total_n} | {total_dx}/{total_n} ({100*total_dx/total_n:.0f}%) | "
        f"{total_pe}/{total_n} ({100*total_pe/total_n:.0f}%) | - | - | {total_inv} |"
    )
    return "\n".join(lines)


def build_evolver_prompt(all_data, all_failures, prev_skill=None):
    """Construct the full Evolver prompt from multiple pathologies."""
    pathologies = [d["pathology"] for d in all_data]
    total_n = sum(d["n_patients"] for d in all_data)
    pathology_str = ", ".join(pathologies)

    # Build gap analysis for failures (limit to avoid token explosion)
    # Pick up to 3 failures per pathology, max 12 total
    gap_analyses = []
    for pathology in pathologies:
        path_failures = [f for f in all_failures if f["pathology"] == pathology]
        for fail in path_failures[:3]:
            admission = fail["admission"]
            reasons = ", ".join(fail["reasons"])
            analysis = f"""
---
{format_trajectory_summary(admission, pathology=fail['pathology'])}

**Failure reasons**: {reasons}

**Real Doctor's Discharge Summary**:
```
{format_discharge_summary(admission)}
```
"""
            gap_analyses.append(analysis)

    # Limit total gap analyses
    gap_analyses = gap_analyses[:12]

    prev_skill_section = ""
    if prev_skill:
        prev_skill_section = f"""
## Previous Skill

The following skill was used in the runs above. Analyze where it helped and where it failed, then IMPROVE it:

```markdown
{prev_skill}
```

"""

    prompt = f"""You are a clinical AI system optimizer. Your task is to analyze diagnostic agent trajectories and real discharge summaries from {total_n} patients across {len(pathologies)} pathologies ({pathology_str}), then generate an improved clinical reasoning skill.

## Current Agent Performance

{build_aggregate_table(all_data)}

{prev_skill_section}## Failed Trajectories with Gap Analysis

Below are patients where the agent failed across different pathologies. For each, you see:
1. What the agent did (its trajectory)
2. What the real doctor did (the discharge summary)
3. Why the agent failed

{chr(10).join(gap_analyses)}

## Your Task

Generate a GENERAL clinical reasoning workflow skill for diagnosing patients presenting with acute abdominal pain. This skill must:

1. **Teach systematic diagnostic reasoning** — the same workflow regardless of final diagnosis
2. **Address the specific failure patterns above** — focus on what went wrong and teach the correct approach
3. **Be grounded in what real doctors did** — use the discharge summary evidence, not abstract guidelines
4. **Work across ALL pathologies** — must handle {pathology_str} and any other acute abdominal pain cause
5. **Stay under 500 tokens** — concise, actionable instructions
6. **NOT use disease names** — use ____ as a mask for any disease or procedure name that would reveal the diagnosis (e.g., write "surgical intervention" instead of a specific procedure name)

The skill should be written as markdown with clear step-by-step instructions that the agent can follow during its diagnostic reasoning loop. Focus on:
- When to do Physical Examination (should always be FIRST)
- How to select labs based on exam findings (not shotgun ordering)
- How to choose imaging modality based on suspected pathology location
- When to recommend surgical vs conservative treatment
- How to interpret lab values in context (normal labs don't rule out surgical conditions)

Output ONLY the skill content in markdown format. Do not include any preamble or explanation outside the skill itself."""

    return prompt


def call_anthropic(prompt, model):
    """Call the Anthropic API to generate the skill."""
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def evolve_skill(trajectories_paths, model, output_path, prev_skill_path=None, dry_run=False):
    """Full evolution pipeline: load -> analyze -> generate -> save."""
    # Load all trajectory files
    all_data = []
    all_failures = []
    for path in trajectories_paths:
        data = load_trajectories(path)
        all_data.append(data)
        failures = identify_failures(data)
        all_failures.extend(failures)
        print(f"Loaded {data['n_patients']} trajectories for {data['pathology']}")
        print(f"  {len(failures)} patients with failures:")
        for fail in failures:
            a = fail["admission"]
            print(f"    hadm_id={a['hadm_id']}: {', '.join(fail['reasons'])}")

    total_n = sum(d["n_patients"] for d in all_data)
    total_fail = len(all_failures)
    print(f"\nTotal: {total_n} patients across {len(all_data)} pathologies, {total_fail} failures")

    # Load previous skill if provided
    prev_skill = None
    if prev_skill_path and os.path.exists(prev_skill_path):
        with open(prev_skill_path, "r") as f:
            prev_skill = f.read()
        # Strip YAML frontmatter
        if prev_skill.startswith("---"):
            parts = prev_skill.split("---", 2)
            if len(parts) >= 3:
                prev_skill = parts[2].strip()
        print(f"Loaded previous skill from {prev_skill_path} ({len(prev_skill)} chars)")

    prompt = build_evolver_prompt(all_data, all_failures, prev_skill=prev_skill)

    if dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN — Evolver prompt:")
        print(f"{'='*60}")
        print(prompt)
        print(f"\nPrompt length: {len(prompt)} chars")
        return prompt

    print(f"\nCalling {model}...")
    skill_content = call_anthropic(prompt, model)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(skill_content)
    print(f"\nGenerated skill ({len(skill_content)} chars) saved to {output_path}")
    print(f"\n{'='*60}")
    print("GENERATED SKILL:")
    print(f"{'='*60}")
    print(skill_content)

    return skill_content


def main():
    parser = argparse.ArgumentParser(
        description="Generate improved clinical reasoning skill from trajectories"
    )
    parser.add_argument(
        "--trajectories", type=str, nargs="+", required=True,
        help="Path(s) to trajectory JSON(s) from extract_trajectories.py"
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514",
        help="Anthropic model to use for skill generation"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for generated skill markdown"
    )
    parser.add_argument(
        "--prev-skill", type=str, default=None,
        help="Path to previous skill for iterative refinement"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the Evolver prompt without calling the API"
    )
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is required")
        sys.exit(1)

    evolve_skill(args.trajectories, args.model, args.output,
                 prev_skill_path=args.prev_skill, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
