"""
Parse open_guidelines.jsonl and extract disease-specific clinical guidelines
for the 4 target pathologies.

Outputs per-pathology directories under guidelines/ with:
  - Individual guideline files (full text, for reference)
  - A condensed summary file (key diagnostic/treatment sections, for Evolver injection)

Usage:
    python scripts/parse_guidelines.py \
        --input open_guidelines.jsonl \
        --output-dir guidelines

    # Only extract condensed summaries (skip full guideline files):
    python scripts/parse_guidelines.py --input open_guidelines.jsonl --output-dir guidelines --summary-only

Prioritizes high-quality sources: pubmed > nice > cdc > who > cma > wikidoc.
"""

import argparse
import json
import os
import re
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Target pathologies and their search terms
# ---------------------------------------------------------------------------
PATHOLOGIES = {
    "appendicitis": {
        "primary": ["appendicitis"],
        "secondary": ["appendectomy", "appendiceal"],
    },
    "cholecystitis": {
        "primary": ["cholecystitis"],
        "secondary": ["cholecystectomy", "gallstone disease", "cholelithiasis"],
    },
    "diverticulitis": {
        "primary": ["diverticulitis"],
        "secondary": ["diverticular disease"],
    },
    "pancreatitis": {
        "primary": ["pancreatitis"],
        "secondary": [],  # "pancreatic" is too broad
    },
}

# Source quality tiers (higher = better)
SOURCE_TIER = {
    "pubmed": 1,   # Peer-reviewed, evidence-based
    "nice": 2,     # UK evidence-based guidelines
    "cdc": 3,      # US public health
    "who": 4,      # International
    "cma": 5,      # Canadian Medical Association
    "spor": 6,     # Outcomes research
    "cco": 7,      # Cancer Care Ontario
    "icrc": 8,     # Red Cross
    "wikidoc": 9,  # Wiki-derived, lowest tier
}

# Curated best guidelines per pathology (line numbers in JSONL, 1-indexed).
# These were manually identified as the most directly relevant, highest-quality
# guidelines from the full JSONL. Used to build condensed summaries.
CURATED_GUIDELINES = {
    "appendicitis": [
        # EAES consensus on diagnosis and management (2015)
        {"line": 3436, "source": "pubmed", "label": "EAES Consensus 2015"},
        # SIFIPAC/WSES guidelines for appendicitis in elderly (2019)
        {"line": 3823, "source": "pubmed", "label": "SIFIPAC/WSES Elderly 2019"},
    ],
    "cholecystitis": [
        # 2020 WSES updated guidelines for acute calculus cholecystitis
        {"line": 3108, "source": "pubmed", "label": "WSES 2020"},
        # Diagnostic criteria and severity assessment (Tokyo Guidelines)
        {"line": 3595, "source": "pubmed", "label": "Tokyo Guidelines: Diagnostic Criteria"},
        # Revised Tokyo Guidelines: new diagnostic criteria
        {"line": 4088, "source": "pubmed", "label": "Tokyo Guidelines: Revised Criteria"},
        # Surgical treatment (Tokyo Guidelines)
        {"line": 3860, "source": "pubmed", "label": "Tokyo Guidelines: Surgical Treatment"},
        # NICE: Gallstone disease
        {"line": 2275, "source": "nice", "label": "NICE: Gallstone Disease"},
    ],
    "diverticulitis": [
        # 2020 WSES guidelines for acute colonic diverticulitis
        {"line": 3396, "source": "pubmed", "label": "WSES 2020"},
        # NICE: Diverticular disease
        {"line": 1706, "source": "nice", "label": "NICE: Diverticular Disease"},
    ],
    "pancreatitis": [
        # NICE: Pancreatitis
        {"line": 1602, "source": "nice", "label": "NICE: Pancreatitis"},
        # JPN: Diagnostic criteria
        {"line": 2984, "source": "pubmed", "label": "JPN: Diagnostic Criteria"},
        # JPN: Severity assessment
        {"line": 2997, "source": "pubmed", "label": "JPN: Severity Assessment"},
        # JPN: Medical management
        {"line": 4133, "source": "pubmed", "label": "JPN: Medical Management"},
        # JPN: Gallstone-induced pancreatitis
        {"line": 3909, "source": "pubmed", "label": "JPN: Gallstone-Induced"},
    ],
}


# ---------------------------------------------------------------------------
# Section extraction heuristics
# ---------------------------------------------------------------------------
# Headings that typically contain the most actionable content for our agent
RELEVANT_HEADING_PATTERNS = [
    r"diagnosis",
    r"diagnostic",
    r"clinical presentation",
    r"signs and symptoms",
    r"physical exam",
    r"laboratory",
    r"lab\b",
    r"imaging",
    r"radiolog",
    r"ultrasound",
    r"ct scan",
    r"severity",
    r"classification",
    r"grading",
    r"treatment",
    r"management",
    r"surgical",
    r"conservative",
    r"antibiotic",
    r"recommendation",
    r"guideline",
    r"algorithm",
    r"differential",
    r"workup",
    r"assessment",
    r"initial evaluation",
    r"emergency",
    r"acute",
]


def extract_relevant_sections(text, max_chars=8000):
    """Extract sections with headings matching diagnostic/treatment patterns.

    Returns a condensed version of the text containing only the most
    relevant sections for clinical decision-making.
    """
    # Split on markdown headings (# or ##)
    sections = re.split(r'(?m)^(#{1,3}\s+.+)$', text)

    relevant = []
    current_heading = ""

    for part in sections:
        part_stripped = part.strip()
        if not part_stripped:
            continue

        # Check if this is a heading
        if re.match(r'^#{1,3}\s+', part_stripped):
            current_heading = part_stripped
            continue

        # Check if the heading matches our patterns
        heading_lower = current_heading.lower()
        for pattern in RELEVANT_HEADING_PATTERNS:
            if re.search(pattern, heading_lower):
                section_text = f"{current_heading}\n{part_stripped}"
                # Truncate individual sections that are too long
                if len(section_text) > 3000:
                    section_text = section_text[:3000] + "\n[...truncated...]"
                relevant.append(section_text)
                break

    if not relevant:
        # Fallback: return first max_chars of the text
        return text[:max_chars] + ("\n[...truncated...]" if len(text) > max_chars else "")

    result = "\n\n".join(relevant)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n[...truncated...]"
    return result


def classify_guideline(obj, pathology):
    """Determine if a guideline is primarily about the target pathology.

    Returns:
        "primary" - disease is the main topic (in title)
        "secondary" - disease is mentioned but not the main topic
        None - not relevant
    """
    title = (obj.get("title") or "").lower()
    text = obj.get("clean_text", "").lower()

    terms = PATHOLOGIES[pathology]

    # Check primary terms in title
    for term in terms["primary"]:
        if term in title:
            return "primary"

    # Check secondary terms in title
    for term in terms["secondary"]:
        if term in title:
            return "primary"

    # Check if primary term appears frequently in text (not just a mention)
    for term in terms["primary"]:
        count = text.count(term)
        # Must appear at least 5 times to be considered relevant
        if count >= 5:
            return "secondary"

    return None


def safe_filename(title, max_len=80):
    """Convert a title to a safe filename."""
    if not title or title == "None":
        return "untitled"
    # Replace non-alphanumeric with underscores
    safe = re.sub(r'[^a-zA-Z0-9]+', '_', title)
    safe = safe.strip('_').lower()
    return safe[:max_len]


# ---------------------------------------------------------------------------
# Main parsing logic
# ---------------------------------------------------------------------------
def parse_guidelines(input_path, output_dir, summary_only=False):
    """Parse JSONL and extract per-pathology guidelines."""
    output_dir = Path(output_dir)

    # Phase 1: Scan for all relevant guidelines
    print("Phase 1: Scanning for relevant guidelines...")
    all_guidelines = {p: [] for p in PATHOLOGIES}

    with open(input_path) as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            obj["_line"] = i + 1  # 1-indexed

            for pathology in PATHOLOGIES:
                relevance = classify_guideline(obj, pathology)
                if relevance:
                    all_guidelines[pathology].append({
                        "obj": obj,
                        "relevance": relevance,
                        "line": i + 1,
                        "source": obj.get("source", ""),
                        "title": obj.get("title") or "Untitled",
                        "text_len": len(obj.get("clean_text", "")),
                    })

    # Sort: primary before secondary, then by source tier, then by text length
    for pathology in PATHOLOGIES:
        all_guidelines[pathology].sort(
            key=lambda g: (
                0 if g["relevance"] == "primary" else 1,
                SOURCE_TIER.get(g["source"], 99),
                -g["text_len"],
            )
        )

    # Print summary
    for pathology in PATHOLOGIES:
        guidelines = all_guidelines[pathology]
        primary = [g for g in guidelines if g["relevance"] == "primary"]
        secondary = [g for g in guidelines if g["relevance"] == "secondary"]
        print(f"\n  {pathology}: {len(primary)} primary, {len(secondary)} secondary")
        for g in primary[:5]:
            print(f"    [{g['source']}] {g['title'][:80]} ({g['text_len']} chars)")

    # Phase 2: Save individual guideline files (full text)
    if not summary_only:
        print("\nPhase 2: Saving individual guideline files...")
        for pathology in PATHOLOGIES:
            path_dir = output_dir / pathology / "full"
            path_dir.mkdir(parents=True, exist_ok=True)

            # Save primary guidelines from high-quality sources
            saved = 0
            for g in all_guidelines[pathology]:
                if g["relevance"] != "primary":
                    continue
                if g["source"] in ("wikidoc",):
                    continue  # Skip low-quality sources for full saves

                obj = g["obj"]
                fname = f"{g['source']}_{safe_filename(g['title'])}.md"
                fpath = path_dir / fname

                content = f"# {g['title']}\n\n"
                content += f"**Source**: {g['source']}  \n"
                if obj.get("url"):
                    content += f"**URL**: {obj['url']}  \n"
                content += f"**Length**: {g['text_len']} characters  \n"
                content += f"**JSONL line**: {g['line']}  \n\n"
                content += "---\n\n"
                content += obj.get("clean_text", "")

                fpath.write_text(content)
                saved += 1

            print(f"  {pathology}: saved {saved} full guidelines to {path_dir}")

    # Phase 3: Build condensed summaries from curated guidelines
    print("\nPhase 3: Building condensed summaries from curated guidelines...")

    # Load curated lines from JSONL
    curated_lines = set()
    for pathology, entries in CURATED_GUIDELINES.items():
        for entry in entries:
            curated_lines.add(entry["line"])

    curated_content = {}
    with open(input_path) as f:
        for i, line in enumerate(f):
            line_num = i + 1
            if line_num in curated_lines:
                curated_content[line_num] = json.loads(line)

    for pathology, entries in CURATED_GUIDELINES.items():
        path_dir = output_dir / pathology
        path_dir.mkdir(parents=True, exist_ok=True)

        summary_parts = []
        summary_parts.append(f"# Clinical Guidelines: {pathology.title()}\n")
        summary_parts.append(
            "Condensed from peer-reviewed guidelines (PubMed, NICE). "
            "Sections most relevant to diagnostic workup and treatment.\n"
        )

        for entry in entries:
            obj = curated_content.get(entry["line"])
            if not obj:
                print(f"  WARNING: line {entry['line']} not found for {pathology}")
                continue

            title = obj.get("title") or "Untitled"
            text = obj.get("clean_text", "")

            summary_parts.append(f"\n---\n\n## {entry['label']}")
            summary_parts.append(f"*{title}* ({entry['source']})\n")

            # Extract relevant sections
            extracted = extract_relevant_sections(text, max_chars=6000)
            summary_parts.append(extracted)

        summary_text = "\n".join(summary_parts)

        # Save condensed summary
        summary_path = path_dir / "summary.md"
        summary_path.write_text(summary_text)
        print(f"  {pathology}: summary.md ({len(summary_text)} chars)")

        # Also save a short version (max 3000 chars) for direct Evolver injection
        short_parts = []
        short_parts.append(f"## Clinical Guidelines: {pathology.title()}\n")
        for entry in entries:
            obj = curated_content.get(entry["line"])
            if not obj:
                continue
            text = obj.get("clean_text", "")
            extracted = extract_relevant_sections(text, max_chars=1500)
            short_parts.append(f"### {entry['label']}\n")
            short_parts.append(extracted)

        short_text = "\n".join(short_parts)
        if len(short_text) > 4000:
            short_text = short_text[:4000] + "\n[...truncated...]"

        short_path = path_dir / "evolver_context.md"
        short_path.write_text(short_text)
        print(f"  {pathology}: evolver_context.md ({len(short_text)} chars)")

    # Phase 4: Create a combined evolver context file (all pathologies)
    print("\nPhase 4: Creating combined evolver context...")
    combined_parts = [
        "# Clinical Practice Guidelines\n",
        "Evidence-based diagnostic and treatment guidelines for acute abdominal pain pathologies.\n",
        "Condensed from PubMed and NICE guidelines.\n",
    ]

    for pathology in PATHOLOGIES:
        context_path = output_dir / pathology / "evolver_context.md"
        if context_path.exists():
            combined_parts.append(f"\n{'='*60}\n")
            combined_parts.append(context_path.read_text())

    combined_text = "\n".join(combined_parts)
    combined_path = output_dir / "all_pathologies_context.md"
    combined_path.write_text(combined_text)
    print(f"  Combined context: {combined_path} ({len(combined_text)} chars)")

    # Summary stats
    print(f"\nDone. Output directory: {output_dir}")
    print(f"  Combined evolver context: {len(combined_text)} chars")


def main():
    parser = argparse.ArgumentParser(
        description="Parse open_guidelines.jsonl and extract disease-specific guidelines"
    )
    parser.add_argument(
        "--input", type=str, default="open_guidelines.jsonl",
        help="Path to open_guidelines.jsonl"
    )
    parser.add_argument(
        "--output-dir", type=str, default="guidelines",
        help="Output directory for extracted guidelines"
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Only generate condensed summaries, skip full guideline files"
    )
    args = parser.parse_args()

    parse_guidelines(args.input, args.output_dir, summary_only=args.summary_only)


if __name__ == "__main__":
    main()
