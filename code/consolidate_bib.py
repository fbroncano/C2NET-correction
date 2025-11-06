"""Consolidate .bib file with only citations used in manuscript."""
import re
from pathlib import Path

def parse_bib_entries(bib_content):
    """Parse .bib file and extract entries with their keys."""
    entries = {}
    current_entry = []
    current_key = None
    in_entry = False

    for line in bib_content.split('\n'):
        # Start of entry
        if line.strip().startswith('@'):
            if current_key and current_entry:
                entries[current_key] = '\n'.join(current_entry)
            current_entry = [line]
            in_entry = True
            # Extract key
            match = re.search(r'@\w+\{([^,]+),', line)
            if match:
                current_key = match.group(1).strip()
        elif in_entry:
            current_entry.append(line)
            # End of entry
            if line.strip() == '}':
                if current_key:
                    entries[current_key] = '\n'.join(current_entry)
                current_entry = []
                current_key = None
                in_entry = False

    return entries

def extract_citations_from_tex(tex_path):
    """Extract all \cite commands from .tex file."""
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all \cite{...} commands
    citations = set()
    for match in re.finditer(r'\\cite(?:\w*)\{([^}]+)\}', content):
        keys = match.group(1).split(',')
        for key in keys:
            citations.add(key.strip())

    return citations

def main():
    base_path = Path(__file__).parent

    # Read existing .bib file
    print("Reading final_60.bib...")
    with open(base_path / 'final_60.bib', 'r', encoding='utf-8') as f:
        bib_content = f.read()

    # Parse .bib entries
    bib_entries = parse_bib_entries(bib_content)
    print(f"  Found {len(bib_entries)} entries in final_60.bib")

    # Extract citations from manuscript
    print("\nExtracting citations from sn-article_revised_v02.tex...")
    cited_keys = extract_citations_from_tex(base_path / 'sn-article_revised_v02.tex')
    print(f"  Found {len(cited_keys)} unique citations")

    # List all cited keys
    print("\nCitations found in manuscript:")
    for key in sorted(cited_keys):
        if key in bib_entries:
            print(f"  ✓ {key}")
        else:
            print(f"  ✗ {key} (MISSING!)")

    # Check which citations are missing
    missing_keys = cited_keys - set(bib_entries.keys())
    if missing_keys:
        print(f"\n  WARNING: {len(missing_keys)} citations not found in .bib:")
        for key in sorted(missing_keys):
            print(f"    - {key}")
    else:
        print("\n  ✓ All citations found in .bib file!")

    # Filter .bib to only include cited entries
    print("\nCreating filtered .bib with only cited entries...")
    filtered_entries = {key: bib_entries[key] for key in cited_keys if key in bib_entries}
    print(f"  Filtered .bib contains {len(filtered_entries)} entries")

    # Write consolidated .bib
    output_path = base_path / 'final_60_consolidated.bib'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("%% Consolidated bibliography for sn-article_revised_v02.tex\n")
        f.write(f"%% Generated automatically - {len(filtered_entries)} entries\n")
        f.write("%% Only includes citations actually used in the manuscript\n\n")

        for key in sorted(filtered_entries.keys()):
            f.write(filtered_entries[key])
            f.write('\n\n')

    print(f"\nConsolidated .bib saved to: {output_path}")

    # Statistics
    print("\n=== STATISTICS ===")
    print(f"Entries in final_60.bib: {len(bib_entries)}")
    print(f"Citations in manuscript: {len(cited_keys)}")
    print(f"Missing citations: {len(missing_keys)}")
    print(f"Consolidated .bib entries: {len(filtered_entries)}")
    print(f"Reduction: {len(bib_entries) - len(filtered_entries)} unused entries removed")

if __name__ == '__main__':
    main()
