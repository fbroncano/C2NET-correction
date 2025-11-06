"""Extract references from Word documents and consolidate with existing .bib file."""
import docx
import re
from pathlib import Path

def extract_references_from_docx(docx_path):
    """Extract reference section from Word document."""
    doc = docx.Document(docx_path)
    references = []
    in_references = False

    for para in doc.paragraphs:
        text = para.text.strip()

        # Detect reference section start
        if text.lower() in ['references', 'bibliography', 'referencias', 'bibliografía']:
            in_references = True
            continue

        # Extract references
        if in_references and text:
            references.append(text)

    return references

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

    # Extract references from Word documents
    print("Extracting references from Art1.docx...")
    try:
        refs1 = extract_references_from_docx(base_path / 'Art1.docx')
        print(f"  Found {len(refs1)} references in Art1.docx")
    except Exception as e:
        print(f"  Error reading Art1.docx: {e}")
        refs1 = []

    print("\nExtracting references from Art2.docx...")
    try:
        refs2 = extract_references_from_docx(base_path / 'Art2.docx')
        print(f"  Found {len(refs2)} references in Art2.docx")
    except Exception as e:
        print(f"  Error reading Art2.docx: {e}")
        refs2 = []

    # Save extracted references to text file for manual review
    with open(base_path / 'extracted_references.txt', 'w', encoding='utf-8') as f:
        f.write("=== REFERENCES FROM ART1.DOCX ===\n\n")
        for ref in refs1:
            f.write(f"{ref}\n\n")
        f.write("\n\n=== REFERENCES FROM ART2.DOCX ===\n\n")
        for ref in refs2:
            f.write(f"{ref}\n\n")

    print(f"\nExtracted references saved to: extracted_references.txt")

    # Read existing .bib file
    print("\nReading final_60.bib...")
    with open(base_path / 'final_60.bib', 'r', encoding='utf-8') as f:
        bib_content = f.read()

    # Parse .bib entries
    bib_entries = parse_bib_entries(bib_content)
    print(f"  Found {len(bib_entries)} entries in final_60.bib")

    # Extract citations from manuscript
    print("\nExtracting citations from sn-article_revised_v02.tex...")
    cited_keys = extract_citations_from_tex(base_path / 'sn-article_revised_v02.tex')
    print(f"  Found {len(cited_keys)} unique citations")

    # Check which citations are missing
    missing_keys = cited_keys - set(bib_entries.keys())
    if missing_keys:
        print(f"\n  WARNING: {len(missing_keys)} citations not found in .bib:")
        for key in sorted(missing_keys):
            print(f"    - {key}")

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
    print(f"References in Art1.docx: {len(refs1)}")
    print(f"References in Art2.docx: {len(refs2)}")
    print(f"Entries in final_60.bib: {len(bib_entries)}")
    print(f"Citations in manuscript: {len(cited_keys)}")
    print(f"Missing citations: {len(missing_keys)}")
    print(f"Consolidated .bib entries: {len(filtered_entries)}")

if __name__ == '__main__':
    main()
