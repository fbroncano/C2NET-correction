#!/usr/bin/env python3
"""
Script to fix float positions in LaTeX document.
Changes all [!t] and [ht] to [!htbp] for better float placement.
"""

import re

# Read file
with open('sn-article_revised_v02.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Count changes
count_fig_t = len(re.findall(r'\\begin{figure}\[!t\]', content))
count_tab_ht = len(re.findall(r'\\begin{table}\[ht\]', content))
count_tab_t = len(re.findall(r'\\begin{table}\[!t\]', content))
count_tab_star = len(re.findall(r'\\begin{table\*}\[!t\]', content))

print(f"Found:")
print(f"  - {count_fig_t} figures with [!t]")
print(f"  - {count_tab_ht} tables with [ht]")
print(f"  - {count_tab_t} tables with [!t]")
print(f"  - {count_tab_star} table* with [!t]")

# Apply replacements
content = re.sub(r'\\begin{figure}\[!t\]', r'\\begin{figure}[!htbp]', content)
content = re.sub(r'\\begin{table}\[ht\]', r'\\begin{table}[!htbp]', content)
content = re.sub(r'\\begin{table}\[!t\]', r'\\begin{table}[!htbp]', content)
content = re.sub(r'\\begin{table\*}\[!t\]', r'\\begin{table*}[!htbp]', content)

# Write back
with open('sn-article_revised_v02.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ Changes applied successfully!")
print("All floats now use [!htbp] for flexible positioning")
