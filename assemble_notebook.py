"""Assemble all parts into the final notebook."""
import json, os, shutil

# Run part builders
exec(open('build_part1.py', encoding='utf-8').read())
exec(open('build_part2.py', encoding='utf-8').read())
exec(open('build_part3.py', encoding='utf-8').read())

# Load all cells
with open('part1_cells.json', 'r', encoding='utf-8') as f:
    cells1 = json.load(f)
with open('part2_cells.json', 'r', encoding='utf-8') as f:
    cells2 = json.load(f)
with open('part3_cells.json', 'r', encoding='utf-8') as f:
    cells3 = json.load(f)

all_cells = cells1 + cells2 + cells3

notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": all_cells
}

out_path = os.path.join('i23-2654_Assignment2_DS-A', 'i23-2654_Assignment2_DS-A.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

# Copy data files into submission folder
for src in ['cleaned.txt', 'raw.txt', 'Metadata.json']:
    if os.path.exists(src):
        shutil.copy2(src, 'i23-2654_Assignment2_DS-A/')

print(f"\nNotebook created: {out_path}")
print(f"Total cells: {len(all_cells)}")

# Cleanup temp files
for f in ['part1_cells.json', 'part2_cells.json', 'part3_cells.json']:
    if os.path.exists(f):
        os.remove(f)
print("Temp files cleaned up.")
