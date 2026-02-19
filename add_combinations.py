#!/usr/bin/env python3
"""
Reads a datafile (3 header lines + one animal per line),
appends all pairwise combinations like "a hedgehog and an octopus".
Usage: python add_combinations.py <input.tsv>
"""
import sys
from itertools import combinations

if len(sys.argv) < 2:
    print("Usage: python add_combinations.py <input.tsv>")
    sys.exit(1)

path = sys.argv[1]

with open(path, encoding="utf-8") as f:
    header_lines = [f.readline() for _ in range(3)]
    animals = [line.strip() for line in f if line.strip()]

# Generate all pairwise combinations in both orders
pairs = []
for a, b in combinations(animals, 2):
    pairs.append(f"{a} and {b}")
    pairs.append(f"{b} and {a}")

# Write back: headers + individual animals + combinations
with open(path, "w", encoding="utf-8") as f:
    for h in header_lines:
        f.write(h)
    for animal in animals:
        f.write(animal + "\n")
    for pair in pairs:
        f.write(pair + "\n")

print(f"Added {len(pairs)} combinations of {len(animals)} animals.", file=sys.stderr)
