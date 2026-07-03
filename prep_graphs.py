import os
import random
import shutil

random.seed(42)

GRAPH_FILES = [
    "diameter_20_1000000_100000.txt",
    "diameter_40_1000000_100000.txt",
    "diameter_60_1000000_100000.txt",
    "diameter_100_1000000_100000.txt",
    "diameter_20_3000000_300000.txt",
    "diameter_40_3000000_300000.txt",
    "diameter_60_3000000_300000.txt",
    "diameter_100_3000000_300000.txt",
    "lcc_30pct_1000000_10000000.txt",
    "lcc_40pct_1000000_10000000.txt",
    "lcc_50pct_1000000_10000000.txt",
    "lcc_70pct_1000000_10000000.txt",
    "lcc_30pct_3000000_98000000.txt",
    "lcc_40pct_3000000_98000000.txt",
    "lcc_50pct_3000000_98000000.txt",
    "lcc_70pct_3000000_98000000.txt",
]

# Clean and recreate
if os.path.exists("syn_datasets"):
    shutil.rmtree("syn_datasets")
os.makedirs("syn_datasets", exist_ok=True)

INSERT_FRAC = 0.10  # 10% of edges go to insert set

for fname in GRAPH_FILES:
    # Derive dataset name from filename (remove .txt)
    dataset_name = fname.replace(".txt", "")
    dataset_dir = f"syn_datasets/{dataset_name}"
    os.makedirs(dataset_dir, exist_ok=True)

    # Read the graph file
    with open(fname, "r") as f:
        lines = f.readlines()

    # Separate header lines and edge lines
    header_lines = []
    edge_lines = []
    for line in lines:
        line_stripped = line.strip()
        if line_stripped == "" or line_stripped.startswith("%"):
            header_lines.append(line)
        else:
            edge_lines.append(line)

    # Shuffle edges and split
    random.shuffle(edge_lines)
    split_idx = int(len(edge_lines) * (1 - INSERT_FRAC))
    base_edges = edge_lines[:split_idx]
    insert_edges = edge_lines[split_idx:]

    # Sort each set back to original order for determinism
    base_edges.sort(key=lambda x: (int(x.split()[0]), int(x.split()[1])))
    insert_edges.sort(key=lambda x: (int(x.split()[0]), int(x.split()[1])))

    # Write refined_edges.txt (base graph)
    refined_path = f"{dataset_dir}/refined_edges.txt"
    with open(refined_path, "w") as f:
        for h in header_lines:
            f.write(h)
        # Update edge count header
        f.write(f"% base_edges: {len(base_edges)}\n")
        f.write(f"% insert_edges: {len(insert_edges)}\n")
        for e in base_edges:
            f.write(e)

    # Write insert_edges.txt (edges to insert)
    insert_path = f"{dataset_dir}/insert_edges.txt"
    with open(insert_path, "w") as f:
        for h in header_lines:
            f.write(h)
        f.write(f"% base_edges: {len(base_edges)}\n")
        f.write(f"% insert_edges: {len(insert_edges)}\n")
        for e in insert_edges:
            f.write(e)

    print(f"  {dataset_name:45s} | total: {len(edge_lines):>9,} | base: {len(base_edges):>9,} | insert: {len(insert_edges):>9,} | size: {os.path.getsize(refined_path)/1024/1024:.0f}MB + {os.path.getsize(insert_path)/1024/1024:.0f}MB")

print("\nDone! Created syn_datasets/ with 16 subdirectories.")
print("Next: Build the scc binary, then run the automation script.")
