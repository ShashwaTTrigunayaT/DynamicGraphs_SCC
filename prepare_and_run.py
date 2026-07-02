#!/usr/bin/env python3
"""
Prepare 16 synthetic graphs for modes 5/6/11 by splitting each into
base (refined_edges.txt) + insert batches at 1%, 3%, 5%, 7%, 15%.

Then run modes 5, 6, 11 on all batches with 14 threads.
"""

import os
import sys
import random
import subprocess
import glob

BATCH_PCTS = [1, 3, 5, 7, 15]
NUM_THREADS = "14"
MODES = ["5", "6", "11"]
# Auto-detect binary and make command (cross-platform: Linux vs Windows MinGW)
import shutil
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) or "."
SCC_BINARY = shutil.which("scc.exe") or shutil.which("scc") or os.path.join(SCRIPT_DIR, "scc")
MAKE_CMD = shutil.which("mingw32-make") or shutil.which("make") or "make"

# The 16 graph files in the project root
GRAPH_FILES = sorted(glob.glob("diameter_*.txt") + glob.glob("lcc_*.txt"))

random.seed(42)  # reproducible splits


def count_total_edges(filename):
    """Count number of edge lines in a graph file (skip % comments)."""
    count = 0
    with open(filename) as f:
        for line in f:
            if line.strip() and not line.startswith("%"):
                count += 1
    return count


def read_edges_and_header(filename):
    """Read header lines and edge pairs from a graph file."""
    header = []
    edges = []
    with open(filename) as f:
        for line in f:
            if line.startswith("%"):
                header.append(line)
            elif line.strip():
                parts = line.strip().split()
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
    return header, edges


def write_graph(filepath, header, edges):
    """Write a graph file with header + edges."""
    with open(filepath, "w") as f:
        for h in header:
            f.write(h)
        for u, v in edges:
            f.write(f"{u} {v}\n")


def print_header(msg):
    print(f"\n{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}")


def build_binary():
    """Build the gm_graph library and scc binary from src/."""
    print_header("Building gm_graph library (libgmgraph.a)")
    result1 = subprocess.run(
        [MAKE_CMD, "-C", "gm_graph", "all"],
        capture_output=True, text=True, shell=True
    )
    if result1.returncode != 0:
        print("gm_graph build failed!")
        print(result1.stderr[-500:])
        # Check if lib already exists
        if os.path.isfile("gm_graph/lib/libgmgraph.a"):
            print("  But libgmgraph.a already exists, continuing...")
        else:
            return False
    else:
        print("gm_graph library built successfully.")

    print_header("Building scc binary")
    result2 = subprocess.run(
        [MAKE_CMD, "-C", "src", "all"],
        capture_output=True, text=True, shell=True
    )
    if result2.returncode != 0:
        print("Build failed!")
        print(result2.stderr[-1000:])
        return False
    # Check if binary exists
    if not os.path.isfile(SCC_BINARY):
        # Try without .exe extension
        if os.path.isfile("scc"):
            os.rename("scc", SCC_BINARY)
        else:
            print(f"ERROR: {SCC_BINARY} not found after build!")
            return False
    print("Build successful.")
    return True


def split_graphs():
    """Split all 16 graphs into base + insert batches."""
    print_header("Splitting graphs into batches")

    os.makedirs("syn_datasets", exist_ok=True)
    os.makedirs("scc_lists", exist_ok=True)

    dataset_dirs = []

    for gf in GRAPH_FILES:
        base_name = os.path.splitext(gf)[0]  # e.g., "diameter_20_1000000_100000"
        header, all_edges = read_edges_and_header(gf)
        total = len(all_edges)

        if total == 0:
            print(f"  WARNING: {gf} has no edges, skipping")
            continue

        print(f"\n  {base_name}: {total} total edges")

        for pct in BATCH_PCTS:
            # Randomly select pct% of edges as insert edges
            n_insert = max(1, int(total * pct / 100))
            indices = list(range(total))
            random.shuffle(indices)
            insert_idx = set(indices[:n_insert])

            insert_edges = [all_edges[i] for i in insert_idx]
            base_edges = [all_edges[i] for i in indices if i not in insert_idx]

            # Directory: syn_datasets/{base_name}_{pct}pct/
            dir_name = f"syn_datasets/{base_name}_{pct}pct"
            os.makedirs(dir_name, exist_ok=True)

            # Adjust headers for the new files
            base_header = []
            insert_header = []
            for h in header:
                if h.startswith("% edges:"):
                    base_header.append(f"% edges: {len(base_edges)}\n")
                    insert_header.append(f"% edges: {len(insert_edges)}\n")
                elif h.startswith("% diameter:") or h.startswith("% layers:") or \
                     h.startswith("% lcc_size:") or h.startswith("% satellite_nodes:"):
                    # Keep same structural metadata
                    base_header.append(h)
                    insert_header.append(h)
                else:
                    base_header.append(h)
                    insert_header.append(h)

            # If no % nodes / % edges header, add minimum
            has_header = any(h.startswith("% nodes:") for h in header)
            if not has_header:
                total_nodes = max(max(u, v) for u, v in all_edges) if all_edges else 0
                base_header.insert(1, f"% nodes: {total_nodes}\n")
                insert_header.insert(1, f"% nodes: {total_nodes}\n")

            write_graph(f"{dir_name}/refined_edges.txt", base_header, base_edges)
            write_graph(f"{dir_name}/insert_edges.txt", insert_header, insert_edges)

            dataset_dirs.append((dir_name, base_name, pct))
            print(f"    {pct:2d}% batch: {len(insert_edges):>8} inserts  +  "
                  f"{len(base_edges):>8} base  →  {dir_name}")

    print(f"\n  Created {len(dataset_dirs)} dataset directories.")
    return dataset_dirs


def generate_scc_lists(dataset_dirs):
    """Run mode 2 -p on each refined_edges.txt to generate SCC list, then copy to scc_lists/."""
    print_header("Generating SCC lists for all base graphs")

    for dir_name, base_name, pct in dataset_dirs:
        refined_path = os.path.abspath(f"{dir_name}/refined_edges.txt")

        # The dataset_name derived by common_main.h is the directory name
        # e.g., "dia20_1M_1pct" for "syn_datasets/dia20_1M_1pct"
        dataset_name = os.path.basename(dir_name)
        scc_list_dest = f"scc_lists/{dataset_name}.txt"

        if os.path.isfile(scc_list_dest):
            print(f"  ✓ SCC list exists for {dataset_name}, skipping")
            continue

        print(f"  Computing SCC list for {dataset_name} ...", end=" ", flush=True)
        result = subprocess.run(
            [SCC_BINARY, refined_path, "1", "2", "-p"],
            capture_output=True, text=True, timeout=1200
        )

        if result.returncode != 0:
            print(f"FAILED")
            print(f"    stderr: {result.stderr[:200]}")
            continue

        # The -p flag writes scc_list.txt in the current working directory
        if os.path.isfile("scc_list.txt"):
            os.rename("scc_list.txt", scc_list_dest)
            print(f"done → {scc_list_dest}")
        else:
            print("done but scc_list.txt not found")
            # Print output for debugging
            for line in result.stdout.split("\n"):
                if any(kw in line for kw in ["running_time", "Total # SCCs", "Error"]):
                    print(f"    {line}")


def run_benchmarks(dataset_dirs):
    """Run modes 5, 6, 11 on all insert_edges.txt files."""
    print_header("Running benchmarks: modes 5, 6, 11")

    results = []

    for dir_name, base_name, pct in dataset_dirs:
        insert_path = os.path.abspath(f"{dir_name}/insert_edges.txt")

        for mode in MODES:
            print(f"  {dir_name} | mode {mode} | {NUM_THREADS}T ...", end=" ", flush=True)
            result = subprocess.run(
                [SCC_BINARY, insert_path, NUM_THREADS, mode],
                capture_output=True, text=True, timeout=1200
            )

            # Parse result
            runtime = None
            scc_count = None
            for line in result.stdout.split("\n"):
                if "running_time(ms)=" in line:
                    runtime = float(line.split("=")[-1].strip())
                if "Total # SCCs" in line:
                    scc_count = int(line.split("=")[-1].strip())

            status = f"RT={runtime:.1f}ms" if runtime else f"FAILED"
            print(status)

            results.append({
                "dataset": base_name,
                "batch_pct": pct,
                "mode": mode,
                "threads": NUM_THREADS,
                "runtime_ms": runtime,
                "scc_count": scc_count,
            })

    # Print summary table
    print_header("SUMMARY TABLE")
    print(f"  {'Dataset':35s} {'Batch':>5s} {'Mode':>5s} {'Threads':>7s} {'Runtime(ms)':>12s} {'SCCs':>8s}")
    print(f"  {'-'*35} {'-'*5} {'-'*5} {'-'*7} {'-'*12} {'-'*8}")
    for r in results:
        rt = f"{r['runtime_ms']:.1f}" if r['runtime_ms'] else "FAIL"
        sc = str(r['scc_count']) if r['scc_count'] else "?"
        print(f"  {r['dataset']:35s} {r['batch_pct']:3d}% {r['mode']:>5s} {r['threads']:>4s}T {rt:>12s} {sc:>8s}")

    return results


def read_existing_datasets():
    """Read already-split datasets from syn_datasets/ without re-splitting."""
    dataset_dirs = []
    for d in sorted(glob.glob("syn_datasets/*/")):
        refined = f"{d}refined_edges.txt"
        insert = f"{d}insert_edges.txt"
        if os.path.isfile(refined) and os.path.isfile(insert):
            dir_name = d.rstrip("/")
            # Parse: syn_datasets/{base_name}_{pct}pct
            parts = os.path.basename(dir_name).rsplit("_", 1)
            if len(parts) == 2 and parts[1].endswith("pct"):
                base_name = parts[0]
                pct = int(parts[1].replace("pct", ""))
                dataset_dirs.append((dir_name, base_name, pct))
    print(f"  Found {len(dataset_dirs)} existing datasets in syn_datasets/")
    return dataset_dirs


if __name__ == "__main__":
    if not os.path.isfile(SCC_BINARY):
        print(f"Building {SCC_BINARY}...")
        if not build_binary():
            sys.exit(1)
    else:
        print(f"{SCC_BINARY} already exists, skipping build.")

    # Step 1: Split all graphs
    dataset_dirs = split_graphs()

    if not dataset_dirs:
        print("No datasets created. Exiting.")
        sys.exit(1)

    # Step 2: Generate SCC lists (needed for modes 6 and 11)
    generate_scc_lists(dataset_dirs)

    # Step 3: Run benchmarks
    run_benchmarks(dataset_dirs)

    print("\nDone! All benchmarks complete.")
