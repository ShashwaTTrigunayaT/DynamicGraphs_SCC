#!/usr/bin/env python3
"""
Convert WebGraph BV .graph files (from law.di.unimi.it) to edge list format.

Usage:
    python3 tools/convert_graph.py <graph_basename> <output_path>

Examples:
    python3 tools/convert_graph.py ~/indochina-2004 datasets/indochina-2004/refined_edges.txt
    python3 tools/convert_graph.py ~/it-2004 datasets/it-2004/refined_edges.txt

<graph_basename> = path WITHOUT .graph extension.
The .graph file and any .properties/.offsets must be alongside it.
"""
import os, sys, subprocess


def convert(basename, output_path):
    print("[INFO] Installing webgraph package...", file=sys.stderr)
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', 'webgraph', '-q'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERR] pip install failed: {r.stderr[:200]}", file=sys.stderr)
        return False

    import webgraph
    print(f"[INFO] Reading {basename}.graph ...", file=sys.stderr)
    try:
        g = webgraph.BvGraph(basename)
    except Exception as e:
        print(f"[ERR] BvGraph failed: {e}", file=sys.stderr)
        print("Try: pip install webgraph --upgrade", file=sys.stderr)
        return False

    n, e = g.num_nodes(), g.num_edges()
    print(f"[INFO] Nodes: {n}, Edges: {e:,}", file=sys.stderr)
    print(f"[INFO] Writing edges...", file=sys.stderr)

    written = 0
    with open(output_path, 'w') as f:
        for src in range(n):
            if src % 500000 == 0 and src > 0:
                print(f"[INFO] Node {src}/{n} — {written:,} edges written", file=sys.stderr)
            for dst in g.successors(src):
                f.write(f"{src} {dst}\n")
                written += 1

    print(f"[OK] Done — {written:,} edges → {output_path}", file=sys.stderr)
    return True


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    basename = sys.argv[1]
    if basename.endswith('.graph'):
        basename = basename[:-6]

    output_path = sys.argv[2]
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if convert(basename, output_path):
        return

    print("[ERR] Conversion failed.", file=sys.stderr)
    print("  Ensure the .graph file exists and try:", file=sys.stderr)
    print(f"  pip install webgraph --upgrade", file=sys.stderr)
    print(f"  python3 {sys.argv[0]} {basename} {output_path}", file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
    main()
