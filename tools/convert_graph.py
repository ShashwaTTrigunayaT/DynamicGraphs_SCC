#!/usr/bin/env python3
"""
Convert WebGraph BV .graph files (from law.di.unimi.it) to plain edge list format.

Usage:
    python3 tools/convert_graph.py <graph_basename> <output_path>

Examples:
    python3 tools/convert_graph.py ~/it-2004 datasets/it-2004/refined_edges.txt
    python3 tools/convert_graph.py ~/indochina-2004 datasets/indochina-2004/refined_edges.txt

The <graph_basename> should point to the .graph file WITHOUT extension.
The script auto-downloads missing .properties and .offsets files from LAW.
"""
import os, sys, struct, urllib.request, subprocess


def download_missing_files(basename):
    """Download .properties and .offsets files if missing."""
    base_name = os.path.basename(basename)
    ok = True
    for ext in ['.properties', '.offsets']:
        path = basename + ext
        if not os.path.exists(path):
            url = f"http://data.law.di.unimi.it/webdata/{base_name}/{base_name}{ext}"
            print(f"[DL] {url}", file=sys.stderr)
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"[ERR] {e}", file=sys.stderr)
                ok = False
    return ok


def convert_with_webgraph(basename, output_path):
    """Convert using the webgraph Python library (pip install webgraph)."""
    print("[INFO] Installing webgraph package...", file=sys.stderr)
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', 'webgraph', '-q'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERR] pip install failed: {r.stderr[:200]}", file=sys.stderr)
        return False

    import webgraph
    print(f"[INFO] Reading {basename}.graph ...", file=sys.stderr)
    g = webgraph.Graph(basename)
    n = g.num_nodes()

    print(f"[INFO] Nodes: {n}, writing edges...", file=sys.stderr)
    with open(output_path, 'w') as f:
        for src in range(n):
            for dst in g.successors(src):
                f.write(f"{src} {dst}\n")
    print(f"[OK] Wrote to {output_path}", file=sys.stderr)
    return True


def convert_java(basename, output_path):
    """Convert using the Java WebGraph tool (if Java is available)."""
    # Check if java is available
    r = subprocess.run(['which', 'java'], capture_output=True)
    if r.returncode != 0:
        return False

    base = os.path.basename(basename)
    # Try to download the WebGraph jar
    jar_url = "https://repo1.maven.org/maven2/it/unimi/dsi/webgraph/webgraph/3.6.9/webgraph-3.6.9.jar"
    deps = [
        "https://repo1.maven.org/maven2/it/unimi/dsi/fastutil/8.5.15/fastutil-8.5.15.jar",
        "https://repo1.maven.org/maven2/com/martinkl/colt/1.2.0/colt-1.2.0.jar",
        "https://repo1.maven.org/maven2/org/slf4j/slf4j-api/2.0.16/slf4j-api-2.0.16.jar",
        "https://repo1.maven.org/maven2/org/slf4j/slf4j-simple/2.0.16/slf4j-simple-2.0.16.jar",
    ]

    lib_dir = os.path.join(os.path.dirname(__file__) or '.', 'lib')
    os.makedirs(lib_dir, exist_ok=True)
    jars = []
    for url in [jar_url] + deps:
        jar = os.path.join(lib_dir, os.path.basename(url))
        if not os.path.exists(jar):
            print(f"[DL] {os.path.basename(url)}", file=sys.stderr)
            urllib.request.urlretrieve(url, jar)
        jars.append(jar)

    cp = ':'.join(jars)
    cmd = ['java', '-cp', cp,
           'it.unimi.dsi.webgraph.BVGraph',
           '-g', base, output_path]
    print(f"[JAVA] {' '.join(cmd)}", file=sys.stderr)
    # Note: BVGraph -g outputs in WebGraph ASCII format, not edge list
    # We'd need additional parsing — this is complex
    return False


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    basename = sys.argv[1]
    if basename.endswith('.graph'):
        basename = basename[:-6]
    
    if sys.argv[2] == '-':
        output_path = '/dev/stdout'
    else:
        output_path = sys.argv[2]
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    download_missing_files(basename)

    if convert_with_webgraph(basename, output_path):
        return

    print("[ERR] Could not convert. Try installing webgraph manually:", file=sys.stderr)
    print(f"  pip install webgraph", file=sys.stderr)
    print(f"  python3 tools/convert_graph.py {basename} {output_path}", file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
    main()
