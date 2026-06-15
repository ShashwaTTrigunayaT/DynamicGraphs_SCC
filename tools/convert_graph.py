#!/usr/bin/env python3
"""
Convert WebGraph BV .graph files (from law.di.unimi.it) to edge list format.

Strategy (tried in order):
  1. Pure-Python decoder — reads .graph + .offsets (or .ef) directly, no Java needed.
  2. Python 'webgraph' package — works when .ef offset file is present; if only
     .offsets exists, auto-generates .ef via Java then retries.
  3. Java WebGraph JAR — dumps edges via BVGraph / ArcListASCIIGraph if Java
     and the fat-jar are available.

Usage:
    python3 tools/convert_graph.py <graph_basename> <output_path> [--jar path/to/webgraph-deps.jar]

Examples:
    python3 tools/convert_graph.py ~/indochina-2004 datasets/indochina-2004/refined_edges.txt
    python3 tools/convert_graph.py ~/it-2004       datasets/it-2004/refined_edges.txt
    python3 tools/convert_graph.py ~/it-2004       datasets/it-2004/refined_edges.txt \\
            --jar ~/webgraph-deps.jar

<graph_basename> = path WITHOUT .graph extension.
The .graph file and .properties / .offsets must be in the same directory.
"""

import os
import sys
import struct
import subprocess
import shutil
import urllib.request
import argparse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WEBGRAPH_JAR_URL = (
    "https://repo1.maven.org/maven2/it/unimi/dsi/webgraph/3.6.10/"
    "webgraph-3.6.10-deps.jar"
)
DEFAULT_JAR = os.path.expanduser("~/webgraph-deps.jar")


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _read_properties(basename):
    """Return dict from .properties file (Java key=value format)."""
    props = {}
    path = basename + ".properties"
    if not os.path.exists(path):
        return props
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                props[k.strip()] = v.strip()
    return props


# ---------------------------------------------------------------------------
# Strategy 1 — Pure Python BVGraph decoder
# ---------------------------------------------------------------------------

class _BitReader:
    """Reads bits from a byte buffer with minimal overhead."""

    __slots__ = ("_buf", "_pos", "_bit")

    def __init__(self, buf):
        self._buf = buf
        self._pos = 0   # current byte index
        self._bit = 0   # bits consumed in current byte (0-7)

    # ---- primitives --------------------------------------------------------

    def read_bit(self):
        if self._pos >= len(self._buf):
            raise EOFError("unexpected end of bitstream")
        b = (self._buf[self._pos] >> (7 - self._bit)) & 1
        self._bit += 1
        if self._bit == 8:
            self._bit = 0
            self._pos += 1
        return b

    def read_bits(self, n):
        """Read n bits as an integer (MSB first)."""
        val = 0
        for _ in range(n):
            val = (val << 1) | self.read_bit()
        return val

    def bit_position(self):
        return self._pos * 8 + self._bit

    def seek_bit(self, pos):
        self._pos = pos >> 3
        self._bit = pos & 7

    # ---- codes -------------------------------------------------------------

    def read_unary(self):
        """Read unary-coded integer (number of 0-bits before a 1-bit)."""
        count = 0
        while self.read_bit() == 0:
            count += 1
        return count

    def read_gamma(self):
        """Elias gamma code."""
        b = self.read_unary()
        if b == 0:
            return 0
        return (1 << b) + self.read_bits(b) - 1

    def read_zeta(self, k=3):
        """Elias zeta-k code (default k=3 as used by BVGraph)."""
        h = self.read_unary()
        left = 1 << (h * k)
        m = self.read_bits(h * k + k)
        if m < left:
            return left + m - 1
        else:
            return (m << 1) | self.read_bit()

    def read_nibble(self):
        """Read a nibble-coded integer."""
        val = 0
        shift = 0
        while True:
            nib = self.read_bits(4)
            val |= (nib & 0x7) << shift
            shift += 3
            if nib & 0x8:
                break
        return val


def _read_offsets(basename):
    """
    Load node bit-offsets from .offsets file (gamma-coded bitstream).
    Returns list of length n+1 where offsets[i] = bit position of node i.
    """
    off_path = basename + ".offsets"
    if not os.path.exists(off_path):
        return None
    _log(f"[INFO] Reading offsets from {off_path} ...")
    with open(off_path, "rb") as f:
        data = f.read()
    br = _BitReader(data)
    offsets = []
    cur = 0
    try:
        while True:
            delta = br.read_gamma()
            cur += delta
            offsets.append(cur)
    except (EOFError, IndexError):
        pass
    _log(f"[INFO] Loaded {len(offsets):,} node offsets")
    return offsets


def _decode_node_list(br, last_node, zeta_k):
    """
    Decode one adjacency list from the bitstream.
    Returns list of successor node IDs.
    """
    # outdegree
    outdegree = br.read_gamma()
    if outdegree == 0:
        return []

    # reference window
    ref = br.read_unary()           # reference back-pointer (0 = no reference)
    # For simplicity we don't handle copy blocks in this decoder — works for
    # most BVGraph files that were compressed with ref=0 (or small ref).
    # Full reference-window handling is complex; fall back to Java if needed.
    if ref != 0:
        return None   # signal caller to use Java fallback

    # extra nodes (interval + residuals)
    # interval count
    interval_count = br.read_gamma()
    nodes = []
    prev = last_node
    for _ in range(interval_count):
        left = br.read_gamma() + prev + 1   # left endpoint (delta from prev)
        length = br.read_gamma() + 2        # interval length
        for v in range(left, left + length):
            nodes.append(v)
        prev = left + length - 1

    # residuals
    residual_count = outdegree - sum(1 for _ in nodes)  # already counted intervals
    residual_count = outdegree - len(nodes)
    if residual_count < 0:
        return None
    prev2 = last_node
    for i in range(residual_count):
        if i == 0:
            delta = br.read_zeta(zeta_k)
            v = prev2 + delta if delta >= 0 else prev2 + delta
            # First residual is signed: actually it is (v - last_node) as zeta
            # but may be negative — stored as 2*|d|-1 if neg or 2*d if pos.
            # Re-read using the signed mapping:
            pass
        # Simpler: residuals are stored as consecutive zeta-coded deltas
        # with the first one relative to last_node (signed).
        delta = br.read_zeta(zeta_k)
        # map from natural to signed: even → positive, odd → negative
        if delta % 2 == 0:
            signed = delta // 2
        else:
            signed = -(delta + 1) // 2
        v = prev2 + signed if i == 0 else prev2 + delta + 1
        nodes.append(v)
        prev2 = v

    nodes.sort()
    return nodes


def _try_pure_python(basename, output_path):
    """
    Attempt pure-Python BVGraph decoding using .graph + .offsets.
    Returns True on success, False if the format is too complex (use Java).
    """
    graph_path = basename + ".graph"
    if not os.path.exists(graph_path):
        _log(f"[ERR] {graph_path} not found")
        return False

    props = _read_properties(basename)
    zeta_k = int(props.get("zetak", 3))
    n = int(props.get("nodes", 0))
    if n == 0:
        _log("[WARN] Could not read node count from .properties; skipping pure-Python path")
        return False

    offsets = _read_offsets(basename)
    if offsets is None:
        _log("[WARN] No .offsets file found; skipping pure-Python path")
        return False

    _log(f"[INFO] Pure-Python decode: {n:,} nodes, zeta_k={zeta_k}")
    _log(f"[INFO] Reading {graph_path} ...")

    with open(graph_path, "rb") as f:
        data = f.read()

    written = 0
    fallback_needed = False

    with open(output_path, "w", buffering=1 << 20) as out:
        br = _BitReader(data)
        for src in range(n):
            if src % 500_000 == 0 and src > 0:
                _log(f"[INFO] Node {src:,}/{n:,} — {written:,} edges written")

            if src < len(offsets):
                br.seek_bit(offsets[src])

            # Outdegree only (simple path — no reference compression)
            outdegree = br.read_gamma()
            if outdegree == 0:
                continue

            # Check if reference-compressed (ref > 0)
            ref = br.read_unary()
            if ref != 0:
                _log(f"[WARN] Node {src} uses reference compression (ref={ref}). "
                     "Pure-Python decoder does not support this. Switching to Java.")
                fallback_needed = True
                break

            # intervals
            interval_count = br.read_gamma()
            succs = []
            prev = src
            for _ in range(interval_count):
                left = br.read_gamma() + prev + 1
                length = br.read_gamma() + 2
                succs.extend(range(left, left + length))
                prev = left + length - 1

            # residuals
            residual_count = outdegree - len(succs)
            prev2 = src
            for i in range(residual_count):
                raw = br.read_zeta(zeta_k)
                if i == 0:
                    # first residual: signed encoding
                    if raw % 2 == 0:
                        v = prev2 + raw // 2
                    else:
                        v = prev2 - (raw + 1) // 2
                else:
                    v = prev2 + raw + 1
                succs.append(v)
                prev2 = v

            succs.sort()
            for dst in succs:
                out.write(f"{src} {dst}\n")
                written += 1

    if fallback_needed:
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

    _log(f"[OK] Pure-Python done — {written:,} edges → {output_path}")
    return True


# ---------------------------------------------------------------------------
# Strategy 2 — Python 'webgraph' package
# ---------------------------------------------------------------------------

def _ensure_ef(basename):
    """
    If .ef is missing but .offsets exists, generate .ef via Java WebGraph.
    Returns True if .ef is ready, False otherwise.
    """
    ef_path = basename + ".ef"
    if os.path.exists(ef_path):
        return True

    off_path = basename + ".offsets"
    if not os.path.exists(off_path):
        _log(f"[WARN] No .offsets file at {off_path}; cannot generate .ef")
        return False

    jar = _find_or_download_jar()
    if jar is None:
        return False

    java = shutil.which("java")
    if java is None:
        _log("[WARN] Java not found; cannot generate .ef file")
        return False

    _log(f"[INFO] Generating {ef_path} via Java BVGraph --list ...")
    cmd = [java, "-cp", jar, "it.unimi.dsi.webgraph.BVGraph", "--list", basename]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        _log(f"[WARN] Java --list failed:\n{r.stderr[:400]}")
        return False

    if os.path.exists(ef_path):
        _log(f"[INFO] .ef file generated: {ef_path}")
        return True

    _log("[WARN] .ef file still missing after Java --list")
    return False


def _try_webgraph_package(basename, output_path):
    """Try the pip 'webgraph' package (needs .ef offsets)."""
    _log("[INFO] Trying Python 'webgraph' package ...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "webgraph", "-q", "--upgrade"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        _log(f"[WARN] pip install webgraph failed: {r.stderr[:200]}")
        return False

    # Make sure .ef exists
    if not _ensure_ef(basename):
        _log("[WARN] .ef not available; skipping webgraph package")
        return False

    try:
        import importlib
        import webgraph as wg
        importlib.reload(wg)
    except ImportError:
        _log("[WARN] Could not import webgraph after install")
        return False

    try:
        g = wg.BvGraph(basename)
    except Exception as e:
        _log(f"[WARN] webgraph.BvGraph failed: {e}")
        return False

    n = g.num_nodes()
    _log(f"[INFO] webgraph: {n:,} nodes, {g.num_edges():,} edges")

    written = 0
    with open(output_path, "w", buffering=1 << 20) as f:
        for src in range(n):
            if src % 500_000 == 0 and src > 0:
                _log(f"[INFO] Node {src:,}/{n:,} — {written:,} edges written")
            for dst in g.successors(src):
                f.write(f"{src} {dst}\n")
                written += 1

    _log(f"[OK] webgraph package done — {written:,} edges → {output_path}")
    return True


# ---------------------------------------------------------------------------
# Strategy 3 — Java WebGraph JAR
# ---------------------------------------------------------------------------

def _find_or_download_jar(jar_hint=None):
    """Return path to webgraph fat-jar, downloading if necessary."""
    candidates = [jar_hint, DEFAULT_JAR] + [
        p for p in (
            os.path.join(os.getcwd(), "webgraph-deps.jar"),
            os.path.join(os.path.dirname(__file__), "webgraph-deps.jar"),
        )
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p

    _log(f"[INFO] Downloading WebGraph JAR from Maven Central ...")
    _log(f"       {WEBGRAPH_JAR_URL}")
    try:
        urllib.request.urlretrieve(WEBGRAPH_JAR_URL, DEFAULT_JAR)
        _log(f"[INFO] JAR saved to {DEFAULT_JAR}")
        return DEFAULT_JAR
    except Exception as e:
        _log(f"[WARN] Could not download JAR: {e}")
        return None


def _try_java(basename, output_path, jar_hint=None):
    """Dump edges via Java WebGraph BVGraph."""
    java = shutil.which("java")
    if java is None:
        _log("[WARN] Java not found in PATH; skipping Java strategy")
        return False

    jar = _find_or_download_jar(jar_hint)
    if jar is None:
        _log("[WARN] No WebGraph JAR available; skipping Java strategy")
        return False

    _log(f"[INFO] Trying Java WebGraph (jar={jar}) ...")

    # BVGraph can write an ASCII arc-list to stdout with -o -O -L flags
    cmd = [
        java, "-Xmx4g",
        "-cp", jar,
        "it.unimi.dsi.webgraph.BVGraph",
        "-o", "-O", "-L",   # offline / sequential / arc-list mode
        basename,
    ]
    _log(f"[INFO] Running: {' '.join(cmd)}")

    written = 0
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        with open(output_path, "w", buffering=1 << 20) as out:
            for line in proc.stdout:
                out.write(line)
                written += 1
                if written % 5_000_000 == 0:
                    _log(f"[INFO] {written:,} edges written ...")
        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read()
            _log(f"[WARN] Java BVGraph exited {proc.returncode}:\n{err[:400]}")
            return False
    except Exception as e:
        _log(f"[WARN] Java strategy failed: {e}")
        return False

    _log(f"[OK] Java done — {written:,} edges → {output_path}")
    return True


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def convert(basename, output_path, jar_hint=None):
    graph_path = basename + ".graph"
    if not os.path.exists(graph_path):
        _log(f"[ERR] Graph file not found: {graph_path}")
        return False

    props = _read_properties(basename)
    _log(f"[INFO] Graph properties: nodes={props.get('nodes','?')}, "
         f"arcs={props.get('arcs','?')}, version={props.get('version','?')}, "
         f"zetak={props.get('zetak','3')}, windowsize={props.get('windowsize','?')}")

    window_size = int(props.get("windowsize", 1))

    # Strategy 1: pure Python (only safe when windowsize == 0, i.e. no references)
    if window_size == 0:
        _log("[INFO] windowsize=0 — attempting pure-Python decoder ...")
        if _try_pure_python(basename, output_path):
            return True
    else:
        _log(f"[INFO] windowsize={window_size} > 0 — skipping pure-Python decoder "
             "(reference compression requires Java)")

    # Strategy 2: Python webgraph package
    if _try_webgraph_package(basename, output_path):
        return True

    # Strategy 3: Java
    if _try_java(basename, output_path, jar_hint=jar_hint):
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert WebGraph BV .graph files to edge list format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("basename",
                        help="Path to graph WITHOUT .graph extension")
    parser.add_argument("output_path",
                        help="Output edge-list file path")
    parser.add_argument("--jar", default=None,
                        help=f"Path to webgraph fat-jar "
                             f"(downloaded automatically to {DEFAULT_JAR} if absent)")
    args = parser.parse_args()

    basename = args.basename
    if basename.endswith(".graph"):
        basename = basename[:-6]

    output_path = args.output_path
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    _log(f"[INFO] basename   : {basename}")
    _log(f"[INFO] output     : {output_path}")

    if convert(basename, output_path, jar_hint=args.jar):
        sys.exit(0)

    _log("[ERR] All conversion strategies failed.")
    _log("  Check that the .graph and .properties files exist.")
    _log("  If Java is installed, make sure it is on your PATH.")
    _log(f"  You can supply the JAR manually: --jar /path/to/webgraph-deps.jar")
    sys.exit(1)


if __name__ == "__main__":
    main()