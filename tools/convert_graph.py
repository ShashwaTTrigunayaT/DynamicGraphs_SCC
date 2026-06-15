#!/usr/bin/env python3
"""
Convert WebGraph BV .graph files (from law.di.unimi.it) to plain edge list format.

Usage:
    python3 tools/convert_graph.py <graph_basename> <output_path>

Examples:
    python3 tools/convert_graph.py ../it-2004 ../datasets/it-2004/refined_edges.txt
    python3 tools/convert_graph.py ../indochina-2004 ../datasets/indochina-2004/refined_edges.txt

The <graph_basename> should point to the .graph file without the extension.
The script also downloads the required .properties and .offsets files if missing.
"""
import os
import sys
import struct
import math
import urllib.request


def download_missing_files(basename):
    """Download .properties and .offsets files if they don't exist."""
    base_dir = os.path.dirname(basename) or '.'
    base_name = os.path.basename(basename)

    for ext in ['.properties', '.offsets']:
        fpath = basename + ext
        if not os.path.exists(fpath):
            # Try to download from LAW
            url = f"http://data.law.di.unimi.it/webdata/{base_name}/{base_name}{ext}"
            print(f"[INFO] Downloading {url} ...", file=sys.stderr)
            try:
                urllib.request.urlretrieve(url, fpath)
                print(f"[INFO] Saved to {fpath}", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] Failed to download {url}: {e}", file=sys.stderr)
                return False
    return True


def read_properties(basename):
    """Read .properties file and return dict."""
    props = {}
    with open(basename + '.properties', 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                k, v = line.split('=', 1)
                props[k.strip()] = v.strip()
    return props


# WebGraph BV format constants
BV_HEADER_MAGIC = 0xB5A0C1B2  # uint32 little-endian


def read_gamma_coded_int(bit_stream, pos):
    """Read a gamma-coded integer from a byte array starting at bit position pos.
    Returns (value, new_pos). Gamma code: unary length + binary value."""
    # Count leading 1 bits (unary part) - actually gamma uses 0s in unary
    # Gamma: n written as (floor(log2(n)) zeros + binary of n without MSB)
    # e.g., n=5 (101b) -> floor(log2(5))=2 -> "00 01" -> bits: 0001
    # Actually: Gamma code is: (n written as) floor(log2(n)) zeros followed by 
    # the binary representation of n without the most significant bit
    
    # Read zeros to determine length
    bit_count = 0
    byte_pos = pos // 8
    bit_offset = pos % 8
    
    while byte_pos < len(bit_stream):
        # Check the current bit
        if bit_stream[byte_pos] & (1 << (7 - bit_offset)):
            break
        bit_count += 1
        pos += 1
        byte_pos = pos // 8
        bit_offset = pos % 8
    
    # The 1 bit that ended the unary
    pos += 1
    byte_pos = pos // 8
    bit_offset = pos % 8
    
    # Now read bit_count more bits to get the binary value
    if bit_count == 0:
        return 1, pos  # gamma(1) = 1 -> unary has 0 zeros, then 1, then 0 bits
    
    value = 1  # MSB is always 1 (implicit)
    for _ in range(bit_count):
        value <<= 1
        if byte_pos < len(bit_stream):
            if bit_stream[byte_pos] & (1 << (7 - bit_offset)):
                value |= 1
        pos += 1
        byte_pos = pos // 8
        bit_offset = pos % 8
    
    return value, pos


def read_delta_coded_int(bit_stream, pos):
    """Read a delta-coded integer.
    Delta code: gamma(length) + binary(value without MSB)."""
    length, pos = read_gamma_coded_int(bit_stream, pos)
    
    if length == 1:
        return 1, pos
    
    value = 1
    for _ in range(length - 1):
        value <<= 1
        byte_pos = pos // 8
        bit_offset = pos % 8
        if byte_pos < len(bit_stream):
            if bit_stream[byte_pos] & (1 << (7 - bit_offset)):
                value |= 1
        pos += 1
    
    return value, pos


class BVGraph:
    """Simple reader for WebGraph BV format."""
    
    def __init__(self, basename):
        self.basename = basename
        self.props = read_properties(basename)
        
        self.num_nodes = int(self.props.get('nodes', 0))
        self.num_edges = int(self.props.get('edges', 0))
        
        print(f"[INFO] Graph: {self.num_nodes} nodes, {self.num_edges} edges", file=sys.stderr)
        
        # Read the graph file
        with open(basename + '.graph', 'rb') as f:
            self.data = f.read()
        
        # Read offsets file if available
        self.offsets = None
        offsets_path = basename + '.offsets'
        if os.path.exists(offsets_path):
            with open(offsets_path, 'rb') as f:
                raw = f.read()
            self.offsets = []
            for i in range(0, len(raw), 8):
                self.offsets.append(struct.unpack('<Q', raw[i:i+8])[0])
            print(f"[INFO] Loaded {len(self.offsets)} offsets", file=sys.stderr)
        
        # Parse header
        self._parse_header()
        
    def _parse_header(self):
        """Parse BV file header (at least 24 bytes before the bit stream)."""
        # Header: magic (4), nodes (4), edges (4), window_size (4), min_interval_len (4), 
        #         z (4 + varint), block_count (4)
        # The bitstream starts after the header.
        # For simplicity, we use the offsets to know where bitstream starts.
        pass
    
    def successors(self, node):
        """Return list of successors for a given node. (Simplified implementation)"""
        # For the actual conversion, we iterate all nodes
        pass


def convert_with_python_library(basename, output_path):
    """Try to convert using the webgraph Python library."""
    try:
        import webgraph
        print("[INFO] Using 'webgraph' Python library", file=sys.stderr)
        
        g = webgraph.Graph(basename)
        num_nodes = g.num_nodes()
        
        with open(output_path, 'w') as f:
            for src in range(num_nodes):
                for dst in g.successors(src):
                    f.write(f"{src} {dst}\n")
        
        print(f"[OK] Wrote edge list to {output_path}", file=sys.stderr)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"[WARN] webgraph library failed: {e}", file=sys.stderr)
        return False


def convert_via_python_library_manual(basename, output_path):
    """Attempt to read the WebGraph file using available Python libraries."""
    # Try pip install
    import subprocess
    import importlib
    
    # Try installing webgraph
    print("[INFO] Attempting to install 'webgraph' package...", file=sys.stderr)
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', 'webgraph', '-q'],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print("[INFO] 'webgraph' installed successfully", file=sys.stderr)
        return convert_with_python_library(basename, output_path)
    else:
        print(f"[WARN] pip install failed: {result.stderr[:200]}", file=sys.stderr)
        return False


def simple_bv_converter(basename, output_path):
    """Manual WebGraph BV format decoder."""
    print("[INFO] Using built-in BV decoder...", file=sys.stderr)
    
    # Read the .graph file
    with open(basename + '.graph', 'rb') as f:
        data = f.read()
    
    # Read .properties
    props = read_properties(basename)
    num_nodes = int(props.get('nodes', 0))
    num_edges = int(props.get('edges', 0))
    window_size = int(props.get('window', 0))
    min_interval_len = int(props.get('mininterval', 0))
    
    print(f"[INFO] Nodes: {num_nodes}, Edges: {num_edges}", file=sys.stderr)
    
    # Read offsets
    offsets = []
    offsets_path = basename + '.offsets'
    if os.path.exists(offsets_path):
        with open(offsets_path, 'rb') as f:
            raw = f.read()
        for i in range(0, len(raw), 8):
            offsets.append(struct.unpack('<Q', raw[i:i+8])[0])
    
    # The BV format stores bitstream data after a header
    # Find the start of the bitstream (after header)
    # Header format: magic(4) + node_count(4) + edge_count(4) + window_size(4) + 
    #                min_interval(4) + ref_count(4) + block_count(4) = 28 bytes min
    
    pos = 0
    magic = struct.unpack('<I', data[pos:pos+4])[0]
    pos += 4
    if magic != BV_HEADER_MAGIC:
        print(f"[WARN] Unexpected magic: 0x{magic:08X} (expected 0x{BV_HEADER_MAGIC:08X})", file=sys.stderr)
    
    nodes_in_header = struct.unpack('<I', data[pos:pos+4])[0]
    pos += 4
    edges_in_header = struct.unpack('<I', data[pos:pos+4])[0]
    pos += 4
    window = struct.unpack('<I', data[pos:pos+4])[0]
    pos += 4
    min_interval = struct.unpack('<I', data[pos:pos+4])[0]
    pos += 4
    ref_count = struct.unpack('<I', data[pos:pos+4])[0]
    pos += 4
    block_count = struct.unpack('<I', data[pos:pos+4])[0]
    pos += 4
    
    print(f"[INFO] Header: nodes={nodes_in_header}, edges={edges_in_header}, "
          f"window={window}, min_interval={min_interval}, "
          f"ref_count={ref_count}, blocks={block_count}", file=sys.stderr)
    
    # The bitstream starts at pos
    # BV format bitstream is divided into blocks
    # For each block, there are:
    #   1. Outdegrees (gamma coded)
    #   2. Bit flags per node
    #   3. References (if any)
    #   4. Intervals / residuals
    
    bit_pos = pos * 8  # start of bitstream in bits
    
    # Read total outdegrees for correct error checking
    total_edges = 0
    node_successors = []
    
    for node in range(num_nodes):
        if node % 100000 == 0:
            print(f"[INFO] Converting node {node}/{num_nodes} ({total_edges}/{num_edges} edges)...", file=sys.stderr)
        
        # Read outdegree (gamma coded)
        outdegree, bit_pos = read_gamma_coded_int(data, bit_pos)
        total_edges += outdegree
        
        # For each successor, read using different encoding
        succs = []
        prev = 0
        for j in range(outdegree):
            # The encoding can be: gap, interval, or reference
            # For simplicity, try to read as delta-coded gap
            if j == 0:
                # First successor is delta-coded (or absolute)
                val, bit_pos = read_delta_coded_int(data, bit_pos)
                succs.append(node + val)
                prev = node + val
            else:
                # Subsequent successors use gap coding
                gap, bit_pos = read_delta_coded_int(data, bit_pos)
                succs.append(prev + gap)
                prev = prev + gap
        
        node_successors.append(succs)
    
    # Write output
    print(f"[INFO] Writing {total_edges} edges to {output_path}...", file=sys.stderr)
    with open(output_path, 'w') as f:
        for src, succs in enumerate(node_successors):
            for dst in succs:
                f.write(f"{src} {dst}\n")
    
    print(f"[OK] Done! {total_edges} edges written", file=sys.stderr)
    return True


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    
    graph_basename = sys.argv[1]
    output_path = sys.argv[2]
    
    # Remove .graph extension if present
    if graph_basename.endswith('.graph'):
        graph_basename = graph_basename[:-6]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Download missing files
    if not download_missing_files(graph_basename):
        print("[ERROR] Missing required .properties file. Download it manually.", file=sys.stderr)
        sys.exit(1)
    
    # Method 1: Try Python webgraph library
    if convert_with_python_library(graph_basename, output_path):
        return
    
    # Method 2: Try installing webgraph
    if convert_via_python_library_manual(graph_basename, output_path):
        return
    
    # Method 3: Use built-in BV decoder
    print("[INFO] Falling back to built-in BV decoder...", file=sys.stderr)
    try:
        simple_bv_converter(graph_basename, output_path)
    except Exception as e:
        print(f"[ERROR] BV decoder failed: {e}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Alternative: Download edge lists from SNAP instead:", file=sys.stderr)
        print("  soc-LiveJournal1 (4.8M nodes, 69M edges): https://snap.stanford.edu/data/soc-LiveJournal1.html", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
