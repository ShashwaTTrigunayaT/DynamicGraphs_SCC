"""
Verify basic properties of the diameter graph:
- Node count (should match expected)
- Edge count (should match expected)
- Node ID range (should be contiguous 0..N-1)
- Min/avg/max degree
- Self-loops check

Usage: python verify_diameter_graph.py <graph_file> [expected_nodes] [expected_edges]
"""
import sys

def verify(filename, expected_nodes=None, expected_edges=None):
    print(f"Verifying: {filename}")
    print("=" * 60)

    max_node = -1
    min_node = float('inf')
    edge_count = 0
    out_deg = {}  # node -> out-degree
    self_loops = 0

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            edge_count += 1
            max_node = max(max_node, u, v)
            min_node = min(min_node, u, v)
            out_deg[u] = out_deg.get(u, 0) + 1
            if u not in out_deg:
                out_deg[u] = 1
            else:
                out_deg[u] += 1
            if v not in out_deg:
                out_deg[v] = out_deg.get(v, 0)  # ensure it exists with 0 out-degree
            if u == v:
                self_loops += 1

    num_nodes = max_node + 1 if min_node == 0 else max_node - min_node + 1
    
    print(f"  Node range:         {min_node} .. {max_node}")
    print(f"  Node count:         {num_nodes}", end="")
    if expected_nodes:
        print(f"  (expected {expected_nodes}) {'✅' if num_nodes == expected_nodes else '❌'}", end="")
    print()
    
    print(f"  Edge count:         {edge_count}", end="")
    if expected_edges:
        print(f"  (expected {expected_edges}) {'✅' if edge_count == expected_edges else '❌'}", end="")
    print()
    
    print(f"  Self-loops:         {self_loops}")
    
    # Degree stats
    degrees = list(out_deg.values())
    if degrees:
        print(f"  Min out-degree:     {min(degrees)}")
        print(f"  Max out-degree:     {max(degrees)}")
        print(f"  Avg out-degree:     {sum(degrees)/len(degrees):.2f}")
    
    # Check if node IDs are contiguous
    unique_nodes = set()
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            unique_nodes.add(u)
            unique_nodes.add(v)
    
    expected_unique = num_nodes
    actual_unique = len(unique_nodes)
    print(f"  Unique nodes:       {actual_unique}  (expected {expected_unique}) {'✅' if actual_unique == expected_unique else '❌'}")
    
    # Check contiguity
    sorted_nodes = sorted(unique_nodes)
    contiguous = all(sorted_nodes[i] == sorted_nodes[0] + i for i in range(len(sorted_nodes)))
    print(f"  Contiguous IDs:     {'✅' if contiguous else '❌ (gaps in node numbering)'}")
    
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_diameter_graph.py <graph_file> [expected_nodes] [expected_edges]")
        sys.exit(1)
    
    filename = sys.argv[1]
    expected_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else None
    expected_edges = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    verify(filename, expected_nodes, expected_edges)
