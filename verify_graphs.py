"""
Verify that generated graph files match the documented properties.

Reads the edge-list files (1-indexed format from graphGenerator) and
checks:
  1. Node and edge counts
  2. SCC decomposition using Kosaraju
  3. Condensation DAG diameter (longest path)
  4. BFS depth from the designated source node

Universe of discourse for each graph:

diameter_45.txt:
  - Structure: 45 layers × 5 nodes/layer = 225 nodes
    Each layer: 5-cycle (intra-SCC)
    Inter-layer: all-pairs forward (layer i → layer i+1)
  - Expected SCCs: 45 (each layer is its own SCC)
  - Condensation DAG: simple chain of 45 nodes → diameter = 45
  - Back edges: 0

bfs_depth_40.txt:
  - Structure: (10 prefix + 1 source + 40 BFS-depth) = 51 layers × 5 = 255 nodes
    Source node (0-indexed) = 10 * 5 = 50
  - Expected SCCs: 51
  - BFS depth from source node 50: should reach depth 40
  - Total diameter: 10 + 40 = 50
"""

import sys
from collections import defaultdict, deque


def read_edge_list(filename: str):
    """Read 1-indexed edge list; return (edges, max_node_id)."""
    edges = []
    max_node = -1
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]) - 1, int(parts[1]) - 1
            edges.append((u, v))
            max_node = max(max_node, u, v)
    return edges, max_node + 1


def build_adjacency(n: int, edges):
    out_adj = [[] for _ in range(n)]
    in_adj = [[] for _ in range(n)]
    for u, v in edges:
        out_adj[u].append(v)
        in_adj[v].append(u)
    return out_adj, in_adj


def kosaraju_scc(n: int, out_adj, in_adj):
    """Return (scc_id per node, number_of_sccs)."""
    visited = [False] * n
    order = []

    def dfs1(u):
        visited[u] = True
        for v in out_adj[u]:
            if not visited[v]:
                dfs1(v)
        order.append(u)

    for i in range(n):
        if not visited[i]:
            dfs1(i)

    comp = [-1] * n

    def dfs2(u, cid):
        comp[u] = cid
        for v in in_adj[u]:
            if comp[v] == -1:
                dfs2(v, cid)

    cid = 0
    for u in reversed(order):
        if comp[u] == -1:
            dfs2(u, cid)
            cid += 1

    return comp, cid


def condensation_dag(n: int, comp, num_scc, out_adj):
    """Build condensation DAG and return its adjacency, plus SCC sizes."""
    scc_size = [0] * num_scc
    for c in comp:
        scc_size[c] += 1

    dag_adj = [set() for _ in range(num_scc)]
    for u in range(n):
        cu = comp[u]
        for v in out_adj[u]:
            cv = comp[v]
            if cu != cv:
                dag_adj[cu].add(cv)
    # Convert to lists
    dag = [sorted(s) for s in dag_adj]
    return dag, scc_size


def longest_path_dag(dag):
    """Compute longest path (in edges) in a DAG via DP / topological sort."""
    n = len(dag)
    in_deg = [0] * n
    for u in range(n):
        for v in dag[u]:
            in_deg[v] += 1

    q = deque([u for u in range(n) if in_deg[u] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in dag[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)

    dp = [0] * n
    for u in topo:
        for v in dag[u]:
            if dp[u] + 1 > dp[v]:
                dp[v] = dp[u] + 1
    return max(dp) if dp else 0


def bfs_max_depth(n: int, out_adj, source: int):
    """Return the maximum BFS depth reachable from source."""
    dist = [-1] * n
    q = deque([source])
    dist[source] = 0
    while q:
        u = q.popleft()
        for v in out_adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    max_d = max(d for d in dist if d >= 0)
    return max_d


# ======================================================================
# Checks
# ======================================================================

def check_diameter_graph(filename: str, expected_layers: int, scc_size: int):
    print("=" * 60)
    print(f"CHECK: {filename}")
    print("=" * 60)

    edges, n = read_edge_list(filename)
    expected_nodes = expected_layers * scc_size
    expected_edges = expected_layers * scc_size + (expected_layers - 1) * scc_size * scc_size
    print(f"  Nodes: {n} (expected {expected_nodes})", end="")
    print("  ✅" if n == expected_nodes else f"  ❌ (expected {expected_nodes})")

    print(f"  Edges: {len(edges)} (expected {expected_edges})", end="")
    print("  ✅" if len(edges) == expected_edges else f"  ❌ (expected {expected_edges})")

    out_adj, in_adj = build_adjacency(n, edges)
    comp, num_scc = kosaraju_scc(n, out_adj, in_adj)
    dag, scc_sizes = condensation_dag(n, comp, num_scc, out_adj)

    print(f"  SCC count: {num_scc} (expected {expected_layers})", end="")
    print("  ✅" if num_scc == expected_layers else f"  ❌ (expected {expected_layers})")

    # Check each SCC has the right size
    correct_sizes = all(sz == scc_size for sz in scc_sizes)
    print(f"  All SCCs size={scc_size}: ", end="")
    print("✅" if correct_sizes else f"❌ Sizes: {scc_sizes}")

    # Check condensation DAG diameter
    diam = longest_path_dag(dag)
    print(f"  Condensation DAG diameter: {diam} (expected {expected_layers - 1} edges, i.e. {expected_layers} nodes)", end="")
    # The longest path edges count = layers - 1
    print("  ✅" if diam == expected_layers - 1 else f"  ❌ (expected {expected_layers - 1} edges)")

    # Check no back edges (DAG property) - the condensation should have no cycles
    # Since we already used Kosaraju and got num_scc = expected_layers, this is implicit

    # Check the chain structure: each SCC i should connect only to SCC i+1
    chain_ok = True
    for u in range(num_scc):
        for v in dag[u]:
            if v != u + 1:
                chain_ok = False
    print(f"  Chain structure (SCC{u}→SCC{u+1} only): ", end="")
    print("✅" if chain_ok else "❌ Non-chain edges found in condensation")

    print()
    return (n == expected_nodes and len(edges) == expected_edges and
            num_scc == expected_layers and correct_sizes and
            diam == expected_layers - 1 and chain_ok)


def check_bfs_depth_graph(filename: str, prefix_layers: int, target_depth: int, scc_size: int):
    total_layers = prefix_layers + 1 + target_depth
    source_node = prefix_layers * scc_size
    expected_nodes = total_layers * scc_size
    expected_edges = total_layers * scc_size + (total_layers - 1) * scc_size * scc_size

    print("=" * 60)
    print(f"CHECK: {filename}")
    print("=" * 60)

    edges, n = read_edge_list(filename)
    print(f"  Nodes: {n} (expected {expected_nodes})", end="")
    print("  ✅" if n == expected_nodes else f"  ❌ (expected {expected_nodes})")

    print(f"  Edges: {len(edges)} (expected {expected_edges})", end="")
    print("  ✅" if len(edges) == expected_edges else f"  ❌ (expected {expected_edges})")

    out_adj, in_adj = build_adjacency(n, edges)
    comp, num_scc = kosaraju_scc(n, out_adj, in_adj)
    dag, scc_sizes = condensation_dag(n, comp, num_scc, out_adj)

    print(f"  SCC count: {num_scc} (expected {total_layers})", end="")
    print("  ✅" if num_scc == total_layers else f"  ❌ (expected {total_layers})")

    correct_sizes = all(sz == scc_size for sz in scc_sizes)
    print(f"  All SCCs size={scc_size}: ", end="")
    print("✅" if correct_sizes else f"❌")

    # BFS depth from source node
    bfs_depth = bfs_max_depth(n, out_adj, source_node)
    print(f"  BFS depth from source node {source_node}: {bfs_depth} (expected {target_depth})", end="")
    print("  ✅" if bfs_depth == target_depth else f"  ❌ (expected {target_depth})")

    # Condensation DAG diameter
    diam = longest_path_dag(dag)
    expected_diam = total_layers - 1
    print(f"  Condensation DAG diameter: {diam} edges (expected {expected_diam})", end="")
    print("  ✅" if diam == expected_diam else f"  ❌ (expected {expected_diam})")

    # Chain structure check
    chain_ok = True
    for u in range(num_scc):
        for v in dag[u]:
            if v != u + 1:
                chain_ok = False
    print(f"  Chain structure (SCC{u}→SCC{u+1} only): ", end="")
    print("✅" if chain_ok else "❌")

    print()
    return (n == expected_nodes and len(edges) == expected_edges and
            num_scc == total_layers and correct_sizes and
            bfs_depth == target_depth and diam == expected_diam and chain_ok)


if __name__ == "__main__":
    ok1 = check_diameter_graph("diameter_45.txt", expected_layers=45, scc_size=5)
    ok2 = check_bfs_depth_graph("bfs_depth_40.txt", prefix_layers=10, target_depth=40, scc_size=5)

    print("=" * 60)
    print("OVERALL RESULT")
    print("=" * 60)
    print(f"  diameter_45.txt:  {'✅ ALL CHECKS PASSED' if ok1 else '❌ SOME CHECKS FAILED'}")
    print(f"  bfs_depth_40.txt: {'✅ ALL CHECKS PASSED' if ok2 else '❌ SOME CHECKS FAILED'}")

    sys.exit(0 if ok1 and ok2 else 1)
