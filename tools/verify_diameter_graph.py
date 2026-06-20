"""
Verify basic properties of the diameter graph:
- Node count, edge count
- Node ID contiguity
- Degree distribution
- SCC count (via Kosaraju, CPU)  <-- NEW
- Condensation DAG diameter        <-- NEW

Usage: python verify_diameter_graph.py <graph_file> [expected_nodes] [expected_edges] [expected_sccs]
"""
import sys
sys.setrecursionlimit(1000000)


def read_graph(filename):
    """Return edges list [(u,v), ...] and number of nodes (max_id+1)."""
    edges = []
    max_id = -1
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
            max_id = max(max_id, u, v)
    return edges, max_id + 1


def build_graph(n, edges):
    """Build forward and reverse adjacency lists."""
    out_adj = [[] for _ in range(n)]
    in_adj = [[] for _ in range(n)]
    for u, v in edges:
        out_adj[u].append(v)
        in_adj[v].append(u)
    return out_adj, in_adj


def kosaraju_scc(n, out_adj, in_adj):
    """Return (scc_id_per_node, num_sccs)."""
    visited = [False] * n
    order = []

    # Iterative first pass (DFS order) to avoid Python recursion on deep chains
    # But since max depth = 45 layers, recursive is fine. Use iterative for safety.
    for start in range(n):
        if visited[start]:
            continue
        stack = [(start, 0, False)]  # (node, edge_idx, processed_flag)
        while stack:
            v, idx, processed = stack.pop()
            if processed:
                order.append(v)
                continue
            if visited[v]:
                continue
            visited[v] = True
            stack.append((v, 0, True))
            for w in out_adj[v]:
                if not visited[w]:
                    stack.append((w, 0, False))

    comp = [-1] * n
    cid = 0

    # Iterative second pass
    for start in reversed(order):
        if comp[start] != -1:
            continue
        stack = [start]
        comp[start] = cid
        while stack:
            v = stack.pop()
            for w in in_adj[v]:
                if comp[w] == -1:
                    comp[w] = cid
                    stack.append(w)
        cid += 1

    return comp, cid


def condensation_dag(n, comp, num_scc, out_adj):
    """Build condensation DAG, return adjacency and SCC sizes."""
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
    dag = [sorted(s) for s in dag_adj]
    return dag, scc_size


def longest_path_dag(dag):
    """Longest path (in edges) in a DAG via topological sort."""
    n = len(dag)
    in_deg = [0] * n
    for u in range(n):
        for v in dag[u]:
            in_deg[v] += 1

    from collections import deque
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


def verify(filename, expected_nodes=None, expected_edges=None, expected_sccs=None):
    import time
    t0 = time.time()

    print(f"Verifying: {filename}")
    print("=" * 60)

    # ---- Phase 1: Basic stats ----
    print("\n[Phase 1] Basic properties...")
    max_node = -1
    min_node = float('inf')
    edge_count = 0
    out_deg = {}
    self_loops = 0

    edges, n = read_graph(filename)
    edge_count = len(edges)
    t1 = time.time()
    print(f"  Graph loaded in {t1-t0:.1f}s")

    print(f"  Node range:         {0} .. {n-1}")
    print(f"  Node count:         {n}", end="")
    if expected_nodes:
        print(f"  (expected {expected_nodes}) {'✅' if n == expected_nodes else '❌'}", end="")
    print()

    print(f"  Edge count:         {edge_count}", end="")
    if expected_edges:
        print(f"  (expected {expected_edges}) {'✅' if edge_count == expected_edges else '❌'}", end="")
    print()

    # Degree stats
    for u, v in edges:
        out_deg[u] = out_deg.get(u, 0) + 1
        if u == v:
            self_loops += 1
    print(f"  Self-loops:         {self_loops}")

    degrees = list(out_deg.values())
    if degrees:
        print(f"  Min out-degree:     {min(degrees)}")
        print(f"  Max out-degree:     {max(degrees)}")
        print(f"  Avg out-degree:     {sum(degrees)/len(degrees):.2f}")

    # Unique nodes
    uniq = set()
    for u, v in edges:
        uniq.add(u)
        uniq.add(v)
    actual_unique = len(uniq)
    print(f"  Unique nodes:       {actual_unique}  (expected {n}) {'✅' if actual_unique == n else '❌'}")
    sorted_nodes = sorted(uniq)
    contiguous = all(sorted_nodes[i] == sorted_nodes[0] + i for i in range(len(sorted_nodes)))
    print(f"  Contiguous IDs:     {'✅' if contiguous else '❌ (gaps in node numbering)'}")

    # ---- Phase 2: SCC analysis (Kosaraju) ----
    print(f"\n[Phase 2] SCC analysis (Kosaraju)...")
    t2 = time.time()
    out_adj, in_adj = build_graph(n, edges)
    print(f"  Adjacency built in {time.time()-t2:.1f}s")

    # Free edges memory
    del edges

    t3 = time.time()
    comp, num_scc = kosaraju_scc(n, out_adj, in_adj)
    t4 = time.time()
    print(f"  Kosaraju: {num_scc} SCCs ({t4-t3:.1f}s)", end="")
    if expected_sccs:
        print(f"  (expected {expected_sccs}) {'✅' if num_scc == expected_sccs else '❌'}", end="")
    print()

    # SCC size distribution
    from collections import Counter
    scc_sizes = Counter()
    for c in comp:
        scc_sizes[c] += 1
    size_dist = Counter(scc_sizes.values())
    print(f"  SCC size distribution:")
    for sz in sorted(size_dist):
        print(f"    size={sz}: {size_dist[sz]} SCCs")

    # Condensation DAG
    t5 = time.time()
    dag, _ = condensation_dag(n, comp, num_scc, out_adj)
    diam = longest_path_dag(dag)
    t6 = time.time()
    print(f"  Condensation DAG diameter: {diam} edges ({t6-t5:.1f}s)")

    # Check if condensation has single-in/single-out chain structure
    chain_nodes = sum(1 for s in dag if len(s) == 1)
    print(f"  SCCs with exactly 1 outgoing neighbor: {chain_nodes}/{num_scc}")

    total_time = time.time() - t0
    print(f"\n  Total verification time: {total_time:.1f}s")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_diameter_graph.py <graph_file> [expected_nodes] [expected_edges] [expected_sccs]")
        sys.exit(1)

    filename = sys.argv[1]
    expected_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else None
    expected_edges = int(sys.argv[3]) if len(sys.argv) > 3 else None
    expected_sccs = int(sys.argv[4]) if len(sys.argv) > 4 else None

    verify(filename, expected_nodes, expected_edges, expected_sccs)
