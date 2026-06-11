import os
import subprocess
import sys

datasets_path='/hdd/thej_par_scc_datasets'
insert_methods=["5","6","11"]
num_threads_list=["72"]
abs_path =""

def parse_output(output):
    """Parse lines and extract running_time and SCC count."""
    data_info = {}
    for line in output.split('\n'):
        # parse data= line
        if 'data=' in line:
            parts = line.strip().split('=')
            if len(parts) >= 2:
                info = parts[1].split(' ')
                if len(info) >= 3:
                    data_info['file'] = info[0]
                    data_info['method'] = int(info[1])
                    data_info['threads'] = int(info[2])
        # parse running_time
        if 'running_time(ms)=' in line:
            parts = line.strip().split('=')
            if len(parts) >= 2:
                data_info['runtime'] = float(parts[1])
        # parse SCC count
        if 'Total # SCCs' in line:
            parts = line.strip().split('=')
            if len(parts) >= 2:
                data_info['scc_count'] = int(parts[1].strip())
    return data_info

def run_binary(binary, args):
    """Run a binary and return stdout+stderr, or None on failure."""
    try:
        result = subprocess.run([binary] + args,
                                capture_output=True, text=True, timeout=600)
        return result.stdout + result.stderr
    except Exception as e:
        return f"ERROR: {e}"

def print_comparison_row(basename, method, threads, omp_info, cuda_info):
    """Print a single comparison row."""
    runtime_omp = omp_info.get('runtime', None)
    runtime_cuda = cuda_info.get('runtime', None)
    scc_omp = omp_info.get('scc_count', None)
    scc_cuda = cuda_info.get('scc_count', None)

    speedup = ""
    match = ""
    if runtime_omp is not None and runtime_cuda is not None and runtime_cuda > 0:
        speedup = f"{runtime_omp / runtime_cuda:.2f}x"
    if scc_omp is not None and scc_cuda is not None:
        match = "✓" if scc_omp == scc_cuda else "✗ MISMATCH"

    print(f"  {basename:40s} | M{method} | {threads:>3}T | "
          f"OMP: {runtime_omp:>8.2f}ms | "
          f"CUDA: {runtime_cuda:>8.2f}ms | "
          f"Speedup: {speedup:>8s} | {match}")

def run_comparison(binary_omp, binary_cuda, graph_path, threads, method, label):
    """Run both binaries and print comparison."""
    print(f"\n{'='*80}")
    print(f"Dataset: {label}  |  Threads: {threads}  |  Method: {method}")
    print(f"{'='*80}")

    # OMP
    omp_output = run_binary(binary_omp, [graph_path, str(threads), str(method)])
    if omp_output is None:
        print("  OpenMP: FAILED")
        return None

    omp_info = parse_output(omp_output)

    # Print OMP output for reference
    for line in omp_output.split('\n'):
        if any(kw in line for kw in ['running_time', 'Total # SCCs', 'data=', 'Running']):
            print(f"  [OMP] {line}")

    # CUDA
    cuda_output = run_binary(binary_cuda, [graph_path, str(threads), str(method)])
    if cuda_output is None:
        print("  CUDA: FAILED")
        return None

    cuda_info = parse_output(cuda_output)

    # Print CUDA output for reference
    for line in cuda_output.split('\n'):
        if any(kw in line for kw in ['running_time', 'Total # SCCs', 'data=', 'Running']):
            print(f"  [CUDA] {line}")

    # Comparison
    print(f"\n  {'─'*75}")
    print_comparison_row(label, method, threads, omp_info, cuda_info)
    print(f"  {'─'*75}")

    return (omp_info, cuda_info)

# ======================================================================
# Main
# ======================================================================
binary_omp  = "./scc"
binary_cuda = "./scc_cuda"

# Check that both binaries exist
if not os.path.isfile(binary_omp):
    print(f"ERROR: {binary_omp} not found. Build it first.")
    sys.exit(1)
if not os.path.isfile(binary_cuda):
    print(f"ERROR: {binary_cuda} not found. Build it first.")
    sys.exit(1)

print("=" * 80)
print("  OpenMP vs CUDA SCC Benchmark Comparison")
print("=" * 80)

results = []

for dataset in sorted(os.listdir(datasets_path)):
    if '.tar.gz' in dataset:
        continue

    dataset_dir = os.path.join(datasets_path, dataset)

    # --- Static graph: method 2 (Trim1 + Global FW-BW + Trim1/2 + WCC + FW-BW) ---
    for filename in os.listdir(dataset_dir):
        if 'refined_edges' in filename:
            abs_path = os.path.abspath(os.path.join(dataset_dir, filename))
            label = f"{dataset}/{filename}"
            res = run_comparison(binary_omp, binary_cuda, abs_path, "1", "2", label)
            if res:
                results.append(("static", label, "2", "1", res[0], res[1]))

    # --- Dynamic graphs: insertions ---
    for i in range(5):
        for filename in os.listdir(dataset_dir):
            if 'insert_edges' in filename:
                abs_path = os.path.abspath(os.path.join(dataset_dir, filename))
                for method in insert_methods:
                    for num_threads in num_threads_list:
                        label = f"{dataset}/{filename}"
                        res = run_comparison(binary_omp, binary_cuda, abs_path, num_threads, method, label)
                        if res:
                            results.append(("dynamic_insert", label, method, num_threads, res[0], res[1]))

# --- Summary Table ---
print("\n\n")
print("=" * 80)
print("  COMPARISON SUMMARY")
print("=" * 80)
print(f"  {'Dataset':40s} | {'Type':17s} | Method | Threads | OMP (ms)   | CUDA (ms)  | Speedup   | SCC Match")
print(f"  {'─'*40}─┼{'─'*17}─┼───────┼────────┼────────────┼────────────┼───────────┼──────────")

for typ, label, method, threads, omp, cuda in results:
    runtime_omp = omp.get('runtime', -1)
    runtime_cuda = cuda.get('runtime', -1)
    scc_omp = omp.get('scc_count', -1)
    scc_cuda = cuda.get('scc_count', -1)

    speedup = ""
    if runtime_omp > 0 and runtime_cuda > 0:
        speedup = f"{runtime_omp / runtime_cuda:.2f}x"

    match = ""
    if scc_omp >= 0 and scc_cuda >= 0:
        match = "✓" if scc_omp == scc_cuda else "✗"

    print(f"  {label:40s} | {typ:17s} | M{method:5s} | {threads:>4s} | "
          f"{'N/A' if runtime_omp<0 else f'{runtime_omp:>8.2f}ms':12s} | "
          f"{'N/A' if runtime_cuda<0 else f'{runtime_cuda:>8.2f}ms':12s} | "
          f"{speedup:>9s} | {match}")
