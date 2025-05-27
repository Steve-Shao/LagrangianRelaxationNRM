import os
import json
import numpy as np

from src.instance import InstanceRM

def run_instance(data_path):
    # Set up data and results paths
    if not os.path.exists(data_path):
        print(f"Test instance file not found: {data_path}")
        return

    os.makedirs("results", exist_ok=True)
    instance_name = os.path.splitext(os.path.basename(data_path))[0]
    results_dir = os.path.join("results", instance_name)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "results.json")
    lambda_txt_path = os.path.join(results_dir, "optimized_lambda.txt")

    # Check if results already exist and are non-empty
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                data = json.load(f)
            if data:  # file is not empty and contains results
                print(f"Results already exist for {data_path}. Skipping this instance.")
                return
        except Exception:
            pass  # If file is empty or corrupted, proceed to rerun

    # Load instance and solve
    print(f"Loading instance {data_path} and solving LR problem...")
    inst = InstanceRM(data_path)
    optimized_lmd, V_history = inst.minimize_lr_relaxation(
        alpha0=100.0,
        eps=1e-4,
        max_iter=10000,
        verbose=True,
        print_every=10,
        patience_iters=20
    )
    print(f"Optimization finished. Final V_lambda: {V_history[-1]:.4f}")
    print(f"Optimized lambda shape: {optimized_lmd.shape}")

    # Simulate revenue
    N_SIMULATIONS = 1000
    simulated_revenues = [
        inst.simulate_revenue_with_bid_prices(1, optimized_lambdas=optimized_lmd)
        for _ in range(N_SIMULATIONS)
    ]
    simulated_revenues = np.array(simulated_revenues)
    estimated_rev = np.mean(simulated_revenues)
    std_err = np.std(simulated_revenues, ddof=1) / np.sqrt(N_SIMULATIONS)
    ci_low = estimated_rev - 1.96 * std_err
    ci_high = estimated_rev + 1.96 * std_err
    print(f"Estimated revenue over {N_SIMULATIONS} simulations: {estimated_rev:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    # Save results (excluding lambda array)
    results_data = {
        "final_V_lambda": V_history[-1] if V_history else None,
        "V_lambda_history": V_history,
        "estimated_revenue": estimated_rev,
        "std_error": std_err,
        "ci_95": [ci_low, ci_high],
        "all_simulated_revenues": simulated_revenues.tolist()
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to {results_path}")

    # Save optimized lambda (first and last time period)
    with open(lambda_txt_path, "w") as f:
        if optimized_lmd.ndim == 3:
            T = optimized_lmd.shape[2]
            # Save t=0
            f.write("# lambda[:, :, 0]\n")
            for leg_idx, mat in enumerate(optimized_lmd):
                f.write(f"# leg {leg_idx}, t=0\n")
                row = mat[:, 0]
                f.write("\t".join(f"{val:.6f}" for val in row) + "\n")
            f.write("\n")
            # Save t=T-1
            f.write(f"# lambda[:, :, {T-1}]\n")
            for leg_idx, mat in enumerate(optimized_lmd):
                f.write(f"# leg {leg_idx}, t={T-1}\n")
                row = mat[:, T-1]
                f.write("\t".join(f"{val:.6f}" for val in row) + "\n")
            f.write("\n")
        elif optimized_lmd.ndim == 2:
            for row in optimized_lmd:
                f.write("\t".join(f"{val:.6f}" for val in row) + "\n")
        elif optimized_lmd.ndim == 1:
            for val in optimized_lmd:
                f.write(f"{val:.6f}\n")
        else:
            f.write(repr(optimized_lmd))
    print(f"Optimized lambda (first and last time period) saved to {lambda_txt_path}")

if __name__ == "__main__":
    # data_dir = "data/200_rm_datasets"
    # instance_files = [
    #     os.path.join(data_dir, fname)
    #     for fname in os.listdir(data_dir)
    #     if fname.endswith(".txt")
    # ]
    # instance_files.sort()
    # for data_path in instance_files:
    #     print(f"\n=== Running instance: {data_path} ===")
    #     run_instance(data_path)

    # Read reported LR revenue from Huseyin
    revenue_json_path = os.path.join("data", "revenue.json")
    if os.path.exists(revenue_json_path):
        with open(revenue_json_path, "r") as f:
            huseyin_revenue_data = json.load(f)
        # Create a dictionary mapping problem name to LR revenue
        huseyin_lr_revenue = {entry["Problem"]: entry["LR"] for entry in huseyin_revenue_data}
        print(f"Loaded reported LR revenue for {len(huseyin_lr_revenue)} problems from Huseyin.")
    else:
        huseyin_lr_revenue = {}
        print(f"Warning: {revenue_json_path} not found. No reported LR revenue loaded.")

    # Read upper bounds from bounds.json
    bounds_json_path = os.path.join("data", "bounds.json")
    if os.path.exists(bounds_json_path):
        with open(bounds_json_path, "r") as f:
            bounds_data = json.load(f)
        # Create a dictionary mapping problem name to LR upper bound
        lr_upper_bound = {entry["Problem"]: entry["LR"] for entry in bounds_data}
        print(f"Loaded LR upper bounds for {len(lr_upper_bound)} problems from bounds.json.")
    else:
        lr_upper_bound = {}
        print(f"Warning: {bounds_json_path} not found. No LR upper bounds loaded.")

    # Save a markdown file with a table of LR revenue and upper bounds from all instances
    os.makedirs("results", exist_ok=True)
    md_path = os.path.join("results", "huseyin_lr_revenue.md")
    with open(md_path, "w") as f:
        f.write("| Problem | LR Revenue (Huseyin) | LR Upper Bound |\n")
        f.write("|---------|----------------------|---------------|\n")
        all_problems = set(huseyin_lr_revenue.keys()) | set(lr_upper_bound.keys())
        for problem in sorted(all_problems):
            lr_rev = huseyin_lr_revenue.get(problem, "-")
            lr_ub = lr_upper_bound.get(problem, "-")
            f.write(f"| {problem} | {lr_rev} | {lr_ub} |\n")
    print(f"Markdown table of LR revenue and upper bounds saved to {md_path}")

    
