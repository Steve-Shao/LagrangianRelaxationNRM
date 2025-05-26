import re
import numpy as np


class InstanceRM:
    """
    InstanceRM parses an airline revenue management test problem file.

    The file encodes a single-leg or network airline RM instance. It contains flight legs, itineraries, fares, and a time-indexed demand probability matrix.

    Attributes:
        T (int): Number of time periods.
        flight_legs (np.ndarray[str]): Array of flight leg names, each as "O{o}D{d}".
        capacities (np.ndarray[int]): Array of capacities for each flight leg.
        itineraries (np.ndarray[str]): Array of itinerary names, each as "O{o}D{d}F{c}".
        A (np.ndarray[int]): Matrix of shape (num_legs, num_itins). A[i, j] = 1 if itinerary j uses leg i, 0 otherwise.
        probabilities (np.ndarray[float]): Matrix of shape (T, num_itins). probabilities[t, j] = probability of request for itinerary j at time t.
        F (np.ndarray[float]): Array of fares for each itinerary.
        lmd (np.ndarray[float]): Array of shape (L, J, T) with Lagrange multipliers. All (i, j) can change independently, and lmd is constant over t.

    File Format:
        - First line: integer τ, the number of time periods.
        - Second line: integer N, the number of flight legs.
        - Next N lines: each line has two integers o d, the origin and destination of a flight leg.
        - Next line: integer M, the number of itineraries.
        - Next M lines: each line has four values: o d c fare, the origin, destination, fare class, and fare of an itinerary.
        - Next τ lines: each line lists demand probabilities for that time period, in the format:
            [o d c] p  [o d c] p  ... (for all itineraries)
          where [o d c] identifies the itinerary and p is the probability of a request for that itinerary at that time.

    Methods:
        __init__(filepath): Loads and parses the instance file.
        _parse_file(filepath): Parses the file and sets attributes.
        _init_lmd(): Initializes the Lagrange multiplier array.
        solve_single_leg_dp(leg_idx): Solves the single-resource dynamic program for a specific flight leg.
        solve_subgradient_test(dim=3, alpha0=0.1, epsilon=1e-5, K=100, delta=1e-4, verbose=False): Test projected subgradient descent on a dummy convex function.
        lr_objective(lmd_full): Compute the LR relaxation V^lambda_1(c_1) for a given (L, J) lambda array.
        minimize_lr_relaxation(lmd0=None, alpha0=10.0, delta=0.1, eps=1e-6, max_iter=2000, verbose=False, print_every=10): Minimize the LR relaxation V^lambda_1(c_1) over lambda using subgradient descent.
    """

    def __init__(self, filepath: str):
        """
        Initialize a InstanceRM by parsing the given file.

        Args:
            filepath (str): Path to the airline RM instance file.
        """
        self._parse_file(filepath)
        self._init_lmd()
        self.vartheta = np.zeros((self.L, max(self.C)+1, self.T+1)) # (L, C+1, T+1)

    def _parse_file(self, filepath: str):
        """
        Parse the airline RM instance file and populate the attributes.

        Args:
            filepath (str): Path to the instance file.

        Populates:
            self.T, self.flight_legs, self.capacities, self.itineraries, self.L, self.J, self.A, self.probabilities, self.F
        """
        # Read all non-comment, non-empty lines
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.lstrip().startswith('#')]

        idx = 0
        tau = int(lines[idx]); idx += 1
        num_legs = int(lines[idx]); idx += 1

        # Parse flight legs
        flight_legs = []
        capacities = []
        for parts in (lines[i].split() for i in range(idx, idx + num_legs)):
            o, d, cap = int(parts[0]), int(parts[1]), int(parts[2])
            flight_legs.append(f"O{o}D{d}")
            capacities.append(cap)
        idx += num_legs

        num_itins = int(lines[idx]); idx += 1

        # Parse itineraries and fares
        itineraries_info = []
        itineraries = []
        fares = []
        for parts in (lines[i].split() for i in range(idx, idx + num_itins)):
            o, d, c, fare = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            itineraries_info.append((o, d, c))
            itineraries.append(f"O{o}D{d}F{c}")
            fares.append(fare)
        idx += num_itins

        # Initialize matrices
        A = np.zeros((num_legs, num_itins), int)
        P = np.zeros((tau, num_itins), float)
        leg_idx = {leg: i for i, leg in enumerate(flight_legs)}
        itin_idx = {info: j for j, info in enumerate(itineraries_info)}

        # Fill A matrix
        for j, (o, d, _) in enumerate(itineraries_info):
            if o == 0 or d == 0:
                A[leg_idx[f"O{o}D{d}"], j] = 1
            else:
                A[leg_idx[f"O{o}D0"], j] = 1
                A[leg_idx[f"O0D{d}"], j] = 1

        pat = re.compile(r"\[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]\s*([\d.eE+-]+)")
        for t in range(tau):
            for o, d, c, p in pat.findall(lines[idx]):
                P[t, itin_idx[(int(o), int(d), int(c))]] = float(p)
            idx += 1

        # Store results as numpy arrays
        self.T = tau
        self.flight_legs = np.array(flight_legs, dtype='<U10')
        self.C = np.array(capacities, dtype=int)
        self.L = len(self.flight_legs)
        self.itineraries = np.array(itineraries, dtype='<U20')
        self.F = np.array(fares, dtype=float)
        self.J = len(self.itineraries)
        self.probabilities = P
        self.A = A

    def _init_lmd(self):
        """
        Initialize Lagrange multiplier array.
        For each i, t, lmd_ijt = f_j / sum_j a_ij if a_ij > 0, and 0 otherwise.
        """
        lmd_ij = np.zeros((self.L, self.J))
        for j in range(self.J):
            a_ij = self.A[:, j]
            denom = np.sum(a_ij)
            if denom > 0:
                for i in range(self.L):
                    if a_ij[i] > 0:
                        lmd_ij[i, j] = self.F[j] / denom
        self.lmd = np.repeat(lmd_ij[:, :, None], self.T, axis=2)

    def solve_single_leg_dp(self, leg_idx):
        """
        Tabular backward induction for a single flight leg.
        Returns:
            vartheta: (C+1, T+1) value function
            y_star: (J, C+1, T) optimal decisions
        """
        C, T, J = self.C[leg_idx], self.T, self.J
        a_ij, p_jt, lmd_ijt = self.A[leg_idx], self.probabilities, self.lmd[leg_idx]

        vartheta = np.zeros((C+1, T+1))
        y_star = np.zeros((J, C+1, T), dtype=int)

        for t in range(T-1, -1, -1):
            # For all x in 0..C, shape (C+1, J)
            x_arr = np.arange(C+1)[:, None]  # (C+1, 1)
            a_ij_broadcast = a_ij[None, :]   # (1, J)
            feasible = (a_ij_broadcast <= x_arr)  # (C+1, J)

            # v0: (C+1, J), v1: (C+1, J)
            v0 = vartheta[x_arr[:, 0], t+1][:, None]  # (C+1, 1) -> (C+1, J)
            x_minus_a = x_arr - a_ij_broadcast  # (C+1, J)
            x_minus_a = np.where(feasible, x_minus_a, 0)
            v1 = lmd_ijt[None, :, t] + vartheta[x_minus_a, t+1]

            # y_star: (C+1, J)
            y_star_t = (v1 > v0) & feasible
            y_star[:, :, t] = y_star_t.T  # (J, C+1)

            # Compute next state for each (x, j)
            x_next = x_arr - a_ij_broadcast * y_star_t  # (C+1, J)
            x_next = np.clip(x_next, 0, C)

            # Compute value for each (x, j)
            value = lmd_ijt[None, :, t] * y_star_t + vartheta[x_next, t+1]

            # Weighted sum over j for each x
            vartheta[:, t] = np.sum(p_jt[t, :] * value, axis=1)

        return vartheta, y_star

    def lr_objective(self, lmd_full=None):
        """
        Compute the LR relaxation V^lambda_1(c_1) for a given (L, J) lambda array.
        """
        if lmd_full is not None:
            # lmd_full should be (L, J)
            self.lmd = np.repeat(lmd_full[:, :, None], self.T, axis=2)

        # Compute the sum of value functions for each leg
        total = 0.0
        for i in range(self.L):
            vartheta, _ = self.solve_single_leg_dp(i)
            total += vartheta[self.C[i], 0]
            print(f"Leg {i} value function: {vartheta[self.C[i], 0]}")

        # Add the expected revenue adjustment term
        for t in range(self.T):
            for j in range(self.J):
                penalty = max(self.F[j] - np.sum(self.lmd[:, j, t]), 0.0)
                total += self.probabilities[t, j] * penalty

        return total

    def subgradient_descent(self, f, x0, alpha0, delta=1e-4, eps=1e-6, max_iter=1000, verbose=False, print_every=0):
        """
        Projected subgradient descent for convex, non-analytical functions.
        Returns final x and objective history.
        """
        x, prev_val, history = x0.copy(), np.inf, []
        for k in range(max_iter):
            v = f(x)
            history.append(v)
            if print_every and k % print_every == 0:
                print(f"Iter {k}: f(x)={v:.6f}") # , x={x}
            if abs(v - prev_val) < eps:
                if verbose: print(f"Converged at iteration {k}, objective = {v:.6f}")
                break
            prev_val = v
            d = np.random.randn(*x.shape); d /= np.linalg.norm(d)
            g = ((f(x + delta * d) - v) / delta) * d
            x = np.maximum(0, x - (alpha0 / np.sqrt(1)) * g) # k + 
        return x, np.array(history)

    def minimize_lr_relaxation(self, lmd0=None, alpha0=1.0, delta=0.1, eps=1e-6, max_iter=2000, verbose=False, print_every=1):
        """
        Minimize LR relaxation V^lambda_1(c_1) over lambda using subgradient descent.
        Returns optimal multipliers (L, J, T) and objective history.
        """
        # Use full (L, J) representation of lambda
        if lmd0 is None:
            lmd0_full = self.lmd[:, :, 0].copy()
        else:
            lmd0_full = lmd0
        # Flatten for optimization
        lmd0_flat = lmd0_full.flatten()

        def obj(flat_lmd):
            lmd_full = flat_lmd.reshape(self.L, self.J)
            return self.lr_objective(lmd_full)

        lmd_star_flat, history = self.subgradient_descent(
            obj,
            lmd0_flat, alpha0, delta, eps, max_iter, verbose, print_every
        )
        lmd_star = lmd_star_flat.reshape(self.L, self.J)
        lmd_star_full = np.repeat(lmd_star[:, :, None], self.T, axis=2)
        return lmd_star_full, history


if __name__ == "__main__":
    data_path = "data/200_rm_datasets/rm_200_4_1.0_4.0.txt"
    inst = InstanceRM(data_path)

    print(f"T: {inst.T}")
    print(f"\nFlight legs ({inst.flight_legs.size}):\n{inst.flight_legs}")
    print(f"\nCapacities ({inst.C.size}):\n{inst.C}")
    print(f"\nItineraries ({inst.itineraries.size}):\n{inst.itineraries}")
    print(f"\nFares ({inst.F.size}):\n{inst.F}")
    print(f"\nA (shape {inst.A.shape}):\n{inst.A}")
    print(f"\nProbabilities (shape {inst.probabilities.shape}):")
    max_t, max_j = min(2, inst.probabilities.shape[0]), min(5, inst.probabilities.shape[1])
    print(f"First {max_t} time periods, {max_j} itineraries:\n{inst.probabilities[:max_t, :max_j]}\n...")

    # Test solve_single_leg_dp for the first flight leg
    leg_idx = 0
    vartheta, y_star = inst.solve_single_leg_dp(leg_idx)
    print(f"\n[Testing] Value function shape for leg {leg_idx}: {vartheta.shape}")
    print(f"[Testing] Optimal decision array shape for leg {leg_idx}: {y_star.shape}")
    # Print a small sample
    np.savetxt("vartheta_output.txt", vartheta)

    # Test lr_objective for the current lambda
    lr_obj_val = inst.lr_objective()
    print(f"\n[Testing] LR objective value for current lambda: {lr_obj_val:.6f}")

    # Test minimize_lr_relaxation for the LR relaxation
    print("\n[Testing] Minimize LR relaxation (this may take a while for large instances)...")
    lmd_star, history = inst.minimize_lr_relaxation()
    # print(f"\n[Testing] Final LR objective value: {history[-1]:.6f}")
    # print(f"[Testing] Optimal lambda shape: {lmd_star.shape}")
    # print(f"[Testing] First 5 objective values: {history[:5]}")
