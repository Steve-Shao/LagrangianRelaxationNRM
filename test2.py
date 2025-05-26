import re
import numpy as np
import numba # Added numba import


class InstanceRM:
    """
    InstanceRM parses an airline revenue management test problem file.

    The file encodes a single-leg or network airline RM instance. It contains flight legs, itineraries, fares, and a time-indexed demand probability matrix.

    Attributes:
        T (int): Number of time periods.
        flight_legs (np.ndarray[str]): Array of flight leg names, each as "O{o}D{d}".
        C (np.ndarray[int]): Array of capacities for each flight leg.
        L (int): Number of flight legs.
        itineraries (np.ndarray[str]): Array of itinerary names, each as "O{o}D{d}F{c}".
        F (np.ndarray[float]): Array of fares for each itinerary.
        J (int): Number of itineraries.
        probabilities (np.ndarray[float]): Matrix of shape (T, J). probabilities[t, j] = probability of request for itinerary j at time t.
        A (np.ndarray[int]): Matrix of shape (L, J). A[i, j] = 1 if itinerary j uses leg i, 0 otherwise.
        lmd (np.ndarray[float]): Array of shape (L, J, T) with Lagrange multipliers.
        vartheta (np.ndarray[float]): Array of shape (L, max(C)+1, T+1) for value functions.

    File Format:
        - First line: integer τ, the number of time periods.
        - Second line: integer N, the number of flight legs.
        - Next N lines: each line has three integers o d cap, the origin, destination, and capacity of a flight leg.
        - Next line: integer M, the number of itineraries.
        - Next M lines: each line has four values: o d c fare, the origin, destination, fare class, and fare of an itinerary.
        - Next τ lines: each line lists demand probabilities for that time period, in the format:
            [o d c] p  [o d c] p  ... (for all itineraries)
          where [o d c] identifies the itinerary and p is the probability of a request for that itinerary at that time.

    Methods:
        __init__(filepath): Loads and parses the instance file.
        _parse_file(filepath): Parses the file and sets attributes.
        _initialize_lambdas(): Initializes the Lagrange multiplier array.
        solve_single_leg_dp(leg_idx): Solves the single-resource dynamic program for a specific flight leg.
        compute_state_probabilities(leg_idx, y_star): Compute state occupancy probabilities mu[t, x] for a resource i.
        compute_vartheta_subgradient(leg_idx, mu, y_star): Compute the subgradient G[j, t] of the single-resource value function with respect to Lagrange multipliers.
        compute_V_lambda_subgradient(): Compute the value of the Lagrangian relaxation and its subgradient with respect to each Lagrange multiplier.
        minimize_lr_relaxation(lmd0=None, alpha0=1.0, eps=1e-5, max_iter=1000, verbose=False, print_every=10): Minimize the Lagrangian relaxation using projected subgradient descent.
        simulate_revenue_with_bid_prices(N_sim: int, optimized_lmd_param: np.ndarray = None) -> float: Estimate total expected revenue using the bid price policy derived from Lagrangian multipliers, via Monte Carlo simulation.
    """

    # Add jitted static helper for solve_single_leg_dp
    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _jit_solve_single_leg_dp(
        T_periods: int,
        J_itineraries: int,
        leg_capacity: int,
        leg_consumptions_j: np.ndarray,  # A[leg_idx, :] -> (J,)
        probs_jt: np.ndarray,          # probabilities.T -> (J, T)
        leg_lambdas_jt: np.ndarray,     # lmd[leg_idx, :, :] -> (J, T)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        JIT-compiled core logic for solving single-resource dynamic program.
        """
        vartheta_table = np.zeros((leg_capacity + 1, T_periods + 1))  # Value function
        optimal_decisions_y = np.zeros((J_itineraries, leg_capacity + 1, T_periods), dtype=np.int_) # Optimal decisions
        capacity_states = np.arange(leg_capacity + 1)  # All possible capacity levels [0,...,leg_capacity]

        # Precompute broadcasted arrays and capacity mask
        # leg_consumptions_j: (J,), capacity_states: (C+1,)
        # consumptions_bc: (J, 1), capacity_states_bc: (1, C+1)
        consumptions_bc = leg_consumptions_j.reshape(-1, 1)
        capacity_states_bc = capacity_states.reshape(1, -1)
        
        # sufficient_capacity_mask: (J, C+1)
        sufficient_capacity_mask = (consumptions_bc <= capacity_states_bc)

        # Backward induction through time
        for t in range(T_periods - 1, -1, -1):
            next_t_value_func = vartheta_table[:, t + 1]  # Next period's value function (C+1,)
            # current_t_lambdas_j: (J,1) for broadcasting against (C+1) states
            current_t_lambdas_j = leg_lambdas_jt[:, t].reshape(-1, 1) 
            
            # Calculate accept/reject values
            # capacity_if_accept: (J, C+1)
            capacity_if_accept = capacity_states_bc - consumptions_bc
            
            # future_value_if_accept: (J, C+1)
            # Initialize with -inf for invalid states (where capacity_if_accept < 0 before clipping)
            future_value_if_accept = np.full((J_itineraries, leg_capacity + 1), -np.inf)
            
            # Valid capacity indices after acceptance (clipped to be >= 0)
            valid_capacity_indices_if_accept = np.maximum(0, capacity_if_accept)

            # Populate future_value_if_accept only where sufficient_capacity_mask is True
            for j_idx in range(J_itineraries):
                for cap_idx in range(leg_capacity + 1):
                    if sufficient_capacity_mask[j_idx, cap_idx]:
                        future_value_if_accept[j_idx, cap_idx] = next_t_value_func[valid_capacity_indices_if_accept[j_idx, cap_idx]]
            
            val_accept = current_t_lambdas_j + future_value_if_accept  # Value if accept (J, C+1)
            val_reject = next_t_value_func.reshape(1, -1)  # Value if reject, broadcast to (1, C+1)
            
            # Determine optimal decisions (J, C+1)
            current_t_optimal_decisions = (val_accept > val_reject).astype(np.int_)
            optimal_decisions_y[:, :, t] = current_t_optimal_decisions
            
            # Calculate capacity after optimal decisions (J, C+1)
            capacity_after_decision = capacity_states_bc - consumptions_bc * current_t_optimal_decisions
            # Clip to ensure capacity is not negative (though logic should prevent this if y is correct)
            capacity_after_decision = np.maximum(0, capacity_after_decision) 
            
            # Bellman equation update
            # term_if_accepted_and_optimal: (J, C+1)
            # (lambda_jt + V(x-a, t+1)) * y_jt
            term_if_accepted_and_optimal = (current_t_lambdas_j + future_value_if_accept) * current_t_optimal_decisions
            # term_if_rejected_or_not_optimal: (J, C+1)
            # V(x, t+1) * (1 - y_jt)
            term_if_rejected_or_not_optimal = val_reject * (1 - current_t_optimal_decisions)

            # Expected value from decision on product j, given state x
            # Sum of [ P(j) * ( Value_if_y_star_for_j ) ]
            # This part was tricky with Numba's direct translation, simplified:
            # vartheta_table[:, t] is (C+1,)
            # probs_jt is (J,T)
            # current_t_optimal_decisions is (J, C+1)
            # current_t_lambdas_j is (J,1)
            # next_t_value_func is (C+1,)
            # capacity_after_decision is (J, C+1)
            
            # For each state x (column in vartheta_table):
            # V(x,t) = sum_j { p_jt * [ y_jxt * (lambda_jt + V(x-a_j, t+1)) + (1-y_jxt) * V(x,t+1) ] }
            # This is: sum_j { p_jt * ( y_jxt*lambda_jt + V(x - a_j*y_jxt, t+1) ) }

            # Re-evaluate the term inside sum for Bellman update:
            # For each state x (cap_idx_):
            #   sum_over_j = 0
            #   for j_idx_ in range(J_itineraries):
            #       prob_j_at_t = probs_jt[j_idx_, t]
            #       decision_y = optimal_decisions_y[j_idx_, cap_idx_, t]
            #       lambda_val = leg_lambdas_jt[j_idx_, t]
            #       
            #       val_from_this_j = 0
            #       if decision_y == 1: # Accept
            #           cap_after = capacity_states[cap_idx_] - leg_consumptions_j[j_idx_]
            #           val_from_this_j = lambda_val + next_t_value_func[max(0, cap_after)]
            #       else: # Reject
            #           val_from_this_j = next_t_value_func[capacity_states[cap_idx_]]
            #       sum_over_j += prob_j_at_t * val_from_this_j
            #   vartheta_table[cap_idx_, t] = sum_over_j
            # The original formulation was more vectorized and likely correct, let's stick to it.
            # Original:
            # vartheta_table[:, t] = np.sum(
            #     probs_jt[:, t, None] * (current_t_lambdas_j * optimal_decisions_y[:, :, t] + next_t_value_func[capacity_after_decision]),
            #     axis=0
            # )
            # This requires next_t_value_func[capacity_after_decision] to work with 2D index.
            # Numba handles advanced indexing: next_t_value_func is 1D, capacity_after_decision is 2D.
            # The result of next_t_value_func[capacity_after_decision] will be 2D.
            
            # Step-by-step for Numba compatibility if direct indexing is an issue (it usually isn't for this simple case)
            value_at_cap_after_decision = np.zeros_like(capacity_after_decision, dtype=np.float64)
            for j_ in range(J_itineraries):
                for c_ in range(leg_capacity + 1):
                    value_at_cap_after_decision[j_, c_] = next_t_value_func[capacity_after_decision[j_, c_]]
            
            sum_val = probs_jt[:, t].reshape(-1,1) * \
                      (current_t_lambdas_j * optimal_decisions_y[:, :, t] + value_at_cap_after_decision)
            
            vartheta_table[:, t] = np.sum(sum_val, axis=0)

        return vartheta_table, optimal_decisions_y

    # Add jitted static helper for compute_state_probabilities
    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _jit_compute_state_probabilities(
        T_periods: int,
        J_itineraries: int,
        leg_capacity: int,
        probs_jt: np.ndarray,           # self.probabilities.T (J, T)
        A_leg_consumption_j: np.ndarray,# self.A[leg_idx, :] (J,)
        y_star: np.ndarray,             # (J, C_i+1, T)
        epsilon_prob: float = 1e-9      # Epsilon for probability checks
    ) -> np.ndarray:
        """
        JIT-compiled core logic for computing state occupancy probabilities.
        """
        mu_table = np.zeros((T_periods, leg_capacity + 1))
        if leg_capacity < 0: # Should not happen
            return mu_table 
            
        mu_table[0, leg_capacity] = 1.0  # Initial state at t=0: full capacity
        
        capacity_states = np.arange(leg_capacity + 1) # (C+1,)
        # A_leg_consumption_j needs to be (J,1) for broadcasting
        A_leg_consumption_j_bc = A_leg_consumption_j.reshape(-1, 1)


        for t in range(T_periods - 1):
            mu_next_t = np.zeros(leg_capacity + 1) # Probabilities for t+1
            
            active_capacity_indices = np.where(mu_table[t] > epsilon_prob)[0]
            
            if active_capacity_indices.shape[0] == 0: # No active states
                mu_table[t+1] = mu_next_t
                continue

            active_capacity_states_arr = capacity_states[active_capacity_indices] # (num_active,)
            active_state_probs_arr = mu_table[t, active_capacity_indices]     # (num_active,)
            
            # Handle no-request case
            current_t_probs_j = probs_jt[:, t] # (J,)
            prob_sum_requests_at_t = 0.0
            for j_idx in range(J_itineraries):
                prob_sum_requests_at_t += current_t_probs_j[j_idx]
            prob_no_request = max(0.0, 1.0 - prob_sum_requests_at_t)

            if prob_no_request > epsilon_prob:
                for i in range(active_capacity_states_arr.shape[0]):
                    cap_state = active_capacity_states_arr[i]
                    prob_mass = active_state_probs_arr[i]
                    mu_next_t[cap_state] += prob_mass * prob_no_request
            
            # Handle product requests
            # y_star_slice: decisions for active capacity states at time t (J, num_active_states)
            # y_star is (J, C+1, T)
            y_star_slice = y_star[:, active_capacity_indices, t] # (J, num_active)
            
            # capacity_next_state: capacity if product j is accepted, for active states (J, num_active_states)
            # active_capacity_states_arr broadcasted to (1, num_active)
            # A_leg_consumption_j_bc is (J,1)
            # y_star_slice is (J, num_active)
            cap_reduction = A_leg_consumption_j_bc * y_star_slice # (J, num_active)
            capacity_next_state_matrix = np.maximum(0, active_capacity_states_arr.reshape(1,-1) - cap_reduction) # (J, num_active)

            # transition_probs_jt_active: prob of transitioning from an active state due to request for j (J, num_active_states)
            # current_t_probs_j broadcasted (J,1) * active_state_probs_arr broadcasted (1, num_active)
            transition_probs_jt_active = current_t_probs_j.reshape(-1,1) * active_state_probs_arr.reshape(1,-1) # (J, num_active)
            
            for j_idx in range(J_itineraries):
                for active_idx in range(active_capacity_states_arr.shape[0]):
                    if transition_probs_jt_active[j_idx, active_idx] > epsilon_prob:
                        next_cap = capacity_next_state_matrix[j_idx, active_idx]
                        prob_to_add = transition_probs_jt_active[j_idx, active_idx]
                        mu_next_t[next_cap] += prob_to_add
            
            mu_table[t+1] = mu_next_t
            # Normalize if necessary (should sum to 1 ideally if probs sum to 1)
            # sum_mu_next_t = np.sum(mu_next_t)
            # if sum_mu_next_t > epsilon_prob:
            #    mu_table[t+1] /= sum_mu_next_t
            # For numerical stability, often let it be rather than re-normalizing if theory implies sum=1.

        return mu_table

    def __init__(self, filepath: str):
        """
        Initialize an InstanceRM by parsing the given file.

        Args:
            filepath (str): Path to the airline RM instance file.
        """
        self._parse_file(filepath)
        self._ensure_probabilities_sum_to_one() # Add dummy itinerary if needed
        self._initialize_lambdas()
        self.vartheta = np.zeros((self.L, max(self.C)+1, self.T+1))  # (L, C+1, T+1)

    def _parse_file(self, filepath: str):
        """
        Parse the airline RM instance file and populate the attributes.

        Args:
            filepath (str): Path to the instance file.

        Populates:
            self.T, self.flight_legs, self.C, self.L, self.itineraries, self.F, self.J, self.probabilities, self.A
        """
        # Read all non-comment, non-empty lines
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.lstrip().startswith('#')]

        idx = 0
        num_time_periods = int(lines[idx]); idx += 1
        num_flight_legs = int(lines[idx]); idx += 1

        # Parse flight legs
        flight_legs_list = []
        capacities_list = []
        for parts in (lines[i].split() for i in range(idx, idx + num_flight_legs)):
            o, d, cap = int(parts[0]), int(parts[1]), int(parts[2])
            flight_legs_list.append(f"O{o}D{d}")
            capacities_list.append(cap)
        idx += num_flight_legs

        num_itineraries = int(lines[idx]); idx += 1

        # Parse itineraries and fares
        itineraries_info_list = []
        itineraries_list = []
        fares_list = []
        for parts in (lines[i].split() for i in range(idx, idx + num_itineraries)):
            o, d, c, fare = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            itineraries_info_list.append((o, d, c))
            itineraries_list.append(f"O{o}D{d}F{c}")
            fares_list.append(fare)
        idx += num_itineraries

        # Initialize matrices
        A_matrix = np.zeros((num_flight_legs, num_itineraries), int)
        prob_matrix = np.zeros((num_time_periods, num_itineraries), float)
        leg_to_idx = {leg: i for i, leg in enumerate(flight_legs_list)}
        itinerary_to_idx = {info: j for j, info in enumerate(itineraries_info_list)}

        # Fill A matrix
        for j, (o, d, _) in enumerate(itineraries_info_list):
            if o == 0 or d == 0: # Single leg itinerary
                A_matrix[leg_to_idx[f"O{o}D{d}"], j] = 1
            else: # Two-leg itinerary (connecting via hub 0)
                A_matrix[leg_to_idx[f"O{o}D0"], j] = 1
                A_matrix[leg_to_idx[f"O0D{d}"], j] = 1
        
        # Regex pattern for parsing probabilities
        # Matches patterns like "[ o d c ] p"
        prob_pattern = re.compile(r"\[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]\s*([\d.eE+-]+)")
        for t in range(num_time_periods):
            for o, d, c, p_val in prob_pattern.findall(lines[idx]):
                prob_matrix[t, itinerary_to_idx[(int(o), int(d), int(c))]] = float(p_val)
            idx += 1

        # Store results as numpy arrays
        self.T = num_time_periods
        self.flight_legs = np.array(flight_legs_list, dtype='<U10')
        self.C = np.array(capacities_list, dtype=int)
        self.L = len(self.flight_legs)
        self.itineraries = np.array(itineraries_list, dtype='<U20')
        self.F = np.array(fares_list, dtype=float)
        self.J = len(self.itineraries)
        self.probabilities = prob_matrix
        self.A = A_matrix

    def _ensure_probabilities_sum_to_one(self):
        """
        Ensures that the sum of probabilities for all itineraries (including a potential new dummy)
        is 1 for each time period. If the sum is less than 1, a dummy itinerary
        is added with zero fare, zero resource consumption, and the necessary
        arrival probability to make the sum 1.
        """
        prob_sum_per_t = np.sum(self.probabilities, axis=1)
        epsilon = 1e-9 # Tolerance for floating point comparison

        # Check if any time period has a sum of probabilities significantly less than 1
        if np.any(prob_sum_per_t < 1.0 - epsilon):
            dummy_itinerary_name = "DUMMY_NO_REQUEST"
            print(f"Info: Sum of probabilities per time period found to be < 1. Adding dummy itinerary '{dummy_itinerary_name}' to compensate.")

            # Update itineraries: Convert to list, append, convert back to handle dtype
            itineraries_list = self.itineraries.tolist()
            itineraries_list.append(dummy_itinerary_name)
            self.itineraries = np.array(itineraries_list) # Dtype will be inferred, typically <U based on max length
            
            # Update fares
            self.F = np.append(self.F, 0.0)
            
            # Update A matrix (add a column of zeros for the dummy itinerary)
            if self.L > 0 : # Only add column if there are legs
                dummy_A_column = np.zeros((self.L, 1), dtype=self.A.dtype)
                self.A = np.hstack((self.A, dummy_A_column))
            elif self.J == 0 : # If there were no itineraries and L=0, A might be uninitialized or 0-dim
                # This case implies A was likely shape (0,0) or similar.
                # If L=0, A should be (0, J+1). It's initialized to (L,J).
                # If self.A was (0, old_J), hstacking (0,1) makes it (0, old_J+1)
                dummy_A_column = np.zeros((0, 1), dtype=self.A.dtype)
                self.A = np.hstack((self.A, dummy_A_column))


            # Calculate probabilities for the dummy itinerary
            dummy_itinerary_probs = 1.0 - prob_sum_per_t
            dummy_itinerary_probs = np.maximum(0.0, dummy_itinerary_probs) # Ensure non-negative
            
            # Update probabilities matrix (add a column for the dummy itinerary)
            self.probabilities = np.hstack((self.probabilities, dummy_itinerary_probs.reshape(-1, 1)))
            
            # Update number of itineraries
            self.J += 1

    def _initialize_lambdas(self):
        """
        Initialize the Lagrange multiplier array.

        For each i, j, set lmd_ij = f_j / sum_k a_kj if a_ij > 0, and 0 otherwise.
        The array is then repeated along the time axis to shape (L, J, T).
        """
        if self.J == 0 or self.L == 0:
            self.lmd = np.zeros((self.L, self.J, self.T))
            return

        num_legs_used_by_itinerary = np.sum(self.A, axis=0)  # Shape (J,)

        fare_per_leg_segment = np.zeros(self.J, dtype=float)
        valid_itineraries_mask = num_legs_used_by_itinerary > 0
        
        # Ensure self.F has J elements; handle cases where valid_itineraries_mask might be all False
        if np.any(valid_itineraries_mask):
             # Check if self.F has enough elements for the mask
            if self.F.shape[0] == self.J:
                 fare_per_leg_segment[valid_itineraries_mask] = self.F[valid_itineraries_mask] / num_legs_used_by_itinerary[valid_itineraries_mask]
            else: # Fallback or error, depends on how self.F is guaranteed to be size J
                  # This case implies a potential mismatch if self.F is not size J.
                  # For now, assume self.F is always correctly sized.
                  # If F can be smaller than J before dummy, this needs adjustment.
                  # Assuming F is updated with dummy itinerary if one is added.
                  pass # Let it be zero if F is not aligned, or raise error.

        # lambdas_leg_itinerary (L, J)
        # If A_ij > 0, lambda_ij = fare_per_leg_segment[j], else 0
        # fare_per_leg_segment[None, :] broadcasts to (1, J)
        lambdas_leg_itinerary = np.where(self.A > 0, fare_per_leg_segment[None, :], 0.0)
        
        self.lmd = np.repeat(lambdas_leg_itinerary[:, :, None], self.T, axis=2)

    def solve_single_leg_dp(self, leg_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the single-resource dynamic program for a flight leg using backward induction.
        This method now calls a JIT-compiled helper function.
        """
        leg_capacity = self.C[leg_idx]
        leg_consumptions_j = self.A[leg_idx, :]  # (J,)
        # Ensure probs_jt and leg_lambdas_jt are C-contiguous for Numba if they come from slices
        # .T creates a view with Fortran order. Numba prefers C order for some ops or will make a copy.
        # Using .copy() can ensure C-order if performance issues arise, but often Numba handles it.
        probs_jt = np.ascontiguousarray(self.probabilities.T) # (J, T)
        leg_lambdas_jt = np.ascontiguousarray(self.lmd[leg_idx, :, :]) # (J, T)

        return InstanceRM._jit_solve_single_leg_dp(
            self.T, self.J, leg_capacity, leg_consumptions_j, probs_jt, leg_lambdas_jt
        )

    def compute_state_probabilities(self, leg_idx: int, y_star: np.ndarray) -> np.ndarray:
        """
        Compute state occupancy probabilities mu[t, x] for resource i.
        This method now calls a JIT-compiled helper function.
        """
        leg_capacity = self.C[leg_idx]
        if leg_capacity < 0:
            return np.zeros((self.T, 0)) # Or handle as error

        # Ensure arrays passed to Numba are contiguous if sliced.
        probs_jt = np.ascontiguousarray(self.probabilities.T) # (J, T)
        A_leg_consumption_j = np.ascontiguousarray(self.A[leg_idx, :]) # (J,)
        # y_star is (J, C_i+1, T). Ensure it's also contiguous if it's a result of operations.
        # Typically, y_star from _jit_solve_single_leg_dp should be C-contiguous by default.
        y_star_cont = np.ascontiguousarray(y_star)

        return InstanceRM._jit_compute_state_probabilities(
            self.T, self.J, leg_capacity, probs_jt, A_leg_consumption_j, y_star_cont
        )

    def compute_vartheta_subgradient(self, leg_idx: int, mu: np.ndarray, y_star: np.ndarray) -> np.ndarray:
        """
        Compute the subgradient G[j, t] for vartheta_i1(c_i1) with respect to lambda_ijs.

        Args:
            leg_idx (int): Flight leg index (unused in vectorized code).
            mu (np.ndarray): State probabilities, shape (T, C_i+1).
            y_star (np.ndarray): Optimal policies, shape (J, C_i+1, T).

        Returns:
            np.ndarray: Subgradient G[j, t] as an array of shape (J, T).
        """
        # Calculate expected y_star[j,t] by summing over reachable states (mu > 1e-9)
        mu_filtered = np.where(mu > 1e-9, mu, 0)
        expected_y_star = np.einsum('tc,jct->jt', mu_filtered, y_star, optimize=True)
        
        # Multiply by request probabilities to get final subgradient
        return self.probabilities.T * expected_y_star

    def compute_V_lambda_subgradient(self) -> tuple[float, np.ndarray]:
        """
        Compute the value of the Lagrangian relaxation V^lambda_1(c_1) and its
        subgradient with respect to each Lagrange multiplier lambda_ijt.

        Returns:
            tuple:
                float: Value of V^lambda_1(c_1).
                np.ndarray: Subgradient array of shape (L, J, T).
        """
        subgradient_matrix = np.zeros_like(self.lmd)
        sum_lambdas_over_legs = np.sum(self.lmd, axis=0)  # Sum over legs (J, T)
        fare_minus_sum_lambdas = self.F[:, None] - sum_lambdas_over_legs # (J, T)
        probs_jt = self.probabilities.T # (J, T)

        # Calculate first term of V_lambda and its contribution to subgradient
        first_term_V_lambda = np.sum(probs_jt * np.maximum(fare_minus_sum_lambdas, 0))
        # Component of subgradient from the first term: -p_jt if (f_j - sum_i l_ijt) >= 0
        subgrad_contrib_from_first_term = probs_jt * (fare_minus_sum_lambdas >= 0).astype(float)

        # Calculate second term of V_lambda (sum of vartheta_i[ci,0]) and its subgradient contribution
        sum_initial_varthetas = 0.0
        for i in range(self.L):
            vartheta_i_table, y_star_i = self.solve_single_leg_dp(i)
            sum_initial_varthetas += vartheta_i_table[self.C[i], 0]  # Value at initial capacity and t=0 for leg i
            
            # G_i is the subgradient of vartheta_i1(c_i) w.r.t. lambda_ijt (shape J, T)
            vartheta_subgrad_i = self.compute_vartheta_subgradient(i, self.compute_state_probabilities(i, y_star_i), y_star_i)
            # Subgradient of V_lambda w.r.t. lambda_ijt for this leg i
            subgradient_matrix[i] = vartheta_subgrad_i - subgrad_contrib_from_first_term

        return first_term_V_lambda + sum_initial_varthetas, subgradient_matrix

    def minimize_lr_relaxation(
        self,
        initial_lambdas: np.ndarray = None,
        alpha0: float = 1.0,
        eps: float = 1e-5,
        max_iter: int = 1000,
        verbose: bool = False,
        print_every: int = 10,
        patience_iters: int = 20  # Number of iterations without improvement to wait
    ) -> tuple[np.ndarray, list]:
        """
        Minimize the Lagrangian relaxation V^lambda_1(c_1) using projected subgradient descent.

        Args:
            initial_lambdas (np.ndarray, optional): Initial multipliers of shape (L, J, T). Defaults to a copy of self.lmd.
            alpha0 (float, optional): Initial step size. Default is 1.0.
            eps (float, optional): Convergence tolerance.
            max_iter (int, optional): Maximum number of iterations. Default is 1000.
            verbose (bool, optional): If True, print progress. Default is False.
            print_every (int, optional): Print frequency. Default is 10.
            patience_iters (int, optional): Stop if no improvement (>= eps) in best V 
                                          for this many iterations. Default 20.

        Returns:
            tuple:
                np.ndarray: Optimized multipliers of shape (L, J, T).
                list: Objective value history.
        """
        # Initialize multipliers and constraints
        self.lmd = np.copy(initial_lambdas if initial_lambdas is not None else self.lmd)
        # Mask for (leg, itinerary) pairs where itinerary j does not use leg i (A[i,j]=0)
        zero_consumption_mask = self.A == 0  # (L, J) 
        # Ensure lmd is 3D (L, J, T) before applying 2D mask across T
        if self.lmd.ndim == 3 and zero_consumption_mask.ndim == 2:
            self.lmd[zero_consumption_mask, :] = 0 
        elif self.lmd.shape[:2] == zero_consumption_mask.shape : # If lmd is (L,J) for some reason
            self.lmd[zero_consumption_mask] = 0

        V_history = []
        best_V = float('inf')
        best_V_iter = 0
        no_improvement_count = 0
        
        if verbose:
            print(f"Starting minimization: alpha0={alpha0}, eps={eps}, max_iter={max_iter}, patience_iters={patience_iters}")

        for k in range(max_iter):
            current_V, current_grad = self.compute_V_lambda_subgradient()
            V_history.append(current_V)
            
            # Update best value and check for improvement
            if current_V < best_V - eps: # Sufficient improvement
                best_V = current_V
                no_improvement_count = 0
                best_V_iter = k
            else:
                no_improvement_count += 1
            # Print and check convergence
            if verbose and (k % print_every == 0 or k == max_iter - 1):
                print(f"Iter {k:4d}: V={current_V:.1f} | Best={best_V:.1f} (@{best_V_iter}) | No improve: {no_improvement_count}/{patience_iters}")

            # Stopping condition: no improvement for patience_iters
            if no_improvement_count >= patience_iters:
                if verbose:
                    print(f"\nStopping at iter {k}: No improvement > {eps} for {patience_iters} iterations.")
                    print(f"Best V remained {best_V:.6f} from iter {best_V_iter}")
                break
                
            # Projected subgradient update
            step_size_k = alpha0 / np.sqrt(k + 1) # Diminishing step size
            self.lmd -= step_size_k * current_grad
            self.lmd = np.maximum(0, self.lmd) # Project onto non-negative orthant
            
            # Enforce lambda_ijt=0 where A[i,j]=0
            if self.lmd.ndim == 3 and zero_consumption_mask.ndim == 2:
                self.lmd[zero_consumption_mask, :] = 0
            elif self.lmd.shape[:2] == zero_consumption_mask.shape: # Fallback if lmd happens to be 2D
                 self.lmd[zero_consumption_mask] = 0

        if verbose and k == max_iter - 1 and len(V_history) == max_iter: # Check if max_iter was reached without other convergence
            print(f"Max iterations ({max_iter}) reached. Final V={V_history[-1]:.6f}")
            
        return self.lmd, V_history

    def simulate_revenue_with_bid_prices(
        self, 
        N_sim: int, 
        optimized_lambdas: np.ndarray = None
    ) -> float:
        """
        Estimate total expected revenue using the bid price policy derived from
        Lagrangian multipliers, via Monte Carlo simulation.

        Args:
            N_sim (int): Number of simulation runs.
            optimized_lambdas (np.ndarray, optional): Optimized Lagrangian multipliers
                of shape (L, J, T). If None, uses `self.lmd`.

        Returns:
            float: Estimated total expected revenue.
        """
        if self.L == 0 or N_sim <= 0:
            return 0.0

        # Backup and set multipliers if provided
        original_lmd_backup = None
        if optimized_lambdas is not None:
            original_lmd_backup = np.copy(self.lmd) 
            self.lmd = np.copy(optimized_lambdas)

        # Pre-compute value functions (vartheta tables) for all legs using current self.lmd
        # self.vartheta stores these tables: (L, max_C+1, T+1)
        for i in range(self.L):
            vartheta_i_table, _ = self.solve_single_leg_dp(i) # vartheta_i_table is (C_i+1, T+1)
            self.vartheta[i, :(self.C[i]+1), :] = vartheta_i_table
            # Extend values for capacities > C_i if vartheta table is larger (due to max_C)
            # This assumes value is constant for capacities beyond actual C_i (reasonable for DP structure)
            if self.C[i] < self.vartheta.shape[1] - 1:
                 self.vartheta[i, (self.C[i]+1):, :] = vartheta_i_table[self.C[i], :][np.newaxis, :]

        # Pre-compute cumulative probability arrays for faster sampling of arriving requests
        # For each time t, cumulative_arrival_probs_t[t, j] = sum_{k<=j} P(request k or no request)
        cumulative_arrival_probs_t = np.zeros((self.T, self.J + 1)) # +1 for no-request
        for t in range(self.T):
            arrival_probs_current_t = self.probabilities[t, :] # Probabilities for itineraries at time t
            prob_no_request_current_t = max(0.0, 1.0 - np.sum(arrival_probs_current_t))
            # Extended probabilities including no-request pseudo-itinerary (index J)
            extended_probs_at_t = np.append(arrival_probs_current_t, prob_no_request_current_t)
            sum_probs_at_t = np.sum(extended_probs_at_t)
            if sum_probs_at_t > 1e-9: # Normalize if sum is not zero
                # Ensure non-negative and normalize
                extended_probs_at_t = np.maximum(extended_probs_at_t, 0) / sum_probs_at_t 
            cumulative_arrival_probs_t[t, :] = np.cumsum(extended_probs_at_t)

        # Pre-compute consumption patterns and involved legs for quick lookup
        itinerary_consumption_vectors = self.A.T  # Shape (J, L); row j is consumption vector for itinerary j
        # List of boolean masks, one per itinerary, indicating which legs it uses
        itinerary_leg_usage_masks = [itinerary_consumption_vectors[j] > 0 for j in range(self.J)]

        # Monte Carlo simulation
        total_simulated_revenue = 0.0

        for _ in range(N_sim):
            current_run_revenue = 0.0
            current_capacities = np.copy(self.C) # Capacities for this simulation run

            for t in range(self.T): # Iterate over time periods
                # Fast sampling of itinerary request (or no request) using pre-computed cumulative probabilities
                if self.J == 0 or cumulative_arrival_probs_t[t, -1] <= 1e-9: # No itineraries or no chance of request
                    continue # Go to next time period
                    
                random_sample = np.random.random() # Uniform random number in [0,1)
                # Find first index where cumulative_arrival_probs_t >= random_sample
                sampled_idx = np.searchsorted(cumulative_arrival_probs_t[t, :], random_sample)
                
                if sampled_idx >= self.J:  # This means no actual itinerary was requested (sampled the no-request part)
                    continue # Go to next time period
                
                # Process request for itinerary req_itinerary_idx
                req_itinerary_idx = sampled_idx 
                req_itinerary_consumption = itinerary_consumption_vectors[req_itinerary_idx] # Consumption vector (L,)
                req_itinerary_legs_mask = itinerary_leg_usage_masks[req_itinerary_idx] # Boolean mask (L,)
                
                # Check capacity constraints for all legs involved in this itinerary
                if not np.all(current_capacities[req_itinerary_legs_mask] >= req_itinerary_consumption[req_itinerary_legs_mask]):
                    continue  # Insufficient capacity on at least one required leg

                # Calculate total opportunity cost for accepting this itinerary request
                # This sums (V(x_i, t+1) - V(x_i - a_ij, t+1)) over all legs i used by itinerary j
                involved_leg_indices = np.where(req_itinerary_legs_mask)[0]
                involved_leg_current_caps = current_capacities[involved_leg_indices]
                involved_leg_consumption_units = req_itinerary_consumption[involved_leg_indices]
                involved_leg_new_caps = involved_leg_current_caps - involved_leg_consumption_units
                
                # Fetch values from pre-computed self.vartheta table
                # self.vartheta has shape (L, max_C+1, T+1)
                # Need to index carefully for involved_leg_indices, their capacities, and time t+1
                val_at_current_caps_involved = self.vartheta[involved_leg_indices, involved_leg_current_caps, t + 1]
                val_at_new_caps_involved = self.vartheta[involved_leg_indices, involved_leg_new_caps, t + 1]
                
                opportunity_cost = np.sum(val_at_current_caps_involved - val_at_new_caps_involved)
                
                # Acceptance decision: accept if fare covers or exceeds opportunity cost
                if self.F[req_itinerary_idx] >= opportunity_cost:
                    current_run_revenue += self.F[req_itinerary_idx]
                    # Update capacities for the legs consumed by the accepted itinerary
                    current_capacities[req_itinerary_legs_mask] -= req_itinerary_consumption[req_itinerary_legs_mask]

            total_simulated_revenue += current_run_revenue
        
        estimated_expected_revenue = total_simulated_revenue / N_sim

        # Restore original multipliers if they were changed for this simulation
        if original_lmd_backup is not None:
            self.lmd = original_lmd_backup
            
        return estimated_expected_revenue


if __name__ == "__main__":
    import os

    data_path = "data/600_rm_datasets/rm_600_4_1.0_4.0.txt"
    inst = InstanceRM(data_path)

    # Print basic instance info
    print(f"T: {inst.T}")
    print(f"Flight legs: {inst.flight_legs}")
    print(f"Capacities: {inst.C}")
    print(f"Itineraries: {inst.itineraries}")
    print(f"Fares: {inst.F}")
    print(f"A shape: {inst.A.shape}\n{inst.A}")
    print(f"Probabilities shape: {inst.probabilities.shape}")
    print(f"First 2 time periods, 5 itineraries:\n{inst.probabilities[:2, :5]}\n...")

    # Prepare debug directory
    os.makedirs("debug", exist_ok=True)

    # Test single leg DP
    leg = 0
    print(f"\nTesting solve_single_leg_dp for leg {leg}:")
    vartheta, y_star = inst.solve_single_leg_dp(leg)
    print(f"  vartheta shape: {vartheta.shape}, y_star shape: {y_star.shape}")
    print(f"  Optimal expected value (full cap, t=0): {vartheta[inst.C[leg], 0]:.4f}")
    print("  y_star (first 3 itineraries, full cap, first 3 time periods):")
    print(y_star[:3, inst.C[leg], :3])
    assert vartheta.shape == (inst.C[leg]+1, inst.T+1)
    assert y_star.shape == (inst.J, inst.C[leg]+1, inst.T)
    assert np.all((y_star == 0) | (y_star == 1))
    print("Single-leg DP test passed.")

    # # Check dimensions of probability array
    # print(f"\nChecking probability array dimensions:")
    # print(f"  probabilities shape: {inst.probabilities.shape}")
    # print(f"  Expected shape: ({inst.T}, {inst.J})")
    # print(f"  Shape matches expected: {inst.probabilities.shape == (inst.T, inst.J)}")
    # print(f"  Sum of probabilities per time period (first 3):")
    # for t in range(min(3, inst.T)):
    #     prob_sum = np.sum(inst.probabilities[t, :])
    #     print(f"    t={t}: {prob_sum:.6f}")
    # # Print row sums (total probability per time period)
    # print(f"\nRow sums (total probability per time period):")
    # for t in range(inst.T):
    #     row_sum = np.sum(inst.probabilities[t, :])
    #     print(f"    t={t}: {row_sum:.6f}")
    # # Save probabilities to debug file
    # np.savetxt(f"debug/probabilities.txt", inst.probabilities, fmt="%.6f")

    # Save results
    np.savetxt(f"debug/lambda_t0.txt", inst.lmd[:, :, 0], fmt="%.6f")
    np.savetxt(f"debug/lambda_leg{leg}_t0.txt", inst.lmd[leg, :, 0], fmt="%.6f")
    np.savetxt(f"debug/vartheta_leg{leg}.txt", vartheta, fmt="%.6f")
    np.savetxt(f"debug/y_star_leg{leg}_t0.txt", y_star[:, :, 0], fmt="%.6f")
    np.savetxt(f"debug/y_star_leg{leg}_x{inst.C[leg]}.txt", y_star[:, inst.C[leg], :], fmt="%.6f")

    # Test compute_mu and subgradient
    print(f"\nTesting compute_mu and compute_vartheta_subgradient for leg {leg}:")
    mu = inst.compute_state_probabilities(leg_idx=leg, y_star=y_star)
    print(f"  mu shape: {mu.shape}")
    for t in range(inst.T):
        assert np.isclose(np.sum(mu[t, :]), 1.0)
    print("  mu sums to 1 for all t.")
    print(f"  mu[0, C_leg]: {mu[0, inst.C[leg]]:.4f}")
    if inst.T > 1 and inst.C[leg] > 0:
        print(f"  mu[1, C_leg]: {mu[1, inst.C[leg]]:.4f}")
        print(f"  mu[1, C_leg-1]: {mu[1, inst.C[leg]-1]:.4f}")

    np.savetxt(f"debug/mu_leg{leg}.txt", mu, fmt="%.6f")

    G = inst.compute_vartheta_subgradient(leg_idx=leg, mu=mu, y_star=y_star)
    print(f"  G shape: {G.shape}")
    print("  G sample (first 3 itineraries, first 3 time periods):")
    print(G[:3, :3])
    np.savetxt(f"debug/G_leg{leg}.txt", G, fmt="%.6f")
    print("mu and subgradient tests passed.")

    # Test compute_V_lambda_subgradient
    print("\nTesting compute_V_lambda_subgradient:")
    V_lambda_val, V_lambda_subgrad = inst.compute_V_lambda_subgradient()
    print(f"  V_lambda_1(c_1): {V_lambda_val:.4f}")
    print(f"  V_lambda_subgradient shape: {V_lambda_subgrad.shape}")
    assert V_lambda_subgrad.shape == (inst.L, inst.J, inst.T)
    print("  V_lambda_subgradient sample (leg 0, first 3 itineraries, first 3 time periods):")
    print(V_lambda_subgrad[0, :3, :3])
    np.savetxt(f"debug/V_lambda_subgradient_leg{leg}.txt", V_lambda_subgrad[leg,:,:], fmt="%.6f")
    print("compute_V_lambda_subgradient test passed.")

    # Test minimize_lr_relaxation
    print("\nTesting minimize_lr_relaxation:")
    inst2 = InstanceRM(data_path)
    optimized_lmd, V_history = inst2.minimize_lr_relaxation(
        alpha0=100.0,
        eps=1e-4,
        max_iter=10000,
        verbose=True,
        print_every=10,
        patience_iters=20  # Added new parameter for testing
    )
    print(f"  Optimization finished. V_lambda history: {len(V_history)}")
    if V_history:
        print(f"  Initial V_lambda: {V_history[0]:.4f}, Final V_lambda: {V_history[-1]:.4f}")
    print(f"  Optimized lambda shape: {optimized_lmd.shape}")
    assert optimized_lmd.shape == inst2.lmd.shape

    np.save("debug/optimized_lmd.npy", optimized_lmd)
    np.savetxt("debug/V_lambda_history.txt", V_history, fmt="%.6f")
    print("minimize_lr_relaxation test run completed.")

    # Test estimate_revenue_monte_carlo_bid_price
    print("\nTesting simulate_revenue_with_bid_prices:")

    N_SIMULATIONS = 1000  # Number of simulation runs for revenue estimation

    # Collect simulated revenues for CI calculation
    simulated_revenues = []
    for _ in range(N_SIMULATIONS):
        rev = inst2.simulate_revenue_with_bid_prices(1)
        simulated_revenues.append(rev)
    simulated_revenues = np.array(simulated_revenues)
    estimated_rev = np.mean(simulated_revenues)
    std_err = np.std(simulated_revenues, ddof=1) / np.sqrt(N_SIMULATIONS)
    ci_low = estimated_rev - 1.96 * std_err
    ci_high = estimated_rev + 1.96 * std_err
    print(f"  Estimated revenue over {N_SIMULATIONS} simulations: {estimated_rev:.4f}")
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    # Example using a specific lambda (e.g., the initial one for comparison)
    inst3 = InstanceRM(data_path)  # Fresh instance with initial lambda
    initial_lmd_for_test = np.copy(inst3.lmd)
    simulated_revenues_init = []
    for _ in range(N_SIMULATIONS):
        rev = inst3.simulate_revenue_with_bid_prices(
            1,
            optimized_lambdas=initial_lmd_for_test
        )
        simulated_revenues_init.append(rev)
    simulated_revenues_init = np.array(simulated_revenues_init)
    estimated_rev_initial_lmd = np.mean(simulated_revenues_init)
    std_err_init = np.std(simulated_revenues_init, ddof=1) / np.sqrt(N_SIMULATIONS)
    ci_low_init = estimated_rev_initial_lmd - 1.96 * std_err_init
    ci_high_init = estimated_rev_initial_lmd + 1.96 * std_err_init
    print(f"  Estimated revenue (using initial lambdas) over {N_SIMULATIONS} simulations: {estimated_rev_initial_lmd:.4f}")
    print(f"  95% CI (initial lambdas): [{ci_low_init:.4f}, {ci_high_init:.4f}]")
    print("simulate_revenue_with_bid_prices test run completed.")
