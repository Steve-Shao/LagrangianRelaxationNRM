import re
import numpy as np


class InstanceRM:
    """
    InstanceRM parses an airline revenue management test problem file.

    The file encodes a single-leg or network airline RM instance. It contains 
    flight legs, itineraries, fares, and a time-indexed demand probability matrix.

    Attributes:
        T (int): Number of time periods.
        flight_legs (np.ndarray[str]): Array of flight leg names, each as "O{o}D{d}".
        C (np.ndarray[int]): Array of capacities for each flight leg.
        L (int): Number of flight legs.
        itineraries (np.ndarray[str]): Array of itinerary names, each as "O{o}D{d}F{c}".
        F (np.ndarray[float]): Array of fares for each itinerary.
        J (int): Number of itineraries.
        probabilities (np.ndarray[float]): Matrix of shape (T, J). probabilities[t, j] = 
            probability of request for itinerary j at time t.
        A (np.ndarray[int]): Matrix of shape (L, J). A[i, j] = 1 if itinerary j uses leg i, 
            0 otherwise.
        lmd (np.ndarray[float]): Array of shape (L, J, T) with Lagrange multipliers.
        vartheta (np.ndarray[float]): Array of shape (L, max(C)+1, T+1) for value functions.

    File Format:
        - First line: integer τ, the number of time periods.
        - Second line: integer N, the number of flight legs.
        - Next N lines: each line has three integers o d cap, the origin, destination, 
          and capacity of a flight leg.
        - Next line: integer M, the number of itineraries.
        - Next M lines: each line has four values: o d c fare, the origin, destination, 
          fare class, and fare of an itinerary.
        - Next τ lines: each line lists demand probabilities for that time period, 
          in the format:
            [o d c] p  [o d c] p  ... (for all itineraries)
          where [o d c] identifies the itinerary and p is the probability of a request 
          for that itinerary at that time.

    Methods:
        __init__(filepath): Loads and parses the instance file.
        _parse_file(filepath): Parses the file and sets attributes.
        _initialize_lambdas(): Initializes the Lagrange multiplier array.
        solve_single_leg_dp(leg_idx): Solves the single-resource dynamic program for 
            a specific flight leg.
        compute_state_probabilities(leg_idx, y_star): Compute state occupancy probabilities 
            mu[t, x] for a resource i.
        compute_vartheta_subgradient(leg_idx, mu, y_star): Compute the subgradient G[j, t] 
            of the single-resource value function with respect to Lagrange multipliers.
        compute_V_lambda_subgradient(): Compute the value of the Lagrangian relaxation 
            and its subgradient with respect to each Lagrange multiplier.
        minimize_lr_relaxation(lmd0=None, alpha0=1.0, eps=1e-5, max_iter=1000, 
            verbose=False, print_every=10): Minimize the Lagrangian relaxation using 
            projected subgradient descent.
        simulate_revenue_with_bid_prices(N_sim: int, optimized_lmd_param: np.ndarray = None) 
            -> float: Estimate total expected revenue using the bid price policy derived 
            from Lagrangian multipliers, via Monte Carlo simulation.
    """

    def __init__(self, filepath: str):
        """
        Initialize an InstanceRM by parsing the given file.

        Args:
            filepath (str): Path to the airline RM instance file.
        """
        self._parse_file(filepath)
        self._ensure_probabilities_sum_to_one()  # Add dummy itinerary if needed
        self._initialize_lambdas()
        self.vartheta = np.zeros((self.L, max(self.C) + 1, self.T + 1))  # (L, C+1, T+1)

    def _parse_file(self, filepath: str):
        """
        Parse the airline RM instance file and populate the instance attributes.

        The method reads the file line by line, interpreting sections based on the
        format specified in the class docstring. It extracts data for time periods,
        flight legs (with capacities), itineraries (with fares), and time-indexed
        demand probabilities. This data is then used to populate the various
        attributes of the `InstanceRM` object, such as `self.T`, `self.flight_legs`,
        `self.C`, `self.A`, `self.probabilities`, etc.

        Args:
            filepath (str): Path to the instance file.

        Populates:
            self.T: Number of time periods.
            self.flight_legs: NumPy array of flight leg names (e.g., "O1D2").
            self.C: NumPy array of capacities for each flight leg.
            self.L: Number of flight legs.
            self.itineraries: NumPy array of itinerary names (e.g., "O1D2F3").
            self.F: NumPy array of fares for each itinerary.
            self.J: Number of itineraries.
            self.probabilities: NumPy array (T, J) of demand probabilities.
            self.A: NumPy array (L, J), the flight leg-itinerary incidence matrix.
        """
        with open(filepath, 'r') as f:
            # Read all lines, stripping whitespace and removing comments/empty lines.
            lines = [line.strip() for line in f 
                    if line.strip() and not line.lstrip().startswith('#')]

        line_iterator = iter(lines)

        # Section 1: Read global parameters
        # First line: τ (number of time periods)
        num_time_periods = int(next(line_iterator))
        
        # Second line: N (number of flight legs)
        num_flight_legs = int(next(line_iterator))

        # Section 2: Parse Flight Leg Definitions
        # Next N lines: origin_id destination_id capacity
        parsed_flight_legs_list = []
        parsed_capacities_list = []
        for _ in range(num_flight_legs):
            line_content = next(line_iterator)
            parts = line_content.split()
            leg_origin_id, leg_dest_id, leg_capacity = (
                int(parts[0]), int(parts[1]), int(parts[2])
            )
            
            # Flight leg names are formatted as "O{origin}D{destination}"
            leg_name = f"O{leg_origin_id}D{leg_dest_id}"
            parsed_flight_legs_list.append(leg_name)
            parsed_capacities_list.append(leg_capacity)

        # Line after flight legs: M (number of itineraries)
        num_itineraries = int(next(line_iterator))

        # Section 3: Parse Itinerary Definitions
        # Next M lines: origin_id destination_id fare_class_id fare_value
        # These o, d, c define the overall itinerary.
        itinerary_key_tuples = []  # Stores (origin_id, dest_id, fare_class_id) tuples for mapping
        parsed_itineraries_list = []  # Stores "O{o}D{d}F{c}" formatted names
        parsed_fares_list = []
        for _ in range(num_itineraries):
            line_content = next(line_iterator)
            parts = line_content.split()
            itin_origin_id, itin_dest_id, itin_fare_class_id, itin_fare_value = (
                int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            )
            
            # Itinerary names are formatted as "O{origin}D{destination}F{fare_class}"
            itinerary_name = f"O{itin_origin_id}D{itin_dest_id}F{itin_fare_class_id}"
            parsed_itineraries_list.append(itinerary_name)
            parsed_fares_list.append(itin_fare_value)
            
            # This tuple (o,d,c) serves as a unique key for the itinerary definition.
            itinerary_key = (itin_origin_id, itin_dest_id, itin_fare_class_id)
            itinerary_key_tuples.append(itinerary_key)

        # Section 4: Initialize Data Structures and Mappings
        # Create mappings for quick lookup of leg/itinerary indices.
        leg_to_idx = {name: i for i, name in enumerate(parsed_flight_legs_list)}
        itinerary_to_idx = {key: j for j, key in enumerate(itinerary_key_tuples)}

        # Initialize A matrix (leg-itinerary incidence) and probability matrix.
        A_matrix = np.zeros((num_flight_legs, num_itineraries), dtype=int)
        probabilities_matrix = np.zeros((num_time_periods, num_itineraries), dtype=float)

        # Section 5: Populate A Matrix (Leg-Itinerary Incidence)
        # This logic determines which flight legs are part of each itinerary.
        # It assumes a hub-spoke model where '0' is the hub.
        for j, (itin_o, itin_d, _) in enumerate(itinerary_key_tuples):
            # (itin_o, itin_d) are the overall origin and destination of itinerary j.
            
            # Case 1: Itinerary's origin or destination is the hub (0).
            # This is treated as a single-leg itinerary.
            # Example: Itinerary (0, 5, 1) (O0D5F1) uses leg "O0D5".
            if itin_o == 0 or itin_d == 0:
                single_leg_name = f"O{itin_o}D{itin_d}"
                if single_leg_name in leg_to_idx:
                    A_matrix[leg_to_idx[single_leg_name], j] = 1
                # else: Consider error handling for missing leg.
            
            # Case 2: Itinerary is between two non-hub airports.
            # This is assumed to be a two-leg itinerary connecting via hub 0.
            # Example: Itinerary (3, 5, 1) (O3D5F1) uses legs "O3D0" and "O0D5".
            else:
                leg1_name = f"O{itin_o}D0"  # Leg from itinerary origin to hub
                leg2_name = f"O0D{itin_d}"  # Leg from hub to itinerary destination
                
                if leg1_name in leg_to_idx:
                    A_matrix[leg_to_idx[leg1_name], j] = 1
                # else: Consider error handling for missing leg.
                
                if leg2_name in leg_to_idx:
                    A_matrix[leg_to_idx[leg2_name], j] = 1
                # else: Consider error handling for missing leg.
        
        # Section 6: Parse Demand Probabilities
        # Next τ lines: each line lists demand probabilities for that time period.
        # Format: "[o d c] p  [o d c] p ..."
        
        # Regex to capture "[ o d c ] p" patterns.
        prob_entry_pattern = re.compile(r"\[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]\s*([\d.eE+-]+)")

        for t in range(num_time_periods):
            current_prob_line = next(line_iterator)
            
            # Find all itinerary probability entries in the current line.
            for o_str, d_str, c_str, p_str in prob_entry_pattern.findall(current_prob_line):
                itinerary_key_for_prob = (int(o_str), int(d_str), int(c_str))
                probability_value = float(p_str)
                
                if itinerary_key_for_prob in itinerary_to_idx:
                    j = itinerary_to_idx[itinerary_key_for_prob]
                    probabilities_matrix[t, j] = probability_value
                # else: Consider error handling for itinerary in prob data not defined earlier.
        
        # Section 7: Store Parsed Data into Instance Attributes
        self.T = num_time_periods
        self.flight_legs = np.array(parsed_flight_legs_list, dtype='<U10')
        self.C = np.array(parsed_capacities_list, dtype=int)
        self.L = num_flight_legs
        
        self.itineraries = np.array(parsed_itineraries_list, dtype='<U20')
        self.F = np.array(parsed_fares_list, dtype=float)
        self.J = num_itineraries
        
        self.probabilities = probabilities_matrix
        self.A = A_matrix

    def _ensure_probabilities_sum_to_one(self):
        """
        Ensures that the sum of probabilities for all itineraries (including a potential 
        new dummy) is 1 for each time period. If the sum is less than 1, a dummy itinerary
        is added with zero fare, zero resource consumption, and the necessary
        arrival probability to make the sum 1.
        """
        prob_sum_per_t = np.sum(self.probabilities, axis=1)
        epsilon = 1e-9  # Tolerance for floating point comparison

        # Check if any time period has a sum of probabilities significantly less than 1
        if np.any(prob_sum_per_t < 1.0 - epsilon):
            dummy_itinerary_name = "DUMMY_NO_REQUEST"
            print(f"Info: Sum of probabilities per time period found to be < 1. "
                  f"Adding dummy itinerary '{dummy_itinerary_name}' to compensate.")

            # Update itineraries: Convert to list, append, convert back to handle dtype
            itineraries_list = self.itineraries.tolist()
            itineraries_list.append(dummy_itinerary_name)
            self.itineraries = np.array(itineraries_list)  # Dtype will be inferred
            
            # Update fares
            self.F = np.append(self.F, 0.0)
            
            # Update A matrix (add a column of zeros for the dummy itinerary)
            if self.L > 0:  # Only add column if there are legs
                dummy_A_column = np.zeros((self.L, 1), dtype=self.A.dtype)
                self.A = np.hstack((self.A, dummy_A_column))
            elif self.J == 0:  # If there were no itineraries and L=0, A might be uninitialized
                # This case implies A was likely shape (0,0) or similar.
                # If L=0, A should be (0, J+1). It's initialized to (L,J).
                # If self.A was (0, old_J), hstacking (0,1) makes it (0, old_J+1)
                dummy_A_column = np.zeros((0, 1), dtype=self.A.dtype)
                self.A = np.hstack((self.A, dummy_A_column))

            # Calculate probabilities for the dummy itinerary
            dummy_itinerary_probs = 1.0 - prob_sum_per_t
            dummy_itinerary_probs = np.maximum(0.0, dummy_itinerary_probs)  # Ensure non-negative
            
            # Update probabilities matrix (add a column for the dummy itinerary)
            self.probabilities = np.hstack((self.probabilities, 
                                          dummy_itinerary_probs.reshape(-1, 1)))
            
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
                fare_per_leg_segment[valid_itineraries_mask] = (
                    self.F[valid_itineraries_mask] / num_legs_used_by_itinerary[valid_itineraries_mask]
                )
            else:  # Fallback or error, depends on how self.F is guaranteed to be size J
                # This case implies a potential mismatch if self.F is not size J.
                # For now, assume self.F is always correctly sized.
                # If F can be smaller than J before dummy, this needs adjustment.
                # Assuming F is updated with dummy itinerary if one is added.
                pass  # Let it be zero if F is not aligned, or raise error.

        # lambdas_leg_itinerary (L, J)
        # If A_ij > 0, lambda_ij = fare_per_leg_segment[j], else 0
        # fare_per_leg_segment[None, :] broadcasts to (1, J)
        lambdas_leg_itinerary = np.where(self.A > 0, fare_per_leg_segment[None, :], 0.0)
        
        self.lmd = np.repeat(lambdas_leg_itinerary[:, :, None], self.T, axis=2)

    def solve_single_leg_dp(self, leg_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the single-resource dynamic program for a flight leg using backward induction.
        Pure numpy implementation (no numba).
        
        Returns:
            tuple: (vartheta_table, y_star)
                vartheta_table: (C+1, T+1)
                y_star: (J, C+1, T)
        """
        leg_capacity = self.C[leg_idx]
        J = self.J
        T = self.T
        leg_consumptions_j = self.A[leg_idx, :]  # Shape: (J,)
        probs_jt = self.probabilities.T  # Shape: (J, T)
        leg_lambdas_jt = self.lmd[leg_idx, :, :]  # Shape: (J, T)

        # vartheta_table[c, t] stores V_t(c) for capacity c at time t
        vartheta_table = np.zeros((leg_capacity + 1, T + 1))
        # y_star[j, c, t] stores optimal decision for product j, capacity c, time t
        y_star = np.zeros((J, leg_capacity + 1, T), dtype=int)
        
        # Array of capacity states [0, 1, ..., leg_capacity]
        capacity_states = np.arange(leg_capacity + 1)  # Shape: (leg_capacity + 1,)

        # Broadcastable versions of consumptions and capacity states
        # consumptions_bc[j, c] = leg_consumptions_j[j]
        consumptions_bc = leg_consumptions_j.reshape(-1, 1)  # Shape: (J, 1)
        # capacity_states_bc[j, c] = capacity_states[c]
        capacity_states_bc = capacity_states.reshape(1, -1)  # Shape: (1, leg_capacity + 1)

        # sufficient_capacity_mask[j, c] is True if product j can be accepted with capacity c
        sufficient_capacity_mask = (consumptions_bc <= capacity_states_bc)  # Shape: (J, leg_capacity + 1)

        # Backward induction loop through time
        for t in range(T - 1, -1, -1):
            # Value function for the next time period V_{t+1}(c)
            next_t_value_func = vartheta_table[:, t + 1]  # Shape: (leg_capacity + 1,)
            
            # Lagrange multipliers for the current time period lambda_{ijt}
            current_t_lambdas_j = leg_lambdas_jt[:, t].reshape(-1, 1)  # Shape: (J, 1)

            # Calculate capacity if product j is accepted: c - a_ij
            # Shape: (J, leg_capacity + 1)
            capacity_if_accept = capacity_states_bc - consumptions_bc
            
            # Clip capacities at 0, as capacity cannot be negative
            # These are the indices for next_t_value_func
            valid_capacity_indices_if_accept = np.maximum(0, capacity_if_accept)  # Shape: (J, leg_capacity + 1)

            # Calculate V_{t+1}(c - a_ij)
            # Initialize with -np.inf for cases where capacity is insufficient
            future_value_if_accept = np.full((J, leg_capacity + 1), -np.inf, dtype=float)
            
            # Get values from next_t_value_func using valid_capacity_indices_if_accept as indices
            potential_future_values = next_t_value_func[valid_capacity_indices_if_accept]  # Shape: (J, leg_capacity + 1)
            
            # Apply these values only where capacity is sufficient
            future_value_if_accept = np.where(sufficient_capacity_mask, potential_future_values, -np.inf)

            # Value if product j is accepted: lambda_ij + V_{t+1}(c - a_ij)
            val_accept = current_t_lambdas_j + future_value_if_accept  # Shape: (J, leg_capacity + 1)
            
            # Value if product j is rejected: V_{t+1}(c)
            # Reshape for broadcasting with val_accept
            val_reject = next_t_value_func.reshape(1, -1)  # Shape: (1, leg_capacity + 1)
            
            # Optimal decision: accept if val_accept > val_reject
            current_t_optimal_decisions = (val_accept > val_reject).astype(int)  # Shape: (J, leg_capacity + 1)
            y_star[:, :, t] = current_t_optimal_decisions

            # Capacity after making the optimal decision: c - a_ij * y*_ij(c)
            # Shape: (J, leg_capacity + 1)
            capacity_after_decision = capacity_states_bc - consumptions_bc * current_t_optimal_decisions
            capacity_after_decision = np.maximum(0, capacity_after_decision)  # Clip at 0

            # V_{t+1}(c - a_ij * y*_ij(c))
            # Shape: (J, leg_capacity + 1)
            value_at_cap_after_decision = next_t_value_func[capacity_after_decision]
            
            # Expected value contribution for each product j:
            # p_jt * (lambda_ijt * y*_ijt(c) + V_{t+1}(c - a_ij * y*_ijt(c)))
            # Shape: (J, leg_capacity + 1)
            expected_value_contribution = probs_jt[:, t].reshape(-1, 1) * (
                current_t_lambdas_j * current_t_optimal_decisions + value_at_cap_after_decision
            )
            
            # V_t(c) = sum over j of expected_value_contribution
            # Sum over axis 0 (products)
            vartheta_table[:, t] = np.sum(expected_value_contribution, axis=0)  # Shape: (leg_capacity + 1,)
            
        return vartheta_table, y_star

    def compute_state_probabilities(self, leg_idx: int, y_star: np.ndarray) -> np.ndarray:
        """
        Compute state occupancy probabilities mu[t, x] for resource i (leg_idx).
        Vectorized numpy implementation.

        Args:
            leg_idx (int): Index of the leg.
            y_star (np.ndarray): Optimal policies for this leg, shape (J, C_i+1, T).
                                 y_star[j, cap, t] is 1 if accept product j at capacity cap 
                                 at time t, else 0.

        Returns:
            mu_table (np.ndarray): State probabilities, shape (T, C_i+1).
                                   mu_table[t, cap] is P(capacity = cap at time t).
        """
        leg_capacity = self.C[leg_idx]  # Initial capacity for this leg C_i
        
        if leg_capacity < 0:  # Should ideally not happen with valid inputs
            return np.zeros((self.T, 0))

        if self.T == 0:
            return np.zeros((0, leg_capacity + 1))

        # mu_table[t, cap_val] stores the probability of having cap_val capacity at time t
        # Dimensions: T rows (time periods), leg_capacity + 1 columns (capacity states 0 to C_i)
        mu_table = np.zeros((self.T, leg_capacity + 1))

        # Initial condition: at t=0, leg has its full initial capacity C_i with probability 1.
        # mu_table[0, C_i] = 1.0. If C_i is 0, mu_table[0,0] = 1.0.
        mu_table[0, leg_capacity] = 1.0
        
        # Local constant for probability threshold
        epsilon_prob = 1e-9

        # probs_jt[j, t] is probability of product j arriving at time t
        probs_jt = self.probabilities.T  # Shape: (J, T)
        
        # A_leg_consumption_j[j] is consumption of leg by product j
        A_leg_consumption_j = self.A[leg_idx, :]  # Shape: (J,)
        # Broadcast A_leg_consumption_j for calculations: (J, 1)
        A_leg_consumption_j_bc = A_leg_consumption_j.reshape(-1, 1)

        # Iterate over time periods t = 0, ..., T-2 to compute mu_table[t+1]
        for t in range(self.T - 1):
            mu_current_t_probs = mu_table[t, :]  # Probabilities of capacity states at current time t. Shape: (C_i+1,)
            mu_next_t_probs = np.zeros(leg_capacity + 1)  # Initialize probs for next time t+1. Shape: (C_i+1,)

            # Find active capacity states at time t (states with P(cap) > epsilon_prob)
            # active_capacity_values are the actual capacity values that are active (e.g., [c1, c2, ...])
            active_capacity_values = np.where(mu_current_t_probs > epsilon_prob)[0]

            if active_capacity_values.shape[0] == 0:  # No reachable states at time t
                mu_table[t + 1, :] = mu_next_t_probs  # which is all zeros
                continue
            
            # Probabilities of these active capacity states
            active_state_probs_arr = mu_current_t_probs[active_capacity_values]  # Shape: (num_active_caps,)

            # Part 1: Transitions if no (actual) product request arrives.
            # This handles the probability mass if sum of p_jt < 1 (i.e. no dummy "no-op" product).
            # If a dummy product is included in J and probs_jt such that sum(p_jt)=1, this part might be redundant.
            current_t_arrival_probs_j = probs_jt[:, t]  # Arrival probs for products at time t. Shape: (J,)
            prob_sum_actual_requests_at_t = np.sum(current_t_arrival_probs_j)
            prob_no_request_transition = max(0.0, 1.0 - prob_sum_actual_requests_at_t) 

            if prob_no_request_transition > epsilon_prob:
                # If no request, capacity remains unchanged.
                # Add P(active_cap) * P(no_request) to mu_next_t_probs[active_cap]
                np.add.at(mu_next_t_probs, 
                          active_capacity_values,  # Target capacity states (indices for mu_next_t_probs)
                          active_state_probs_arr * prob_no_request_transition)  # Probabilities to add

            # Part 2: Transitions if an (actual) product j requests
            # y_star shape is (J, C_i+1, T)
            # y_star_slice_t_active_caps[j, k] is optimal decision for product j if current capacity is active_capacity_values[k] at time t
            y_star_slice_t_active_caps = y_star[:, active_capacity_values, t]  # Shape: (J, num_active_caps)

            # cap_reduction_matrix[j, k]: capacity reduction for product j if current capacity is active_capacity_values[k]
            cap_reduction_matrix = A_leg_consumption_j_bc * y_star_slice_t_active_caps  # Shape: (J, num_active_caps)

            # active_capacity_values_bc is (1, num_active_caps)
            active_capacity_values_bc = active_capacity_values.reshape(1, -1) 
            # capacity_next_state_matrix[j, k]: next capacity if product j arrives and current capacity is active_capacity_values[k]
            capacity_next_state_matrix = np.maximum(0, active_capacity_values_bc - cap_reduction_matrix)  # Shape: (J, num_active_caps)

            # transition_probs_matrix[j, k]: P(product j arrives AND current capacity is active_capacity_values[k])
            #   = P(product j arrives at t) * P(current capacity is active_capacity_values[k] at t)
            current_t_arrival_probs_j_bc = current_t_arrival_probs_j.reshape(-1, 1)  # Shape: (J, 1)
            active_state_probs_arr_bc = active_state_probs_arr.reshape(1, -1)  # Shape: (1, num_active_caps)
            
            transition_probs_matrix = current_t_arrival_probs_j_bc * active_state_probs_arr_bc  # Shape: (J, num_active_caps)

            # Flatten matrices for np.add.at
            # target_next_capacity_states_flat are the capacity states in mu_next_t_probs to update
            target_next_capacity_states_flat = capacity_next_state_matrix.ravel().astype(int)
            # probability_mass_to_add_flat are the corresponding probability masses
            probability_mass_to_add_flat = transition_probs_matrix.ravel()

            # Filter out transitions with negligible probability
            significant_transitions_mask = probability_mass_to_add_flat > epsilon_prob
            if np.any(significant_transitions_mask):
                filtered_target_next_caps = target_next_capacity_states_flat[significant_transitions_mask]
                filtered_prob_mass_to_add = probability_mass_to_add_flat[significant_transitions_mask]
                
                np.add.at(mu_next_t_probs, 
                          filtered_target_next_caps,
                          filtered_prob_mass_to_add)
            
            mu_table[t + 1, :] = mu_next_t_probs
            
        return mu_table

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
        mu_filtered = np.where(mu > 1e-9, mu, 0)
        expected_y_star = np.einsum('tc,jct->jt', mu_filtered, y_star, optimize=True)
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
        fare_minus_sum_lambdas = self.F[:, None] - sum_lambdas_over_legs  # (J, T)
        probs_jt = self.probabilities.T  # (J, T)

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
            vartheta_subgrad_i = self.compute_vartheta_subgradient(
                i, self.compute_state_probabilities(i, y_star_i), y_star_i
            )
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
            initial_lambdas (np.ndarray, optional): Initial multipliers of shape (L, J, T). 
                Defaults to a copy of self.lmd.
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
        zero_consumption_mask = self.A == 0  # (L, J) 
        if self.lmd.ndim == 3 and zero_consumption_mask.ndim == 2:
            self.lmd[zero_consumption_mask, :] = 0 
        elif self.lmd.shape[:2] == zero_consumption_mask.shape:
            self.lmd[zero_consumption_mask] = 0

        V_history = []
        best_V = float('inf')
        best_V_iter = 0
        no_improvement_count = 0
        
        if verbose:
            print(f"Starting minimization: alpha0={alpha0}, eps={eps}, max_iter={max_iter}, "
                  f"patience_iters={patience_iters}")

        for k in range(max_iter):
            current_V, current_grad = self.compute_V_lambda_subgradient()
            V_history.append(current_V)
            
            if current_V < best_V - eps:
                best_V = current_V
                no_improvement_count = 0
                best_V_iter = k
            else:
                no_improvement_count += 1
                
            if verbose and (k % print_every == 0 or k == max_iter - 1):
                print(f"Iter {k:4d}: V={current_V:.1f} | Best={best_V:.1f} (@{best_V_iter}) | "
                      f"No improve: {no_improvement_count}/{patience_iters}")

            if no_improvement_count >= patience_iters:
                if verbose:
                    print(f"\nStopping at iter {k}: No improvement > {eps} for {patience_iters} iterations.")
                    print(f"Best V remained {best_V:.6f} from iter {best_V_iter}")
                break
                
            step_size_k = alpha0 / np.sqrt(k + 1)
            self.lmd -= step_size_k * current_grad
            self.lmd = np.maximum(0, self.lmd)
            if self.lmd.ndim == 3 and zero_consumption_mask.ndim == 2:
                self.lmd[zero_consumption_mask, :] = 0
            elif self.lmd.shape[:2] == zero_consumption_mask.shape:
                self.lmd[zero_consumption_mask] = 0

        if verbose and k == max_iter - 1 and len(V_history) == max_iter:
            print(f"Max iterations ({max_iter}) reached. Final V={V_history[-1]:.6f}")
            
        return self.lmd, V_history

    def simulate_revenue_with_bid_prices(
        self, 
        N_sim: int, 
        optimized_lambdas: np.ndarray = None
    ) -> float:
        if self.L == 0 or N_sim <= 0:
            return 0.0

        original_lmd_backup = None
        if optimized_lambdas is not None:
            original_lmd_backup = np.copy(self.lmd)
            self.lmd = np.copy(optimized_lambdas)

        # Precompute value functions for all legs (vectorized)
        for i in range(self.L):
            vartheta_i_table, _ = self.solve_single_leg_dp(i)
            self.vartheta[i, :self.C[i]+1, :] = vartheta_i_table
            if self.C[i] < self.vartheta.shape[1] - 1:
                self.vartheta[i, self.C[i]+1:, :] = vartheta_i_table[self.C[i], :]

        # Vectorized simulation setup
        current_capacities = np.tile(self.C, (N_sim, 1))  # (N_sim, L)
        revenues = np.zeros(N_sim)
        cum_probs = np.cumsum(self.probabilities, axis=1)  # (T, J)
        rand_vals = np.random.random((N_sim, self.T))      # (N_sim, T)

        # Vectorized request generation
        # For each sim and t, find the first j where rand < cum_prob
        # Fix: np.searchsorted expects 1D cum_probs for each t, so loop over t
        requested_j = np.empty((N_sim, self.T), dtype=int)
        for t in range(self.T):
            requested_j[:, t] = np.searchsorted(cum_probs[t], rand_vals[:, t], side="right")

        # Precompute for vectorized access
        A = self.A  # (L, J)
        F = self.F  # (J,)
        vartheta = self.vartheta  # (L, max_C+1, T+1)

        for t in range(self.T):
            j_t = requested_j[:, t]  # (N_sim,)
            # a_ij: (N_sim, L) resource consumption for each sim
            a_ij = A[:, j_t].T  # (N_sim, L)
            fares = F[j_t]      # (N_sim,)

            # Capacity check
            sufficient = np.all(current_capacities >= a_ij, axis=1)  # (N_sim,)

            # Vectorized opportunity cost calculation
            # For each sim and leg, get x_i and a_i
            x_i = current_capacities  # (N_sim, L)
            a_i = a_ij                # (N_sim, L)
            t1 = t + 1

            # Only compute for sufficient and a_i > 0
            valid = (a_i > 0) & sufficient[:, None]  # (N_sim, L)

            # Prepare indices for advanced indexing
            sim_idx, leg_idx = np.nonzero(valid)
            x_vals = x_i[sim_idx, leg_idx]
            a_vals = a_i[sim_idx, leg_idx]
            # Clamp x_vals - a_vals to >= 0
            x_minus_a = np.maximum(x_vals - a_vals, 0)

            # Get opportunity cost for all valid (sim, leg)
            opp_costs = np.zeros((N_sim, self.L), dtype=float)
            opp_costs[sim_idx, leg_idx] = (
                vartheta[leg_idx, x_vals, t1] - vartheta[leg_idx, x_minus_a, t1]
            )

            # Sum over legs for each sim
            opportunity_cost = np.sum(opp_costs, axis=1)  # (N_sim,)

            # Acceptance decision and updates
            accept = (fares >= opportunity_cost) & sufficient  # (N_sim,)
            revenues += fares * accept
            # Only update capacities for accepted sims
            current_capacities -= a_ij * accept[:, None]

        if original_lmd_backup is not None:
            self.lmd = original_lmd_backup

        return np.mean(revenues)


if __name__ == "__main__":
    import os

    data_path = "data/200_rm_datasets/rm_200_4_1.0_4.0.txt"
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
    assert vartheta.shape == (inst.C[leg] + 1, inst.T + 1)
    assert y_star.shape == (inst.J, inst.C[leg] + 1, inst.T)
    assert np.all((y_star == 0) | (y_star == 1))
    print("Single-leg DP test passed.")

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
        print(f"  mu[1, C_leg-1]: {mu[1, inst.C[leg] - 1]:.4f}")

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
    np.savetxt(f"debug/V_lambda_subgradient_leg{leg}.txt", V_lambda_subgrad[leg, :, :], fmt="%.6f")
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
