import re
import numpy as np


class DataInstance:
    """
    DataInstance parses an airline revenue management test problem file of the form rm-τ-N-α-κ.txt.

    The file encodes a single-leg or network airline RM instance, as described in the technical documentation (see description.pdf).
    The instance consists of a set of flight legs, itineraries, and a tau-indexed demand probability matrix.

    Attributes:
        tau (int): Number of time periods (τ). Each period represents a discrete time step in the booking horizon.
        flight_legs (np.ndarray[str]): Array of flight leg names, each in the form "O{o}D{d}", where o is the origin and d is the destination.
        itineraries (np.ndarray[str]): Array of itinerary names, each in the form "O{o}D{d}F{c}", where o is origin, d is destination, and c is fare class.
        A (np.ndarray[int]): Consumption matrix of shape (num_legs, num_itins). A[i, j] = 1 if itinerary j uses leg i, 0 otherwise.
        probabilities (np.ndarray[float]): Demand probability matrix of shape (tau, num_itins). probabilities[t, j] = Pr(request for itinerary j at time t).

    File Format (see description.pdf for details):
        - First line: integer τ, the number of time periods.
        - Second line: integer N, the number of flight legs.
        - Next N lines: each line contains two integers o d, the origin and destination of a flight leg.
        - Next line: integer M, the number of itineraries.
        - Next M lines: each line contains three integers o d c, the origin, destination, and fare class of an itinerary.
        - Next τ lines: each line contains a list of demand probabilities for that time period, in the format:
            [o d c] p  [o d c] p  ... (for all itineraries)
          where [o d c] identifies the itinerary and p is the probability of a request for that itinerary at that time.

    Example:
        4
        2
        0 1
        1 0
        3
        0 1 0
        1 0 0
        0 0 1
        [0 1 0] 0.1 [1 0 0] 0.2 [0 0 1] 0.05
        ...

    The class builds the following:
        - flight_legs: ["O0D1", "O1D0"]
        - itineraries: ["O0D1F0", "O1D0F0", "O0D0F1"]
        - A: a binary matrix indicating which legs are used by which itineraries
        - probabilities: a (tau, M) matrix of demand probabilities

    Methods:
        __init__(filepath): Loads and parses the instance file.
        _parse_file(filepath): Internal method to parse the file and populate attributes.
    """

    def __init__(self, filepath: str):
        """
        Initialize a DataInstance by parsing the given file.

        Args:
            filepath (str): Path to the airline RM instance file.
        """
        self._parse_file(filepath)

    def _parse_file(self, filepath: str):
        """
        Parse the airline RM instance file and populate the attributes.

        Args:
            filepath (str): Path to the instance file.

        Populates:
            self.tau, self.flight_legs, self.itineraries, self.A, self.probabilities
        """
        # Read all non-comment, non-empty lines from the file
        with open(filepath, 'r') as f:
            lines = []
            for line in f:
                line = line.strip()
                # Ignore empty lines and lines starting with '#'
                if line and not line.lstrip().startswith('#'):
                    lines.append(line)

        idx = 0

        # Parse number of time periods (τ)
        self.tau = int(lines[idx])
        idx += 1

        # Parse number of flight legs (N)
        num_legs = int(lines[idx])
        idx += 1

        # Parse flight leg definitions
        # Each leg is defined by origin and destination (o, d)
        flight_legs = []
        for _ in range(num_legs):
            parts = lines[idx].split()
            # Format: "O{o}D{d}"
            flight_legs.append(f"O{int(parts[0])}D{int(parts[1])}")
            idx += 1

        # Parse number of itineraries (M)
        num_itins = int(lines[idx])
        idx += 1

        # Parse itinerary definitions
        # Each itinerary is defined by origin, destination, and fare class (o, d, c)
        itineraries = []
        itineraries_info = []
        for _ in range(num_itins):
            parts = lines[idx].split()
            o, d, c = int(parts[0]), int(parts[1]), int(parts[2])
            itineraries_info.append((o, d, c))
            # Format: "O{o}D{d}F{c}"
            itineraries.append(f"O{o}D{d}F{c}")
            idx += 1

        # Initialize consumption matrix A and probability matrix P
        # A[i, j] = 1 if itinerary j uses leg i
        A = np.zeros((num_legs, num_itins), int)
        # P[t, j] = probability of request for itinerary j at time t
        P = np.zeros((self.tau, num_itins), float)

        # Build index maps for legs and itineraries for fast lookup
        leg_idx = {leg: i for i, leg in enumerate(flight_legs)}
        itin_idx = {info: j for j, info in enumerate(itineraries_info)}

        # Populate the consumption matrix A
        # For each itinerary, determine which legs it uses:
        #   - If o == 0 or d == 0, the itinerary uses a single leg O{o}D{d}
        #   - Otherwise, the itinerary uses two legs: O{o}D0 and O0D{d}
        for j, (o, d, _) in enumerate(itineraries_info):
            if o == 0 or d == 0:
                # Direct itinerary: uses one leg
                leg = f"O{o}D{d}"
                A[leg_idx[leg], j] = 1
            else:
                # Connecting itinerary: uses two legs
                A[leg_idx[f"O{o}D0"], j] = 1
                A[leg_idx[f"O0D{d}"], j] = 1

        # Compile regex pattern to extract [o d c] p entries from probability lines
        pat = re.compile(r"\[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]\s*([\d.eE+-]+)")

        # Parse demand probability matrix for each time period
        for t in range(self.tau):
            line = lines[idx]
            idx += 1
            # Each line contains multiple [o d c] p entries
            for o, d, c, p in pat.findall(line):
                o, d, c = int(o), int(d), int(c)
                j = itin_idx[(o, d, c)]
                P[t, j] = float(p)

        # Store results as numpy arrays
        self.flight_legs = np.array(flight_legs, dtype='<U10')
        self.itineraries = np.array(itineraries, dtype='<U20')
        self.A = A
        self.probabilities = P


if __name__ == "__main__":
    # Example usage: load a sample airline RM instance and print its contents
    data_path = "data/200_rm_datasets/rm_200_4_1.0_4.0.txt"
    inst = DataInstance(data_path)

    print("Number of time periods (tau):", inst.tau)
    print("\nFlight legs ({}):".format(inst.flight_legs.shape[0]))
    print(inst.flight_legs)

    print("\nItineraries ({}):".format(inst.itineraries.shape[0]))
    print(inst.itineraries)

    print("\nConsumption matrix A (shape {}):".format(inst.A.shape))
    print(inst.A)

    print("\nProbability matrix (shape {}):".format(inst.probabilities.shape))
    # Print first 2 time periods and first 5 itineraries for brevity
    max_t = min(2, inst.probabilities.shape[0])
    max_j = min(5, inst.probabilities.shape[1])
    print("First {} time periods and {} itineraries:".format(max_t, max_j))
    print(inst.probabilities[:max_t, :max_j])
    print("...")
