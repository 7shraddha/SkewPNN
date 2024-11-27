Simulation Results:
In this code, we are demonstrating decision boundaries for HDDTs, PNNs. and SkewPNNs for three different random sample data. 1. HalfMoon, 2. Full Circle and 3. Interwined_spiral. 

Halfmoon is already in the existing code. 

This is for makecircles.....
# Define parameters for the half moons
n_points = 300  # Number of points for each half-moon
noise = 0.35   # Noise level to add variability

# Generate the inner circle (Class 0)
theta1 = np.linspace(0, 2 * np.pi, n_points)  # Angles for the circle
radius1 = 1  # Fixed radius for the inner circle
X1 = np.column_stack((radius1 * np.cos(theta1), radius1 * np.sin(theta1)))  # Coordinates
X1 += np.random.normal(scale=noise, size=X1.shape)  # Add noise
y1 = np.zeros(n_points, dtype=int)  # Label 0 for Class 0

# Generate the outer circle (Class 1)
theta2 = np.linspace(0, 2 * np.pi, n_points)  # Angles for the circle
radius2 = 2  # Fixed radius for the outer circle
X2 = np.column_stack((radius2 * np.cos(theta2), radius2 * np.sin(theta2)))  # Coordinates
X2 += np.random.normal(scale=noise, size=X2.shape)  # Add noise
y2 = np.ones(n_points, dtype=int)  # Label 1 for Class 1

# Combine the datasets and labels
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))


This is for Interwined_spiral.....
# Define parameters for the intertwined spirals
n_points = 350  # Number of points for each spiral
angle = np.linspace(0, 2 * np.pi, n_points)

# Generate points for the first spiral (Class 0)
radius1 = np.linspace(0.5, 2.5, n_points)
X1 = np.column_stack((radius1 * np.cos(angle), radius1 * np.sin(angle)))
y1 = np.zeros(n_points, dtype=int)  # Label 0 for Class 0

# Generate points for the second spiral (Class 1)
radius2 = np.linspace(0.5, 2.5, n_points)
X2 = np.column_stack((radius2 * np.cos(angle + np.pi), radius2 * np.sin(angle + np.pi)))
y2 = np.ones(n_points, dtype=int)  # Label 1 for Class 1

# Combine the datasets and labels
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

Yaaayyy!! 


