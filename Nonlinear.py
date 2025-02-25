import numpy as np
from scipy.interpolate import RegularGridInterpolator

class NonlinearSystem2D:
    def __init__(self, mass=1.0, base_friction=1):
        self.mass = mass
        self.base_friction = base_friction
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        
        # Resistance map
        self.grid_size = 10
        self.resistance_map = np.ones((self.grid_size, self.grid_size)) * base_friction
        self.circular_resistance()
        
    def randomize_resistance(self):
        """Create a nonlinear resistance field with discrete grid-aligned regions"""
        # Start with base friction
        self.resistance_map = np.ones((self.grid_size, self.grid_size)) * self.base_friction
        
        # Add obstacle-like areas with higher resistance that align with grid squares
        num_obstacles = np.random.randint(2, 5)
        for _ in range(num_obstacles):
            # Choose random grid square
            grid_x = np.random.randint(0, self.grid_size)
            grid_y = np.random.randint(0, self.grid_size)
            
            # Assign higher resistance value to the entire grid square
            intensity = np.random.uniform(3.0, 5.0)
            self.resistance_map[grid_y, grid_x] = intensity
        
        # No need for interpolation anymore since we're using discrete grid values

    def circular_resistance(self):
        """Create a radial resistance field where resistance is highest at the center and decreases outward."""
        min_intensity = self.base_friction
        max_intensity = np.random.uniform(3.0, 5.0)  # Maximum resistance at the center

        center = np.array([self.grid_size / 2, self.grid_size / 2])  # Grid center
        max_distance = np.linalg.norm(center)  # Maximum possible distance to the edge

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                # Compute distance from the center
                distance = np.linalg.norm(np.array([x, y]) - center)
                
                # Normalize distance (0 at center, 1 at farthest edge)
                norm_distance = distance / max_distance

                # Compute resistance (higher at center, lower outward)
                self.resistance_map[y, x] = min_intensity + (max_intensity - min_intensity) * (1 - norm_distance)
        
    def get_local_resistance(self, position):
        """Get resistance value at the grid cell containing the given position"""
        # Simple grid cell lookup without interpolation
        x = int(np.clip(position[0], 0, self.grid_size - 1))
        y = int(np.clip(position[1], 0, self.grid_size - 1))
        return self.resistance_map[y, x]
    
    def update(self, force, dt):
        # Get local resistance based on current position
        local_resistance = self.get_local_resistance(self.position)
        
        # Nonlinear system: apply a position-dependent force
        # Force = mass * acceleration, solve for acceleration
        acceleration = force / self.mass
        
        # Apply the system dynamics
        self.velocity += acceleration * dt - (self.velocity * local_resistance * local_resistance) * dt
        self.position += self.velocity * dt
        
        # Keep within bounds
        self.position = np.clip(self.position, 0, self.grid_size - 0.01)
        
        return self.position.copy()
    
    def reset(self, position=None):
        if position is not None:
            self.position = np.array(position)
        else:
            self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])