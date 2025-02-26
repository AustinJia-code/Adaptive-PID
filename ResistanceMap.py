import numpy as np

class ResistanceMap:
  def __init__(self, mass=1.0, base_friction=1):
    self.mass = mass
    self.base_friction = base_friction
    self.position = np.array([0.0, 0.0])
    self.velocity = np.array([0.0, 0.0])
    
    # Resistance map
    self.grid_size = 10
    self.resistance_map = np.ones((self.grid_size, self.grid_size)) * base_friction
    self.circular_resistance()

  def circular_resistance(self):
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
    # Simple grid cell lookup
    x = int(np.clip(position[0], 0, self.grid_size - 1))
    y = int(np.clip(position[1], 0, self.grid_size - 1))
    return self.resistance_map[y, x]
  
  def update(self, force, dt):
    # Get local resistance based on current position
    local_resistance = self.get_local_resistance(self.position)

    acceleration = force / self.mass
    
    # Apply the system dynamics, add resistance
    self.velocity += acceleration * dt - (self.velocity * local_resistance * local_resistance) * dt
    self.position += self.velocity * dt
    
    # Keep within bounds
    self.position = np.clip(self.position, 0, self.grid_size - 0.01)
    
    return self.position