import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PID import PIDController, AdaptivePIDController
from Nonlinear import NonlinearSystem2D

# Create a visualization of the controllers
class PIDVisualization:
    def __init__(self):
        plt.rcParams['toolbar'] = 'None'
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("Adaptive PID")
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        
        # Create systems and controllers
        self.standard_system = NonlinearSystem2D(mass=1.0, base_friction=1)
        self.adaptive_system = NonlinearSystem2D(mass=1.0, base_friction=1)
        
        # Make both systems use the same resistance map for fair comparison
        self.adaptive_system.resistance_map = self.standard_system.resistance_map.copy()
        
        # Start with more aggressive PID values for the adaptive controller
        self.standard_controller = PIDController(kp=2.0, ki=0.1, kd=1.0)
        self.adaptive_controller = AdaptivePIDController(kp=2.0, ki=0.1, kd=1.0)
        
        # Initialize positions and target
        self.standard_position = self.standard_system.position.copy()
        self.adaptive_position = self.adaptive_system.position.copy()
        self.target_position = np.array([5.0, 5.0])
        
        # Set initial setpoints
        self.standard_controller.setpoint = self.target_position.copy()
        self.adaptive_controller.setpoint = self.target_position.copy()
        
        # Create plot elements
        self.standard_dot, = self.ax.plot([], [], 'ro', markersize=10, label='Standard')
        self.adaptive_dot, = self.ax.plot([], [], 'bo', markersize=10, label='Adaptive')
        self.target_dot, = self.ax.plot([], [], 'go', markersize=7, label='Target')
        
        self.standard_path, = self.ax.plot([], [], 'r-', alpha=0.5)
        self.adaptive_path, = self.ax.plot([], [], 'b-', alpha=0.5)
        
        self.standard_path_x = []
        self.standard_path_y = []
        self.adaptive_path_x = []
        self.adaptive_path_y = []
        
        # Show resistance map
        self.resistance_img = self.ax.imshow(self.standard_system.resistance_map, 
                                           extent=[0, 10, 0, 10], 
                                           origin='lower', alpha=0.5, cmap='YlOrRd')
        self.resistance_colorbar = plt.colorbar(self.resistance_img, ax=self.ax)
        self.resistance_colorbar.set_label('Resistance')
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # State variables
        self.running = True
        self.time_elapsed = 0
        self.show_resistance = True
        
        # Parameter display
        self.param_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      verticalalignment='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
        
        # Add velocity vectors
        self.standard_vel_arrow = self.ax.quiver(0, 0, 0, 0, color='r', scale=20)
        self.adaptive_vel_arrow = self.ax.quiver(0, 0, 0, 0, color='b', scale=20)
        
        # Animation setup
        self.dt = 0.05
        
        # Draw initial frame
        self.update(0)
        
        # Create the animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update, interval=50, blit=True)
    
    def toggle_resistance_map(self, label):
        self.show_resistance = not self.show_resistance
        self.resistance_img.set_alpha(0.5 if self.show_resistance else 0)
        
    def randomize_resistance(self, event):
        self.standard_system.randomize_resistance()
        self.adaptive_system.resistance_map = self.standard_system.resistance_map.copy()
        self.resistance_img.set_data(self.standard_system.resistance_map)
        self.fig.canvas.draw_idle()
        
    def on_click(self, event):
        if event.inaxes == self.ax and event.button == 1:  # Left click
            # Set new target
            self.target_position = np.array([event.xdata, event.ydata])
            self.standard_controller.setpoint = self.target_position.copy()
            self.adaptive_controller.setpoint = self.target_position.copy()
            
            # Reset controller integral terms
            self.standard_controller.integral = np.array([0.0, 0.0]) 
            self.adaptive_controller.integral = np.array([0.0, 0.0])
            
            # Clear paths
            self.standard_path_x = [self.standard_position[0]]
            self.standard_path_y = [self.standard_position[1]]
            self.adaptive_path_x = [self.adaptive_position[0]]
            self.adaptive_path_y = [self.adaptive_position[1]]
            
            # Reset time
            self.time_elapsed = 0
        
    def update(self, frame):
        # Update standard controller
        standard_control = self.standard_controller.control(self.standard_position, self.dt)
        self.standard_position = self.standard_system.update(standard_control, self.dt)
        
        # Update adaptive controller
        adaptive_control = self.adaptive_controller.control(self.adaptive_position, self.dt)
        self.adaptive_position = self.adaptive_system.update(adaptive_control, self.dt)
        
        # Get current local resistance values for display
        standard_resistance = self.standard_system.get_local_resistance(self.standard_position)
        adaptive_resistance = self.adaptive_system.get_local_resistance(self.adaptive_position)
        
        # Update paths
        self.standard_path_x.append(self.standard_position[0])
        self.standard_path_y.append(self.standard_position[1])
        self.adaptive_path_x.append(self.adaptive_position[0])
        self.adaptive_path_y.append(self.adaptive_position[1])
        
        # Update plot data
        self.standard_dot.set_data([self.standard_position[0]], [self.standard_position[1]])
        self.adaptive_dot.set_data([self.adaptive_position[0]], [self.adaptive_position[1]])
        self.target_dot.set_data([self.target_position[0]], [self.target_position[1]])
        
        self.standard_path.set_data(self.standard_path_x, self.standard_path_y)
        self.adaptive_path.set_data(self.adaptive_path_x, self.adaptive_path_y)
        
        # Update velocity arrows
        self.standard_vel_arrow.set_offsets(self.standard_position)
        self.standard_vel_arrow.set_UVC(self.standard_system.velocity[0], self.standard_system.velocity[1])
        
        self.adaptive_vel_arrow.set_offsets(self.adaptive_position)
        self.adaptive_vel_arrow.set_UVC(self.adaptive_system.velocity[0], self.adaptive_system.velocity[1])
        
        # Calculate distances to target
        standard_dist = np.linalg.norm(self.standard_position - self.target_position)
        adaptive_dist = np.linalg.norm(self.adaptive_position - self.target_position)
        
        # Update parameter display
        param_text = f"Time: {self.time_elapsed:.1f}s\n"
        param_text += f"Standard PID: Kp={self.standard_controller.kp:.2f}, Ki={self.standard_controller.ki:.2f}, Kd={self.standard_controller.kd:.2f}\n"
        param_text += f"Adaptive PID: Kp={self.adaptive_controller.kp:.2f}, Ki={self.adaptive_controller.ki:.2f}, Kd={self.adaptive_controller.kd:.2f}\n"
        param_text += f"Distance: Stand={standard_dist:.2f}, Adapt={adaptive_dist:.2f}\n"
        param_text += f"Local Resistance: Stand={standard_resistance:.2f}, Adapt={adaptive_resistance:.2f}"
        self.param_text.set_text(param_text)
        
        self.time_elapsed += self.dt
        
        return (self.standard_dot, self.adaptive_dot, self.target_dot, 
                self.standard_path, self.adaptive_path, self.param_text,
                self.standard_vel_arrow, self.adaptive_vel_arrow)
    
    def show(self):
        plt.tight_layout()
        plt.show()


# Run the visualization
if __name__ == "__main__":
    vis = PIDVisualization()
    vis.show()