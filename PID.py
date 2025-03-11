import numpy as np

class PIDController:
  def __init__(self, kp, ki, kd):
    self.kp = kp
    self.ki = ki 
    self.kd = kd 
    self.prev_error = 0.0
    self.integral = 0.0
    self.setpoint = np.array([0.0, 0.0])
      
  def control(self, state, dt):
    # Calculate error
    error = self.setpoint - state
    
    # Calculate proportional term
    p_term = self.kp * error
    
    # Calculate integral term
    if (abs(error) < 0.1).all():
        self.integral = 0.0
    else:
        self.integral = np.clip(error * dt, -10.0, 10.0)
    i_term = self.ki * self.integral
    
    # Calculate derivative term
    d_term = self.kd * (error - self.prev_error) / dt
    self.prev_error = error
    
    # Sum the terms to get the control signal
    control_signal = p_term + i_term + d_term
    
    return control_signal


class AdaptivePIDController(PIDController):
  def __init__(self, kp, ki, kd):
    super().__init__(kp, ki, kd)
    self.performance_history = []
    self.error_history = []
    self.last_update_time = 0
    self.oscillation_count = 0
    self.last_error_sign = 0
      
  def control(self, state, dt):
    # Calculate error
    error = self.setpoint - state
    error_magnitude = np.linalg.norm(error)
    
    # Store error history
    self.error_history.append(error_magnitude)
    if len(self.error_history) > 20:
        self.error_history.pop(0)
    
    # Basic control signal calculation from parent class
    control_signal = super().control(state, dt)
    
    # Self-tuning logic
    self.last_update_time += dt
    
    # Only tune parameters every 0.5 seconds to allow system to respond
    if self.last_update_time >= 0.5 and len(self.error_history) > 5:
      self.last_update_time = 0
      
      # Calculate error trend
      recent_errors = self.error_history[-5:]
      error_derivative = (recent_errors[-1] - recent_errors[0]) / (5 * dt)
      
      # Detect oscillations by checking for sign changes in error derivative
      if len(self.error_history) > 10:
        current_sign = 1 if error_derivative > 0 else -1
        if current_sign != self.last_error_sign and self.last_error_sign != 0:
            self.oscillation_count += 1
        self.last_error_sign = current_sign
    
      # Calculate rise time, indicator will be large if response is good
      if len(self.error_history) > 10:
        rise_time_indicator = np.mean(self.error_history[-10:]) - np.mean(self.error_history[-5:])
      else:
        rise_time_indicator = 0
          
      # Measure the peak overshoot in recent history
      peak_error = max(self.error_history[-10:]) if len(self.error_history) > 10 else max(self.error_history)

      # Determine how far the peak overshoot is beyond the setpoint
      overshoot = peak_error - np.linalg.norm(self.setpoint)

      # Tune proportional gain based on overshoot
      if overshoot > 0.2:  
        # Large overshoot - decrease Kp based on how excessive the overshoot is
        self.kp *= max(0.7, 1 - overshoot)  # Reduce proportionally but not below 70%
      elif error_magnitude > 0.5 and overshoot < 0.1:
        # Slow response - increase Kp, but not if overshooting
        self.kp *= 1.3
    
      # Tune integral gain based on steady-state error
      if np.std(recent_errors) < 0.1 and np.mean(recent_errors) > 0.1:
        # Persistent steady state error on one side - increase Ki
        self.ki += 0.05
      elif self.oscillation_count > 2:
        # Oscillations - decrease Ki
        self.ki *= 0.8
      
      # Tune derivative gain based on oscillations
      if self.oscillation_count > 2:
        # Oscillations - increase Kd to dampen
        self.kd += 0.2
      elif np.std(recent_errors) < 0.05 and self.kd > 0.1:
        # Smooth response - slightly decrease Kd
        self.kd *= 0.9
      
      # Keep gains within reasonable limits
      self.kp = np.clip(self.kp, 0.5, 30.0)
      self.ki = np.clip(self.ki, 0.0, 5.0)
      self.kd = np.clip(self.kd, 0.1, 5.0)
  
    return control_signal