import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import math

class DoublePendulum:
    """
    A comprehensive double pendulum simulation system with adjustable parameters.
    
    Physical parameters (all adjustable):
    - L1, L2: lengths of rods (m)
    - m1, m2: masses of bobs (kg)
    - g: gravitational acceleration (m/s²)
    - θ1, θ2: initial angles (rad)
    - ω1, ω2: initial angular velocities (rad/s)
    """
    
    def __init__(self, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81, 
                 theta1=90.0, theta2=0.0, omega1=0.0, omega2=0.0):
        # Convert angles to radians
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.theta1 = math.radians(theta1)
        self.theta2 = math.radians(theta2)
        self.omega1 = omega1
        self.omega2 = omega2
        
        # State vector: [θ1, θ2, ω1, ω2]
        self.state = np.array([self.theta1, self.theta2, self.omega1, self.omega2])
        
        # History for trajectory plotting
        self.history = [self.state.copy()]
        self.time_points = [0.0]
        
    def equations_of_motion(self, state, t):
        """
        Defines the differential equations for the double pendulum.
        Uses Lagrangian mechanics to derive the equations of motion.
        """
        theta1, theta2, omega1, omega2 = state
        
        # Mass matrix elements
        M11 = (self.m1 + self.m2) * self.L1**2
        M12 = self.m2 * self.L1 * self.L2 * np.cos(theta1 - theta2)
        M21 = M12
        M22 = self.m2 * self.L2**2
        
        # Coriolis and centrifugal terms
        C1 = -self.m2 * self.L1 * self.L2 * omega2**2 * np.sin(theta1 - theta2)
        C2 = self.m2 * self.L1 * self.L2 * omega1**2 * np.sin(theta1 - theta2)
        
        # Gravitational terms
        G1 = (self.m1 + self.m2) * self.g * self.L1 * np.sin(theta1)
        G2 = self.m2 * self.g * self.L2 * np.sin(theta2)
        
        # Solve for angular accelerations
        det = M11 * M22 - M12 * M21
        
        alpha1 = (M22 * (C1 - G1) - M12 * (C2 - G2)) / det
        alpha2 = (-M21 * (C1 - G1) + M11 * (C2 - G2)) / det
        
        return [omega1, omega2, alpha1, alpha2]
    
    def simulate(self, t_max=10, dt=0.01):
        """
        Run the simulation for specified time duration.
        """
        # Reset history
        self.history = [self.state.copy()]
        self.time_points = [0.0]
        
        # Time points
        t_points = np.arange(0, t_max, dt)
        
        # Runge-Kutta 4th order integration
        for t in t_points[1:]:
            k1 = np.array(self.equations_of_motion(self.state, t))
            k2 = np.array(self.equations_of_motion(self.state + 0.5 * dt * k1, t + 0.5 * dt))
            k3 = np.array(self.equations_of_motion(self.state + 0.5 * dt * k2, t + 0.5 * dt))
            k4 = np.array(self.equations_of_motion(self.state + dt * k3, t + dt))
            
            self.state += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            
            # Normalize angles to [-π, π]
            self.state[0] = np.arctan2(np.sin(self.state[0]), np.cos(self.state[0]))
            self.state[1] = np.arctan2(np.sin(self.state[1]), np.cos(self.state[1]))
            
            self.history.append(self.state.copy())
            self.time_points.append(t)
        
        self.history = np.array(self.history)
        self.time_points = np.array(self.time_points)
    
    def get_positions(self):
        """
        Calculate the (x, y) positions of both pendulum bobs.
        """
        theta1, theta2 = self.history[:, 0], self.history[:, 1]
        
        x1 = self.L1 * np.sin(theta1)
        y1 = -self.L1 * np.cos(theta1)
        
        x2 = x1 + self.L2 * np.sin(theta2)
        y2 = y1 - self.L2 * np.cos(theta2)
        
        return x1, y1, x2, y2
    
    def get_energy(self):
        """
        Calculate total energy (kinetic + potential) of the system.
        Useful for checking numerical stability.
        """
        theta1, theta2, omega1, omega2 = self.history[:, 0], self.history[:, 1], self.history[:, 2], self.history[:, 3]
        
        # Kinetic energy
        T1 = 0.5 * self.m1 * (self.L1 * omega1)**2
        T2 = 0.5 * self.m2 * ((self.L1 * omega1)**2 + (self.L2 * omega2)**2 + 
                               2 * self.L1 * self.L2 * omega1 * omega2 * np.cos(theta1 - theta2))
        T = T1 + T2
        
        # Potential energy
        U1 = -self.m1 * self.g * self.L1 * np.cos(theta1)
        U2 = -self.m2 * self.g * (self.L1 * np.cos(theta1) + self.L2 * np.cos(theta2))
        U = U1 + U2
        
        return T + U

class DoublePendulumVisualizer:
    """
    Interactive visualization for the double pendulum simulation.
    Includes toggleable parameter controls for real-time adjustment.
    """
    
    def __init__(self, pendulum):
        self.pendulum = pendulum
        self.controls_visible = False  # Initially hidden
        
        # Create figure with gridspec for better layout control
        self.fig = plt.figure(figsize=(14, 10))
        self.gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3, 
                                       height_ratios=[3, 3, 0.1])  # Last row for toggle button
        
        # Main simulation plot (top left)
        self.ax_main = self.fig.add_subplot(self.gs[0:2, 0:2])
        self.ax_main.set_xlim(-3, 3)
        self.ax_main.set_ylim(-3, 2)
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True)
        self.ax_main.set_title('Double Pendulum Motion', fontsize=14, fontweight='bold')
        
        # Energy plot (top right)
        self.ax_energy = self.fig.add_subplot(self.gs[0, 2:])
        self.ax_energy.set_title('Energy vs Time', fontsize=12)
        self.ax_energy.set_xlabel('Time (s)')
        self.ax_energy.set_ylabel('Energy (J)')
        
        # Phase space plot (middle right)
        self.ax_phase = self.fig.add_subplot(self.gs[1, 2:])
        self.ax_phase.set_title('Phase Space: θ₁ vs ω₁', fontsize=12)
        self.ax_phase.set_xlabel('θ₁ (rad)')
        self.ax_phase.set_ylabel('ω₁ (rad/s)')
        self.ax_phase.grid(True)
        
        # Control panel area (initially hidden)
        self.control_ax = None
        self.sliders = {}
        self.reset_button = None
        
        # Toggle button (always visible at bottom)
        self.toggle_ax = self.fig.add_subplot(self.gs[2, :])
        self.toggle_button = Button(self.toggle_ax, 'Show Controls (Press C)')
        self.toggle_button.on_clicked(self.toggle_controls)
        
        # Initialize plots
        self.line, = self.ax_main.plot([], [], 'o-', lw=2, markersize=8)
        self.trace, = self.ax_main.plot([], [], 'r-', alpha=0.3, lw=1)
        self.energy_line, = self.ax_energy.plot([], [], 'b-', lw=2)
        self.phase_line, = self.ax_phase.plot([], [], 'g-', lw=1)
        
        # Animation variables
        self.anim = None
        self.is_running = False
        
        # Keyboard event for 'c' key
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Run initial simulation
        self.pendulum.simulate()
        self.update_plots(0)
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'c' or event.key == 'C':
            self.toggle_controls(None)
    
    def toggle_controls(self, event):
        """Toggle visibility of the control panel."""
        self.controls_visible = not self.controls_visible
        
        if self.controls_visible:
            self.create_control_panel()
            self.toggle_button.label.set_text('Hide Controls (Press C)')
        else:
            self.remove_control_panel()
            self.toggle_button.label.set_text('Show Controls (Press C)')
        
        plt.draw()
    
    def create_control_panel(self):
        """Create the control panel with sliders."""
        # Remove existing control panel if present
        self.remove_control_panel()
        
        # Create control axes at the bottom
        self.control_ax = self.fig.add_subplot(self.gs[2, :])
        self.control_ax.set_position([0.1, 0.02, 0.8, 0.15])  # Position at bottom
        self.control_ax.clear()
        self.control_ax.axis('off')
        
        # Define slider positions
        slider_params = [
            ('L1', 'L1 (m)', 0.1, 2.0, self.pendulum.L1, 0.05, 0.8),
            ('L2', 'L2 (m)', 0.1, 2.0, self.pendulum.L2, 0.25, 0.8),
            ('m1', 'm1 (kg)', 0.1, 3.0, self.pendulum.m1, 0.45, 0.8),
            ('m2', 'm2 (kg)', 0.1, 3.0, self.pendulum.m2, 0.65, 0.8),
            ('theta1', 'θ1 (°)', -180, 180, np.degrees(self.pendulum.theta1), 0.05, 0.5),
            ('theta2', 'θ2 (°)', -180, 180, np.degrees(self.pendulum.theta2), 0.25, 0.5),
            ('g', 'g (m/s²)', 1.0, 20.0, self.pendulum.g, 0.45, 0.5)
        ]
        
        self.sliders = {}
        
        # Create sliders
        for param_name, label, min_val, max_val, init_val, x_pos, y_pos in slider_params:
            # Adjust positions relative to control panel
            ax_slider = plt.axes([0.1 + x_pos*0.8, 0.02 + y_pos*0.12, 0.15, 0.03])
            self.sliders[param_name] = Slider(
                ax_slider, label, min_val, max_val, valinit=init_val)
            self.sliders[param_name].on_changed(self.update_parameters)
        
        # Create reset button
        self.reset_button_ax = plt.axes([0.8, 0.05, 0.1, 0.04])
        self.reset_button = Button(self.reset_button_ax, 'Reset')
        self.reset_button.on_clicked(self.reset_simulation)
    
    def remove_control_panel(self):
        """Remove the control panel."""
        if self.control_ax:
            self.control_ax.remove()
            self.control_ax = None
            self.sliders = {}
            self.reset_button = None
            
            # Remove slider axes
            for slider in self.sliders.values():
                slider.ax.remove()
                
            if self.reset_button:
                self.reset_button.ax.remove()
    
    def update_parameters(self, val):
        """Update pendulum parameters when sliders change."""
        # Update pendulum parameters
        self.pendulum.L1 = self.sliders['L1'].val
        self.pendulum.L2 = self.sliders['L2'].val
        self.pendulum.m1 = self.sliders['m1'].val
        self.pendulum.m2 = self.sliders['m2'].val
        self.pendulum.g = self.sliders['g'].val
        self.pendulum.theta1 = math.radians(self.sliders['theta1'].val)
        self.pendulum.theta2 = math.radians(self.sliders['theta2'].val)
        
        # Reset simulation with new parameters
        self.reset_simulation(None)
    
    def reset_simulation(self, event):
        """Reset the simulation with current parameters."""
        # Stop current animation
        if self.anim:
            self.anim.event_source.stop()
        
        # Reset pendulum state
        self.pendulum.state = np.array([
            self.pendulum.theta1,
            self.pendulum.theta2,
            0.0,  # Reset angular velocities
            0.0
        ])
        
        # Re-run simulation
        self.pendulum.simulate()
        
        # Update plots
        self.update_plots(0)
        
        # Restart animation
        self.start_animation()
    
    def update_plots(self, frame):
        """Update all plots for given frame."""
        x1, y1, x2, y2 = self.pendulum.get_positions()
        
        # Update main plot
        self.line.set_data([0, x1[frame], x2[frame]], 
                          [0, y1[frame], y2[frame]])
        self.trace.set_data(x2[:frame+1], y2[:frame+1])
        
        # Update energy plot
        energy = self.pendulum.get_energy()
        self.energy_line.set_data(self.pendulum.time_points[:frame+1], 
                                 energy[:frame+1])
        self.ax_energy.relim()
        self.ax_energy.autoscale_view()
        
        # Update phase space plot
        self.phase_line.set_data(self.pendulum.history[:frame+1, 0], 
                               self.pendulum.history[:frame+1, 2])
        self.ax_phase.relim()
        self.ax_phase.autoscale_view()
        
        return self.line, self.trace, self.energy_line, self.phase_line
    
    def start_animation(self):
        """Start the animation."""
        if self.anim:
            self.anim.event_source.stop()
        
        self.anim = FuncAnimation(self.fig, self.update_plots, 
                                frames=len(self.pendulum.time_points),
                                interval=20, blit=True, repeat=True)
        plt.draw()
    
    def show(self):
        """Display the interactive visualization."""
        self.start_animation()
        plt.show()

# Example usage and parameter adjustment
def main():
    """
    Main function demonstrating the double pendulum simulation.
    You can easily adjust parameters here or use the interactive sliders.
    """
    
    # Create pendulum with initial parameters
    pendulum = DoublePendulum(
        L1=1.0,      # Length of first rod (m)
        L2=1.0,      # Length of second rod (m)
        m1=1.0,      # Mass of first bob (kg)
        m2=1.0,      # Mass of second bob (kg)
        g=9.81,      # Gravitational acceleration (m/s²)
        theta1=120.0,  # Initial angle of first rod (degrees)
        theta2=0.0,  # Initial angle of second rod (degrees)
        omega1=0.0,  # Initial angular velocity of first rod (rad/s)
        omega2=0.0   # Initial angular velocity of second rod (rad/s)
    )
    
    # Create and show visualization
    visualizer = DoublePendulumVisualizer(pendulum)
    visualizer.show()

# Alternative: Run a simple simulation without GUI
def quick_simulation():
    """Run a quick simulation without interactive GUI for testing."""
    pendulum = DoublePendulum(
        L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81,
        theta1=90.0, theta2=0.0, omega1=0.0, omega2=0.0
    )
    
    pendulum.simulate(t_max=10, dt=0.01)
    
    # Create simple plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    x1, y1, x2, y2 = pendulum.get_positions()
    
    # Plot trajectory
    ax1.plot(x2, y2, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_aspect('equal')
    ax1.set_title('Pendulum Tip Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True)
    
    # Plot angles over time
    ax2.plot(pendulum.time_points, np.degrees(pendulum.history[:, 0]), 
             'r-', label='θ₁')
    ax2.plot(pendulum.time_points, np.degrees(pendulum.history[:, 1]), 
             'b-', label='θ₂')
    ax2.set_title('Angles vs Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (degrees)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot angular velocities
    ax3.plot(pendulum.time_points, pendulum.history[:, 2], 'r-', label='ω₁')
    ax3.plot(pendulum.time_points, pendulum.history[:, 3], 'b-', label='ω₂')
    ax3.set_title('Angular Velocities vs Time')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angular velocity (rad/s)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot energy
    energy = pendulum.get_energy()
    ax4.plot(pendulum.time_points, energy, 'g-', linewidth=2)
    ax4.set_title('Total Energy vs Time')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Energy (J)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Uncomment one of the following:
    
    # For interactive GUI with parameter controls:
    main()
    
    # For quick static simulation:
    # quick_simulation()
