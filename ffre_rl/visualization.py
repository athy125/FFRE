import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import io
from PIL import Image
import base64
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.gridspec as gridspec

# Configure logging
logger = logging.getLogger(__name__)


class Arrow3D(FancyArrowPatch):
    """
    Custom 3D arrow class for matplotlib.
    
    This allows adding directional arrows to 3D plots to show vectors.
    """
    
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """Initialize with points and optional arguments."""
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    
    def draw(self, renderer):
        """Draw the arrow."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class MagneticFieldVisualization:
    """
    Visualization utilities for magnetic field configurations and particle trajectories.
    
    This class provides methods for plotting magnetic field configurations, particle
    trajectories, and various performance metrics in the FFRE optimization process.
    """
    
    def __init__(self, dpi: int = 100, style: str = 'dark_background', 
               color_map: str = 'viridis', fig_size: Tuple[int, int] = (10, 6)):
        """
        Initialize visualization utilities.
        
        Args:
            dpi: DPI for plots
            style: Matplotlib style
            color_map: Color map for plots
            fig_size: Default figure size
        """
        self.dpi = dpi
        self.style = style
        self.color_map = color_map
        self.fig_size = fig_size
        
        # Set plot style
        plt.style.use(style)
    
    def plot_field_configuration(self, field_strengths: np.ndarray, max_field: float, 
                              n_points: int = 100, ax: Optional[plt.Axes] = None, 
                              title: Optional[str] = "Magnetic Field Configuration",
                              show_coils: bool = True, coil_positions: Optional[np.ndarray] = None,
                              return_fig: bool = False) -> Union[plt.Figure, plt.Axes]:
        """
        Plot magnetic field configuration along the chamber length.
        
        Args:
            field_strengths: Array of field strengths at coil locations
            max_field: Maximum field strength in Tesla
            n_points: Number of points to plot
            ax: Matplotlib axes to plot on (optional)
            title: Plot title
            show_coils: Whether to show coil positions
            coil_positions: Array of coil positions (optional)
            return_fig: Whether to return the figure object
            
        Returns:
            Matplotlib axes or figure
        """
        n_coils = len(field_strengths)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        else:
            fig = ax.figure
        
        # Coil positions
        if coil_positions is None:
            coil_positions = np.linspace(0, 1, n_coils)
        
        # Interpolated field
        x = np.linspace(0, 1, n_points)
        B = np.zeros(n_points)
        
        for i in range(n_points):
            # Find segment
            segment_length = 1.0 / (n_coils - 1) if n_coils > 1 else 1.0
            segment_idx = int(x[i] / segment_length)
            segment_idx = min(segment_idx, n_coils - 2)
            
            # Linear interpolation
            if n_coils == 1:
                B[i] = field_strengths[0]
            else:
                segment_pos = (x[i] - segment_idx * segment_length) / segment_length
                B[i] = (1 - segment_pos) * field_strengths[segment_idx] + \
                       segment_pos * field_strengths[segment_idx + 1]
        
        # Scale B field
        B *= max_field
        
        # Plot interpolated field
        ax.plot(x, B, 'b-', linewidth=2, label='Magnetic Field')
        
        # Plot coil positions
        if show_coils:
            ax.scatter(coil_positions, field_strengths * max_field, 
                      c='r', s=50, zorder=3, label='Coil Positions')
        
        # Axis labels
        ax.set_xlabel('Normalized Chamber Position')
        ax.set_ylabel('Magnetic Field Strength (Tesla)')
        
        # Title
        if title:
            ax.set_title(title)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Legend
        if show_coils:
            ax.legend()
        
        # Tight layout
        fig.tight_layout()
        
        # Return figure or axes
        if return_fig:
            return fig
        else:
            return ax
    
    def plot_particle_trajectories(self, trajectories: List[List[Dict[str, Any]]], 
                                 chamber_length: float, chamber_radius: float,
                                 ax: Optional[plt.Axes] = None,
                                 title: Optional[str] = "Particle Trajectories",
                                 color_by_status: bool = True,
                                 show_chamber: bool = True,
                                 show_field_arrows: bool = False,
                                 field_function: Optional[callable] = None,
                                 plot_3d: bool = True,
                                 return_fig: bool = False) -> Union[plt.Figure, plt.Axes]:
        """
        Plot particle trajectories in 3D or 2D.
        
        Args:
            trajectories: List of particle trajectory data
            chamber_length: Length of the chamber in meters
            chamber_radius: Radius of the chamber in meters
            ax: Matplotlib axes to plot on (optional)
            title: Plot title
            color_by_status: Whether to color trajectories by particle status
            show_chamber: Whether to show chamber outline
            show_field_arrows: Whether to show magnetic field direction arrows
            field_function: Function to calculate magnetic field (required if show_field_arrows=True)
            plot_3d: Whether to plot in 3D (True) or 2D (False)
            return_fig: Whether to return the figure object
            
        Returns:
            Matplotlib axes or figure
        """
        # Create figure if needed
        if ax is None:
            if plot_3d:
                fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        else:
            fig = ax.figure
        
        if plot_3d and show_chamber:
            # Plot chamber outline in 3D
            # Create cylinder
            theta = np.linspace(0, 2*np.pi, 100)
            z = np.linspace(0, chamber_length, 100)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = chamber_radius * np.cos(theta_grid)
            y_grid = chamber_radius * np.sin(theta_grid)
            
            # Plot surface
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, color='b')
            
            # Plot end caps
            end_theta = np.linspace(0, 2*np.pi, 100)
            end_r = np.linspace(0, chamber_radius, 20)
            end_theta_grid, end_r_grid = np.meshgrid(end_theta, end_r)
            
            # Start cap
            end_x = end_r_grid * np.cos(end_theta_grid)
            end_y = end_r_grid * np.sin(end_theta_grid)
            end_z = np.zeros_like(end_x)
            ax.plot_surface(end_x, end_y, end_z, alpha=0.1, color='b')
            
            # End cap
            end_z = np.ones_like(end_x) * chamber_length
            ax.plot_surface(end_x, end_y, end_z, alpha=0.1, color='b')
        
        elif not plot_3d and show_chamber:
            # Plot chamber outline in 2D (r-z projection)
            # Bottom line
            ax.plot([0, chamber_length], [-chamber_radius, -chamber_radius], 'b-', alpha=0.5)
            # Top line
            ax.plot([0, chamber_length], [chamber_radius, chamber_radius], 'b-', alpha=0.5)
            # Left cap
            theta = np.linspace(-np.pi/2, np.pi/2, 100)
            ax.plot(np.zeros(100), chamber_radius * np.sin(theta), 'b-', alpha=0.5)
            # Right cap
            ax.plot(np.ones(100) * chamber_length, chamber_radius * np.sin(theta), 'b-', alpha=0.5)
        
        # Plot trajectories
        for traj in trajectories:
            # Skip empty trajectories
            if len(traj) < 2:
                continue
                
            # Extract positions
            positions = np.array([state['position'] for state in traj])
            
            # Get trajectory status
            escaped = traj[-1].get('escaped_thrust', False)
            active = traj[-1].get('active', False)
            
            # Set color based on status
            if color_by_status:
                if escaped:
                    color = 'lime'  # Successfully escaped
                    alpha = 0.8
                    linewidth = 1.5
                elif not active:
                    color = 'red'   # Hit wall or went backward
                    alpha                    color = 'red'   # Hit wall or went backward
                    alpha = 0.5
                    linewidth = 1.0
                else:
                    color = 'yellow'  # Still active
                    alpha = 0.7
                    linewidth = 1.0
            else:
                # Use default
                color = None
                alpha = 0.7
                linewidth = 1.0
            
            # Plot trajectory
            if plot_3d:
                ax.plot(positions[:, 1], positions[:, 2], positions[:, 0], 
                       color=color, alpha=alpha, linewidth=linewidth)
                
                # Plot start and end points
                if len(positions) > 1:
                    # Start point
                    ax.scatter(positions[0, 1], positions[0, 2], positions[0, 0], 
                              color='white', s=20, alpha=0.8)
                    
                    # End point
                    if escaped:
                        marker = '^'  # Triangle for escaped
                    elif not active:
                        marker = 'x'  # X for wall collision
                    else:
                        marker = 'o'  # Circle for active
                        
                    ax.scatter(positions[-1, 1], positions[-1, 2], positions[-1, 0], 
                              color=color, s=30, alpha=1.0, marker=marker)
            else:
                # 2D plot (r-z projection)
                # Calculate radial position
                r = np.sqrt(positions[:, 1]**2 + positions[:, 2]**2) * np.sign(positions[:, 2])
                
                # Plot trajectory
                ax.plot(positions[:, 0], r, color=color, alpha=alpha, linewidth=linewidth)
                
                # Plot start and end points
                if len(positions) > 1:
                    # Start point
                    ax.scatter(positions[0, 0], r[0], color='white', s=20, alpha=0.8)
                    
                    # End point
                    if escaped:
                        marker = '^'  # Triangle for escaped
                    elif not active:
                        marker = 'x'  # X for wall collision
                    else:
                        marker = 'o'  # Circle for active
                        
                    ax.scatter(positions[-1, 0], r[-1], color=color, s=30, alpha=1.0, marker=marker)
        
        # Show magnetic field arrows if requested
        if show_field_arrows and field_function is not None:
            # Create grid of points to show field direction
            grid_points = 5
            x_grid = np.linspace(0, chamber_length, grid_points)
            y_grid = np.linspace(-chamber_radius, chamber_radius, grid_points)
            z_grid = np.linspace(-chamber_radius, chamber_radius, grid_points)
            
            if plot_3d:
                # 3D field arrows
                for x in x_grid:
                    for y in y_grid:
                        for z in z_grid:
                            # Skip points outside chamber
                            if np.sqrt(y**2 + z**2) > chamber_radius:
                                continue
                                
                            # Get field at this point
                            pos = np.array([x, y, z])
                            field = field_function(pos)
                            
                            # Normalize and scale for visibility
                            field_mag = np.linalg.norm(field)
                            if field_mag > 1e-6:
                                field = field / field_mag * (chamber_radius * 0.2)
                                
                                # Add arrow
                                arrow = Arrow3D([y, y + field[1]], 
                                              [z, z + field[2]], 
                                              [x, x + field[0]],
                                              mutation_scale=10, lw=1, 
                                              arrowstyle='->', color='cyan', alpha=0.5)
                                ax.add_artist(arrow)
            else:
                # 2D field arrows (show only in r-z plane where y=0)
                for x in x_grid:
                    for z in z_grid:
                        # Skip points outside chamber
                        if abs(z) > chamber_radius:
                            continue
                            
                        # Get field at this point
                        pos = np.array([x, 0, z])
                        field = field_function(pos)
                        
                        # Get components in r-z plane
                        field_r = field[2]  # z-component becomes r
                        field_z = field[0]  # x-component becomes z
                        
                        # Normalize and scale for visibility
                        field_mag = np.sqrt(field_r**2 + field_z**2)
                        if field_mag > 1e-6:
                            scale = chamber_radius * 0.1
                            field_r = field_r / field_mag * scale
                            field_z = field_z / field_mag * scale
                            
                            # Add arrow
                            ax.arrow(x, z, field_z, field_r, head_width=0.02, head_length=0.03, 
                                    fc='cyan', ec='cyan', alpha=0.5)
        
        # Set labels and limits
        if plot_3d:
            ax.set_xlabel('Y (m)')
            ax.set_ylabel('Z (m)')
            ax.set_zlabel('X (m)')
            
            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, chamber_length/chamber_radius])
            
            # Set limits
            ax.set_xlim(-chamber_radius, chamber_radius)
            ax.set_ylim(-chamber_radius, chamber_radius)
            ax.set_zlim(0, chamber_length)
        else:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('R (m)')
            
            # Set limits
            ax.set_xlim(0, chamber_length)
            ax.set_ylim(-chamber_radius, chamber_radius)
            
            # Equal aspect ratio
            ax.set_aspect('equal')
        
        # Title
        if title:
            ax.set_title(title)
        
        # Tight layout
        fig.tight_layout()
        
        # Return figure or axes
        if return_fig:
            return fig
        else:
            return ax
    
    def plot_trajectory_animation(self, trajectories: List[List[Dict[str, Any]]], 
                               chamber_length: float, chamber_radius: float,
                               field_function: Optional[callable] = None,
                               n_frames: int = 100, interval: int = 50,
                               save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create an animation of particle trajectories.
        
        Args:
            trajectories: List of particle trajectory data
            chamber_length: Length of the chamber in meters
            chamber_radius: Radius of the chamber in meters
            field_function: Function to calculate magnetic field
            n_frames: Number of frames in animation
            interval: Interval between frames in milliseconds
            save_path: Path to save animation (optional)
            
        Returns:
            Matplotlib animation object
        """
        # Create figure
        fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot chamber outline
        theta = np.linspace(0, 2*np.pi, 100)
        z = np.linspace(0, chamber_length, 100)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = chamber_radius * np.cos(theta_grid)
        y_grid = chamber_radius * np.sin(theta_grid)
        
        # Plot surface
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, color='b')
        
        # Find max trajectory length
        max_len = max(len(traj) for traj in trajectories)
        
        # Create particle objects
        particles = []
        colors = []
        
        # Initialize particles
        for traj in trajectories:
            if len(traj) > 1:
                # Initial position
                initial_pos = traj[0]['position']
                
                # Get trajectory status
                escaped = traj[-1].get('escaped_thrust', False)
                active = traj[-1].get('active', False)
                
                # Set color based on status
                if escaped:
                    color = 'lime'  # Successfully escaped
                elif not active:
                    color = 'red'   # Hit wall or went backward
                else:
                    color = 'yellow'  # Still active
                
                # Add particle
                particle = ax.scatter(initial_pos[1], initial_pos[2], initial_pos[0], 
                                     color=color, s=10, alpha=0.8)
                particles.append(particle)
                colors.append(color)
        
        # Set labels and limits
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('Z (m)')
        ax.set_zlabel('X (m)')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, chamber_length/chamber_radius])
        
        # Set limits
        ax.set_xlim(-chamber_radius, chamber_radius)
        ax.set_ylim(-chamber_radius, chamber_radius)
        ax.set_zlim(0, chamber_length)
        
        # Title
        ax.set_title("Particle Trajectories Animation")
        
        # Animation update function
        def update(frame):
            # Calculate frame index in trajectories
            t_idx = int(frame * max_len / n_frames)
            
            # Update each particle
            for i, (particle, traj) in enumerate(zip(particles, trajectories)):
                if t_idx < len(traj):
                    # Get position
                    pos = traj[t_idx]['position']
                    
                    # Update position
                    particle._offsets3d = ([pos[1]], [pos[2]], [pos[0]])
                else:
                    # Particle trajectory ended
                    particle._offsets3d = ([], [], [])
            
            return particles
        
        # Create animation
        animation = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
        
        # Save if path provided
        if save_path:
            animation.save(save_path, writer='ffmpeg', dpi=self.dpi)
        
        return animation
    
    def plot_performance_metrics(self, metrics: Dict[str, List[float]], 
                              ax: Optional[plt.Axes] = None,
                              title: Optional[str] = "Training Performance",
                              return_fig: bool = False) -> Union[plt.Figure, plt.Axes]:
        """
        Plot training performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            ax: Matplotlib axes to plot on (optional)
            title: Plot title
            return_fig: Whether to return the figure object
            
        Returns:
            Matplotlib axes or figure
        """
        # Create figure if needed
        if ax is None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=self.dpi)
            axes = axes.flatten()
        else:
            fig = ax.figure
            # If single axis provided, create grid
            gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax)
            axes = [plt.subplot(gs[i]) for i in range(4)]
        
        # Required metrics
        required_metrics = ['episode_rewards', 'thrust_values', 'efficiency_values', 'escape_rates']
        
        # Check if all required metrics are present
        for metric in required_metrics:
            if metric not in metrics:
                logger.warning(f"Required metric '{metric}' not found in metrics dictionary")
        
        # Plot episode rewards
        if 'episode_rewards' in metrics:
            axes[0].plot(metrics['episode_rewards'], 'b-')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Reward')
            axes[0].set_title('Episode Rewards')
            axes[0].grid(True, alpha=0.3)
        
        # Plot thrust values
        if 'thrust_values' in metrics:
            axes[1].plot(metrics['thrust_values'], 'r-')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Thrust')
            axes[1].set_title('Thrust per Episode')
            axes[1].grid(True, alpha=0.3)
        
        # Plot efficiency values
        if 'efficiency_values' in metrics:
            axes[2].plot(metrics['efficiency_values'], 'g-')
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Efficiency')
            axes[2].set_title('Efficiency per Episode')
            axes[2].grid(True, alpha=0.3)
        
        # Plot escape rates
        if 'escape_rates' in metrics:
            axes[3].plot(metrics['escape_rates'], 'y-')
            axes[3].set_xlabel('Episode')
            axes[3].set_ylabel('Escape Rate')
            axes[3].set_title('Particle Escape Rate per Episode')
            axes[3].grid(True, alpha=0.3)
        
        # Optional metrics
        optional_metrics = ['actor_losses', 'critic_losses', 'entropy_values']
        
        # Plot optional metrics if present
        if any(metric in metrics for metric in optional_metrics):
            # Create additional figure
            fig_losses, axes_losses = plt.subplots(1, len(optional_metrics), figsize=(5*len(optional_metrics), 4), dpi=self.dpi, squeeze=False)
            axes_losses = axes_losses.flatten()
            
            for i, metric in enumerate([m for m in optional_metrics if m in metrics]):
                axes_losses[i].plot(metrics[metric], 'c-')
                axes_losses[i].set_xlabel('Update')
                axes_losses[i].set_ylabel(metric.replace('_', ' ').title())
                axes_losses[i].set_title(metric.replace('_', ' ').title())
                axes_losses[i].grid(True, alpha=0.3)
            
            fig_losses.tight_layout()
            
            # If return_fig, add additional figure to a list
            if return_fig:
                return [fig, fig_losses]
        
        # Tight layout
        fig.tight_layout()
        
        # Return figure or axes
        if return_fig:
            return fig
        else:
            return axes
    
    def plot_field_heatmap(self, field_function: callable, chamber_length: float, chamber_radius: float,
                        resolution: int = 50, slice_pos: float = 0.0, slice_axis: str = 'y',
                        ax: Optional[plt.Axes] = None, cmap: str = 'viridis',
                        title: Optional[str] = "Magnetic Field Strength",
                        return_fig: bool = False) -> Union[plt.Figure, plt.Axes]:
        """
        Plot a heatmap of magnetic field strength.
        
        Args:
            field_function: Function to calculate magnetic field
            chamber_length: Length of the chamber in meters
            chamber_radius: Radius of the chamber in meters
            resolution: Resolution of the grid
            slice_pos: Position of the slice along the slice_axis
            slice_axis: Axis to slice along ('x', 'y', or 'z')
            ax: Matplotlib axes to plot on (optional)
            cmap: Colormap to use
            title: Plot title
            return_fig: Whether to return the figure object
            
        Returns:
            Matplotlib axes or figure
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        else:
            fig = ax.figure
        
        # Create grid based on slice axis
        if slice_axis == 'x':
            # x-slice: fixed x, varying y and z
            y = np.linspace(-chamber_radius, chamber_radius, resolution)
            z = np.linspace(0, chamber_length, resolution)
            y_grid, z_grid = np.meshgrid(y, z)
            x_grid = np.ones_like(y_grid) * slice_pos
        elif slice_axis == 'y':
            # y-slice: fixed y, varying x and z
            x = np.linspace(0, chamber_length, resolution)
            z = np.linspace(-chamber_radius, chamber_radius, resolution)
            x_grid, z_grid = np.meshgrid(x, z)
            y_grid = np.ones_like(x_grid) * slice_pos
        else:  # slice_axis == 'z'
            # z-slice: fixed z, varying x and y
            x = np.linspace(0, chamber_length, resolution)
            y = np.linspace(-chamber_radius, chamber_radius, resolution)
            x_grid, y_grid = np.meshgrid(x, y)
            z_grid = np.ones_like(x_grid) * slice_pos
        
        # Calculate field strength at each grid point
        field_strength = np.zeros_like(x_grid)
        
        for i in range(resolution):
            for j in range(resolution):
                # Get position
                if slice_axis == 'x':
                    pos = np.array([x_grid[i, j], y_grid[i, j], z_grid[i, j]])
                elif slice_axis == 'y':
                    pos = np.array([x_grid[i, j], y_grid[i, j], z_grid[i, j]])
                else:  # slice_axis == 'z'
                    pos = np.array([x_grid[i, j], y_grid[i, j], z_grid[i, j]])
                
                # Get field
                field = field_function(pos)
                
                # Calculate magnitude
                field_strength[i, j] = np.linalg.norm(field)
        
        # Plot heatmap
        if slice_axis == 'x':
            im = ax.pcolormesh(y_grid, z_grid, field_strength, cmap=cmap, shading='auto')
            ax.set_xlabel('Y (m)')
            ax.set_ylabel('Z (m)')
        elif slice_axis == 'y':
            im = ax.pcolormesh(x_grid, z_grid, field_strength, cmap=cmap, shading='auto')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
        else:  # slice_axis == 'z'
            im = ax.pcolormesh(x_grid, y_grid, field_strength, cmap=cmap, shading='auto')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
        
        # Add colorbar
        fig.colorbar(im, ax=ax, label='Field Strength (T)')
        
        # Set aspect ratio
        ax.set_aspect('equal')
        
        # Title
        if title:
            ax.set_title(f"{title} ({slice_axis}={slice_pos:.2f}m)")
        
        # Tight layout
        fig.tight_layout()
        
        # Return figure or axes
        if return_fig:
            return fig
        else:
            return ax
    
    def plot_3d_field_lines(self, field_function: callable, chamber_length: float, chamber_radius: float,
                         n_lines: int = 20, line_length: float = 0.5, step_size: float = 0.01,
                         ax: Optional[plt.Axes] = None, colors: Optional[str] = None,
                         title: Optional[str] = "Magnetic Field Lines",
                         return_fig: bool = False) -> Union[plt.Figure, plt.Axes]:
        """
        Plot 3D magnetic field lines.
        
        Args:
            field_function: Function to calculate magnetic field
            chamber_length: Length of the chamber in meters
            chamber_radius: Radius of the chamber in meters
            n_lines: Number of field lines to plot
            line_length: Length of each field line
            step_size: Step size for field line integration
            ax: Matplotlib axes to plot on (optional)
            colors: Colors for field lines
            title: Plot title
            return_fig: Whether to return the figure object
            
        Returns:
            Matplotlib axes or figure
        """
        # Create figure if needed
        if ax is None:
            fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Plot chamber outline
        theta = np.linspace(0, 2*np.pi, 100)
        z = np.linspace(0, chamber_length, 100)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = chamber_radius * np.cos(theta_grid)
        y_grid = chamber_radius * np.sin(theta_grid)
        
        # Plot surface
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, color='b')
        
        # Get color map if colors not provided
        if colors is None:
            cmap = cm.get_cmap(self.color_map)
        
        # Generate seed points for field lines
        seed_points = []
        
        # Random seed points within chamber
        for _ in range(n_lines):
            # Random position within chamber
            x = np.random.uniform(0, chamber_length)
            r = np.random.uniform(0, chamber_radius * 0.8)
            theta = np.random.uniform(0, 2*np.pi)
            y = r * np.cos(theta)
            z = r * np.sin(theta)
            
            seed_points.append(np.array([x, y, z]))
        
        # Compute field lines using numerical integration
        for i, seed in enumerate(seed_points):
            # Initialize line
            line = [seed]
            pos = seed.copy()
            
            # Forward integration
            for _ in range(int(line_length / step_size)):
                # Get field at current position
                field = field_function(pos)
                
                # Normalize field
                field_mag = np.linalg.norm(field)
                if field_mag < 1e-10:
                    break
                    
                field = field / field_mag
                
                # Update position
                pos = pos + field * step_size
                
                # Check if still within chamber
                if (pos[0] < 0 or pos[0] > chamber_length or
                    np.sqrt(pos[1]**2 + pos[2]**2) > chamber_radius):
                    break
                
                # Add to line
                line.append(pos.copy())
            
            # Convert to array
            line = np.array(line)
            
            # Plot line
            if colors is None:
                color = cmap(i / n_lines)
            else:
                color = colors
                
            ax.plot(line[:, 1], line[:, 2], line[:, 0], color=color, linewidth=1.5, alpha=0.8)
        
        # Set labels and limits
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('Z (m)')
        ax.set_zlabel('X (m)')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, chamber_length/chamber_radius])
        
        # Set limits
        ax.set_xlim(-chamber_radius, chamber_radius)
        ax.set_ylim(-chamber_radius, chamber_radius)
        ax.set_zlim(0, chamber_length)
        
        # Title
        if title:
            ax.set_title(title)
        
        # Tight layout
        fig.tight_layout()
        
        # Return figure or axes
        if return_fig:
            return fig
        else:
            return ax

# Configure logging
logger = logging.getLogger(__name__)


class Arrow3D(FancyArrowPatch):
    """
    Custom 3D arrow class for matplotlib.
    
    This allows adding directional arrows to 3D plots to show vectors.
    """
    
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """Initialize with points and optional arguments."""
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    
    def draw(self, renderer):
        """Draw the arrow."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class MagneticFieldVisualization:
    """
    Visualization utilities for magnetic field configurations and particle trajectories.
    
    This class provides methods for plotting magnetic field configurations, particle
    trajectories, and various performance metrics in the FFRE optimization process.
    """
    
    def __init__(self, dpi: int = 100, style: str = 'dark_background', 
               color_map: str = 'viridis', fig_size: Tuple[int, int] = (10, 6)):
        """
        Initialize visualization utilities.
        
        Args:
            dpi: DPI for plots
            style: Matplotlib style
            color_map: Color map for plots
            fig_size: Default figure size
        """
        self.dpi = dpi
        self.style = style
        self.color_map = color_map
        self.fig_size = fig_size
        
        # Set plot style
        plt.style.use(style)
    
    def plot_field_configuration(self, field_strengths: np.ndarray, max_field: float, 
                              n_points: int = 100, ax: Optional[plt.Axes] = None, 
                              title: Optional[str] = "Magnetic Field Configuration",
                              show_coils: bool = True, coil_positions: Optional[np.ndarray] = None,
                              return_fig: bool = False) -> Union[plt.Figure, plt.Axes]:
        """
        Plot magnetic field configuration along the chamber length.
        
        Args:
            field_strengths: Array of field strengths at coil locations
            max_field: Maximum field strength in Tesla
            n_points: Number of points to plot
            ax: Matplotlib axes to plot on (optional)
            title: Plot title
            show_coils: Whether to show coil positions
            coil_positions: Array of coil positions (optional)
            return_fig: Whether to return the figure object
            
        Returns:
            Matplotlib axes or figure
        """
        n_coils = len(field_strengths)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        else:
            fig = ax.figure
        
        # Coil positions
        if coil_positions is None:
            coil_positions = np.linspace(0, 1, n_coils)
        
        # Interpolated field
        x = np.linspace(0, 1, n_points)
        B = np.zeros(n_points)
        
        for i in range(n_points):
            # Find segment
            segment_length = 1.0 / (n_coils - 1) if n_coils > 1 else 1.0
            segment_idx = int(x[i] / segment_length)
            segment_idx = min(segment_idx, n_coils - 2)
            
            # Linear interpolation
            if n_coils == 1:
                B[i] = field_strengths[0]
            else:
                segment_pos = (x[i] - segment_idx * segment_length) / segment_length
                B[i] = (1 - segment_pos) * field_strengths[segment_idx] + \
                       segment_pos * field_strengths[segment_idx + 1]
        
        # Scale B field
        B *= max_field
        
        # Plot interpolated field
        ax.plot(x, B, 'b-', linewidth=2, label='Magnetic Field')
        
        # Plot coil positions
        if show_coils:
            ax.scatter(coil_positions, field_strengths * max_field, 
                      c='r', s=50, zorder=3, label='Coil Positions')
        
        # Axis labels
        ax.set_xlabel('Normalized Chamber Position')
        ax.set_ylabel('Magnetic Field Strength (Tesla)')
        
        # Title
        if title:
            ax.set_title(title)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Legend
        if show_coils:
            ax.legend()
        
        # Tight layout
        fig.tight_layout()
        
        # Return figure or axes
        if return_fig:
            return fig
        else:
            return ax
    
    def plot_particle_trajectories(self, trajectories: List[List[Dict[str, Any]]], 
                                 chamber_length: float, chamber_radius: float,
                                 ax: Optional[plt.Axes] = None,
                                 title: Optional[str] = "Particle Trajectories",
                                 color_by_status: bool = True,
                                 show_chamber: bool = True,
                                 show_field_arrows: bool = False,
                                 field_function: Optional[callable] = None,
                                 plot_3d: bool = True,
                                 return_fig: bool = False) -> Union[plt.Figure, plt.Axes]:
        """
        Plot particle trajectories in 3D or 2D.
        
        Args:
            trajectories: List of particle trajectory data
            chamber_length: Length of the chamber in meters
            chamber_radius: Radius of the chamber in meters
            ax: Matplotlib axes to plot on (optional)
            title: Plot title
            color_by_status: Whether to color trajectories by particle status
            show_chamber: Whether to show chamber outline
            show_field_arrows: Whether to show magnetic field direction arrows
            field_function: Function to calculate magnetic field (required if show_field_arrows=True)
            plot_3d: Whether to plot in 3D (True) or 2D (False)
            return_fig: Whether to return the figure object
            
        Returns:
            Matplotlib axes or figure
        """
        # Create figure if needed
        if ax is None:
            if plot_3d:
                fig = plt.figure(figsize=self.fig_size, dpi=self.dpi)
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        else:
            fig = ax.figure
        
        if plot_3d and show_chamber:
            # Plot chamber outline in 3D
            # Create cylinder
            theta = np.linspace(0, 2*np.pi, 100)
            z = np.linspace(0, chamber_length, 100)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = chamber_radius * np.cos(theta_grid)
            y_grid = chamber_radius * np.sin(theta_grid)
            
            # Plot surface
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, color='b')
            
            # Plot end caps
            end_theta = np.linspace(0, 2*np.pi, 100)
            end_r = np.linspace(0, chamber_radius, 20)
            end_theta_grid, end_r_grid = np.meshgrid(end_theta, end_r)
            
            # Start cap
            end_x = end_r_grid * np.cos(end_theta_grid)
            end_y = end_r_grid * np.sin(end_theta_grid)
            end_z = np.zeros_like(end_x)
            ax.plot_surface(end_x, end_y, end_z, alpha=0.1, color='b')
            
            # End cap
            end_z = np.ones_like(end_x) * chamber_length
            ax.plot_surface(end_x, end_y, end_z, alpha=0.1, color='b')
        
        elif not plot_3d and show_chamber:
            # Plot chamber outline in 2D (r-z projection)
            # Bottom line
            ax.plot([0, chamber_length], [-chamber_radius, -chamber_radius], 'b-', alpha=0.5)
            # Top line
            ax.plot([0, chamber_length], [chamber_radius, chamber_radius], 'b-', alpha=0.5)
            # Left cap
            theta = np.linspace(-np.pi/2, np.pi/2, 100)
            ax.plot(np.zeros(100), chamber_radius * np.sin(theta), 'b-', alpha=0.5)
            # Right cap
            ax.plot(np.ones(100) * chamber_length, chamber_radius * np.sin(theta), 'b-', alpha=0.5)
        
        # Plot trajectories
        for traj in trajectories:
            # Skip empty trajectories
            if len(traj) < 2:
                continue
                
            # Extract positions
            positions = np.array([state['position'] for state in traj])
            
            # Get trajectory status
            escaped = traj[-1].get('escaped_thrust', False)
            active = traj[-1].get('active', False)
            
            # Set color based on status
            if color_by_status:
                if escaped:
                    color = 'lime'  # Successfully escaped
                    alpha = 0.8
                    linewidth = 1.5
                elif not active:
                    color = 'red'   # Hit wall or went backward
                    alpha