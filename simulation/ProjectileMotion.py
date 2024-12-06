import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
g = 9.81  # Acceleration due to gravity (m/s^2)
v0 = 20   # Initial velocity (m/s)
theta = 45  # Launch angle (degrees)

# Convert angle to radians
theta_rad = np.radians(theta)

# Time of flight
t_flight = (2 * v0 * np.sin(theta_rad)) / g

# Time array
t = np.linspace(0, t_flight, num=500)

# Equations for trajectory
x = v0 * np.cos(theta_rad) * t  # Horizontal distance
y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2  # Vertical distance

# Apply Seaborn theme
sns.set_theme(style="whitegrid")

# Plot the trajectory
plt.figure(figsize=(10, 5))
sns.lineplot(x=x, y=y, label=f"v0={v0} m/s, angle={theta}°", color="b")
plt.title("Projectile Motion", fontsize=16)
plt.xlabel("Horizontal Distance (m)", fontsize=12)
plt.ylabel("Vertical Distance (m)", fontsize=12)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.legend()
plt.show()
