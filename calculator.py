import matplotlib.pyplot as plt
import numpy as np

# Load data from the file, lb-to-watts
data = []
with open('raw-data.txt', 'r') as file:
    for line in file:
        x, y = line.strip().split('                    ')
        data.append((float(x), float(y)))

x_values = np.array([point[0] * 0.45359237 for point in data])
y_values = np.array([point[1] for point in data])

# Perform quadratic regression
coefficients = np.polyfit(x_values, y_values, 2)
quadratic_curve = coefficients[0] * x_values**2 + coefficients[1] * x_values + coefficients[2]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# First subplot: Full data
ax1.scatter(x_values, y_values, marker='o', color='blue', label='Data')
ax1.plot(x_values, quadratic_curve, color='red', label='Quadratic Curve')
ax1.set_xlabel('Weight, kg')
ax1.set_ylabel('Power, W')
ax1.set_title('Transformer power handling capability by weight')
ax1.legend()
ax1.grid(True)

# Second subplot: Zoomed to 2 kg max
ax2.scatter(x_values, y_values, marker='o', color='blue', label='Data')
ax2.plot(x_values, quadratic_curve, color='red', label='Quadratic Curve')
ax2.set_xlabel('Weight, kg')
ax2.set_ylabel('Power, W')
ax2.set_title('Zoomed to 1 kg max')
ax2.set_xlim(0, 1)  # Zoom to 2 kg max
ax2.set_ylim(0, 75)  # Zoom to 2 kg max
ax2.legend()
ax2.grid(True)

# Adjust layout and display the plots
plt.tight_layout()

# Save the figure to an image file
plt.savefig('transformer_power_plot.png')

# plt.show()


# Example usage of the function
weight_to_check = 0.4 # Replace with the weight you want to check
power_value = np.poly1d(coefficients)(weight_to_check)
print(f"Power value for weight {weight_to_check} kg: {power_value:.2f} W")