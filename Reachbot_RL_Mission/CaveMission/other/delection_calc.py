
import numpy as np

EI_value = 1
length_of_beam = 1
force_on_tip = 1

def calculate_EI(length_of_beam, force_on_tip, deflection):
    # Calculate the EI (modulus of elasticity times moment of inertia) using the formula:
    # EI = (force * length^3) / (3 * deflection)
    # where:
    # - force is the load applied at the free end of the beam
    # - length is the length of the beam
    # - deflection is the deflection at the free end of the beam

    EI = (force_on_tip * length_of_beam**3) / (3 * deflection)
    return EI

def calculate_deflection(EI_value, length_of_beam, force_on_tip):
    # Calculate the deflection of a cantilever beam using the formula:
    # deflection = (force * length^3) / (3 * E * I)
    # where:
    # - force is the load applied at the free end of the beam
    # - length is the length of the beam
    # - E is the modulus of elasticity (EI_value)
    # - I is the moment of inertia (assumed to be 1 for simplicity)

    # Assuming moment of inertia I = 1 for simplicity
    I = 1

    deflection = (force_on_tip * length_of_beam**3) / (3 * EI_value * I)
    return deflection

def calculate_needed_stiffness(EI_value, length_of_beam, force_on_tip):
    # Calculate the stiffness needed to approximate similar deflection using two beams with a rot stiff joint at the beginning and the middle
    # of the beam.
    # The formula used is:
    # stiffness = (force * length) / deflection
    # where:
    # - force is the load applied at the free end of the beam
    # - length is the length of the beam
    # - deflection is the deflection at the free end of the beam

    # Calculate the deflection at the middle
    joint_angle = (force_on_tip * length_of_beam**2) / (4 * EI_value)
    print(f"Joint Angle: {joint_angle:.4f}")
    # Calculate the stiffness needed
    joint_stiffness = (force_on_tip * length_of_beam) / joint_angle
    return joint_stiffness, joint_angle


import matplotlib.pyplot as plt

def plot_bending_beam(EI_value, length, force_on_tip, num_points=100):
    """
    Plot a beam with the given length and deflection at the tip
    """
    # X coordinates along the beam
    x = np.linspace(0, length, num_points)
    
    # For a cantilever beam with point load at the end, deflection follows:
    # y = (F*x^2)/(6*E*I) * (3*L - x) for 0 <= x <= L
    # Simplified for our case since we know the max deflection:
    y = - ((force_on_tip * x**2) / (6 * EI_value)) * (3 * length - x)
    #print(y)
    
    return x, y

def plot_discretized_beam(joint_angle, length_of_beam):
    """
    Plot a discretized beam with the given length and deflection at the tip
    """
    # X coordinates along the beam - 3 points
    x = np.linspace(0, length_of_beam, 3)
    
    # Calculate y values at each point
    y = np.zeros(3)
    y[0] = 0  # First point is fixed (cantilever beam)
    y[1] = np.cos(np.deg2rad(joint_angle)) * (length_of_beam / 2)  # Middle point deflection
    y[2] = y[1] + np.cos(np.deg2rad(joint_angle) * 2) * (x[2] - x[1])  # End point with doubled angle
    
    return x, y


length_of_beam = 1
force_on_tip = 20
deflection = 0.08
EI_value = calculate_EI(length_of_beam, force_on_tip, deflection)
test_lengths = np.linspace(0.1, 2, 10)

deflection = calculate_deflection(EI_value, length_of_beam, force_on_tip)

print(f"EI Value: {EI_value:.4f}")
print(f"Deflection: {deflection:.4f}")

#for length in test_lengths:
#    deflection = calculate_deflection(EI_value, length, force_on_tip)
#    print(f"Length: {length:.2f}, Force: {force_on_tip:.1f}, Deflection: {deflection:.4f}")
#    needed_stiffness = calculate_needed_stiffness(length, force_on_tip, deflection)
#    print(f"Needed Stiffness: {needed_stiffness:.4f}")

a, joint_angle = calculate_needed_stiffness(length_of_beam, force_on_tip, deflection)

x, y = plot_bending_beam(EI_value, length_of_beam, force_on_tip)
x2, y2 = plot_discretized_beam(-joint_angle, length_of_beam)
# Plot the beam deflection
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Deflection Curve', color='blue')
plt.plot(x2, y2, label='Discretized Beam', color='red')
# Add labels and legend
plt.xlabel('Position along beam (m)')
plt.ylabel('Deflection (m)')
plt.title('Cantilever Beam Deflection under Point Load')
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.show()
