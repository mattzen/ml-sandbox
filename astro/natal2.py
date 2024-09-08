import matplotlib.pyplot as plt
import numpy as np

def create_simple_natal_chart():
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create a circle for the chart
    circle = plt.Circle((0, 0), 1, fill=False)
    ax.add_artist(circle)

    # Add zodiac signs
    zodiac_signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
                    'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    angles = np.linspace(0, 2*np.pi, len(zodiac_signs), endpoint=False)

    for angle, sign in zip(angles, zodiac_signs):
        x = 1.1 * np.cos(angle)
        y = 1.1 * np.sin(angle)
        ax.text(x, y, sign, ha='center', va='center', rotation=np.degrees(angle)-90)

    # Add lines for each house
    for angle in angles:
        x = np.cos(angle)
        y = np.sin(angle)
        ax.plot([0, x], [0, y], 'k-', linewidth=0.5)

    # Set aspect ratio and remove axes
    ax.set_aspect('equal', 'box')
    ax.axis('off')

    # Add title
    plt.title("Simple Natal Chart Representation", fontsize=16)

    # Show the plot
    plt.show()

# Call the function to create the chart
create_simple_natal_chart()