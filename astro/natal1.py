import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge

def calculate_positions(birth_date):
    # This function would normally use complex calculations and ephemeris data
    # For this example, we'll use placeholder positions
    return {
        'Sun': 25,  # Aries
        'Moon': 355,  # Pisces
        'Mercury': 45,
        'Venus': 70,
        'Mars': 120,
        'Jupiter': 180,
        'Saturn': 220,
        'Uranus': 280,
        'Neptune': 300,
        'Pluto': 330
    }

def create_detailed_natal_chart(birth_date, birth_time, birth_place):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Create the main circle
    main_circle = Circle((0, 0), 1, fill=False, color='black')
    ax.add_artist(main_circle)
    
    # Add zodiac signs
    zodiac_signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
                    'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    angles = np.linspace(0, 2*np.pi, len(zodiac_signs), endpoint=False)
    
    for angle, sign in zip(angles, zodiac_signs):
        x = 1.1 * np.cos(angle)
        y = 1.1 * np.sin(angle)
        ax.text(x, y, sign, ha='center', va='center', rotation=np.degrees(angle)-90)
    
    # Add house cusps (simplified)
    for i, angle in enumerate(np.linspace(0, 2*np.pi, 12, endpoint=False), 1):
        x = np.cos(angle)
        y = np.sin(angle)
        ax.plot([0, x], [0, y], 'k-', linewidth=0.5)
        ax.text(0.5*x, 0.5*y, str(i), ha='center', va='center', fontsize=8)
    
    # Add planets
    planet_positions = calculate_positions(birth_date)
    planet_symbols = {'Sun': '☉', 'Moon': '☽', 'Mercury': '☿', 'Venus': '♀', 'Mars': '♂', 
                      'Jupiter': '♃', 'Saturn': '♄', 'Uranus': '♅', 'Neptune': '♆', 'Pluto': '♇'}
    
    for planet, angle in planet_positions.items():
        x = 0.8 * np.cos(np.radians(angle))
        y = 0.8 * np.sin(np.radians(angle))
        ax.text(x, y, planet_symbols[planet], ha='center', va='center', fontsize=12)
    
    # Set aspect ratio and remove axes
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    
    # Add title and birth information
    plt.title(f"Natal Chart for {birth_date}, {birth_time}\n{birth_place}", fontsize=16)
    
    plt.tight_layout()
    plt.show()

# Example usage
create_detailed_natal_chart("April 15, 1990", "12:00 PM", "Bialystok, Poland")