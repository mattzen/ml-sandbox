import swisseph as swe
import datetime
import pytz
import math
import matplotlib.pyplot as plt
import numpy as np

# Zodiac signs
zodiac_signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']

# Helper function to calculate the difference between two angles
def calculate_aspect(planet1, planet2, orb=6):
    angle = abs(planet1 - planet2)
    if angle > 180:
        angle = 360 - angle
    if abs(angle - 0) <= orb:
        return 'Conjunction'
    elif abs(angle - 60) <= orb:
        return 'Sextile'
    elif abs(angle - 90) <= orb:
        return 'Square'
    elif abs(angle - 120) <= orb:
        return 'Trine'
    elif abs(angle - 180) <= orb:
        return 'Opposition'
    else:
        return None



# Function to convert degrees to radians for polar plot
def degrees_to_radians(degrees):
    return np.deg2rad(360 - degrees + 90)  # Shift to start from the top of the circle
# Birth data
birth_date = datetime.datetime(1990, 4, 15, 12, 0)  # 15th April 1990 at 12:00 PM
timezone = pytz.timezone('Europe/Warsaw')  # Bialystok's timezone
birth_date = timezone.localize(birth_date)

# Birth location (latitude and longitude for Bialystok, Poland)
latitude = 53.1325
longitude = 23.1688


# Set Swiss Ephemeris to use the files needed for planetary calculations
swe.set_ephe_path('/Users/mattzen/Downloads/swisseph-master/ephe')  # Set the path to the ephemeris files (adjust to your local path)

# Calculate Julian Day Number (JDN) for the birth date
julian_day = swe.julday(birth_date.year, birth_date.month, birth_date.day, 
                        birth_date.hour + birth_date.minute / 60.0)

# List of planets to calculate (Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto)
planets = [swe.SUN, swe.MOON, swe.MERCURY, swe.VENUS, swe.MARS, 
           swe.JUPITER, swe.SATURN, swe.URANUS, swe.NEPTUNE, swe.PLUTO]

# Calculate planetary positions
planet_positions = {}
for planet in planets:
    position, _ = swe.calc_ut(julian_day, planet)
    print(position)
    planet_positions[planet] = position[0]

print (planet_positions)

# Ascendant Calculation
flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL
ascendant, ascmc = swe.houses(julian_day, latitude, longitude, b'P')  # 'P' for Placidus house system
asc_deg = ascmc[0]  # Ascendant is the first value returned

# House Cusps Calculation
house_cusps = ascmc[1:13]  # House cusps are from index 1 to 12

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
ax.set_ylim(0, 360)
ax.set_yticklabels([])  # Remove radial labels
ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
ax.set_xticklabels(zodiac_signs)  # Label zodiac signs

# Plot zodiac wheel background
for i in range(12):
    ax.fill_between(np.linspace(degrees_to_radians(i * 30), degrees_to_radians((i+1) * 30), 100),
                    0, 360, color='lightgrey' if i % 2 == 0 else 'white', alpha=0.3)

# Plot planet positions
for planet, degree in planet_positions.items():
    planet_name = swe.get_planet_name(planet)
    ax.plot(degrees_to_radians(degree), 330, 'o', label=planet_name)
    ax.text(degrees_to_radians(degree), 340, planet_name, fontsize=10, ha='center')

# Plot house cusps
for i, cusp in enumerate(house_cusps):
    ax.plot([degrees_to_radians(cusp)] * 2, [0, 360], color='black', linewidth=1)
    ax.text(degrees_to_radians(cusp), 370, f'House {i+1}', fontsize=8, ha='center')

# Plot aspects between planets
for i, planet1 in enumerate(planets):
    for planet2 in planets[i+1:]:
        aspect = calculate_aspect(planet_positions[planet1], planet_positions[planet2])
        if aspect:
            angle1 = degrees_to_radians(planet_positions[planet1])
            angle2 = degrees_to_radians(planet_positions[planet2])
            ax.plot([angle1, angle2], [330, 330], color='red' if aspect in ['Square', 'Opposition'] else 'green', alpha=0.7)

# Add title and legend
plt.title(f'Natal Chart for {birth_date.strftime("%Y-%m-%d %H:%M")} (Bialystok, Poland)')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Show the plot
plt.show()
