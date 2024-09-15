import swisseph as swe
from datetime import datetime
import pytz

def calculate_precise_positions(birth_date, birth_time, latitude, longitude):
    # Convert birth date and time to Julian day
    birth_datetime = datetime.strptime(f"{birth_date} {birth_time}", "%Y-%m-%d %H:%M")
    birth_datetime = pytz.timezone('Europe/Warsaw').localize(birth_datetime)
    jd = swe.julday(birth_datetime.year, birth_datetime.month, birth_datetime.day,
                    birth_datetime.hour + birth_datetime.minute/60.0)

    # Set ephemeris path (you need to download these files separately)
    swe.set_ephe_path('/Users/mattzen/Downloads/swisseph-master/ephe')  # Update this path

    # List of planets to calculate (using Swiss Ephemeris constants)
    planets = [(swe.SUN, "Sun"), (swe.MOON, "Moon"), (swe.MERCURY, "Mercury"),
               (swe.VENUS, "Venus"), (swe.MARS, "Mars"), (swe.JUPITER, "Jupiter"),
               (swe.SATURN, "Saturn"), (swe.URANUS, "Uranus"), (swe.NEPTUNE, "Neptune"),
               (swe.PLUTO, "Pluto")]

    positions = {}

    for planet_id, planet_name in planets:
        # Calculate planet position
        flags = swe.FLG_SWIEPH | swe.FLG_SPEED
        result, flag = swe.calc_ut(jd, planet_id, flags)
        
        # Extract longitude (position in zodiac)
        longitude = result[0]  # The first element of the tuple is the longitude
        
        # Convert longitude to degrees in 360 system
        position_360 = longitude % 360
        
        positions[planet_name] = position_360

    # Calculate Ascendant and Midheaven
    houses = swe.houses(jd, latitude, longitude, b'P')
    positions['Ascendant'] = houses[0][0]  # First house cusp
    positions['Midheaven'] = houses[0][9]  # Tenth house cusp

    return positions

# Example usage
birth_date = "1990-04-15"
birth_time = "12:00"
latitude = 53.1325  # Latitude of Bialystok, Poland
longitude = 23.1688  # Longitude of Bialystok, Poland

try:
    precise_positions = calculate_precise_positions(birth_date, birth_time, latitude, longitude)

    for body, position in precise_positions.items():
        print(f"{body}: {position:.2f}°")
except swe.Error as e:
    print(f"SwissEph Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()