"""Curated geography reference seed — version-controlled, verifiable, intentionally small.

Coordinates are each country's capital (well-known, stable). Populations are approximate,
rounded seed values — both country totals AND city figures are given by year for 2000/2010/2020
(city numbers are city-proper/metro approximations). This is a STARTER set — replace or extend it
with an authoritative source
(GeoNames, World Bank, …) via `reasoning.reference.import_csv`. Numbers here are deliberately
round so a reader can sanity-check them; do not treat them as census-precise.
"""

from __future__ import annotations

# (name, iso2, capital, continent, area_km2, capital_lat, capital_lon)
COUNTRIES = [
    ("United States", "US", "Washington", "North America", 9833517, 38.9072, -77.0369),
    ("Canada", "CA", "Ottawa", "North America", 9984670, 45.4215, -75.6972),
    ("Mexico", "MX", "Mexico City", "North America", 1964375, 19.4326, -99.1332),
    ("Brazil", "BR", "Brasilia", "South America", 8515767, -15.7939, -47.8828),
    ("Argentina", "AR", "Buenos Aires", "South America", 2780400, -34.6037, -58.3816),
    ("United Kingdom", "GB", "London", "Europe", 243610, 51.5074, -0.1278),
    ("France", "FR", "Paris", "Europe", 551695, 48.8566, 2.3522),
    ("Germany", "DE", "Berlin", "Europe", 357022, 52.5200, 13.4050),
    ("Spain", "ES", "Madrid", "Europe", 505992, 40.4168, -3.7038),
    ("Italy", "IT", "Rome", "Europe", 301340, 41.9028, 12.4964),
    ("Russia", "RU", "Moscow", "Europe", 17098242, 55.7558, 37.6173),
    ("China", "CN", "Beijing", "Asia", 9596961, 39.9042, 116.4074),
    ("India", "IN", "New Delhi", "Asia", 3287263, 28.6139, 77.2090),
    ("Japan", "JP", "Tokyo", "Asia", 377975, 35.6762, 139.6503),
    ("South Korea", "KR", "Seoul", "Asia", 100210, 37.5665, 126.9780),
    ("Indonesia", "ID", "Jakarta", "Asia", 1904569, -6.2088, 106.8456),
    ("Saudi Arabia", "SA", "Riyadh", "Asia", 2149690, 24.7136, 46.6753),
    ("Turkey", "TR", "Ankara", "Asia", 783562, 39.9334, 32.8597),
    ("Iran", "IR", "Tehran", "Asia", 1648195, 35.6892, 51.3890),
    ("Thailand", "TH", "Bangkok", "Asia", 513120, 13.7563, 100.5018),
    ("Australia", "AU", "Canberra", "Oceania", 7692024, -35.2809, 149.1300),
    ("New Zealand", "NZ", "Wellington", "Oceania", 268021, -41.2865, 174.7762),
    ("Egypt", "EG", "Cairo", "Africa", 1001450, 30.0444, 31.2357),
    ("Nigeria", "NG", "Abuja", "Africa", 923768, 9.0765, 7.3986),
    ("South Africa", "ZA", "Pretoria", "Africa", 1221037, -25.7479, 28.2293),
    ("Kenya", "KE", "Nairobi", "Africa", 580367, -1.2864, 36.8172),
    ("Netherlands", "NL", "Amsterdam", "Europe", 41850, 52.3676, 4.9041),
    ("Sweden", "SE", "Stockholm", "Europe", 450295, 59.3293, 18.0686),
    ("Poland", "PL", "Warsaw", "Europe", 312696, 52.2297, 21.0122),
    ("Greece", "GR", "Athens", "Europe", 131957, 37.9838, 23.7275),
]

# country population by year: name -> {year: population}
_POP = {
    "United States": (282000000, 309000000, 331000000),
    "Canada": (31000000, 34000000, 38000000),
    "Mexico": (98000000, 114000000, 126000000),
    "Brazil": (175000000, 196000000, 213000000),
    "Argentina": (37000000, 41000000, 45000000),
    "United Kingdom": (59000000, 63000000, 67000000),
    "France": (61000000, 65000000, 67000000),
    "Germany": (82000000, 81000000, 83000000),
    "Spain": (41000000, 47000000, 47000000),
    "Italy": (57000000, 59000000, 59000000),
    "Russia": (147000000, 143000000, 144000000),
    "China": (1270000000, 1340000000, 1411000000),
    "India": (1060000000, 1230000000, 1380000000),
    "Japan": (127000000, 128000000, 126000000),
    "South Korea": (47000000, 49000000, 52000000),
    "Indonesia": (212000000, 242000000, 271000000),
    "Saudi Arabia": (21000000, 27000000, 35000000),
    "Turkey": (64000000, 73000000, 84000000),
    "Iran": (65000000, 74000000, 84000000),
    "Thailand": (63000000, 67000000, 70000000),
    "Australia": (19000000, 22000000, 26000000),
    "New Zealand": (3900000, 4400000, 5100000),
    "Egypt": (68000000, 82000000, 102000000),
    "Nigeria": (122000000, 160000000, 206000000),
    "South Africa": (45000000, 51000000, 59000000),
    "Kenya": (31000000, 41000000, 53000000),
    "Netherlands": (16000000, 17000000, 17000000),
    "Sweden": (8900000, 9400000, 10400000),
    "Poland": (38000000, 38000000, 38000000),
    "Greece": (11000000, 11000000, 10700000),
}
_YEARS = (2000, 2010, 2020)
# (place, year, population)
POPULATION = [(name, y, _POP[name][i]) for name in _POP for i, y in enumerate(_YEARS)]

# city geography: (name, country, lat, lon)
_CITY_GEO = [
    ("Washington", "United States", 38.9072, -77.0369),
    ("New York", "United States", 40.7128, -74.0060),
    ("Los Angeles", "United States", 34.0522, -118.2437),
    ("Ottawa", "Canada", 45.4215, -75.6972),
    ("Toronto", "Canada", 43.6532, -79.3832),
    ("Mexico City", "Mexico", 19.4326, -99.1332),
    ("Brasilia", "Brazil", -15.7939, -47.8828),
    ("Sao Paulo", "Brazil", -23.5505, -46.6333),
    ("Buenos Aires", "Argentina", -34.6037, -58.3816),
    ("London", "United Kingdom", 51.5074, -0.1278),
    ("Paris", "France", 48.8566, 2.3522),
    ("Berlin", "Germany", 52.5200, 13.4050),
    ("Madrid", "Spain", 40.4168, -3.7038),
    ("Barcelona", "Spain", 41.3851, 2.1734),
    ("Rome", "Italy", 41.9028, 12.4964),
    ("Moscow", "Russia", 55.7558, 37.6173),
    ("Beijing", "China", 39.9042, 116.4074),
    ("Shanghai", "China", 31.2304, 121.4737),
    ("New Delhi", "India", 28.6139, 77.2090),
    ("Mumbai", "India", 19.0760, 72.8777),
    ("Tokyo", "Japan", 35.6762, 139.6503),
    ("Seoul", "South Korea", 37.5665, 126.9780),
    ("Jakarta", "Indonesia", -6.2088, 106.8456),
    ("Riyadh", "Saudi Arabia", 24.7136, 46.6753),
    ("Ankara", "Turkey", 39.9334, 32.8597),
    ("Istanbul", "Turkey", 41.0082, 28.9784),
    ("Tehran", "Iran", 35.6892, 51.3890),
    ("Bangkok", "Thailand", 13.7563, 100.5018),
    ("Canberra", "Australia", -35.2809, 149.1300),
    ("Sydney", "Australia", -33.8688, 151.2093),
    ("Wellington", "New Zealand", -41.2865, 174.7762),
    ("Cairo", "Egypt", 30.0444, 31.2357),
    ("Abuja", "Nigeria", 9.0765, 7.3986),
    ("Lagos", "Nigeria", 6.5244, 3.3792),
    ("Pretoria", "South Africa", -25.7479, 28.2293),
    ("Nairobi", "Kenya", -1.2864, 36.8172),
    ("Amsterdam", "Netherlands", 52.3676, 4.9041),
    ("Stockholm", "Sweden", 59.3293, 18.0686),
    ("Warsaw", "Poland", 52.2297, 21.0122),
    ("Athens", "Greece", 37.9838, 23.7275),
]

# city populations by year (2000, 2010, 2020) — approximate, round seed values. The 2020 figure
# is the cities table's representative population, so the two never diverge.
_CITY_POP = {
    "Washington": (570000, 600000, 690000),
    "New York": (8000000, 8200000, 8400000),
    "Los Angeles": (3700000, 3800000, 3900000),
    "Ottawa": (800000, 900000, 1000000),
    "Toronto": (2500000, 2700000, 2900000),
    "Mexico City": (8600000, 8900000, 9200000),
    "Brasilia": (2000000, 2600000, 3100000),
    "Sao Paulo": (10400000, 11300000, 12300000),
    "Buenos Aires": (2800000, 2900000, 3100000),
    "London": (7200000, 8000000, 8900000),
    "Paris": (2130000, 2240000, 2100000),
    "Berlin": (3380000, 3460000, 3700000),
    "Madrid": (2880000, 3270000, 3200000),
    "Barcelona": (1500000, 1620000, 1600000),
    "Rome": (2550000, 2760000, 2800000),
    "Moscow": (10000000, 11500000, 12500000),
    "Beijing": (13600000, 19600000, 21540000),
    "Shanghai": (16400000, 22300000, 24870000),
    "New Delhi": (12400000, 14900000, 16800000),
    "Mumbai": (11900000, 12400000, 12400000),
    "Tokyo": (12000000, 13000000, 13960000),
    "Seoul": (10000000, 10300000, 9700000),
    "Jakarta": (8400000, 9600000, 10600000),
    "Riyadh": (4200000, 5800000, 7700000),
    "Ankara": (3600000, 4500000, 5700000),
    "Istanbul": (8800000, 13000000, 15500000),
    "Tehran": (7000000, 8000000, 8700000),
    "Bangkok": (6300000, 8300000, 10500000),
    "Canberra": (320000, 380000, 460000),
    "Sydney": (4000000, 4600000, 5300000),
    "Wellington": (340000, 390000, 420000),
    "Cairo": (7600000, 8700000, 9500000),
    "Abuja": (1000000, 2000000, 3500000),
    "Lagos": (7200000, 10400000, 14000000),
    "Pretoria": (600000, 680000, 740000),
    "Nairobi": (2200000, 3200000, 4400000),
    "Amsterdam": (730000, 780000, 870000),
    "Stockholm": (760000, 850000, 980000),
    "Warsaw": (1700000, 1700000, 1800000),
    "Athens": (750000, 700000, 660000),
}

# (name, country, lat, lon, population) — population is the latest (2020) by-year figure
CITIES = [(n, c, la, lo, _CITY_POP[n][-1]) for (n, c, la, lo) in _CITY_GEO]

# city populations join the SAME by-year table as countries, so a city query for a seed year is
# honored (a non-seed year still falls back to the flat city figure with a year-fidelity caveat)
POPULATION += [(n, y, _CITY_POP[n][i]) for n in _CITY_POP for i, y in enumerate(_YEARS)]

# table name -> (columns, rows)
TABLES = {
    "countries": (["name", "iso2", "capital", "continent", "area_km2", "lat", "lon"], COUNTRIES),
    "population": (["place", "year", "population"], POPULATION),
    "cities": (["name", "country", "lat", "lon", "population"], CITIES),
}
