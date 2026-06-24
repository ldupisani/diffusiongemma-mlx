"""End-to-end regression guard for the model-free orchestrators.

Each case routes a query (LexicalMatcher — no model) and runs its orchestrator, asserting
the answer and that ZERO model calls were made. No sidecar needed: a model call would try
to contact it and these must not. This pins the bind + plan + routing for every coded
query type, including the ones grown from captured Claude traces (17, 50%, 37.78°C).
"""

import unittest

from reasoning import router
from reasoning.compiled import model_calls

CASES = [
    # query, expected answer, expected orchestrator id
    ("What is 17% off $240 then 8% tax?", "$215.14", "percent_discount_tax"),
    ("How many days are between 2024-03-03 and 2024-07-15?", "134 days", "days_between_dates"),
    ("How many times does 'ss' appear in 'mississippi'?", "2", "count_occurrences"),
    ("What is the average of 4, 8, 15, 16, and 42?", "17", "list_statistic"),
    ("What is the median of 3, 9, 2, 7, and 5?", "5", "list_statistic"),
    ("What is the highest of 12, 45, 7, and 33?", "45", "list_statistic"),
    ("What is the percentage increase from 50 to 75?", "50%", "percentage_change"),
    ("What is the percent decrease from 200 to 150?", "-25%", "percentage_change"),
    ("Convert 100 degrees Fahrenheit to Celsius.", "37.78°C", "unit_conversion"),
    ("How many km is 5 miles?", "8.05 km", "unit_conversion"),
    ("Convert 150 pounds to kg.", "68.04 kg", "unit_conversion"),
    ("What is the final balance on $1000 invested at 5% annual interest compounded annually for 3 years?",
     "$1157.63", "compound_interest"),
    # finance family — each pinned to the captured Claude answer
    ("I bought shares for $4000 and sold them for $5200. What is my return on investment as a percentage?",
     "30%", "return_on_investment"),
    ("A product costs $40 to make and sells for $60. What is the profit margin as a percentage?",
     "33.33%", "profit_margin"),
    ("An investment grew from $10000 to $16000 over 4 years. What is the compound annual growth rate?",
     "12.47%", "cagr"),
    ("What is the present value of $10000 received in 5 years at a 4% annual discount rate?",
     "$8219.27", "present_value"),
    ("If I save $200 per month at 5% annual interest for 10 years, how much will I have?",
     "$31056.46", "annuity_future_value"),
    ("What is the monthly payment on a $20000 loan at 6% annual interest over 5 years?",
     "$386.66", "loan_payment"),
    ("A machine costs $15000 with a salvage value of $3000 over 6 years. What is the annual straight-line depreciation?",
     "$2000.00", "straight_line_depreciation"),
    ("Fixed costs are $5000, each unit sells for $25 and costs $15 to make. How many units to break even?",
     "500 units", "break_even"),
    ("A dinner bill is $120. Add an 18% tip and split it among 4 people. How much does each person pay?",
     "$35.40", "tip_split"),
    # stats series (captured)
    ("What is the mode of 4, 7, 4, 9, 4, and 7?", "4", "list_statistic"),
    ("What is the standard deviation of 2, 4, 4, 4, 5, 5, 7, and 9?", "2", "list_statistic"),
    ("What is the variance of 2, 4, 4, 4, 5, 5, 7, and 9?", "4", "list_statistic"),
    ("What is the product of 2, 3, 4, and 5?", "120", "list_statistic"),
    ("What is the geometric mean of 2, 8, and 32?", "8", "list_statistic"),
    ("What is the 90th percentile of 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10?", "9.1", "percentile"),
    ("What is the weighted average of 80, 90, and 100 with weights 1, 2, and 3?", "93.33",
     "weighted_average"),
    # date-math series (captured)
    ("What day of the week is 2024-07-15?", "Monday", "weekday_of_date"),
    ("What date is 90 days after 2024-01-01?", "2024-03-31", "date_add"),
    ("What date is 12 weeks after 2024-06-01?", "2024-08-24", "date_add"),
    ("How many weeks are between 2024-01-01 and 2024-12-31?", "52.14 weeks", "days_between_dates"),
    ("How many months are between 2023-02-15 and 2024-08-15?", "18 months", "days_between_dates"),
    ("How many years are between 2000-05-20 and 2024-05-20?", "24 years", "days_between_dates"),
    ("Is 2024 a leap year?", "Yes, 2024 is a leap year.", "leap_year"),
    ("What is the day of the year for 2024-03-01?", "61", "day_of_year"),
    # named-month dates (day-first and month-first; both unambiguous)
    ("What is the day of the year for 10 Feb 2020", "41", "day_of_year"),
    ("How many days are between 1 Jan 2024 and 10 Feb 2024?", "40 days", "days_between_dates"),
    # unit-conversion series (captured) — one factor table spans every dimension
    ("Convert 5 kilometers to meters", "5000 m", "unit_conversion"),
    ("How many feet are in 10 meters?", "32.81 ft", "unit_conversion"),
    ("Convert 100 centimeters to inches", "39.37 in", "unit_conversion"),
    ("Convert 2 gallons to liters", "7.57 L", "unit_conversion"),
    ("How many ounces are in 2 pounds?", "32 oz", "unit_conversion"),
    ("Convert 300 Kelvin to Celsius", "26.85°C", "unit_conversion"),
    ("Convert 1.5 hours to minutes", "90 min", "unit_conversion"),
    ("How many seconds are in 5 minutes?", "300 s", "unit_conversion"),
    ("Convert 500 grams to kilograms", "0.5 kg", "unit_conversion"),
    ("How many milliliters are in 2 liters?", "2000 mL", "unit_conversion"),
    ("Convert 60 miles per hour to kilometers per hour", "96.56 km/h", "unit_conversion"),
    ("How many inches are in 3 feet?", "36 in", "unit_conversion"),
    # text/string series (captured)
    ("Reverse the string 'hello world'", "dlrow olleh", "string_operation"),
    ("How many vowels are in 'encyclopedia'?", "5", "string_operation"),
    ("How many characters are in 'hello world'?", "11", "string_operation"),
    ("Convert 'hello world' to uppercase", "HELLO WORLD", "string_operation"),
    ("How many words are in 'the quick brown fox jumps'?", "5", "string_operation"),
    ("Capitalize each word in 'the lazy dog'", "The Lazy Dog", "string_operation"),
    ("How many consonants are in 'python'?", "5", "string_operation"),
    ("What is the longest word in 'the quick brown fox'?", "quick", "string_operation"),
    ("Convert 'HELLO' to lowercase", "hello", "string_operation"),
    ("What is the acronym of 'North Atlantic Treaty Organization'?", "NATO", "string_operation"),
    ("Is 'racecar' a palindrome?", "Yes, 'racecar' is a palindrome.", "palindrome"),
    ("Replace 'cat' with 'dog' in 'the cat sat on the mat'", "the dog sat on the mat",
     "replace_text"),
    # geometry/measurement series (captured)
    ("What is the area of a circle with radius 5?", "78.54", "geometry"),
    ("What is the circumference of a circle with radius 7?", "43.98", "geometry"),
    ("What is the area of a rectangle with length 4 and width 6?", "24", "geometry"),
    ("What is the perimeter of a rectangle with length 8 and width 3?", "22", "geometry"),
    ("What is the area of a triangle with base 10 and height 5?", "25", "geometry"),
    ("What is the area of a square with side 9?", "81", "geometry"),
    ("What is the volume of a cube with side 3?", "27", "geometry"),
    ("What is the volume of a sphere with radius 4?", "268.08", "geometry"),
    ("What is the volume of a cylinder with radius 3 and height 10?", "282.74", "geometry"),
    ("What is the hypotenuse of a right triangle with legs 3 and 4?", "5", "geometry"),
    ("What is the surface area of a sphere with radius 6?", "452.39", "geometry"),
    ("What is the diagonal of a rectangle with length 3 and width 4?", "5", "geometry"),
    # physics-formula series (captured)
    ("What is the speed of an object that travels 100 meters in 5 seconds?", "20 m/s", "physics"),
    ("What is the force on a mass of 10 kg accelerating at 3 m/s^2?", "30 N", "physics"),
    ("What is the kinetic energy of a 2 kg object moving at 10 m/s?", "100 J", "physics"),
    ("What is the gravitational potential energy of a 5 kg object at a height of 10 meters?",
     "490 J", "physics"),
    ("How much work is done by a force of 20 N over a distance of 4 meters?", "80 J", "physics"),
    ("What is the power if 100 joules of work is done in 5 seconds?", "20 W", "physics"),
    ("What is the density of an object with mass 300 g and volume 100 cm^3?", "3 g/cm³", "physics"),
    ("What is the voltage across a 5 ohm resistor with a current of 2 amps?", "10 V", "physics"),
    ("What is the momentum of a 4 kg object moving at 8 m/s?", "32 kg·m/s", "physics"),
    ("What is the pressure of a 50 N force over an area of 10 square meters?", "5 Pa", "physics"),
    ("What is the acceleration of an object that goes from 0 to 20 m/s in 4 seconds?",
     "5 m/s²", "physics"),
    ("What is the weight of a 10 kg object on Earth?", "98 N", "physics"),
    # number-theory series (captured)
    ("What is the GCD of 24 and 36?", "12", "number_theory"),
    ("What is the LCM of 4 and 6?", "12", "number_theory"),
    ("What is the factorial of 5?", "120", "number_theory"),
    ("Is 17 a prime number?", "Yes, 17 is prime.", "number_theory"),
    ("What is the prime factorization of 60?", "2^2 × 3 × 5", "number_theory"),
    ("How many divisors does 28 have?", "6", "number_theory"),
    ("What is 8 choose 3?", "56", "number_theory"),
    ("In how many ways can 5 distinct items be arranged?", "120", "number_theory"),
    ("What is 7 factorial?", "5040", "number_theory"),
    ("Is 100 a perfect square?", "Yes, 100 is a perfect square.", "number_theory"),
    # chemistry / stoichiometry series (captured)
    ("What is the molar mass of H2O?", "18.02 g/mol", "chemistry"),
    ("What is the molar mass of CO2?", "44.01 g/mol", "chemistry"),
    ("What is the molar mass of NaCl?", "58.44 g/mol", "chemistry"),
    ("What is the molar mass of C6H12O6?", "180.16 g/mol", "chemistry"),
    ("What is the molar mass of H2SO4?", "98.07 g/mol", "chemistry"),
    ("How many moles are in 36 grams of water given a molar mass of 18 g/mol?", "2 mol",
     "chemistry"),
    ("What is the molarity of 2 moles of solute dissolved in 4 liters of solution?", "0.5 M",
     "chemistry"),
    ("What is the pH of a solution with a hydrogen ion concentration of 0.001 M?", "3",
     "chemistry"),
    ("What is the final concentration when 50 mL of 2 M solution is diluted to 200 mL?", "0.5 M",
     "chemistry"),
    # base / radix conversion (captured)
    ("Convert 255 to binary.", "11111111", "base_conversion"),
    ("Convert 1010 from binary to decimal.", "10", "base_conversion"),
    ("What is 255 in hexadecimal?", "FF", "base_conversion"),
    ("Convert hexadecimal FF to decimal.", "255", "base_conversion"),
    ("Convert 64 to octal.", "100", "base_conversion"),
    ("Convert the decimal number 100 to binary.", "1100100", "base_conversion"),
    ("Convert binary 11111111 to decimal.", "255", "base_conversion"),
    ("Convert hexadecimal A to decimal.", "10", "base_conversion"),  # hex digit A vs article "a"
    # data-size conversion (captured) — reuses the units factor table
    ("How many bytes are in 5 kilobytes?", "5120 bytes", "unit_conversion"),
    ("Convert 2 gigabytes to megabytes.", "2048 MB", "unit_conversion"),
    ("How many megabytes are in 1 terabyte?", "1048576 MB", "unit_conversion"),
    # trigonometry (captured)
    ("What is the sine of 30 degrees?", "0.5", "trigonometry"),
    ("What is the cosine of 60 degrees?", "0.5", "trigonometry"),
    ("What is the tangent of 45 degrees?", "1", "trigonometry"),
    ("What is the sine of 90 degrees?", "1", "trigonometry"),
    ("Convert 180 degrees to radians.", "3.1416 rad", "trigonometry"),
    ("Convert 90 degrees to radians.", "1.5708 rad", "trigonometry"),
    ("Convert pi radians to degrees.", "180°", "trigonometry"),
    ("Convert 1.5708 radians to degrees.", "90°", "trigonometry"),
    # coordinate geometry (captured)
    ("What is the distance between the points (0,0) and (3,4)?", "5", "coordinate_geometry"),
    ("What is the distance from (0,0) to (5,12)?", "13", "coordinate_geometry"),
    ("What is the midpoint between (2,4) and (6,8)?", "(4, 6)", "coordinate_geometry"),
    ("What is the midpoint between (-4,2) and (4,-2)?", "(0, 0)", "coordinate_geometry"),
    ("What is the slope of the line through (1,2) and (3,6)?", "2", "coordinate_geometry"),
    ("What is the slope between (2,3) and (5,3)?", "0", "coordinate_geometry"),
    # clock arithmetic (captured)
    ("What time is it 3 hours after 2:30 PM?", "5:30 PM", "clock_arithmetic"),
    ("What time is 90 minutes after 11:15 AM?", "12:45 PM", "clock_arithmetic"),
    ("Convert 14:30 to 12-hour format.", "2:30 PM", "clock_arithmetic"),
    ("Convert 3:45 PM to 24-hour format.", "15:45", "clock_arithmetic"),
    ("How many minutes are between 9:15 AM and 10:45 AM?", "90 minutes", "clock_arithmetic"),
    ("What time is it 5 hours before 1:00 AM?", "8:00 PM", "clock_arithmetic"),
    ("How long is it from 8:00 AM to 5:30 PM?", "9 hours and 30 minutes", "clock_arithmetic"),
    ("What time is 45 minutes after 11:30 PM?", "12:15 AM", "clock_arithmetic"),
    ("Convert 00:00 to 12-hour format.", "12:00 AM", "clock_arithmetic"),
    ("What is 6:00 PM in 24-hour time?", "18:00", "clock_arithmetic"),
    # set / inclusion-exclusion (captured)
    ("In a group of 30 people, 18 like coffee and 15 like tea, and 8 like both. "
     "How many like coffee or tea?", "25", "set_problems"),
    ("Out of 50 people, 30 own a dog and 25 own a cat, and 10 own both. How many own neither?",
     "5", "set_problems"),
    ("In a class of 40, 25 play soccer and 20 play basketball, and 12 play both. "
     "How many play only soccer?", "13", "set_problems"),
    ("If set A has 12 elements, set B has 8, and 3 are in both, how many are in A union B?",
     "17", "set_problems"),
    ("Of 80 students, 50 passed math, 45 passed english, and 30 passed both. "
     "How many passed at least one?", "65", "set_problems"),
    # probability (captured, computable subset)
    ("If there is a 30 percent chance of rain, what is the chance it does not rain?", "70%",
     "probability"),
    ("What is the probability of flipping heads twice in a row?", "0.25", "probability"),
    ("If you flip a coin 3 times, what is the probability of getting all heads?", "0.125",
     "probability"),
    ("What is the probability of rolling a 6 on a fair die?", "0.1667", "probability"),
    ("What is the probability of rolling a sum of 7 with two dice?", "0.1667", "probability"),
    ("What is the expected value of rolling a fair six-sided die?", "3.5", "probability"),
    ("In how many ways can 4 people be seated in a row?", "24", "number_theory"),
    # ratio / proportion / fraction (captured)
    ("Simplify the fraction 18/24.", "3/4", "ratio_proportion"),
    ("What is 3/4 as a decimal?", "0.75", "ratio_proportion"),
    ("What is 0.75 as a fraction?", "3/4", "ratio_proportion"),
    ("What is 3/5 as a percentage?", "60%", "ratio_proportion"),
    ("Solve the proportion 2:5 = x:20.", "8", "ratio_proportion"),
    ("Simplify the ratio 12:18.", "2:3", "ratio_proportion"),
    ("What is the unit rate if 150 miles takes 3 hours?", "50", "ratio_proportion"),
    ("If a recipe for 4 servings needs 2 cups of flour, how much for 6 servings?", "3 cups",
     "ratio_proportion"),
    ("If 3 apples cost $2, how much do 12 apples cost?", "8", "ratio_proportion"),
    # health metrics (captured)
    ("What is the BMI of a person who weighs 70 kg and is 1.75 meters tall?", "22.86",
     "health_metrics"),
    ("What is the BMI of a person who weighs 90 kg and is 2 meters tall?", "22.5",
     "health_metrics"),
    ("What is the maximum heart rate for a 25 year old?", "195 bpm", "health_metrics"),
    ("What is the target heart rate at 70 percent intensity for a 40 year old?", "126 bpm",
     "health_metrics"),
    ("What is the pace per mile if you run 5 miles in 40 minutes?", "8 minutes per mile",
     "health_metrics"),
    ("What is the pace per kilometer if you run 10 km in 50 minutes?", "5 minutes per kilometer",
     "health_metrics"),
    ("What is the BMR of a 30 year old man weighing 80 kg and 180 cm tall?", "1780 kcal/day",
     "health_metrics"),
    ("What is the ideal body weight for a man who is 180 cm tall?", "75 kg", "health_metrics"),
    # formal logic (captured + symbolic)
    ("Is (P OR NOT P) a tautology?", "Yes, it is a tautology.", "logic"),
    ("Is (P AND NOT P) a contradiction?", "Yes, it is a contradiction.", "logic"),
    ("Is (P AND Q) satisfiable?", "Yes, it is satisfiable.", "logic"),
    ("Are (P implies Q) and (NOT P OR Q) logically equivalent?",
     "Yes, they are logically equivalent.", "logic"),
    ("Evaluate P AND Q when P is true and Q is false.", "False", "logic"),
    ("What is the negation of (A AND B)?", "(NOT A) OR (NOT B)", "logic"),
    ("What is the converse of 'if P then Q'?", "If Q then P", "logic"),
    ("What is the contrapositive of 'if P then Q'?", "If NOT Q then NOT P", "logic"),
    ("What is true XOR true?", "False", "logic"),
    ("If P is true and Q is false, what is P implies Q?", "False", "logic"),
    # raw arithmetic (closes the model-fallback gap)
    ("What is 17 percent of 240?", "40.8", "arithmetic"),
    ("What is 25 plus 17?", "42", "arithmetic"),
    ("What is 100 minus 30?", "70", "arithmetic"),
    ("What is 7 times 8?", "56", "arithmetic"),
    ("What is 100 divided by 4?", "25", "arithmetic"),
    ("What is 2 to the power of 10?", "1024", "arithmetic"),
    ("What is the square root of 144?", "12", "arithmetic"),
    ("What is 15% of 200?", "30", "arithmetic"),
    # compound arithmetic expressions (≥2 ops, order-of-operations) — `expression` orchestrator
    ("What is (5 + 3) * 2 - 4?", "12", "expression"),
    ("What is 15% of 200 plus 10?", "40", "expression"),
    ("What is 100 divided by 4 plus 7?", "32", "expression"),
    ("What is 3 times 4 plus 5 times 6?", "42", "expression"),
    ("What is 2 to the power of 10 minus 24?", "1000", "expression"),
    # composed: delegates the unit conversion to unit_conversion via a SubStep (0 model calls)
    ("What is the total cost of 10 miles at $3 per km?", "$48.27", "converted_cost"),
    ("What is the cost of 5 miles at $2 per km?", "$16.1", "converted_cost"),
    ("What is the total cost of 100 yards at $3 per km?", "$0.27", "converted_cost"),
    # synthesis-backed physics: a multi-step derivation (force→work→power) no single formula does
    ("What is the power if a 2 kg mass accelerates at 3 m/s^2 over 4 meters in 5 seconds?",
     "4.8 W", "synthesized_physics"),
    ("How much work is done accelerating a 10 kg mass at 2 m/s^2 over 5 meters?",
     "100 J", "synthesized_physics"),
    # synthesis-backed finance: discount→tax→split (a chain percent_discount_tax can't do)
    ("A $240 jacket is 17% off with 8% tax, split between 3 people — how much each?",
     "$71.71 each", "synthesized_finance"),
    # synthesis-backed geometry → cost: area × unit price (was hijacked by ratio's cost branch)
    ("How much does it cost to tile a 4 meter by 6 meter floor at $8 per square meter?",
     "$192", "synthesized_geometry"),
    # synthesis-backed chemistry: molar_mass(formula) → moles / mass (formula → derivation)
    ("How many moles are in 36 grams of H2O?", "2 mol", "synthesized_chemistry"),
    ("What is the mass of 2 moles of H2O?", "36.03 g", "synthesized_chemistry"),
    # extended synthesis chains (one new chain per live domain)
    ("What is the pressure if a 5 kg mass accelerates at 2 m/s^2 over an area of 4 square meters?",
     "2.5 Pa", "synthesized_physics"),
    ("What is the force on a 2 kg object accelerating from 0 to 10 m/s in 5 seconds?",
     "4 N", "synthesized_physics"),
    ("What is the total for 5 shirts at $20 each with 10% off and 8% tax?",
     "$97.2", "synthesized_finance"),
    ("How much to fence a 4 by 6 yard at $5 per meter?", "$100", "synthesized_geometry"),
    ("How much to fill a cube with a side of 3 at $2 per cubic meter?", "$54", "synthesized_geometry"),
    ("What is the molarity of 58.44 grams of NaCl in 2 liters?", "0.5 M", "synthesized_chemistry"),
    # set theory
    ("What is {1, 2, 3} union {2, 3, 4}?", "{1, 2, 3, 4}", "set_operations"),
    ("Is {1, 2} a subset of {1, 2, 3}?", "Yes.", "set_operations"),
    ("What is the cardinality of {1, 2, 3, 4}?", "4", "set_operations"),
    ("What is the power set of {1, 2}?", "{∅, {1}, {2}, {1, 2}}", "set_operations"),
    ("What is the Cartesian product of {1, 2} and {3}?", "{(1, 3), (2, 3)}", "set_operations"),
    # group theory
    ("What is the order of the element 2 in Z_6?", "3", "group_theory"),
    ("What are the generators of Z_8?", "{1, 3, 5, 7}", "group_theory"),
    ("What is the order of the permutation (2, 3, 1)?", "3", "group_theory"),
    ("Is the permutation (2, 1, 3) even or odd?", "odd", "group_theory"),
    ("What is the order of the symmetric group S_4?", "24", "group_theory"),
    ("What is 3 + 4 in Z_5?", "2", "group_theory"),
    # topology
    ("Is {∅, {1}, {1, 2}} a topology on {1, 2}?", "Yes.", "topology"),
    ("What is the closure of {1} in the topology {∅, {1}, {1, 2}} on {1, 2}?", "{1, 2}", "topology"),
    ("What is the interior of {2} in the topology {∅, {1}, {1, 2}} on {1, 2}?", "∅", "topology"),
    ("Is the topology {∅, {1}, {2}, {1, 2}} on {1, 2} Hausdorff?", "Yes.", "topology"),
    # algebra (SymPy)
    ("Expand (x + 1)(x + 2).", "x**2 + 3*x + 2", "algebra"),
    ("Factor x^2 - 5x + 6.", "(x - 3)*(x - 2)", "algebra"),
    ("Solve x^2 - 5x + 6 = 0 for x.", "[2, 3]", "algebra"),
    # parametric solve: roots in terms of y — must order by string, not crash on symbolic compare
    ("Solve x^2 - 5x + 6 + y = 0 for x",
     "[5/2 - sqrt(1 - 4*y)/2, sqrt(1 - 4*y)/2 + 5/2]", "algebra"),
    # calculus (SymPy)
    ("What is the derivative of x^3 + 3x^2?", "3*x**2 + 6*x", "calculus"),
    ("What is the integral of x^2?", "x**3/3", "calculus"),
    ("What is the integral of x^2 from 0 to 1?", "1/3", "calculus"),
    ("What is the limit of sin(x)/x as x approaches 0?", "1", "calculus"),
    # --- domain extensions ---
    ("Is {1, 2} a proper subset of {1, 2, 3}?", "Yes.", "set_operations"),
    ("How many subsets does {1, 2, 3} have?", "8", "set_operations"),
    ("What is the inverse of 3 in Z_5?", "2", "group_theory"),
    ("What is the order of the dihedral group D_4?", "8", "group_theory"),
    ("What is the subgroup of Z_6 generated by 2?", "{0, 2, 4}", "group_theory"),
    ("What is the exterior of {2} in the topology {∅, {1}, {1, 2}} on {1, 2}?", "{1}", "topology"),
    ("Is {1} dense in the topology {∅, {1}, {1, 2}} on {1, 2}?", "Yes.", "topology"),
    ("Evaluate x^2 + 1 at x = 3.", "10", "algebra"),
    ("What is the degree of x^3 + 2x?", "3", "algebra"),
    ("Solve the system x + y = 5 and x - y = 1.", "{x: 3, y: 2}", "algebra"),
    ("What is the second derivative of x^4?", "12*x**2", "calculus"),
    ("What are the critical points of x^3 - 3x?", "[-1, 1]", "calculus"),
    ("What is the Taylor series of exp(x) around 0 to order 4?",
     "x**3/6 + x**2/2 + x + 1", "calculus"),
    # geospatial
    ("What is the distance in km between (40.7128, -74.0060) and (51.5074, -0.1278)?",
     "5570.22 km", "geospatial"),
    ("What is the bearing from (0, 0) to (0, 1) on Earth?", "90°", "geospatial"),
    ("Convert 40 degrees 30 minutes 0 seconds to decimal degrees.", "40.5", "geospatial"),
    ("Convert 40.5 degrees to degrees minutes seconds.", "40° 30' 0\"", "geospatial"),
    # genetics
    ("What are the Hardy-Weinberg genotype frequencies if the dominant allele frequency is 0.7?",
     "AA: 0.49, Aa: 0.42, aa: 0.09", "genetics"),
    ("What is the offspring ratio of a cross of Aa and Aa?", "1 AA : 2 Aa : 1 aa", "genetics"),
    ("What is the allele frequency of 35 out of 50 alleles?", "0.7", "genetics"),
    # material science
    ("What is the atomic packing factor of an FCC structure?", "0.7405", "materials"),
    ("What is the stress of a 100 N force over an area of 0.5 square meters?", "200 Pa", "materials"),
    ("What is Young's modulus if the stress is 200 Pa and the strain is 0.001?",
     "200000 Pa", "materials"),
    # discrete mathematics
    ("How many connected components are in the graph with edges (1,2), (2,3), (4,5)?",
     "2", "discrete_math"),
    ("What is the 10th Fibonacci number?", "55", "discrete_math"),
    ("What is the 4th Catalan number?", "14", "discrete_math"),
    ("What is the Stirling number of the second kind for 4 and 2?", "7", "discrete_math"),
    # applied mathematics (linear algebra)
    ("What is the determinant of [[1, 2], [3, 4]]?", "-2", "applied_math"),
    ("What are the eigenvalues of [[2, 0], [0, 3]]?", "[2, 3]", "applied_math"),
    ("What is the dot product of (1, 2, 3) and (4, 5, 6)?", "32", "applied_math"),
    ("What is the magnitude of the vector (3, 4)?", "5.0", "applied_math"),
    # practical chemistry
    ("What is the pressure of 2 moles of gas at 300 K in 10 liters?", "4.9234 atm",
     "practical_chemistry"),
    ("What is the percent composition of O in H2O?", "88.81%", "practical_chemistry"),
    ("What is the pOH if the pH is 3?", "11", "practical_chemistry"),
    ("What is the mass percent of 20 g solute in 80 g of solution?", "25%", "practical_chemistry"),
    # molecular science
    ("How many molecules are in 2 moles?", "1.2044e+24", "molecular_science"),
    ("What are the degrees of unsaturation of C6H6?", "4", "molecular_science"),
    # biochemistry
    ("What is the GC content of ATGC?", "50%", "biochemistry"),
    ("What is the reverse complement of ATGC?", "GCAT", "biochemistry"),
    ("Transcribe ATGC to mRNA.", "AUGC", "biochemistry"),
    ("What is the buffer pH with pKa 4.76, base 10, and acid 1?", "5.76", "biochemistry"),
    ("What is the reaction rate with Vmax 100, Km 5, and substrate 15?", "75", "biochemistry"),
    # N-dimensional shapes
    ("What is the volume of a 4-dimensional ball of radius 2?", "78.9568", "nd_geometry"),
    ("What is the volume of a 4-dimensional cube of side 2?", "16", "nd_geometry"),
    ("How many edges does a 4-dimensional hypercube have?", "32", "nd_geometry"),
    # tensor maths
    ("What is the rank of [[1, 2], [3, 4]]?", "2", "tensor_math"),
    ("What is the trace of [[1, 2], [3, 4]]?", "5", "tensor_math"),
    ("What is the outer product of [1, 2] and [3, 4]?", "[[3, 4], [6, 8]]", "tensor_math"),
    # complex numbers
    ("What is the modulus of 3+4i?", "5", "complex_numbers"),
    ("What is the conjugate of 3+4i?", "3-4i", "complex_numbers"),
    ("What is (1+1i) raised to the power 3?", "-2+2i", "complex_numbers"),
    # complex calculus
    ("What is the residue of 1/z at 0?", "1", "complex_calculus"),
    ("What is the residue of 1/(z^2+1) at i?", "-I/2", "complex_calculus"),
    ("What is the contour integral of 1/z around 0?", "2*I*pi", "complex_calculus"),
    # cross-domain synthesis: geometry area → cost → finance-style split
    ("What does it cost to tile a 4 by 6 meter floor at $8 per square meter, split between "
     "3 people?", "$64 each", "synthesized_geometry"),
    # unit-aware synthesis: 5 km is canonicalized to metres before the force→work chain
    ("How much work to push a 2 kg object accelerating at 3 m/s^2 over 5 km?",
     "30000 J", "synthesized_physics"),
    # multi-goal synthesis: two properties in one plan
    ("What are the area and perimeter of a 4 by 6 rectangle?",
     "area: 24, perimeter: 20", "geometry_multi"),
    # inverse synthesis: solve for the discount (input) given the target final price (output)
    ("What discount on a $80 item results in a final price of $60?", "25%", "inverse_finance"),
    # expanded cross-domain: geometry cost + tax tail, fence_cost split, chemistry → finance
    ("How much to tile a 4 by 6 meter floor at $8 per square meter plus 8% tax?",
     "$207.36", "synthesized_geometry"),
    ("What does it cost to fence a 10 by 20 meter yard at $15 per meter, split between 4 people?",
     "$225 each", "synthesized_geometry"),
    ("What is the cost of 3 moles of NaCl at $0.50 per gram?", "$87.66", "synthesized_chemistry"),
    # chemistry → molecular (mass → moles → molecules)
    ("How many molecules are in 36 grams of H2O?", "1.2034e+24", "synthesized_chemistry"),
    # physics force + geometry shape → materials stress / Young's modulus
    ("What is the stress on a 3 by 4 m plate under a 1200 N force?", "100 Pa",
     "synthesized_materials"),
    ("What is Young's modulus of a 3 by 4 m plate under 1200 N with strain 0.001?",
     "100000 Pa", "synthesized_materials"),
    # §A ready seams: geometry shape → physics pressure; motion → physics kinetic energy
    ("What is the pressure of a 100 N force on a circular plate of radius 2 m?",
     "7.96 Pa", "synthesized_physics"),
    ("What is the kinetic energy of a 1000 kg car that travels 200 m in 10 s?",
     "200000 J", "synthesized_physics"),
    # §B priced quantities → cost (labour; electricity = kW×h×$/kWh, a physics→finance chain)
    ("How much is a 5-hour job at $40 per hour?", "$200", "priced_cost"),
    ("What does a 6-hour job at $50 per hour cost, split between 2 people?",
     "$150 each", "priced_cost"),
    ("What is the cost of running a 2 kW heater for 3 hours at $0.15 per kWh?",
     "$0.9", "priced_cost"),
    # geo → finance (travel cost) and geo → motion (flight time)
    ("What is the travel cost between (40.7128, -74.0060) and (51.5074, -0.1278) at $0.10 per km?",
     "$557.02", "synthesized_geo"),
    ("How long is a flight between (40.7128, -74.0060) and (51.5074, -0.1278) at 900 km/h?",
     "6.19 hours", "synthesized_geo"),
    # the bind normalizes a price per ANY length unit to $/km (mile, inch, … all work)
    ("What is the travel cost between (40.7128, -74.0060) and (51.5074, -0.1278) at $0.10 per mile?",
     "$346.12", "synthesized_geo"),
    ("What is the travel cost between (40.7128, -74.0060) and (51.5074, -0.1278) at $0.10 per inch?",
     "$21930008.58", "synthesized_geo"),
    # coordinate pairs accepted bare or wrapped in (), [], {} (geo-scoped, so it can't leak)
    ("What is the travel cost between 40.7128, -74.0060 and 51.5074, -0.1278 at $0.10 per km?",
     "$557.02", "synthesized_geo"),
    ("What is the travel cost between {40.7128, -74.0060} and {51.5074, -0.1278} at $0.10 per km?",
     "$557.02", "synthesized_geo"),
    # §D multi-hop: chemistry → finance → split (a 5-op chain)
    ("What is the cost of 3 moles of NaCl at $0.50 per gram, split between 2 people?",
     "$43.83 each", "synthesized_chemistry"),
    # §F unit coercion: chem mass (g) → units(g→kg) → physics weight
    ("What is the weight of 2 moles of H2O?", "0.35 N", "synthesized_chemistry"),
    # p-adic numbers: valuation, absolute value, expansion, distance
    ("What is the 2-adic valuation of 24?", "3", "padic"),
    ("What is the 5-adic absolute value of 100?", "0.04", "padic"),
    ("Give the 5-adic expansion of 100 to 3 digits.", "0 + 0·5 + 4·5^2", "padic"),
    ("What is the 3-adic distance between 1 and 28?", "0.037", "padic"),
    # hypercomplex (quaternion): conjugate, norm, inverse
    ("What is the conjugate of 1+2i+3j+4k?", "1-2i-3j-4k", "hypercomplex"),
    ("What is the norm of 1+2i+3j+4k?", "5.4772", "hypercomplex"),
    ("What is the inverse of the quaternion i?", "-i", "hypercomplex"),
    ("What is the magnitude of the quaternion 2+3i+6k?", "7", "hypercomplex"),
    # ordinals (Cantor normal form, non-commutative): 'transfinite' folds in here too
    ("What is the Cantor normal form of 1+ω?", "ω", "ordinals"),
    ("Simplify the ordinal ω·2+ω.", "ω·3", "ordinals"),
    ("Is ω+1 greater than ω?", "ω+1 > ω", "ordinals"),
    ("What is the transfinite ordinal ω·ω?", "ω^2", "ordinals"),
    # cardinals (ℵ₀, 𝔠 = 2^ℵ₀): 'transfinite' folds in here too
    ("What is aleph_0 + aleph_0?", "ℵ₀", "cardinals"),
    ("What is 2 to the power of aleph_0?", "𝔠", "cardinals"),
    ("What is the cardinality of the real numbers?", "𝔠", "cardinals"),
    ("Is aleph_0 less than the continuum?", "ℵ₀ < 𝔠", "cardinals"),
    # hyperreals: ε infinitesimal, ω = 1/ε infinite (gated apart from ordinal ω)
    ("What is the standard part of 3+ε?", "3", "hyperreal"),
    ("Classify the hyperreal ε.", "infinitesimal", "hyperreal"),
    ("What is ω·ε?", "1", "hyperreal"),
    ("Is ω greater than 5 in the hyperreals?", "ω > 5", "hyperreal"),
]


class TestOrchestrators(unittest.TestCase):
    def test_coded_orchestrators_reproduce_and_stay_model_free(self):
        for query, expected, orch_id in CASES:
            with self.subTest(query=query):
                m = router.recognize(query)
                self.assertIsNotNone(m, f"no match for {query!r}")
                self.assertEqual(m.orchestrator.id, orch_id)
                r = m.orchestrator.run(query)
                self.assertEqual(r.answer, expected)
                self.assertEqual(model_calls(r), 0, f"{query!r} used the model")

    def test_list_statistic_declines_on_scattered_numbers(self):
        # the bug: "total" + two scattered numbers (100, 3) were summed into "103"
        from reasoning.bind import bind_list_statistic
        self.assertFalse(bind_list_statistic("What is the total cost of 100 yards at $3?")[0])
        # a genuine list still binds
        self.assertTrue(bind_list_statistic("What is the average of 4, 8, and 15?")[0])

    def test_financial_binds_require_exact_dollar_count(self):
        # positional-$ binds must decline when a stray $ makes the operands ambiguous,
        # rather than shifting positions into a confident wrong answer
        from reasoning import bind as B
        for fn, clean, stray in [
            (B.bind_roi, "bought for $4000 and sold for $5200, ROI?",
             "a $50 fee, bought for $4000 and sold for $5200, ROI?"),
            (B.bind_break_even, "fixed $5000, sells for $10, costs $4, break even?",
             "a $200 license, fixed $5000, sells for $10, costs $4, break even?"),
            (B.bind_loan, "payment on a $20000 loan at 6% over 5 years?",
             "a $500 fee, payment on a $20000 loan at 6% over 5 years?"),
        ]:
            self.assertTrue(fn(clean)[0])
            self.assertFalse(fn(stray)[0], "stray $ should make operands ambiguous → decline")

    def test_geometry_cost_declines_per_unit_rate(self):
        # "cost PER square inch …" asks for a rate this orchestrator can't compute (it returns
        # a total) — must decline, not silently answer the total or be mis-scaled by ratio
        from reasoning.bind import bind_synthesis_geometry
        rate = "How much does it cost per square inch to tile a 4 by 6 floor at $8 per square meter?"
        total = "How much does it cost to tile a 4 meter by 6 meter floor at $8 per square meter?"
        self.assertFalse(bind_synthesis_geometry(rate)[0])
        self.assertTrue(bind_synthesis_geometry(total)[0])
        m = router.recognize(rate)
        self.assertNotIn(m.orchestrator.id if m else None,
                         ("synthesized_geometry", "ratio_proportion"))

    def test_days_between_ignores_year_substring(self):
        # "lightYEAR" must NOT trip the year keyword (substring false-positive), so an
        # unsupported-unit travel-cost query falls to the teacher instead of being mis-claimed
        import reasoning.store.orchestrators.days_between_dates as d
        self.assertFalse(d.precondition("travel cost between (1,2) and (3,4) at $1 per lightyear"))
        self.assertTrue(d.precondition("how many days between 2024-01-01 and 2024-02-01"))
        m = router.recognize("What is the travel cost between (40.7128, -74.0060) and "
                             "(51.5074, -0.1278) at $0.10 per lightyear?")
        self.assertNotEqual(m.orchestrator.id if m else None, "days_between_dates")

    def test_algebra_declines_malformed_solve_variable(self):
        # "for z-1" is not a valid variable — must decline, not silently solve for z
        from reasoning.bind import bind_algebra
        self.assertFalse(bind_algebra("Solve x^2 + 2z = 0 for z-1")[0])
        self.assertTrue(bind_algebra("Solve x^2 + 2z = 0 for z")[0])


if __name__ == "__main__":
    unittest.main()
