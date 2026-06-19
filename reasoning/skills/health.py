"""Health / body-metric skills — BMI, heart rate, pace, BMR, ideal weight."""

from __future__ import annotations

from .base import register


@register("health.bmi", "body mass index = weight(kg) / height(m)^2",
          [((70, 1.75), 22.857143), ((90, 2), 22.5)])
def bmi(weight_kg: float, height_m: float) -> float:
    return weight_kg / height_m ** 2


@register("health.max_heart_rate", "maximum heart rate (bpm) = 220 - age",
          [((25,), 195), ((40,), 180)])
def max_heart_rate(age: float) -> float:
    return 220 - age


@register("health.target_heart_rate", "target heart rate = (220-age) * intensity%/100",
          [((40, 70), 126.0)])
def target_heart_rate(age: float, intensity_pct: float) -> float:
    return (220 - age) * intensity_pct / 100


@register("health.pace", "pace = minutes / distance (minutes per unit distance)",
          [((40, 5), 8.0), ((50, 10), 5.0)])
def pace(minutes: float, distance: float) -> float:
    return minutes / distance


@register("health.bmr_male", "Mifflin-St Jeor BMR for men: 10w + 6.25h - 5a + 5 (kcal/day)",
          [((80, 180, 30), 1780.0)])
def bmr_male(weight_kg: float, height_cm: float, age: float) -> float:
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5


@register("health.bmr_female", "Mifflin-St Jeor BMR for women: 10w + 6.25h - 5a - 161 (kcal/day)",
          [((60, 165, 25), 1345.25)])
def bmr_female(weight_kg: float, height_cm: float, age: float) -> float:
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


@register("health.ideal_body_weight", "Devine ideal body weight (kg) from height in cm",
          [((180,), 74.992126)])
def ideal_body_weight(height_cm: float) -> float:
    inches = height_cm / 2.54
    return 50 + 2.3 * (inches - 60)
