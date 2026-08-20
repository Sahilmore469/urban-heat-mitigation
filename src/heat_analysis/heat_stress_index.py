"""
heat_stress_index.py
--------------------
Computes heat stress indices from meteorological and remote sensing data.

Implements:
  - Universal Thermal Climate Index (UTCI) approximation
  - Physiological Equivalent Temperature (PET) approximation  
  - Wet Bulb Globe Temperature (WBGT) - simplified outdoor
  - Heat Stress Classification (6 levels)
"""

import numpy as np


def wbgt_outdoor(air_temp: np.ndarray, humidity: np.ndarray,
                 lst: np.ndarray, wind_speed: np.ndarray) -> np.ndarray:
    """
    Simplified outdoor Wet Bulb Globe Temperature.

    WBGT ≈ 0.7 * Tw + 0.2 * Tg + 0.1 * Ta
    where:
        Tw = wet bulb temp (approx from Ta and RH)
        Tg = globe temp (approx from LST)
        Ta = air temperature
    """
    # Wet bulb temperature approximation (Stull 2011)
    tw = (air_temp * np.arctan(0.151977 * np.sqrt(humidity + 8.313659))
          + np.arctan(air_temp + humidity)
          - np.arctan(humidity - 1.676331)
          + 0.00391838 * humidity**1.5 * np.arctan(0.023101 * humidity)
          - 4.686035)

    # Globe temperature approximation (solar radiation proxy via LST)
    # Tg ≈ LST + wind cooling correction
    tg = lst + 0.25 * (1.0 / (wind_speed + 0.5))

    wbgt = 0.7 * tw + 0.2 * tg + 0.1 * air_temp
    return wbgt


def utci_approximation(air_temp: np.ndarray, humidity: np.ndarray,
                        lst: np.ndarray, wind_speed: np.ndarray) -> np.ndarray:
    """
    Simplified UTCI approximation using linear regression model.
    Based on Bröde et al. (2012) simplified form.
    
    Full polynomial UTCI requires mean radiant temperature (MRT),
    which we approximate from LST and SVF.
    """
    # Mean Radiant Temperature (simplified)
    mrt = 0.7 * lst + 0.3 * air_temp

    # Vapour pressure from RH and Ta
    e_s = 6.112 * np.exp(17.67 * air_temp / (air_temp + 243.5))
    e = (humidity / 100.0) * e_s

    # UTCI approximation
    utci = (air_temp
            + 0.607562052 * (mrt - air_temp)
            - 0.028217 * wind_speed
            + 0.000394 * e
            - 0.0001 * wind_speed * (mrt - air_temp))

    return utci


def classify_heat_stress(utci: np.ndarray) -> np.ndarray:
    """
    Classify UTCI into 6 heat stress categories.
    
    Returns integer array:
        0 = No Thermal Stress  (UTCI < 9°C)
        1 = Moderate Stress    (9–26°C) 
        2 = Strong Stress      (26–32°C)
        3 = Very Strong Stress (32–38°C)
        4 = Extreme Stress     (38–46°C)
        5 = Very Extreme Stress(> 46°C)
    """
    classes = np.zeros_like(utci, dtype=int)
    classes[utci >= 9] = 1
    classes[utci >= 26] = 2
    classes[utci >= 32] = 3
    classes[utci >= 38] = 4
    classes[utci >= 46] = 5
    return classes


HEAT_STRESS_LABELS = {
    0: "No Thermal Stress",
    1: "Moderate Stress",
    2: "Strong Stress",
    3: "Very Strong Stress",
    4: "Extreme Stress",
    5: "Very Extreme Stress",
}

HEAT_STRESS_COLORS = {
    0: "#2196F3",   # Blue
    1: "#4CAF50",   # Green
    2: "#FFEB3B",   # Yellow
    3: "#FF9800",   # Orange
    4: "#F44336",   # Red
    5: "#7B1FA2",   # Purple
}

