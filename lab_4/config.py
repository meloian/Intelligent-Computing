# universe ranges
LUX_RANGE   = (0, 1000)           # lx
HOUR_RANGE  = (0, 24)             # h (0 = midnight)
BRIGHT_RANGE = (0, 100)           # % output

# membership-function control points
MF_POINTS: dict[str, dict[str, list[float]]] = {
    "lux": {
        "Dark":   [0,   0,  50],
        "Dim":    [30, 150, 300],
        "Bright": [200, 600, 1000]
    },
    "hour": {
        "Night":   [22, 22, 24, 6], 
        "Morning": [4,   8,  12],
        "Day":     [10, 14, 18],
        "Evening": [16, 20, 24]
    },
    "presence": {
        "No":  [0.0, 0.0, 1.0],
        "Yes": [0.0, 1.0, 1.0]
    },
    "eco": {
        "Standard": [0.0, 0.0, 1.0],
        "Eco":      [0.0, 1.0, 1.0]
    },
    # output: brightness
    "brightness": {
        "Off":    [0, 0, 0, 5],
        "Low":    [0, 5, 30, 50],
        "Medium": [30, 50, 60, 80],
        "High":   [60, 80, 100, 100]
    }
}

COLOR_TERMS = ["Warm", "Cold"]