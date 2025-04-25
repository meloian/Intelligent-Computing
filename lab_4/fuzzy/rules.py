RULES: list[dict] = [
    
    # --------- Brightness -----------------------------------------------
    
    # кімната порожня → Off
    {"if": {"presence": "No"},                        "then": {"brightness": "Off"}},

    # lux × presence Yes
    {"if": {"lux": "Bright", "presence": "Yes"},      "then": {"brightness": "Low"}},
    {"if": {"lux": "Dim",    "presence": "Yes"},      "then": {"brightness": "Medium"}},
    {"if": {"lux": "Dark",   "presence": "Yes"},      "then": {"brightness": "High"}},

    # eco-режим з людьми → трохи глушимо
    {"if": {"eco": "Eco", "presence": "Yes"},         "then": {"brightness": "Low"}},

    # --------- Colour temperature ---------------------------------------
    
    # night: завжди тепле (атмосферне), незалежно від Eco
    {"if": {"hour": "Night"},                         "then": {"color_temp": "Warm"}},

    # morning
    {"if": {"hour": "Morning", "eco": "Standard"},    "then": {"color_temp": "Warm"}},
    {"if": {"hour": "Morning", "eco": "Eco"},         "then": {"color_temp": "Cold"}},

    # day – завжди холодне
    {"if": {"hour": "Day"},                           "then": {"color_temp": "Cold"}},

    # evening
    {"if": {"hour": "Evening", "eco": "Standard"},    "then": {"color_temp": "Warm"}},
    {"if": {"hour": "Evening", "eco": "Eco"},         "then": {"color_temp": "Cold"}},
]