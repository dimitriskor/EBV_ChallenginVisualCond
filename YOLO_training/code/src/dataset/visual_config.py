night = [
    'zurich_city_03_a',
    'zurich_city_09_a',
    'zurich_city_09_b',
    'zurich_city_09_c',
    'zurich_city_09_d',
    'zurich_city_09_e',
    'zurich_city_10_a',     # night
]

dawn_street_lights_on = [
    'zurich_city_01_a',
    'zurich_city_01_b',
    'zurich_city_01_c',
    'zurich_city_01_d',
    'zurich_city_01_e',
    'zurich_city_01_f',
    'zurich_city_02_c',
    'zurich_city_02_d',
    'zurich_city_02_e',     # maybe night
]

car_lights_on = [
    'zurich_city_00_b',
    'zurich_city_14_a',
    'zurich_city_14_b',
]

clear = [
    'interlaken_00_a',
    'interlaken_00_c',
    'interlaken_00_d',
    'interlaken_00_e',
    'interlaken_00_f',
    'interlaken_00_g',
    'interlaken_01_a',
    'thun_00_a',
    'zurich_city_05_a',
    'zurich_city_05_b',
    'zurich_city_06_a',
    'zurich_city_07_a',
    'zurich_city_11_a',
    'zurich_city_11_b',
    'zurich_city_11_c',
    'zurich_city_13_a',
    'zurich_city_13_b',

    # moved from car linghts on 
    'zurich_city_04_a',     # maybe clear
    'zurich_city_04_b',     # maybe clear
    'zurich_city_04_c',     # maybe clear
    'zurich_city_04_d',     # maybe clear
    'zurich_city_04_e',     # maybe clear
    'zurich_city_04_f',     # maybe clear

]

existing_glare = [
    'interlaken_01_a',  # car lights off # exists in clear as well
]

hold_out = [
    'interlaken_00_b',      # clear
    'thun_01_a',            # clear
    'thun_01_b',            # clear
    'zurich_city_08_a',     # clear
    'zurich_city_00_a',     # car_light_on
    'zurich_city_02_a',     # street_light_on     
    'zurich_city_02_b',     # street_light_on
    'zurich_city_14_c',     # street_light_on
    'zurich_city_10_b',     # night
    'zurich_city_12_a',     # night
    'zurich_city_15_a',     # maybe clear
]

# everything on car_lights_on, dawn_street_lights_on, night has exposure time ~ 15ms
# galre could use this information