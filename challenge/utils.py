import logging
from datetime import datetime


def get_min_diff(data):
    leaving_in_advance_tolerance = 3600.0 # seconds, an hour
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    if (fecha_i - fecha_o).total_seconds() > leaving_in_advance_tolerance:
        message = ("Found flight leaving way before departure time: "
                   "flew at {}, intended time {}".format(fecha_o, fecha_i))
        logging.warning(message)
    minutes_difference = ((fecha_o - fecha_i).total_seconds()) / 60.0
    return minutes_difference
