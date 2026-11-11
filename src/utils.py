import json
from fractions import Fraction

def json_read(fname, **kwargs):
    with open(fname) as j:
        data = json.load(j, **kwargs)
    return data

def fraction_from_json(json_object):
    if 'Fraction' in json_object:
        return Fraction(*json_object['Fraction'])
    return json_object