"""Optics — thin-lens imaging, magnification, lens power, Snell's refraction,
critical angle for total internal reflection, f-number, and angular resolution.
Coded, 0 calls."""
from reasoning.bind import bind_optics
from reasoning.bind._common import guard_residue
from reasoning.compiled import dispatch

META = {
    "id": "optics", "query_type": "optics",
    "description": "optics & light physics: thin-lens equation (image distance, magnification), "
                   "lens power in diopters, Snell's law refraction angle, critical angle for "
                   "total internal reflection, and f-number (focal ratio)",
    "example_queries": [
        "What is the image distance for a focal length of 10 cm and object distance of 30 cm?",
        "Calculate the magnification for an image distance of 15 and object distance of 30.",
        "What is the lens power in diopters for a focal length of 0.2 meters?",
        "Using Snell's law, what is the refraction angle for n1=1, angle=30 degrees, n2=1.5?",
        "What is the critical angle for total internal reflection when n1=1.5 and n2=1?",
        "What is the f-number of a lens with focal length 50 mm and aperture diameter 12.5 mm?",
    ],
    "keywords": ["optics", "lens", "focal length", "magnification", "image distance",
                 "refractive index", "refraction", "Snell", "critical angle", "TIR",
                 "total internal reflection", "diopter", "lens power",
                 "f-number", "focal ratio", "aperture"],
    "model_free": True,
}


def precondition(query):
    return bind_optics(query)[0]


def run(query):
    return dispatch(guard_residue(query, bind_optics(query)),
                    parse_message="could not parse the optics query")