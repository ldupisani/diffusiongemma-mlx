"""bind layer — fluid mechanics. Gated strictly on fluid-specific context keywords
(fluid, density, hydrostatic, dynamic pressure, flow rate, Reynolds, viscosity, depth).
Each operation is gated on its specific numeric inputs via num_near for bidirectional extraction.
Declines on unexplained residue to avoid silent-wrong answers."""

from __future__ import annotations

from reasoning.bind._common import *  # noqa: F401,F403


def bind_fluid_mechanics(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    # Conservative gate: must have clear fluid-mechanics context.
    # Decline if the query is pure physics concepts without fluid-specific signal.
    has_fluid_context = any(x in q for x in [
        "fluid", "density", "hydrostatic", "dynamic pressure", "flow rate", "flow",
        "reynolds", "viscosity", "depth", "pressure in water", "underwater",
        "fluid flow", "streamline", "laminar", "turbulent", "continuity", "inlet", "outlet"
    ])
    if not has_fluid_context:
        return False, None, "code", 0

    # Decline physics queries about pressure = force/area (non-fluid). Require force AND area —
    # "area" alone is a legitimate flow-rate / continuity input that must NOT be declined.
    if "force" in q and "area" in q and "fluid" not in q and "density" not in q:
        return False, None, "code", 0

    def ok(skill, dims, vals, **extra):
        if any(v is None for v in vals):
            return False, None, "code", 0
        return True, {"skill": skill, "dims": dims, **dict(zip(dims, vals)), **extra}, "code", 0

    # Op 1: Hydrostatic pressure P = ρ·g·h. Gate strictly on density AND depth.
    if "hydrostatic" in q or ("pressure" in q and "depth" in q and "density" in q):
        rho = num_near(query, "density", "ρ", "kg/m³", "kg/m³")
        h = num_near(query, "depth", "height", "h =", "m")
        result = ok("fluid_mechanics.hydrostatic_pressure", ["density", "depth"],
                    [rho, h], round=True, places=6, suffix=" Pa")
        return guard_residue(query, result)

    # Op 2: Dynamic pressure q = 0.5·ρ·v². Gate strictly on density AND velocity.
    if "dynamic pressure" in q or ("pressure" in q and "velocity" in q and "density" in q and
                                   "depth" not in q):
        rho = num_near(query, "density", "ρ", "kg/m³")
        v = num_near(query, "velocity", "speed", "v =", "m/s")
        result = ok("fluid_mechanics.dynamic_pressure", ["density", "velocity"],
                    [rho, v], round=True, places=6, suffix=" Pa")
        return guard_residue(query, result)

    # Op 3: Volumetric flow rate Q = A·v. Gate on area AND velocity.
    if "flow rate" in q or ("flow" in q and "area" in q and "velocity" in q):
        A = num_near(query, "area", "cross", "cross-section", "cross section", "m²")
        v = num_near(query, "velocity", "speed", "v =", "m/s")
        result = ok("fluid_mechanics.volumetric_flow_rate", ["area", "velocity"],
                    [A, v], round=True, places=6, suffix=" m³/s")
        return guard_residue(query, result)

    # Op 4: Reynolds number Re = ρ·v·D/μ. Gate on all four inputs.
    if "reynolds" in q or "re =" in q:
        rho = num_near(query, "density", "ρ", "kg/m³")
        v = num_near(query, "velocity", "speed", "v =", "m/s")
        D = num_near(query, "diameter", "d =", "characteristic length", "m")
        mu = num_near(query, "viscosity", "dynamic viscosity", "μ", "pa·s", "pa*s")
        result = ok("fluid_mechanics.reynolds_number", ["density", "velocity", "diameter", "dynamic_viscosity"],
                    [rho, v, D, mu], round=True, places=6)
        return guard_residue(query, result)

    # Op 5: Continuity equation v₂ = A₁·v₁/A₂. Gate on inlet area, inlet velocity, and outlet area.
    if "continuity" in q or ("outlet" in q and "inlet" in q):
        A1 = num_near(query, "inlet area", "inlet", "a1", "m²")
        v1 = num_near(query, "inlet velocity", "inlet", "v1", "m/s")
        A2 = num_near(query, "outlet area", "outlet", "a2", "m²")
        result = ok("fluid_mechanics.continuity_equation", ["inlet_area", "inlet_velocity", "outlet_area"],
                    [A1, v1, A2], round=True, places=6, suffix=" m/s")
        return guard_residue(query, result)

    return False, None, "code", 0