def evidence(actions: list, candidates: list) -> dict:
    masses = {c["name"]: c["mass"] for c in candidates}
    feas: dict = {}
    motors = any((a.get("log") or {}).get("rule") == "legs_dead"
                 for a in actions)
    for act in actions:
        log = act.get("log") or {}
        nm = (act.get("action") or {}).get("try_lift")
        if nm not in masses:
            continue
        if log.get("ok"):
            feas[nm] = "feasible"
        elif log.get("rule") == "mu":
            feas[nm] = "infeasible"
        elif log.get("rule") == "rho" and motors:
            feas.setdefault(nm, "infeasible")
    ranked = sorted(masses, key=lambda n: -masses[n])
    ent = None
    for nm in ranked:
        st = feas.get(nm)
        if st == "feasible":
            ent = nm
            break
        if st != "infeasible":
            break
    return {"feas": feas, "motors_observed": motors, "entailed": ent}
