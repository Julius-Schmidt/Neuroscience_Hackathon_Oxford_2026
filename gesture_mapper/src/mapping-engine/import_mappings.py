"""Turn the original camera mappings (Mappings page / data/mappings.json) into Blocks rules.

Each binding  (app context, hand, pose or pinch) -> command  becomes:

    when  camera  <hand>.<gesture>  starts / is held        (held commands: is held)
    only if  app in front is  <context>
    run / hold built-in command  <command>

Imported rules are tagged {"imported": "camera-mappings"}; importing again replaces them, so
it's safe to re-run after editing the Mappings page. Bindings for "either" hand become one
rule per hand. The camera pointer, pinch clicking on the pointer and the Minecraft joysticks
stay built into the camera control loop (Home > Start control) and aren't rules.
"""

TAG = "camera-mappings"


def rules_from_tables(tables: dict) -> list[dict]:
    import rules as R
    out = []
    for ctx, table in tables.items():
        for (hand, gesture), cmd in table.items():
            for h in (("left", "right") if hand == "either" else (hand,)):
                held = getattr(cmd, "mode", "tap") == "held"
                rule = R.new_rule("camera", f"{h}.{gesture}", "hold" if held else "press")
                rule["imported"] = TAG
                rule["title"] = f"{ctx}: {h} {gesture.replace('_', ' ')} -> {cmd.name}"
                rule["conditions"] = [{"type": "context", "value": ctx}]
                rule["actions"] = [{"type": "command_hold" if held else "command", "name": cmd.name}]
                out.append(rule)
    return out


def merge(data: dict, imported: list[dict]) -> int:
    """Replace previously imported camera rules in `data` with `imported`. Returns how many."""
    keep = [r for r in data.get("rules", []) if r.get("imported") != TAG]
    data["rules"] = keep + imported
    return len(imported)


def import_current(data: dict) -> int:
    from mapper import tables
    return merge(data, rules_from_tables(tables()))
