def calculate_node_value(node: dict, state: dict):
    match node["type"]:
        case "Monster":
            return combat_value(node, state)
        case "Elite":
            return elite_value(node, state)
        case "RestSite":
            return rest_value(node, state)
        case "Shop":
            return shop_value(node, state)
        case "Treasure":
            return 0
        case "Boss":
            return 0
        case "Ancient":
            return 0
        case _:
            raise ValueError(f"Unknown room type: {node["type"]}")


def combat_value(node, start):
    return


def elite_value(node, state):
    return


def rest_value(node, state):
    return


def shop_value(node, state):
    return