import numpy as np


w_g = 1
w_hp = 21.75
w_hp_low = 32.65
w_r = 150
w_cr = 63.5
w_s = 1
w_rm = 100
EST_CMBT_HP_LOSS = 7.5
EST_CMBT_GOLD = 15
EST_ELITE_HP_LOSS = 17.5
EST_ELITE_GOLD = 40


def get_best_path(state: dict):
    all_paths = get_all_paths(state['full_map'], state['choices'])
    all_path_values = list(map(lambda path: calculate_path_value(path, state['player']), all_paths))
    if all_paths[np.argmax(all_path_values)][0] not in state['choices']:
        raise ValueError(f"something's wrong with pathing\nselected path: {all_paths[np.argmax(all_path_values)]}\nchoices: {state['choices']}")

    return all_paths[np.argmax(all_path_values)]


def get_all_paths(full_map: dict, choices: dict):
    all_paths = []
    for choice in choices:
        node = get_nodes(full_map, choice["col"], choice["row"])
        if node:
            all_paths.extend(get_all_paths_rec(node, []))
    return all_paths

def get_nodes(root, col, row):
    if root["col"] == col and root["row"] == row:
        return root
    for child in root["children"]:
        result = get_nodes(child, col, row)
        if result:
            return result
    return None

def get_all_paths_rec(node: dict, current_path: list):
    if not current_path:
        current_path = []
    
    new_path = current_path + [{"col": node["col"], "row": node["row"], "type": node["type"]}]

    if len(node["children"]) == 0:
        return [new_path]
    
    return [path for child in node["children"] for path in get_all_paths_rec(child, new_path)]


def calculate_path_value(path: list, player_state: dict):
    return sum(calculate_node_value(node, player_state, path) for node in path)


def calculate_node_value(node: dict, player_state: dict, path: list):
    match node["type"]:
        case "Monster" | "Elite":
            return combat_value(node)
        case "RestSite":
            return rest_value(player_state)
        case "Shop":
            return shop_value(player_state, path)
        case "Treasure":
            return 0
        case "Boss":
            return 0
        case "Ancient":
            return 0
        case "Unknown":
            return float(np.mean([combat_value(node), shop_value(player_state, path), 47.5 + w_cr])) # the last item is the value of a treasure room
        case _:
            raise ValueError(f"Unknown room type in calculate_node_value: {node['type']}")


def combat_value(node):
    p_relic = 0
    p_card_reward = 1
    est_gold, est_hp_loss = EST_CMBT_GOLD, EST_CMBT_HP_LOSS

    if node['type'] == 'Elite':
        p_relic = 1
        est_gold, est_hp_loss = EST_ELITE_GOLD, EST_ELITE_HP_LOSS
    
    return w_g * est_gold + w_r * p_relic + w_cr * p_card_reward - w_hp * est_hp_loss


def rest_value(player_state):
    bonus_hp = 15 if any(relic["name"] == "Regal Pillow" for relic in player_state['relics']) else 0
    delta_hp = player_state['max_hp'] - (player_state['hp'] + 0.15 * player_state['max_hp'] + bonus_hp)
    return float(2 * w_cr + w_hp * delta_hp)


def shop_value(player_state, path):
    expected_gold = player_state['gold'] + get_expected_gold(path)
    scale_factor = 1 * expected_gold // 150
    return float(w_s * scale_factor * expected_gold + w_rm * strike_sum(player_state) * (player_state['gold'] >= 75))

def get_expected_gold(path):
    expected_gold = 0
    for node in path:
        match node['type']:
            case "Monster":
                expected_gold += EST_CMBT_GOLD
            case "Elite":
                expected_gold += EST_ELITE_GOLD
            case "RestSite" | "Boss" | "Ancient":
                expected_gold += 0
            case "Treasure":
                expected_gold += 47.5 # average gold reward
            case "Unknown":
                expected_gold += np.mean([EST_CMBT_GOLD, 0, 47.5]) # 0 is for shop
            case "Shop":
                break
            case _:
                raise ValueError(f"Unknown room type in get_expected_gold: {node['type']}")
    return expected_gold

def strike_sum(player_state):
    return sum(card['name'] == 'Strike' for card in player_state['deck'])