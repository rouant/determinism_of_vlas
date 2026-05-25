import random

# items
sources = [
    [ # cylinders
    "pepsi_can", "coke_can", "sprite_can", "7up_can", "fanta_can", 
    "redbull_can", "blue_plastic_bottle" 
    ],
    [ # cubes
        "green_cube_3cm", "yellow_cube_3cm"
    ],
    [ # rectangular
        "sponge"
    ],
    [ # spherical
        "apple", "orange"
    ],
    [ # elongated
        "bridge_carrot_generated_modified", "bridge_spoon_generated_modified"
        #, "eggplant"
    ]
]
source_rot = [0.7071067811865476, 0.7071067811865476, 0.0, 0.0]
sources_map = {
    "pepsi_can": "pepsi can", 
    "coke_can": "coke can", 
    "sprite_can" : "sprite can", 
    "7up_can": "7up can", 
    "fanta_can": "fanta can", 
    "redbull_can": "redbull can", 
    "blue_plastic_bottle": "blue plastic bottle", 
    "apple": "apple", 
    "orange": "orange", 
    "sponge": "sponge",
    "bridge_carrot_generated_modified": "carrot", 
    "bridge_spoon_generated_modified": "spoon",
    "green_cube_3cm": "green cube", 
    "yellow_cube_3cm": "yellow cube", 
    "eggplant": "eggplant"
}

targets = [
    [ # round
        "bridge_plate_objaverse", "bridge_plate_objaverse_larger"
    ],
    [ # square
        "table_cloth_generated", "table_cloth_generated_shorter"
    ] 
]
target_rot = [0.5, 0.0, 0.0, 0.0]
targets_map = {
    "bridge_plate_objaverse": "plate", 
    "bridge_plate_objaverse_larger": "plate", 
    "table_cloth_generated": "cloth", 
    "table_cloth_generated_shorter": "cloth"
}


def getOptions(src_shape, tgt_shape, discr, x, y):
    src = random.sample(sources[src_shape], k=1)
    target = random.sample(targets[tgt_shape], k=1)
    remaining = [el for sub in sources for el in sub]
    remaining.remove(src[0])
    items = random.sample(remaining, k=len(x)-2)
    ids = src + target

    options = {
      "model_ids": ids, 
      "obj_init_options": {
            src[0]: {
              "init_xy": [x[0], y[0]], 
              "init_rot_quat": source_rot
            }, 
            target[0]: {
              "init_xy": [x[1], y[1]], 
              "init_rot_quat": target_rot
            }
        },
        "source_obj_id": 0, 
        "target_obj_id": 1
    }

    for idx, presence in enumerate(discr):
        if presence:
            ids.append(items[idx])
            options["obj_init_options"][items[idx]] = {
              "init_xy": [x[idx+2], y[idx+2]], 
              "init_rot_quat": source_rot
            }
            
    options["model_ids"] = ids
    
    return options, sources_map[items[0]], targets_map[target[0]]