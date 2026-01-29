ALL_ATTRIBUTES = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
                  "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee",
                  "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
                  "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
                  "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

CONCEPTS = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry",
            "Bushy_Eyebrows", "Chubby", "Double_Chin", "Goatee", "Heavy_Makeup", "High_Cheekbones", "No_Beard", "Pale_Skin", "Receding_Hairline",
            "Sideburns", "Smiling", "Wavy_Hair", "Wearing_Earrings", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

# CAUSAL_CONCEPTS = ["5_o_Clock_Shadow", "Bald", "Bangs",
#                   "Bushy_Eyebrows", "Goatee",
#                   "Heavy_Makeup", "Mustache", "No_Beard",
#                   "Receding_Hairline", "Sideburns",
#                   "Wearing_Earrings", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie"]

# ATTRIBUTES = ["Big_Lips", "Big_Nose", "Black_Hair","Bushy_Eyebrows",
#               "Heavy_Makeup", "High_Cheekbones", "Male", "Narrow_Eyes",
#               "Oval_Face", "Pointy_Nose", "Smiling", "Straight_Hair",
#               "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick"]

# ATTRIBUTES = ["Big_Lips", "Black_Hair",
#               "Heavy_Makeup", "High_Cheekbones", "Male",
#               "Pointy_Nose", "Straight_Hair",
#               "Wearing_Earrings", "Wearing_Lipstick"]

ATTRIBUTES = ["Eyeglasses", "Male", "Smiling", "Wearing_Lipstick"]

CAUSAL_CONCEPTS = ["Male", "Young"]

COEFFICIENTS = {
    "base": 0.45,
    "Male": -0.3,
    "Young": 0.5,
    "Male_Young": 0.1
}

COEFFICIENTS_OOD = {
    "base": 0.45,
    "Male": -0.3,
    "Young": 0.2,
    "Male_Young": 0.1
}

RESIDUAL_CONCEPTS = []

SEED = 42

TASK = "Attractive"

SHORTCUT = "Young"
BALANCE = SHORTCUT
