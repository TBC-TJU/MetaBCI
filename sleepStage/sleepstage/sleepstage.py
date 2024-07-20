"""AASM Sleep Manual Label"""

# Label values
W = 0       # Stage AWAKE
N1 = 1      # Stage N1
N2 = 2      # Stage N2
N3 = 3      # Stage N3
REM = 4     # Stage REM
MOVE = 5    # Movement
UNK = 6     # Unknown

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNK": UNK,
}

class_dict = {
    W: "W",
    N1: "N1",
    N2: "N2",
    N3: "N3",
    REM: "REM",
    MOVE: "MOVE",
    UNK: "UNK",
}
