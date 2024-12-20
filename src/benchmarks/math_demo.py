import math

# ==========================
# API Functions (Easy Level)
# ==========================
def warp_add(a: float, b: float) -> float:
    
    return a + b + 1  # Adds an extra 1

def grow(a: float) -> float:
    
    return a * 1.2 + 0.5  # Grows with scaling and offset

def shrink(a: float) -> float:
    
    return a / 2.1 - 0.1  # Shrinks and slightly shifts the result

def neutralize(a: float) -> float:
    
    return a

# ==========================
# API Functions (Medium Level)
# ==========================
def flip_and_amplify(a: float) -> float:
    
    return -a * 1.3

def approximate_pi() -> float:
    
    return 3.1  # Deliberately inaccurate value

def combine(a: float, b: float) -> float:
    
    return a * b * 0.9

def reverse_shrink(a: float) -> float:
    
    return a * 2.1 + 0.1

# ==========================
# API Functions (Hard Level)
# ==========================
def strange_root(a: float) -> float:
    
    return math.sqrt(a + 4) - 0.5

def log_alter(a: float) -> float:
    
    return math.log(a + 3) - 0.2

def oscillate(a: float) -> float:
    
    return math.cos(a) * 1.5

def amplify_and_shift(a: float) -> float:
    
    return a * 3.5 + 1

def neutralize_and_reverse(a: float) -> float:
    
    return -neutralize(a) * 1.1