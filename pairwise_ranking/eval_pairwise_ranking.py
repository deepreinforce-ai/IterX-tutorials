import os
ordered_pairs = [
    (20, 20),
    (11, 99),
    (11, 98),
    (11, 97),
    (11, 96),
    (11, 95),
    (11, 94),
    (11, 93),
    (11, 92),
    (11, 91),
    (11, 90),
    (11, 89),
    (11, 88),
    (11, 87),
    (11, 86),
    (11, 85),
    (11, 84),
    (11, 83),
    (11, 82),
    (11, 81),
    (11, 80),
    (11, 79),
    (11, 78),
    (11, 77),
    (11, 76),
    (11, 75),
    (11, 74),
    (11, 73),
    (11, 72),
    (11, 71),
    (11, 70),
    (11, 69),
    (11, 68),
    (11, 67),
    (11, 66),
    (11, 65),
    (11, 64),
    (11, 63),
    (11, 62),
    (11, 61),
    (11, 60),
    (11, 59),
    (11, 58),
    (11, 57),
    (11, 56),
    (12, 39),
    (11, 55),
    (11, 54),
    (12, 38),
    (11, 53),
    (10, 99),
    (11, 52),
    (10, 98),
    (10, 97),
    (12, 37),
    (10, 96),
    (11, 51),
    (10, 95),
    (10, 94),
    (10, 93),
    (10, 92),
    (11, 50),
    (10, 91),
    (10, 90),
    (10, 89),
    (11, 49),
    (10, 88),
    (10, 87),
    (10, 86),
    (11, 48),
    (10, 85),
    (10, 84),
    (10, 83),
    (11, 47),
    (10, 82),
    (10, 81),
    (10, 80),
    (11, 46),
    (10, 79),
    (12, 34),
    (10, 78),
    (10, 77),
    (11, 45),
    (10, 76),
    (10, 75),
    (10, 74),
    (11, 44),
    (10, 73),
    (10, 72),
    (10, 71),
    (11, 43),
    (10, 70),
    (10, 69),
    (12, 32),
    (10, 68),
    (11, 42),
    (10, 67),
    (10, 66),
    (11, 41),
    (10, 65),
    (12, 31),
    (10, 64),
    (10, 63),
    (11, 40),
    (10, 62),
    (10, 61),
    (11, 39),
    (10, 60),
    (10, 59),
    (11, 38),
    (10, 58),
    (10, 57),
    (12, 29),
    (10, 56),
    (11, 37),
    (10, 55),
    (10, 54),
    (11, 36),
    (10, 53),
    (12, 28),
    (10, 52),
    (11, 35),
    (10, 51),
    (10, 50),
    (11, 34),
    (12, 27),
    (10, 49),
    (10, 48),
    (11, 33),
    (10, 47),
    (10, 46),
    (12, 26),
    (11, 32),
    (10, 45),
    (10, 44),
    (11, 31),
    (10, 43),
    (12, 25),
    (10, 42),
    (11, 30),
    (10, 41),
    (10, 40),
    (11, 29),
    (10, 39),
    (10, 38),
    (11, 28),
    (12, 23),
    (10, 37),
    (10, 36),
    (11, 27),
    (10, 35),
    (12, 22),
    (11, 26),
    (10, 34),
    (10, 33),
    (11, 25),
    (12, 21),
    (10, 32),
    (10, 31),
    (11, 24),
    (10, 30),
    (11, 23),
    (10, 29),
    (10, 28),
    (12, 19),
    (11, 22),
    (10, 27),
    (10, 26),
    (11, 21),
    (12, 18),
    (10, 25),
    (11, 20),
    (10, 24),
    (12, 17),
    (10, 23),
    (11, 19),
    (10, 22),
    (12, 16),
    (11, 18),
    (10, 21),
    (11, 17),
    (10, 20),
    (10, 19),
    (11, 16),
    (12, 14),
    (10, 18),
    (11, 15),
    (10, 17),
    (12, 13),
    (11, 14),
    (10, 16),
    (10, 15),
    (11, 13),
    (10, 14),
    (11, 12),
    (10, 13),
    (11, 11),
    (10, 12),
    (10, 11),
    (10, 10),
]

def get_reward(code: str):
    """
    Evaluate a scoring function based on how well it ranks pairs.
    
    The ideal ranking: pairs with harmonic mean closest to 20 should score highest.
    ordered_pairs is already sorted by HM distance from 20 (best first).
    
    Args:
        code: Python code string containing the score function
        
    Returns:
        Tuple of (reward, error_msg, details)
        - reward: fraction of concordant pairs (0 to 1)
        - error_msg: "" if successful, error description otherwise
        - details: "" (reserved for future use)
    """
    try:
        # Execute the code and extract the score function
        namespace = {}
        exec(code, namespace)
        score_func = namespace['score']
        
        # Compute scores for all pairs
        scores = [score_func(a, b) for a, b in ordered_pairs]
        
        # Count concordant pairs: for i < j in ordered_pairs (i is better),
        # the score of i should be >= score of j
        n = len(ordered_pairs)
        concordant = 0
        total = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                # Pair i should have higher or equal score than pair j
                if scores[i] >= scores[j]:
                    concordant += 1
        
        reward = concordant / total if total > 0 else 0.0
        return reward, "", ""
        
    except Exception as e:
        return 0.0, str(e), ""


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "initial_code.py")) as f:
        code = f.read()
    reward, error_msg, details = get_reward(code)
    print(f"Reward: {reward:.4f}, Error: {error_msg}")
