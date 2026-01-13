import string
import os


def get_reward(code: str):
    """
    Evaluate a guessed lyric against the reference lyric.
    
    Args:
        code: The guessed lyric content as a string
        
    Returns:
        Tuple of (reward, error_msg, details)
        - reward: Jaccard similarity between guess and reference (0 to 1)
        - error_msg: "" if successful, error description otherwise
        - details: "" (reserved for future use)
    """
    try:
        input_text = code.strip()
        
        reward = _compute_jaccard_similarity(input_text)
        return reward, "", ""
        
    except Exception as e:
        return 0.0, str(e), ""


def _compute_jaccard_similarity(input_text: str) -> float:
    """
    Compute the Jaccard similarity between the input text and the reference lyric.
    Both strings are lowercased, punctuation and stop words are removed, and split into lists of words.
    
    Jaccard similarity = |A ∩ B| / |A ∪ B|
    """
    reference = "hey Jude, don't make it bad"
    
    # Remove punctuation, lowercase, and split into words
    translator = str.maketrans('', '', string.punctuation)
    input_clean = input_text.translate(translator).lower()
    reference_clean = reference.translate(translator).lower()
    
    # Split and remove stop words
    input_words = set(w for w in input_clean.split())
    reference_words = set(w for w in reference_clean.split())
    
    # Handle edge case where both sets are empty
    if not input_words and not reference_words:
        return 0.0
    
    # Compute Jaccard similarity
    intersection = input_words & reference_words
    union = input_words | reference_words
    
    jaccard_similarity = len(intersection) / len(union)
    
    return jaccard_similarity


if __name__ == "__main__":
    # Test with initial_guess.txt
    with open(os.path.join(os.path.dirname(__file__), "initial_guess.txt"), 'r') as f:
        initial_guess = f.read()
    reward, error_msg, details = get_reward(initial_guess)
    print(f"Reward: {reward:.4f}, Error: {error_msg}")
