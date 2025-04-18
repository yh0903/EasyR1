import re
from typing import Dict

def extract_reasoning_content(predict_str: str) -> str:
    """Extract content between <reasoning> tags."""
    reasoning_pattern = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
    match = reasoning_pattern.search(predict_str)
    return match.group(1).strip() if match else ""

def extract_answer_content(predict_str: str) -> str:
    """Extract content between <answer> tags."""
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match = answer_pattern.search(predict_str)
    return match.group(1).strip() if match else ""

def format_reward(predict_str: str) -> float:
    """Check if the prediction follows the required format with both reasoning and answer tags."""
    pattern = re.compile(r"<reasoning>.*</reasoning>.*<answer>.*</answer>.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0

def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """Compare the extracted answer with the ground truth."""
    answer = extract_answer_content(predict_str)
    # You may need to define how to compare sentiment analysis answers
    # This is a simple exact match, but you might want sentiment-specific comparisons
    return 1.0 if answer.strip() == ground_truth.strip() else 0.0

def reasoning_quality_reward(predict_str: str) -> float:
    """Optional: Evaluate the quality of reasoning provided."""
    reasoning = extract_reasoning_content(predict_str)
    # Implement logic to evaluate reasoning quality
    # For example: check for minimum length, presence of key analysis elements, etc.
    min_length = 100  # Example: require at least 100 characters
    return 1.0 if len(reasoning) >= min_length else 0.0

def compute_score(predict_str: str, ground_truth: str, format_weight: float = 0.1, 
                  reasoning_weight: float = 0.3) -> Dict[str, float]:
    """Compute overall score with weights for format, accuracy, and reasoning quality."""
    predict_str = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str)  # handle formatting variations
    
    format_score = format_reward(predict_str)
    accuracy_score = accuracy_reward(predict_str, ground_truth)
    reasoning_score = reasoning_quality_reward(predict_str)
    
    # Calculate overall score with all components
    overall = (
        (1 - format_weight - reasoning_weight) * accuracy_score + 
        format_weight * format_score +
        reasoning_weight * reasoning_score
    )
    
    return {
        "overall": overall,
        "format": format_score,
        "accuracy": accuracy_score,
        "reasoning": reasoning_score,
    }