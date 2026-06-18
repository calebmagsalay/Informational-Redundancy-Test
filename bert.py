import re
from transformers import BertTokenizer, BertModel

# ============================================================
# Load model/tokenizer ONCE
# ============================================================

TOKENIZER = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

MODEL = BertModel.from_pretrained(
    "bert-base-uncased",
    attn_implementation="eager"
)

# ============================================================
# Math regex (compiled once)
# ============================================================

MATH_REGEX = re.compile(
    r"""
    ^
    (?:
        # ----------------------------------------------------
        # Special Tokens
        # ----------------------------------------------------
        \[UNK\] |

        # ----------------------------------------------------
        # Numbers
        # ----------------------------------------------------
        [+\-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)? |

        # Fractions
        [+\-]?\d+\s*/\s*\d+ |

        # ----------------------------------------------------
        # Arithmetic / Relational Operators
        # ----------------------------------------------------
        [+\-*/^=<>%!] |
        [≤≥≠±×÷∓≈∝∼≡≅] |

        # ----------------------------------------------------
        # Grouping / Delimiters
        # ----------------------------------------------------
        [()\[\]{}|] |

        # ----------------------------------------------------
        # Greek Letters / Mathematical Symbols
        # ----------------------------------------------------
        [πΠαβγδεζηθιλμνξρστυφχψωΩ∑∏∫∬∂∇∞√] |

        # ----------------------------------------------------
        # Set Theory Symbols
        # ----------------------------------------------------
        [∪∩⊆⊇∅] |

        # ----------------------------------------------------
        # Logic Symbols
        # ----------------------------------------------------
        [¬∧∨⇒⇔→←↔⇐∀∃∈∉⊂⊃⊕⊗] |

        # ----------------------------------------------------
        # Common Math Functions
        # ----------------------------------------------------
        \b(?:sin|cos|tan|cot|sec|csc
          |arcsin|arccos|arctan
          |sinh|cosh|tanh
          |log|ln|exp|sqrt|cbrt|root
          |abs|sgn|gcd|lcm|mod
          |lim|sup|inf|max|min)\b |

        # ----------------------------------------------------
        # Calculus
        # ----------------------------------------------------
        \b(?:derivative|integral|gradient|diverge|curl
          |laplacian|limit)\b |

        # ----------------------------------------------------
        # Differential Notation
        # ----------------------------------------------------
        \b(?:d|dx|dy|dz|dt|du|dv)\b |

        # ----------------------------------------------------
        # Linear Algebra
        # ----------------------------------------------------
        \b(?:matrix|vector|scalar|tensor
          |eigenvalue|eigenvector
          |determinant|rank
          |nullspace|columnspace|rowspace)\b |

        # ----------------------------------------------------
        # Statistics / Probability
        # ----------------------------------------------------
        \b(?:mean|median|variance
          |probability|distribution)\b |

        # ----------------------------------------------------
        # General Mathematical Terms
        # ----------------------------------------------------
        \b(?:polynomial
          |sigma|delta|theta|epsilon
          |lambda|omega
          |infinity
          |equation|formula
          |coefficient|exponent)\b |

        # ----------------------------------------------------
        # Common Mathematical Variables
        # (avoid labeling every English letter)
        # ----------------------------------------------------
        [xyzijvnrctpb]
    )
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ============================================================
# Embeddings + Attention
# ============================================================

def get_outputs(prompt):
    """
    Returns the full BERT output object.
    """
    inputs = TOKENIZER(
        prompt,
        return_tensors="pt"
    )

    outputs = MODEL(
        **inputs,
        output_attentions=True
    )

    return outputs


def attention_weights(prompt):
    outputs = get_outputs(prompt)

    attentions = outputs.attentions

    if not attentions:
        raise RuntimeError(
            "No attention tensors returned. "
            "Load BertModel with "
            "attn_implementation='eager'."
        )

    last_layer = attentions[-1]

    attention_matrix = last_layer.mean(dim=1)

    attention_matrix = attention_matrix[0]

    attention_matrix = attention_matrix.tolist()

    attention_matrix = attention_matrix[1:-1]

    attention_matrix = [
        row[1:-1]
        for row in attention_matrix
    ]

    return attention_matrix


# ============================================================
# Tokenization + Math Labeling
# ============================================================

def label_prompt(tokens):
    labels = [0] * len(tokens)

    for i, token in enumerate(tokens):
        clean_token = token.removeprefix("##")

        if MATH_REGEX.match(clean_token):
            labels[i] = 1

    return tokens, labels


def math_separate(prompt):
    """
    Tokenizes using the SAME tokenizer used by BERT.

    Removes:
        [CLS]
        [SEP]

    Returns:
        tokens
        labels
    """

    inputs = TOKENIZER(
        prompt,
        return_tensors="pt"
    )

    tokens = TOKENIZER.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    tokens = tokens[1:-1]

    return label_prompt(tokens)


# ============================================================
# Token Importance
# ============================================================

def token_importance(attention_matrix):
    """
    Computes attention received by each token.

    Ignores self-attention terms.
    """

    n = len(attention_matrix)

    importance = [0.0] * n

    for j in range(n):
        for i in range(n):
            if i != j:
                importance[j] += attention_matrix[i][j]

    return importance


# ============================================================
# Convenience function
# ============================================================

def analyze_prompt(prompt):
    """
    Returns:
        tokens
        labels
        scores
    """

    tokens, labels = math_separate(prompt)

    attention_matrix = attention_weights(prompt)

    scores = token_importance(attention_matrix)

    if not (
        len(tokens)
        == len(labels)
        == len(scores)
    ):
        raise ValueError(
            f"Length mismatch: "
            f"{len(tokens)=}, "
            f"{len(labels)=}, "
            f"{len(scores)=}"
        )

    return tokens, labels, scores


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    prompt = (
        "Gemini, please help me solve the following "
        "math equation: y=x^2+5x-50. "
        "Explain why the sky appears blue during the day. "
        "Write a short summary of the Industrial Revolution. "
        "What are the main causes of climate change? "
        "Tell me a story about a robot exploring Mars. "
        "What is the result of (7 + 4) * (9 - 2)? "
        "Solve the system: x + y = 10, 2x - y = 5 "
        "Find the roots of x^3 - 6x^2 + 11x - 6. "
        "Find the area of a circle with radius 8. "
        "Calculate the circumference of a circle using C = 2πr. "
        "Find the derivative of sin(x). "
        "Evaluate the integral ∫ x^2 dx. "
        "Compute the eigenvalues of A = [[2,1],[1,2]]. "
        "Find P(A|B). "
        "Determine whether the graph is bipartite. "
        "∀x∈R, x^2≥0 "
        "Solve √(x+1)"
    )

    tokens, labels, scores = analyze_prompt(prompt)

    print(
        f"{'Token':<20}"
        f"{'Score':<25}"
        f"{'Label'}"
    )

    for token, score, label in zip(
        tokens,
        scores,
        labels
    ):
        print(
            f"{token:<20}"
            f"{score:<25.6f}"
            f"{label}"
        )