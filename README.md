# BERT-Based Token Importance and Mathematical Token Detection

## Overview

This project analyzes a prompt using the **BERT Base Uncased** model and produces three outputs for every token:

1. **Token Text**
2. **Mathematical Label**

   * `1` = token identified as mathematical
   * `0` = token identified as non-mathematical
3. **Attention-Based Importance Score**

   * Higher values indicate that more tokens in the sequence attend to that token.

The purpose of this script is to identify mathematically significant tokens while measuring how much attention they receive from the rest of the prompt.

---

# Model

The project uses:

```python
bert-base-uncased
```

from Hugging Face Transformers.

BERT (Bidirectional Encoder Representations from Transformers) is a transformer encoder model trained on large text corpora using:

* Masked Language Modeling (MLM)
* Next Sentence Prediction (NSP)

This implementation loads:

```python
BertTokenizer
BertModel
```

and extracts attention weights from the final transformer layer.

---

# Imported Libraries

## re

```python
import re
```

Python's built-in regular expression engine.

Used for:

* Compiling the mathematical token detector
* Matching tokens against mathematical patterns

Example:

```python
MATH_REGEX.match(token)
```

---

## transformers

```python
from transformers import BertTokenizer, BertModel
```

### BertTokenizer

Converts raw text into BERT tokens.

Example:

```python
TOKENIZER(prompt)
```

Output:

```text
"Calculus is fun"
```

becomes

```text
[CLS]
calculus
is
fun
[SEP]
```

and is converted into token IDs.

---

### BertModel

Loads the pretrained BERT encoder.

Example:

```python
MODEL(**inputs)
```

Returns:

* Hidden states
* Attention matrices
* Other transformer outputs

This implementation enables:

```python
output_attentions=True
```

to retrieve self-attention weights.

---

# Model Initialization

The tokenizer and model are loaded once:

```python
TOKENIZER = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

MODEL = BertModel.from_pretrained(
    "bert-base-uncased",
    attn_implementation="eager"
)
```

Advantages:

* Faster execution
* No repeated model downloads
* Reduced memory overhead

---

# Mathematical Token Detection

## MATH_REGEX

A large compiled regular expression is used to detect mathematical tokens.

Categories include:

### Numbers

```text
42
3.14
-8
1e5
```

### Fractions

```text
3/4
5/8
```

### Operators

```text
+
-
*
/
^
=
<
>
```

and symbols such as:

```text
≤ ≥ ≠ ± × ÷
```

### Delimiters

```text
(
)
[
]
{
}
```

### Greek Symbols

```text
π
θ
λ
Ω
```

### Set Theory Symbols

```text
∪
∩
⊆
∅
```

### Logic Symbols

```text
∀
∃
⇒
⇔
∈
```

### Mathematical Functions

```text
sin
cos
tan
log
ln
sqrt
```

### Calculus Terms

```text
derivative
integral
gradient
limit
```

### Linear Algebra Terms

```text
matrix
vector
tensor
eigenvalue
rank
```

### Statistics Terms

```text
mean
variance
distribution
```

### Mathematical Variables

```text
x
y
z
i
j
v
n
```

The regex is compiled only once:

```python
re.compile(...)
```

for performance.

---

# Functions

---

## get_outputs(prompt)

### Purpose

Runs BERT and returns the full model output.

### Input

```python
prompt: str
```

### Process

1. Tokenize prompt
2. Run BERT
3. Enable attention extraction

```python
outputs = MODEL(
    **inputs,
    output_attentions=True
)
```

### Returns

```python
BaseModelOutputWithPoolingAndCrossAttentions
```

containing:

* hidden states
* pooled output
* attention tensors

---

## attention_weights(prompt)

### Purpose

Extracts a token-to-token attention matrix.

### Process

#### Step 1

Retrieve all attention layers:

```python
attentions = outputs.attentions
```

Shape:

```text
[num_layers,
 batch_size,
 num_heads,
 seq_len,
 seq_len]
```

For BERT Base:

```text
12 layers
12 heads
```

---

#### Step 2

Select final layer:

```python
last_layer = attentions[-1]
```

Shape:

```text
[1, 12, seq_len, seq_len]
```

---

#### Step 3

Average across heads:

```python
attention_matrix = last_layer.mean(dim=1)
```

Result:

```text
[seq_len, seq_len]
```

---

#### Step 4

Remove special tokens:

```python
[CLS]
[SEP]
```

using:

```python
attention_matrix[1:-1]
```

---

### Returns

```python
list[list[float]]
```

A square attention matrix.

---

## label_prompt(tokens)

### Purpose

Assign mathematical labels.

### Logic

Each token is checked against:

```python
MATH_REGEX
```

If matched:

```python
label = 1
```

otherwise:

```python
label = 0
```

---

### Example

Input:

```text
["solve", "x", "+", "5"]
```

Output:

```text
tokens = ["solve","x","+","5"]
labels = [0,1,1,1]
```

---

## math_separate(prompt)

### Purpose

Tokenizes the prompt using the same tokenizer used by BERT.

### Process

```python
TOKENIZER(prompt)
```

Convert IDs back into readable tokens:

```python
convert_ids_to_tokens(...)
```

Remove:

```text
[CLS]
[SEP]
```

Then call:

```python
label_prompt(tokens)
```

---

### Returns

```python
tokens
labels
```

---

## token_importance(attention_matrix)

### Purpose

Compute an attention-based importance score for each token.

### Formula

For token j:

```text
Importance(j)
=
Σ Attention(i → j)
```

for all tokens:

```text
i ≠ j
```

In other words:

Each token receives attention from every other token.

The total received attention becomes its importance score.

---

### Algorithm

```python
for j in range(n):
    for i in range(n):
        if i != j:
            importance[j] += attention_matrix[i][j]
```

---

### Interpretation

Large score:

```text
Many tokens attend to this token.
```

Small score:

```text
Few tokens attend to this token.
```

---

## analyze_prompt(prompt)

### Purpose

Main analysis pipeline.

### Process

```python
tokens, labels = math_separate(prompt)

attention_matrix = attention_weights(prompt)

scores = token_importance(
    attention_matrix
)
```

Verifies:

```python
len(tokens)
==
len(labels)
==
len(scores)
```

---

### Returns

```python
tokens
labels
scores
```

---

# Scoring System

The score for a token is:

```text
Score(token_j)
=
Σ Attention(i → j)
```

where:

```text
i ≠ j
```

Interpretation:

| Score  | Meaning                                             |
| ------ | --------------------------------------------------- |
| High   | Many other tokens depend on or reference this token |
| Medium | Some contextual relevance                           |
| Low    | Little influence on surrounding tokens              |

The score is not a probability.

The score is not normalized.

The score is simply the accumulated attention received from other tokens in the final BERT layer after averaging all attention heads.

---

# Example Prompt

```text
Gemini, please help me solve the following math equation:
y=x^2+5x-50.

Explain why the sky appears blue during the day.

Write a short summary of the Industrial Revolution.

What are the main causes of climate change?

Tell me a story about a robot exploring Mars.

What is the result of (7 + 4) * (9 - 2)?

Solve the system:
x + y = 10,
2x - y = 5

Find the roots of x^3 - 6x^2 + 11x - 6.

Find the area of a circle with radius 8.

Calculate the circumference of a circle using
C = 2πr.

Find the derivative of sin(x).

Evaluate the integral ∫ x^2 dx.

Compute the eigenvalues of
A = [[2,1],[1,2]].

Find P(A|B).

Determine whether the graph is bipartite.

∀x∈R, x^2≥0

Solve √(x+1)
```

---

# Example Output

```text
Token               Score                    Label

solve               0.824100                 0
x                   1.621900                 1
+                   1.882700                 1
5                   1.203400                 1
```

Where:

* Token = BERT token
* Score = attention importance
* Label = mathematical classification

---

# Current Pipeline

```text
Prompt
   │
   ▼
BERT Tokenizer
   │
   ▼
Tokens
   │
   ├──────────────► Regex Math Detector
   │                      │
   │                      ▼
   │                   Labels
   │
   ▼
BERT Encoder
   │
   ▼
Final Attention Layer
   │
   ▼
Average Heads
   │
   ▼
Attention Matrix
   │
   ▼
Token Importance Scores
   │
   ▼
(tokens, labels, scores)
```

---

# Output Format

```python
tokens, labels, scores = analyze_prompt(prompt)
```

Returns:

```python
(
    tokens,
    labels,
    scores
)
```

where:

```python
tokens: List[str]
labels: List[int]
scores: List[float]
```
