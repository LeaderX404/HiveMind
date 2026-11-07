"""Model catalog for the Download tab (issue #40, wave #38).

Category 1 is downloadable today: five small instruct models,
each with its Hugging Face repo id and an approximate fp16
size - an ESTIMATE used only for the pre-download prediction;
the real size is computed from the downloaded files.

Categories 2-5 are placeholders: no buttons, no invented
models, no promises - "Coming after testing" is the only
text they get.
"""

CATEGORY_1 = [
    {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "name": "Qwen2.5-0.5B",
        "size_gb": 1.0,
    },
    {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "name": "Qwen2.5-1.5B",
        "size_gb": 3.1,
    },
    {
        "repo_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "name": "SmolLM2-1.7B",
        "size_gb": 3.4,
    },
    {
        "repo_id": "google/gemma-2-2b-it",
        "name": "Gemma-2-2B",
        "size_gb": 5.4,
    },
    {
        "repo_id": "microsoft/Phi-3.5-mini-instruct",
        "name": "Phi-3.5-mini",
        "size_gb": 7.6,
    },
]

CATEGORY_2_5 = [
    {"name": "Category 2", "status": "Coming after testing"},
    {"name": "Category 3", "status": "Coming after testing"},
    {"name": "Category 4", "status": "Coming after testing"},
    {"name": "Category 5", "status": "Coming after testing"},
]
