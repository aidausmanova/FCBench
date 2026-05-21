# Global parameters
EXAMPLES_VERACITY_PATH = "./prompts/few_shot_VP_system.txt"

# LLM Configuration - QG
BATCH_SIZE_QG = 4
LLM_MODEL_NAME_QG = "Qwen/Qwen2.5-7B-Instruct"
GPU_MEMORY_UTILIZATION_QG = 0.95
SAMPLING_PARAMS_QG = {"temperature": 0.5, "top_p": 0.8, "min_p": 0.1, "skip_special_tokens":False, "max_tokens": 2048}

# BM25
TOKENIZATION_WORKERS = 8
MAX_SENTENCES = 12
OVERLAP = 0
TOP_N_PER_QUERY = 125

# LLM Configuration - VP
K = 8 
SIM_THRESH = 0.52
BATCH_SIZE_VP = 4
LLM_MODEL_NAME_VP =  "jakiAJK/microsoft-phi-4_GPTQ-int4"
LLM_QUANTIZATION_VP = "gptq"
GPU_MEMORY_UTILIZATION_VP = 0.95
SAMPLING_PARAMS_VP = {"temperature": 0.9, "top_p": 0.7, "top_k": 1, "skip_special_tokens":False, "max_tokens": 2048}

# Embedding configuration
RETRIEVAL_MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v2.0"
RETRIEVAL_BATCH_SIZE = 512

# Semantic chunker configuration
SEMANTIC_CHUNKER_MODEL_NAME = "Lajavaness/bilingual-embedding-small"
MIN_CHUNK_SIZE = 30
MAX_CHUNK_SIZE = 140
CHUNKER_MIN_SENTENCES = 2
CHUNKER_WINDOW_OVERLAP = 1
SIMILARITY_WINDOW = 1

# Evidence Aggregation
EVIDENCES_PER_QUERY = 20
RETRIEVAL_SIMILARITY_THRESHOLD = 0.52
EVIDENCE_SIMILARITY_THRESHOLD = 0.905 # 0.9