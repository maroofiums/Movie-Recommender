from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"


MOVIES_METADATA_PATH = RAW_DATA_DIR / "movies_metadata.csv"

CLEAN_MOVIES_PATH = INTERIM_DATA_DIR / "clean_movies.csv"

FEATURED_MOVIES_PATH = PROCESSED_DATA_DIR / "featured_movies.csv"

PAIR_DATASET_PATH = PROCESSED_DATA_DIR / "movie_pairs.csv"

EMBEDDINGS_PATH = MODELS_DIR / "movie_embeddings.npy"

FAISS_INDEX_PATH = MODELS_DIR / "movie_index.faiss"


RANDOM_SEED = 42


TRAIN_SIZE = 0.8
VALID_SIZE = 0.1
TEST_SIZE = 0.1

BATCH_SIZE = 16

NUM_EPOCHS = 10

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 1e-2


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_SEQUENCE_LENGTH = 256

EMBEDDING_DIM = 384


TOP_K = 10

SIMILARITY_THRESHOLD = 0.70


LOG_LEVEL = "INFO"

LOG_DIR = ROOT_DIR / "logs"

LOG_FILE = LOG_DIR / "project.log"