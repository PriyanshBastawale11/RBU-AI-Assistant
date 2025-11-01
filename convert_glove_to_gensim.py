from gensim.models import KeyedVectors
import os

def convert_glove_to_gensim(glove_file, output_file):
    """Convert GloVe file to Gensim format."""
    print(f"Converting {glove_file} to Gensim format...")
    glove_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
    glove_model.save(output_file)
    print(f"Conversion complete. Saved to {output_file}")

if __name__ == "__main__":
    # Define file paths
    glove_file = os.path.join("models", "glove.6B", "glove.6B.300d.txt")

 # Adjust path if needed
    output_file = os.path.join("models", "word_vectors.bin")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Convert GloVe to Gensim format
    convert_glove_to_gensim(glove_file, output_file)
