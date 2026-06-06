import numpy as np
import tiktoken
from datasets import load_dataset

def build_binary_dataset(repo_id, config_name, output_filename):
    print(f"Connecting to {repo_id} ({config_name})...")
    
    # Initialize the exact tokenizer used in your model
    tokenizer = tiktoken.get_encoding('gpt2')
    
    # Load the dataset in streaming mode. 
    # This acts as an iterator, pulling data over the network continuously rather than downloading it all at once.
    dataset = load_dataset(repo_id, name=config_name, split="train", streaming=True)
    
    # Open the target file in write-binary ('wb') mode
    with open(output_filename, 'wb') as f:
        token_count = 0
        doc_count = 0
        
        for example in dataset:
            # Extract the raw text from the dataset row
            text = example['text']
            
            # Encode the text into token IDs
            # encode_ordinary ignores special tokens in the raw text, preventing injection errors
            tokens = tokenizer.encode_ordinary(text)
            
            # Append the End-Of-Text token (50256 for gpt2) to separate documents
            tokens.append(tokenizer.eot_token)
            
            # Convert the Python list to a highly compressed numpy array of 16-bit unsigned integers.
            # (Since your vocab size is ~50,304, it fits perfectly inside the 65,535 limit of uint16)
            tokens_np = np.array(tokens, dtype=np.uint16)
            
            # Write the raw bytes directly to the NVMe drive
            f.write(tokens_np.tobytes())
            
            token_count += len(tokens)
            doc_count += 1
            
            if doc_count % 10000 == 0:
                print(f"Processed {doc_count:,} documents | Total Tokens Saved: {token_count:,}")

    print(f"\nFinished! Dataset saved to {output_filename}")
    print(f"Total Tokens: {token_count:,}")


build_binary_dataset(
    repo_id="HuggingFaceFW/fineweb-edu", 
    config_name="sample-10BT", 
    output_filename="datasets/train_data.bin"
)