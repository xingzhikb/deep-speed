import os
import json
import shutil

def process_jsonl(input_jsonl, root_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    unique_sources = set()  # Set to store unique sources

    with open(input_jsonl, 'r') as infile:
        for line in infile:
            data = json.loads(line.strip())
            source = data.get('source')
            if source:
                unique_sources.add(source)  # Add source to the set

            if source == 'mimic_cxr':
                file_name = data.get('file_name')
                if file_name:
                    file_path = os.path.join(root_dir, file_name)
                    if os.path.exists(file_path):
                        # Copy the file to the new directory
                        shutil.copy(file_path, output_dir)
                        
                        # Write the JSON line to a new file in the new directory
                        output_file = os.path.join(output_dir, 'output.jsonl')
                        with open(output_file, 'a') as outfile:
                            json.dump(data, outfile)
                            outfile.write('\n')
    
    # Print the number of unique sources
    print(f"Number of unique sources: {len(unique_sources)}")
    print(f"Unique sources: {unique_sources}")

# Example usage
input_jsonl = '/workspace/part_left/metadata.jsonl'
root_dir = '/workspace/part_left'
output_dir = '/workspace/mimic_cxr'

process_jsonl(input_jsonl, root_dir, output_dir)
