import glob
import pickle

def output_pickle_file(filename, output_file):
    output_file.write(f"\n============================================================== {filename} =============================================================\n")
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        # Check if the unpickled data is a dictionary and output accordingly
        if isinstance(data, dict):
            for key, value in data.items():
                output_file.write(f"{key}: {value}\n")
        else:
            output_file.write(f"{data}\n")
    except Exception as e:
        output_file.write(f"Error loading {filename}: {e}\n")
    output_file.write("====================================================================================================================================================\n")

# Define the output file name
output_filename = "pickle_output.txt"

# Open the output file in write mode
with open(output_filename, "w") as output_file:
    # Find all files with the .pkl extension in the current directory and sort them alphabetically
    pickle_files = sorted(glob.glob("*.pkl"))
    
    # Process and output each pickle file's content to the text file
    for pkl_file in pickle_files:
        output_pickle_file(pkl_file, output_file)

print(f"All pickle file outputs have been saved to {output_filename}")
