import os
import gzip

FOLDER_PATH = "./data"

def extract_gz_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".tsv.gz"):
                file_path = os.path.join(root, file)
                output_file_path = os.path.splitext(file_path)[0]
                with gzip.open(file_path, 'rb') as f_in:
                    with open(output_file_path, 'wb') as f_out:
                        f_out.write(f_in.read())


if __name__ == '__main__':
    extract_gz_files(FOLDER_PATH)
