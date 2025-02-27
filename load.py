import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dynamic dataset importer")
    parser.add_argument('--path', type=str, default='.', help="dataset path")
    args = parser.parse_args()

    # Invoke next file in pipeline, sending it the dataset path
    subprocess.run(['python', 'dpre.py', args.path])