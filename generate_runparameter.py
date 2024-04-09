import json
import argparse

def update_dictionary(file_path, transform_value, feature_value, output_file):
    # Read the dictionary from the file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    for key in data.keys():
        key2 = key.split('$')[2]
        if key2 == 'tensor_transform':
            data[key] = transform_value
        elif key2 == 'features':
            data[key] = feature_value
    
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)
    print(f'new parameter file {output_file}')
def main():
    parser = argparse.ArgumentParser(description="Update parameter values.")
    parser.add_argument('--base_parameter', type=str, help="Path to the base parameter file.", default = "./config/r18base_metaparameters.json")
    parser.add_argument('--transform_value', type=str, help="New value for transform", default = "pca_eigenvalues")
    parser.add_argument('--feature_value', type=str, help="New value for feature", default = "white")
    parser.add_argument('--new_parameter', type=str, help="Path to the output file for the updated dictionary.", default = "./config/r18new_metaparameters.json")

    args = parser.parse_args()

    update_dictionary(args.base_parameter, args.transform_value, args.feature_value, args.new_parameter)

if __name__ == "__main__":
    main()

