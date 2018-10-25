import yaml

def yaml_loader(file_path):
    with open(file_path, "r") as file_descriptor:
        data = yaml.load(file_descriptor)
    return data


if __name__ == "__main__":
    file_path = "config.yaml"
    data = yaml_loader(file_path)
    print(data.get("video").get("name"))

