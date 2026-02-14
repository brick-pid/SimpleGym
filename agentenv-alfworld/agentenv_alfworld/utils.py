import yaml

def load_config(config_file):
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
    return config
