import toml

data = {}

with open("../config.toml", "w") as f:
    toml.dump(data, f)