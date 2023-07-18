import toml

data = {
    "database": {
        "db_name": "database",
        "db_password": "docker",
        "db_username": "docker",
        "docker_host": "0.0.0.0",
        "db_port": 5432
    },
    "api_key": "",
    "logging": {
        "log_level": "DEBUG",
        "log_file": "app.log"
    }
}

with open("../config.toml", "w") as f:
    toml.dump(data, f)