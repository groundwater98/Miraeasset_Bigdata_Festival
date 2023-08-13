import logging
import os
import sys
import toml

config_path = os.getenv('TOOLS_CONFIG_PATH', '..')
print(config_path)

with open(config_path, "r") as f:
    _data = toml.load(f)
    data = _data["logging"]

class Logger:
    def __init__(self, logger_name, save=True, file_name=data['log_file'], level=data['log_level']):
        self.logger = logging.getLogger(logger_name)
        self._set_logger_configuration(save, file_name, level)
        
    def _set_logger_configuration(self, save, file_name, level):
        formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
        level = getattr(logging, level.upper())
        
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        
        if save:
            file_handler = logging.FileHandler(file_name)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.setLevel(level)
              
    def get_logger(self):
        return self.logger
