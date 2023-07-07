import logging

class Logger:
    def __init__(self, logger_name, save=False, file_name="app.log", level="INFO"):
        self.logger = logging.getLogger(logger_name)
        self.set_logger_configuration(save, file_name, level)
        
    def set_logger_configuration(self, save, file_name, level):
        # EX) 2023-07-03 13:37:31,123 - INFO - This is an info message
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        level = getattr(logging, level.upper())
        
        if save:
            file_handler = logging.FileHandler(file_name)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
              
    def get_logger(self):
        return self.logger
        