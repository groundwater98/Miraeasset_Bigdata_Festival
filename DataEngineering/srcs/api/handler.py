class APIHandler:
    def __init__(self):
        self.configs = None
        self.headers = None
        self.error_codes = None
        
    def init_config(self, config):
        raise NotImplementedError
    
    def init_credential(self, credentials):
        raise NotImplementedError
    
    def init_error_codes(self, error_codes):
        raise NotImplementedError
    
    def get_response(self):
        raise NotImplementedError