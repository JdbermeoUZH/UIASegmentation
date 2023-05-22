import argparse
from types import ModuleType
from argparse import Namespace
from importlib.machinery import SourceFileLoader

# consider maybe add .json parser or arg parser or both of them. For now, I will continue

class MyParser:
    
    def __init__(self, config_file = ''):
        self.init_parser()
        self.config_dict      = dict()  
        self.config_file      = config_file
        self.config_namespace = Namespace()
        
        if self.config_file == '':
            self.parser_inputs = self.parser.parse_args()
            self.config_file   = self.parser_inputs.config_file
    
        if self.config_file != '' and self.config_file.endswith('.py'):
            print(f'\n ### Reading the configuration from python file: ### \n{self.config_file} \n')
            self.load_from_py(self.config_file)
            
    
    def init_parser(self):
        self.parser = argparse.ArgumentParser(description='Train UIASegmentation model')
        self.parser.add_argument('-c', '--config_file', type = str, default='', help='Config file containing all the parameters')

    def load_from_py(self, config_file):
        '''
        Load the configuration file in python format and store it in config dict.

        Parameters
        ----------
        config_file: The file path to the configuration file.
        
        Returns
        ----------
        None
        
        '''
        module = 'module'
        loader = SourceFileLoader(module, config_file).load_module()
        for key in loader.__dict__.keys():
            if key[:2] != '__' and not isinstance(loader.__dict__[key], ModuleType):
                self.config_dict[key] = loader.__dict__[key]
        self.config_namespace = Namespace(**self.config_dict)
        # load missing values ?? we dont care for now?