import argparse
from types import ModuleType
from argparse import Namespace
from importlib.machinery import SourceFileLoader

# consider maybe add .json parser or arg parser or both of them. For now, I will continue

class MyParser:
    
    def __init__(self, config_file = '', print_config=True):

        self.init_parser()
        self.config_dict                = dict()  
        self.config_file                = config_file
        self.config_namespace           = Namespace()
        
        if self.config_file == '':
            self.parser_inputs = self.parser.parse_args()
            self.populate_config_dict()
    
        if self.config_file != '' and self.config_file.endswith('.py'):
            print(f'\n ### Reading the configuration from python file: ### \n{self.config_file} \n')
            self.load_from_py(self.config_file)
        else:
            raise ValueError("Configuration file in .py format is not given")
        
        self.config_namespace = Namespace(**self.config_dict)
        if print_config == True: self.print_configuration()
    
    def init_parser(self):
        self.parser = argparse.ArgumentParser(description='Train UIASegmentation model')
        # general arguments
        self.parser.add_argument('-c', '--config_file', type = str, default='', help = 'Config file containing all the parameters')
        #--- only for development
        # graph arguments
        self.parser.add_argument('-gh', '--hidden_channels_g', type = int, default=0, help = 'hidden channels for graph neural network')
    
    def populate_config_dict(self):
        parser_dict = vars(self.parser_inputs)
        if parser_dict['config_file'] != '': self.config_file = parser_dict['config_file']
        if parser_dict['hidden_channels_g'] != 0: self.config_dict['hidden_channels_g'] = parser_dict['hidden_channels_g']

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
        existed_keys = self.config_dict.keys()
        for key in loader.__dict__.keys():
            if key[:2] != '__' and not isinstance(loader.__dict__[key], ModuleType):
                if key not in existed_keys:
                    self.config_dict[key] = loader.__dict__[key]
        
    def print_configuration(self):
        temp_dict = vars(self.config_namespace)
        msg       = ''
        msg      += 'PRINTING TOTAL OPTIONS: \n'
        for key, value in sorted(temp_dict.items()):    msg += f'{str(key)}: {str(value)} \n'
        msg      += 'FINISH PRINTING \n' 
        print(msg)