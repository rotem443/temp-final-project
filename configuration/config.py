import os
from configuration.dev_config import DevConfig
from configuration.prod_config import ProdConfig


env = os.environ.get('SIGNAL_ENV', 'DevConfig')

try:
    current_config = eval(env)()
except ImportError:
    pass  # TODO: LOG
