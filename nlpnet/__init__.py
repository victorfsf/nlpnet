# import to provide easier access for nlpnet users

from . import taggers  # noqa
from . import utils  # noqa
from .config import set_data_dir  # noqa
from .taggers import POSTagger, SRLTagger, DependencyParser  # noqa
from .utils import tokenize  # noqa

__version__ = '1.2.1'
