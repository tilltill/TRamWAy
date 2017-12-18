# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .lazy import *
from .namedcolumns import *
from .chain import *
from . import exceptions
from .analyses import *

__all__ = ['PermissionError', 'ro_property_assert', 'Lazy', 'lightcopy', 'isstructured', 'columns', 'Matrix', 'ArrayChain', 'ChainArray', 'Analyses', 'select_analysis']

