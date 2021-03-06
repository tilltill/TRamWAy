# -*- coding: utf-8 -*-

# Copyright © 2017-2018, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from rwa import HDF5Store, lazytype, lazyvalue
from ..lazy import Lazy
from ..analyses import Analyses, coerce_labels, format_analyses, append_leaf
import tramway.core.analyses.base as ba
import os.path
import traceback
import errno

try:
    input = raw_input # Py2
except NameError:
    pass


__all__ = ['RWAStore', 'load_rwa', 'save_rwa']


class RWAStore(HDF5Store):

    __slots__ = ('unload', '__special__')

    def __init__(self, resource, mode='auto', unload=False, verbose=False, **kwargs):
        HDF5Store.__init__(self, resource, mode, verbose, **kwargs)
        self.unload = unload
        self.__special__ = {}

    def poke(self, objname, obj, container=None, visited=None, _stack=None, unload=None):
        if unload is not None:
            previous = self.unload
            self.unload = unload
        if visited is None:
            visited = {}
        if container is None:
            container = self.store
        try:
            obj = obj.lazy_unwrap()
        except AttributeError:
            pass
        obj = lazyvalue(obj, deep=True) # in the case it was stored not unloaded
        if self.unload and isinstance(obj, Lazy):
            # set lazy attributes to None (unset them so that memory is freed)
            #obj = copy(obj) # this causes a weird bug
            for name in obj.__lazy__:
                # `_lazy` should contain `name`;
                # if it is not, the object definition has changed
                if obj._lazy.get(name, True):
                    setattr(obj, obj.__fromlazy__(name), None)
        obj = self.special_unload(obj)
        HDF5Store.poke(self, objname, obj, container, visited=visited, _stack=_stack)
        if unload is not None:
            self.unload = previous

    def peek(self, *args, **kwargs):
        obj = HDF5Store.peek(self, *args, **kwargs)
        obj = self.special_load(obj)
        return obj

    def special_load(self, obj):
        if 'data0' in self.__special__:
            import tramway.tessellation.base as tessellation
            if lazytype(obj) is tessellation.CellStats:
                obj = lazyvalue(obj, deep=True)
                try:
                    if obj._points is None:
                        raise AttributeError
                except AttributeError:
                    obj._points = self.__special__['data0']
        else:
            import pandas
            if lazytype(obj) is pandas.DataFrame:
                obj = lazyvalue(obj, deep=True)
                self.__special__['data0'] = obj
        return obj

    def special_unload(self, obj):
        if 'data0' in self.__special__:
            import tramway.tessellation.base as tessellation
            if isinstance(obj, tessellation.CellStats) and obj.points is self.__special__['data0']:
                import copy
                obj = copy.copy(obj)
                obj._points = None
        else:
            import pandas
            if isinstance(obj, pandas.DataFrame):
                self.__special__['data0'] = obj
        return obj




def load_rwa(path, verbose=None, lazy=False):
    """
    Load a .rwa file.

    Note about laziness: the analysis tree uses an active handle to the opened file.
    As a consequence, the file should be read only once.
    It is safe to load, modify, save and then load again, but the first loaded data
    should be terminated before loading the data again.

    Arguments:

        path (str): path to .rwa file

        verbose (bool or int): verbosity level

        lazy (bool): reads the file lazily

    Returns:

        tramway.core.analyses.base.Analyses:
            analysis tree;
            if `lazy` is ``True``, return type is
            :class:`tramway.core.analyses.lazy.Analyses` instead
    """
    try:
        hdf = RWAStore(path, 'r', verbose=max(0, int(verbose) - 2) if verbose else False)
        #hdf._default_lazy = PermissivePeek
        hdf.lazy = lazy
        try:
            analyses = lazyvalue(hdf.peek('analyses'))
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            if verbose is not False:
                print('cannot load file: {}'.format(path))
            raise
        finally:
            hdf.close()
    except EnvironmentError as e:
        if hasattr(e, 'errno') and e.errno == errno.ENOENT:
            raise
        elif e.args[1:]:
            if verbose:
                print(traceback.format_exc())
            raise OSError('HDF5 libraries may not be installed')
        else:
            raise
    return coerce_labels(analyses)



def save_rwa(path, analyses, verbose=False, force=False, compress=True, append=False):
    """
    Save an analysis tree into a .rwa file.

    Arguments:

        path (str): path to .rwa file

        analyses (tramway.core.analyses.base.Analyses): analysis tree

        verbose (bool or int): verbose mode

        force (bool): do not ask whether to overwrite an existing file or not

        compress (bool): delete the lazy attributes that can be computed again automatically

        append (bool): do not overwrite; reload the file instead and append the analyses as
            a subtree

    """
    if not isinstance(analyses, ba.Analyses):
        raise TypeError('`analyses` is not an `Analyses` instance')
    if os.path.isfile(path):
        if append:
            extra_analyses = analyses
            analyses = load_rwa(path)
            append_leaf(analyses, extra_analyses, overwrite=force)
        elif not force:
            answer = input("overwrite file '{}': [N/y] ".format(path))
            if not (answer and answer[0].lower() == 'y'):
                return
    elif append:
        if verbose:
            print('file not found; flushing all the analyses')
    try:
        store = RWAStore(path, 'w', verbose=max(0, int(verbose) - 2))
        try:
            store.unload = compress
            if verbose:
                print('writing file: {}'.format(path))
            store.poke('analyses', analyses)
        finally:
            store.close()
    except EnvironmentError as e:
        if hasattr(e, 'errno') and e.errno == errno.ENOENT:
            raise
        print(traceback.format_exc())
        raise ImportError('HDF5 libraries may not be installed')
    if 1 < int(verbose):
        print('written analysis tree:')
        print(format_analyses(analyses, global_prefix='\t', node=lazytype))


