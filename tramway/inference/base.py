# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core import *
from tramway.tessellation import format_cell_index, nearest_cell
import tramway.tessellation as tessellation
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.spatial.qhull
from copy import copy
from collections import OrderedDict
from multiprocessing import Pool, Lock
import six
from functools import partial
from warnings import warn
import traceback
from numpy.polynomial import polynomial as poly



class Local(Lazy):
        """
        Spatially local subset of elements (e.g. translocations). Abstract class.

        Attributes:

                index (int):
                        This cell's index as referenced in :class:`Distributed`.

                data (collection of terminal elements or :class:`Local`):
                        Elements, either terminal or not.

                center (array-like):
                        Cell center coordinates.

                span (array-like):
                        Difference vectors from this cell's center to adjacent centers.

                boundary/hull (scipy.spatial.qhull.ConvexHull):
                        Convex hull of the contained locations.

        """
        __slots__ = ('index', 'data', 'center', 'span', '_boundary', '_volume')
        __lazy__ = Lazy.__lazy__ + ('volume',)

        def __init__(self, index, data, center=None, span=None, boundary=None):
                Lazy.__init__(self)
                self.index = index
                self.data = data
                self.center = center
                self.span = span
                self._boundary = boundary
                self._volume = None

        @property
        def boundary(self):
                """
                *any*, property

                Cell boundary/hull.
                """
                return self._boundary

        @boundary.setter
        def boundary(self, b):
                self.volume = None
                self._boundary = b

        @property
        def hull(self):
                """
                `scipy.spatial.qhull.ConvexHull` (or *any* if not convex), property

                Alias for `boundary`.
                """
                return self.boundary

        @hull.setter
        def hull(self, h):
                self.boundary = h

        @property
        def volume(self):
                """
                `float`, property

                Surface area (2D) or volume (3D+) of the convex hull.

                Note that the hull should be convex so that the volume can be
                transparently extracted.
                In other cases, `volume` can be manually set just like a normal
                attribute.
                """
                if self._volume is None and self.hull is not None:
                        if self.dim == 2:
                                area = 0
                                u = self.center
                                for i in self.hull.simplices:
                                        v, w = self.hull.points[i]
                                        area += .5 * abs(\
                                                        (v[0] - u[0]) * (w[1] - u[1]) - \
                                                        (w[0] - u[0]) * (v[1] - u[1]) \
                                                )
                                self._volume = area
                        else:
                                self._volume = self.hull.volume
                return self.__returnlazy__('volume', self._volume)

        @volume.setter
        def volume(self, v):
                self.__setlazy__('volume', v)

        @property
        def dim(self):
                """
                `int`, property

                Dimension of the terminal elements.
                """
                raise NotImplementedError('abstract method')

        @dim.setter
        def dim(self, d):
                self.__assertlazy__('dim', d, related_attribute='data')

        @property
        def tcount(self):
                """
                `int`, property

                Total number of terminal elements (e.g. translocations).
                """
                raise NotImplementedError('abstract method')

        @tcount.setter
        def tcount(self, c):
                self.__assertlazy__('tcount', c, related_attribute='data')



class Distributed(Local):
        """
        Attributes:

                central (array of bools):
                        margin cells are not central.

        """
        __slots__ = ('_reverse', '_adjacency', 'central', '_degree', '_ccount', '_tcount')
        __lazy__  = Local.__lazy__ + ('reverse', 'degree', 'ccount', 'tcount')

        def __init__(self, cells, adjacency, index=None, center=None, span=None, central=None, \
                boundary=None):
                Local.__init__(self, index, OrderedDict(), center, span, boundary)
                self.cells = cells # let's `cells` setter perform the necessary checks 
                self.adjacency = adjacency
                self.central = central

        @property
        def cells(self):
                """
                `list` or `OrderedDict`, rw property for :attr:`data`

                Collection of :class:`Local`. Indices may not match with the global 
                :attr:`~Local.index` attribute of the elements, but match with attributes 
                :attr:`central`, :attr:`adjacency` and :attr:`degree`.
                """
                return self.data

        @cells.setter
        def cells(self, cells):
                celltype = type(self.cells)
                assert(celltype is dict or celltype is OrderedDict)
                if not isinstance(cells, celltype):
                        if isinstance(cells, dict):
                                cells = celltype(sorted(cells.items(), key=lambda t: t[0]))
                        elif isinstance(cells, list):
                                cells = celltype(sorted(enumerate(cells), key=lambda t: t[0]))
                        else:
                                raise TypeError('`cells` argument is not a dictionnary (`dict` or `OrderedDict`)')
                if not all([ isinstance(cell, Local) for cell in cells.values() ]):
                        raise TypeError('`cells` argument is not a dictionnary of `Local`')
                #try:
                #       if self.ccount == len(cells): #.keys().reversed().next(): # max
                #               self.adjacency = None
                #               self.central = None
                #except:
                #       pass
                self.reverse = None
                self.ccount = None
                self.data = cells

        @property
        def indices(self):
                return np.array([ cell.index for cell in self.cells.values() ])

        @property
        def reverse(self):
                """
                `dict of ints`, ro lazy property

                Get "local" indices from global ones.
                """
                if self._reverse is None:
                        self._reverse = {cell.index: i for i, cell in self.cells.items()}
                return self._reverse

        @reverse.setter
        def reverse(self, r): # ro
                self.__assertlazy__('reverse', r, related_attribute='cells')

        @property
        def space_cols(self):
                """
                `list of ints` or `strings`, ro property

                Column indices for coordinates of the terminal points.
                """
                return self.any_cell().space_cols

        @property
        def dim(self):
                """
                `int`, ro property

                Dimension of the terminal points.
                """
                if self.center is None:
                        return self.any_cell().dim
                else:
                        return self.center.size

        @property
        def tcount(self):
                """
                `int`, ro property

                Total number of terminal points. Duplicates are ignored.
                """
                if self._tcount is None:
                        if self.central is None:
                                self._tcount = sum([ cell.tcount \
                                        for i, cell in self.cells.items() if self.central[i] ])
                        else:
                                self._tcount = sum([ cell.tcount for cell in self.cells.values() ])
                return self._tcount

        @tcount.setter
        def tcount(self, c):
                # write access allowed for performance issues, but `c` should equal self.tcount
                self.__setlazy__('tcount', c)

        @property
        def ccount(self):
                """
                `int`, ro property

                Total number of terminal cells. Duplicates are ignored.
                """
                #return self.adjacency.shape[0] # or len(self.cells)
                if self._ccount is None:
                        self._ccount = sum([ cell.ccount if isinstance(cell, Distributed) else 1 \
                                for cell in self.cells.values() ])
                return self._ccount # not mutable (no need for __returnlazy__)

        @ccount.setter
        def ccount(self, c): # rw for performance issues, but `c` should equal self.ccount
                self.__setlazy__('ccount', c)

        @property
        def adjacency(self):
                """
                `scipy.sparse.csr_matrix`, rw property

                Cell adjacency matrix. Row and column indices are to be mapped with `indices`.
                """
                return self._adjacency

        @adjacency.setter
        def adjacency(self, a):
                if a is not None:
                        a = a.tocsr()
                self._adjacency = a
                self._degree = None # `degree` is ro, hence set `_degree` instead

        @property
        def degree(self):
                """
                `array of ints`, ro lazy property

                Number of adjacent cells.
                """
                if self._degree is None:
                        self._degree = np.diff(self._adjacency.indptr)
                return self._degree

        @degree.setter
        def degree(self, d): # ro
                self.__lazyassert__(d, 'adjacency')

        def grad(self, i, X, index_map=None, **kwargs):
                """
                Local gradient.

                Sub-classes are free to return multi-component gradients as matrices provided that
                they exhibit as many columns as there are dimensions in the (trans-)location data.
                :meth:`grad_sum` is then responsible for summing all the elements of such matrices.

                Arguments:

                        i (int):
                                cell index at which the gradient is evaluated.

                        X (array):
                                vector of a scalar measurement at every cell.

                        index_map (array):
                                index map that converts cell indices to indices in X.

                Returns:

                        array:
                                gradient as a vector with as many elements as there are dimensions
                                in the (trans-)location data.
                """
                return grad1(self, i, X, index_map, **kwargs)

        def grad_sum(self, i, grad, index_map=None):
                """
                Mixing operator for the gradient at a given cell.

                Arguments:

                        i (int):
                                cell index.

                        grad (numpy.ndarray):
                                local gradient.

                        index_map (array):
                                index mapping, useful to convert cell indices to positional indices in 
                                an optimization array for example.

                Returns:

                        float:
                                weighted sum of the elements of `grad`.

                """
                cell = self.cells[i]
                if cell.volume:
                        return cell.volume * np.sum(grad)
                else:
                        return np.sum(grad)

        def flatten(self):
                def concat(arrays):
                        if isinstance(arrays[0], tuple):
                                raise NotImplementedError
                        elif isinstance(arrays[0], pd.DataFrame):
                                return pd.concat(arrays, axis=0)
                        else:
                                return np.stack(arrays, axis=0)

                new = copy(self)
                new.cells = {i: Cell(i, concat([cell.data for cell in dist.cells.values()]), \
                                dist.center, dist.span, hull=dist.hull) \
                        if isinstance(dist, Distributed) else dist \
                        for i, dist in self.cells.items() }

                return new

        def group(self, ngroups=None, max_cell_count=None, cell_centers=None, \
                adjacency_margin=2):
                """
                Make groups of cells.

                This builds up an extra hierarchical level. 
                For example if `self` is a `Distributed` of `Cell`, then this returns a `Distributed` 
                of `Distributed` (the groups) of `Cell`.

                Several grouping strategies are proposed.

                Arguments:

                        ngroups (int):
                                number of groups.

                        max_cell_count (int):
                                maximum number of cells per group.

                        cell_centers (array-like):
                                spatial centers of the groups. A cell is associated to the group which 
                                center is the nearest one. If not provided, :meth:`group` will use 
                                a k-means approach to positionning the centers.

                        adjacency_margin (int):
                                groups are dilated to the adjacent cells `adjacency_margin` times. 
                                Defaults to 2.

                Returns:

                        Distributed:
                                a new object of the same type as `self` that contains other such
                                objects as cells.

                """
                ncells = self.adjacency.shape[0]
                new = copy(self)
                if ngroups or (max_cell_count and max_cell_count < ncells) or cell_centers is not None:
                        any_cell = self.any_cell()

                        points = np.zeros((ncells, self.dim), dtype=any_cell.center.dtype)
                        ok = np.zeros(points.shape[0], dtype=bool)
                        for i in self.cells:
                                if self.cells[i]: # non-empty
                                        points[i] = self.cells[i].center
                                        ok[i] = True

                        if cell_centers is None:
                                if max_cell_count == 1:
                                        grid = copy(self)
                                else:
                                        avg_probability = 1.0
                                        if ngroups:
                                                avg_probability = min(1.0 / float(ngroups), avg_probability)
                                        if max_cell_count:
                                                avg_probability = min(float(max_cell_count) / \
                                                        float(points.shape[0]), avg_probability)
                                        import tramway.tessellation.kmeans as kmeans
                                        grid = kmeans.KMeansMesh(avg_probability=avg_probability)
                                        try:
                                                grid.tessellate(points[ok])
                                        except scipy.spatial.qhull.QhullError:
                                                print('Qhull errors are handled; full map optimization')
                                                return new
                                        except ValueError:
                                                print(points)
                                                raise

                        else:
                                grid = tessellation.Voronoi()
                                grid.cell_centers = cell_centers

                        I = np.full(ok.size, -1, dtype=int)
                        I[ok] = grid.cell_index(points[ok], min_location_count=1)
                        #if not np.all(ok):
                        #       print(ok.nonzero()[0])
                        new.adjacency = grid.simplified_adjacency(format='csr') # macro-cell adjacency matrix
                        J = np.unique(I)
                        J = J[0 <= J]
                        assert 0 < J.size
                        new.data = type(self.cells)()

                        for j in J: # for each macro-cell
                                K = I == j # find corresponding cells
                                assert np.any(K)

                                if 0 < adjacency_margin:
                                        L = np.copy(K)
                                        for k in range(adjacency_margin):
                                                # add adjacent cells for future gradient calculations
                                                K[self.adjacency[K,:].indices] = True
                                        L = L[K]

                                A = self.adjacency[K,:].tocsc()[:,K].tocsr() # point adjacency matrix
                                C = grid.cell_centers[j]
                                D = OrderedDict([ (i, self.cells[k]) \
                                        for i, k in enumerate(K.nonzero()[0]) if k in self.cells ])

                                for i in D:
                                        adj = A[i].indices
                                        if 0 < D[i].tcount and adj.size:
                                                span = np.stack([ D[k].center for k in adj ], axis=0)
                                        else:
                                                span = np.empty((0, D[i].center.size), \
                                                        dtype=D[i].center.dtype)
                                        if span.shape[0] < D[i].span.shape[0]:
                                                D[i] = copy(D[i])
                                                D[i].span = span - D[i].center

                                R = grid.cell_centers[new.adjacency[j].indices] - C

                                new.cells[j] = type(self)(D, A, index=j, center=C, span=R)

                                if 0 < adjacency_margin:
                                        new.cells[j].central = L
                                assert bool(new.cells[j])

                        new.ccount = self.ccount
                        # _tcount is not supposed to change

                else:
                        pass
                        #raise KeyError('`group` expects more input arguments')

                return new

        def run(self, function, *args, **kwargs):
                """
                Apply a function to the groups (:class:`Distributed`) of terminal cells.

                The results are merged into a single :class:`~pandas.DataFrame` array, 
                handling adjacency margins if any.

                Although this method was designed for `Distributed` of `Distributed`, its usage is 
                advised to call any function that returns a DataFrame with cell indices as indices.

                Multiples processes may be spawned.

                Arguments:

                        function (callable):
                                the function to be called on each terminal :class:`Distributed`. 
                                Its first argument is the :class:`Distributed` object.
                                It should return a :class:`~pandas.DataFrame`.

                        args (list):
                                positional arguments for `function` after the first one.

                        kwargs (dict):
                                keyword arguments for `function` from which are removed the ones below.

                        worker_count (int):
                                number of simultaneously working processing units.

                        profile (bool or str or tuple):
                                profile each child job if any;
                                if `str`, dump the output stats into *.prof* files;
                                if `tuple`, print a report with :func:`~pstats.Stats.print_stats` and
                                tuple elements as input arguments.

                Returns:

                        pandas.DataFrame:
                                single merged array.

                """
                # clear the caches
                self.clear_caches()

                if all(isinstance(cell, Distributed) for cell in self.cells.values()):
                        # parallel for-loop over the subsets of cells
                        # if `worker_count` is `None`, `Pool` will use `multiprocessing.cpu_count()`
                        worker_count = kwargs.pop('worker_count', None)
                        profile = kwargs.pop('profile', False)
                        pool = Pool(worker_count)
                        fargs = (function, args, kwargs)
                        if profile:
                                fargs = (profile, fargs)
                                cells = [ (i, self.cells[i]) for i in self.cells if bool(self.cells[i]) ]
                        else:
                                cells = [ self.cells[i] for i in self.cells if bool(self.cells[i]) ]
                        if six.PY3:
                                if profile:
                                        _run = __profile_run__
                                else:
                                        _run = __run__
                                ys = pool.map(partial(_run, fargs), cells)
                        elif six.PY2:
                                import itertools
                                if profile:
                                        _run = __profile_run_star__
                                else:
                                        _run = __run_star__
                                ys = pool.map(_run,
                                        itertools.izip(itertools.repeat(fargs), cells))
                        ys = [ y for y in ys if y is not None ]
                        if ys:
                                if ys[1:]:
                                        result = pd.concat(ys, axis=0).sort_index()
                        else:
                                result = None

                else:
                        # direct function application
                        result = function(self, *args, **kwargs)

                return result

        # `dict` interface
        def __len__(self):
                return self.adjacency.shape[0]

        def __nonzero__(self):
                return bool(self.cells)

        def __iter__(self):
                return iter(self.cells)

        def __getitem__(self, i):
                return self.cells[i]

        def __setitem__(self, i, cell):
                self.cells[i] = cell

        def __delitem__(self, i):
                self.cells.__delitem__(i)

        def keys(self):
                try:
                        return self.cells.keys()
                except AttributeError:
                        return range(len(self.cells))

        def values(self):
                try:
                        return self.cells.values()
                except AttributeError:
                        return self.cells

        def items(self):
                try:
                        return self.cells.items()
                except AttributeError:
                        return enumerate(self.cells)

        def any_cell(self):
                return self.cells[next(iter(self.cells))]

        def neighbours(self, i):
                """
                Indices of neighbour cells.

                Argument:

                        i (int): cell index.

                Returns:

                        numpy.ndarray: indices of the neighbour cells of cell *i*.

                """
                return self.adjacency.indices[self.adjacency.indptr[i]:self.adjacency.indptr[i+1]]

        def clear_caches(self):
                try:
                        first = True
                        for c in self.values():
                                c.clear_cache()
                                first = False
                except AttributeError as e:
                        if first:
                                try:
                                        first = True
                                        for c in self.values():
                                                c.clear_caches()
                                                first = False
                                        return
                                except:
                                        if first:
                                                raise e
                        if not first:
                                raise


def __run__(func, cell):
        function, args, kwargs = func
        try:
                x = cell.run(function, *args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
                raise
        except:
                print(traceback.format_exc())
                raise
        if x is None:
                return None
        else:
                i = cell.indices
                if cell.central is not None:
                        try:
                                x = x.iloc[cell.central[x.index]]
                        except IndexError as e:
                                if cell.central.size < x.index.max():
                                        raise IndexError('dataframe indices do no match with group-relative cell indices (maybe are they global ones)')
                                else:
                                        print(x.shape)
                                        print((cell.central.shape, cell.central.max()))
                                        print(x.index.max())
                                        raise e
                        i = i[x.index]
                        if x.shape[0] != i.shape[0]:
                                raise IndexError('not as many indices as values')
                x.index = i
                return x

def __run_star__(args):
        return __run__(*args)


def __profile_run__(func, args):
        import cProfile, pstats
        proptions, func = func
        try:
                process, cells = args
        except TypeError:
                process = None
        profile = cProfile.Profile()
        profile.enable()
        result = __run__(func, cells)
        profile.disable()
        if process is not None and isinstance(proptions, str):
                profile.dump_stats('{}{}.prof'.format(filename, process))
        else:
                if proptions in (True, False):
                        proptions = (.1, )
                elif not isinstance(proptions, (tuple, list)):
                        proptions = (proptions, )
                stats = pstats.Stats(profile).sort_stats('cumulative')
                stats.print_stats(*proptions)
        return result

def __profile_run_star__(args):
        return __profile_run__(*args)



class Cell(Local):
        """
        Spatially constrained subset of (trans-)locations with associated intermediate calculations.

        Attributes:

                cache (any):
                        Depending on the inference approach and objective, caching an intermediate
                        result may avoid repeating many times a same computation. Usage of this cache
                        is totally free and comes without support for concurrency.

        """
        __slots__ = ('_time_col', '_space_cols', 'cache', 'fuzzy')
        __lazy__  = Local.__lazy__ + ('time_col', 'space_cols')

        def __init__(self, index, data, center=None, span=None, boundary=None):
                """
                Arguments:

                        index (int):
                                this cell's index as granted in :class:`Distributed` 's `cells` dict.

                        center (array-like):
                                cell center coordinates.

                        span (array-like):
                                difference vectors from this cell's center to adjacent centers.

                """
                if not (isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame)):
                        raise TypeError('unsupported (trans-)location type `{}`'.format(type(data)))
                Local.__init__(self, index, data, center, span, boundary)
                #self._tcount = data.shape[0]
                self._time_col = None
                self._space_cols = None
                self.cache = None
                #self.data = (self.space_data, self.time_data)
                self.fuzzy = None

        @property
        def time_col(self):
                """
                `int` or `string`, lazy

                Column index for time.
                """
                if self._time_col is None:
                        if isstructured(self.data):
                                self._time_col = 't'
                        else:
                                self._time_col = 0
                return self._time_col

        @time_col.setter
        def time_col(self, col):
                # space_cols is left unchanged
                self.__lazysetter__(col)

        @property
        def space_cols(self):
                """
                `list of ints` or `strings`, lazy

                Column indices for coordinates.
                """
                if self._space_cols is None:
                        if isstructured(self.data):
                                self._space_cols = columns(self.data)
                                if isinstance(self._space_cols, pd.Index):
                                        self._space_cols = self._space_cols.drop(self.time_col)
                                else:
                                        self._space_cols.remove(self.time_col)
                        else:
                                if self.time_col == 0:
                                        self._space_cols = np.arange(1, self.data.shape[1])
                                else:
                                        self._space_cols = np.ones(self.data.shape[1], \
                                                dtype=bool)
                                        self._space_cols[self.time_col] = False
                                        self._space_cols, = self._space_cols.nonzero()
                return self._space_cols

        @space_cols.setter
        def space_cols(self, cols):
                # time_col is left unchanged
                self.__lazysetter__(cols)

        @property
        def dim(self):
                return len(self.space_cols)

        @dim.setter
        def dim(self, d): # ro
                self.__lazyassert__(d, 'data')

        @property
        def tcount(self):
                """
                `int`

                Number of (trans-)locations.
                """
                return self.space_data.shape[0]

        @tcount.setter
        def tcount(self, c): # ro
                self.__lazyassert__(c, 'data')

        @property
        def time_data(self):
                if not isinstance(self.data, tuple):
                        self.data = (self._extract_space(), self._extract_time())
                return self.data[1]

        @time_data.setter
        def time_data(self, t):
                self.data = (self.space_data, t)

        def _extract_time(self):
                if isstructured(self.data):
                        return np.asarray(self.data[self.time_col])
                else:
                        return np.asarray(self.data[:,self.time_col])

        @property
        def space_data(self):
                if not isinstance(self.data, tuple):
                        self.data = (self._extract_space(), self._extract_time())
                return self.data[0]

        @space_data.setter
        def space_data(self, xy):
                self.data = (xy, self.time_data)

        def _extract_space(self):
                if isstructured(self.data):
                        return np.asarray(self.data[self.space_cols])
                else:
                        return np.asarray(self.data[:,self.space_cols])

        def __len__(self):
                return self.tcount

        def __nonzero__(self):
                if isinstance(self.data, tuple):
                        return 0 < int(self.data[0].size)
                else:
                        return 0 < int(self.data.size)

        def clear_cache(self):
                self.cache = None


class Locations(Cell):
        """

        """
        __slots__ = ()

        @property
        def locations(self):
                """
                `array-like`, property

                Locations as a matrix of coordinates and times with as many 
                columns as dimensions; this is an alias for :attr:`~Local.data`.
                """
                return self.data

        @locations.setter
        def locations(self, tr):
                self.data = tr

        @property
        def t(self):
                """
                `array-like`, ro property

                Location timestamps.
                """
                return self.time_data

        @t.setter
        def t(self, t):
                self.time_data = t

        @property
        def r(self):
                """
                `array-like`, ro property

                Location coordinates; `xy` is an alias
                """
                return self.space_data

        @r.setter
        def r(self, r):
                self.space_data = r

        @property
        def xy(self):
                return self.space_data

        @xy.setter
        def xy(self, xy):
                self.space_data = xy


class Translocations(Cell):
        """
        Attributes:

                origins (array-like, ro property):
                        Initial locations (both spatial coordinates and times).

        """
        __slots__ = ('origins', 'destinations')

        def __init__(self, index, translocations, center=None, span=None, boundary=None):
                Cell.__init__(self, index, translocations, center, span, boundary)
                self.origins = None
                self.destinations = None

        @property
        def translocations(self):
                """
                `array-like`, property

                Translocations as a matrix of variations of coordinate and time with as many 
                columns as dimensions; this is an alias for :attr:`~Local.data`.
                """
                return self.data

        @translocations.setter
        def translocations(self, tr):
                self.data = tr

        @property
        def dt(self):
                """
                `array-like`, ro property

                Translocation durations.
                """
                return self.time_data

        @dt.setter
        def dt(self, dt):
                self.time_data = dt

        @property
        def dr(self):
                """
                `array-like`, ro property

                Translocation displacements in space; `dxy` is an alias.
                """
                return self.space_data

        @dr.setter
        def dr(self, dr):
                self.space_data = dr

        @property
        def dxy(self):
                return self.space_data

        @dxy.setter
        def dxy(self, dxy):
                self.space_data = dxy

        @property
        def t(self):
                """
                `array-like`, ro property

                Initial timestamps.
                """
                if isstructured(self.origins):
                        return np.asarray(self.origins[self.time_col])
                else:
                        return np.asarray(self.origins[:,self.time_col])



def identify_columns(points, trajectory_col=True):
        _has_trajectory = trajectory_col not in [False, None]
        _traj_undefined = trajectory_col is True

        if isstructured(points):
                coord_cols = columns(points)
                if _has_trajectory:
                        if _traj_undefined:
                                trajectory_col = 'n'
                        try:
                                if isinstance(coord_cols, pd.Index):
                                        coord_cols = coord_cols.drop(trajectory_col)
                                else:
                                        coord_cols.remove(trajectory_col)
                        except (KeyboardInterrupt, SystemExit):
                                raise
                        except:
                                if _traj_undefined:
                                        _has_trajectory = False
                                else:
                                        raise
        else:
                if not _has_trajectory:
                        coord_cols = np.arange(points.shape[1])
                elif _traj_undefined:
                        trajectory_col = 0
                        coord_cols = np.arange(1, points.shape[1])
                else:
                        coord_cols = np.r_[np.arange(trajectory_col), \
                                np.arange(trajectory_col+1, points.shape[1])]
        if not _has_trajectory:
                trajectory_col = None

        if isinstance(points, pd.DataFrame):
                def get_point(a, i):
                        return a.iloc[i]
                def get_var(a, j):
                        return a[j]
        else:
                if isstructured(points):
                        def get_point(a, i):
                                return a[i,:]
                        def get_var(a, j):
                                return a[j]
                else:
                        def get_point(a, i):
                                return a[i]
                        def get_var(a, j):
                                return a[:,j]

        return coord_cols, trajectory_col, get_var, get_point


def get_locations(points, index=None, coord_cols=None, get_var=None, get_point=None):
        if coord_cols is None:
                coord_cols, _, get_var, get_point = identify_columns(points)
        locations = get_var(points, coord_cols)
        location_cell = format_cell_index(index, 'array', nearest_cell, (locations.shape[0],))
        return locations, location_cell, get_point


def get_translocations(points, index=None, coord_cols=None, trajectory_col=True, get_var=None, get_point=None):
        if coord_cols is None:
                coord_cols, trajectory_col, get_var, get_point = identify_columns(points, trajectory_col)
        if trajectory_col is None:
                raise ValueError('cannot find trajectory indices')
        n = np.asarray(get_var(points, trajectory_col))
        final = np.r_[False, np.diff(n, axis=0) == 0]
        initial = np.r_[final[1:], False]

        # points
        points = get_var(points, coord_cols)
        initial_point = get_point(points, initial)
        final_point = get_point(points, final)

        # cell indices
        if index is None:
                initial_cell = final_cell = None

        elif isinstance(index, np.ndarray):
                initial_cell = index[initial]
                final_cell = index[final]

        elif isinstance(index, tuple):
                ix = np.full(points.shape[0], -1, dtype=index[1].dtype)
                if isinstance(points, pd.DataFrame):
                        ix = pd.DataFrame(data=ix, index=points.index, columns=('cell',))
                        ix.loc[index[0], 'cell'] = index[1]
                        initial_cell = np.asarray(ix['cell'].iloc[initial])
                        final_cell = np.asarray(ix['cell'].iloc[final])
                else:
                        ix[index[0]] = index[1]
                        initial_cell = ix[initial]
                        final_cell = ix[final]

        elif sparse.issparse(index): # sparse matrix
                index = index.tocsr(True)
                initial_cell = index[initial].indices # and not cells.cell_index!
                final_cell = index[final].indices

        else:
                raise ValueError('wrong index format')

        return initial_point, final_point, initial_cell, final_cell, get_point


def distributed(cells, new_cell=None, new_group=Distributed, fuzzy=None,
                new_cell_kwargs={}, new_group_kwargs={}, fuzzy_kwargs={},
                new=None, verbose=False):
        """
        Build a `Distributed`-like object from a :class:`~tramway.tessellation.base.CellStats` object.

        Arguments:

                cells (CellStats): tessellation and partition; must contain (trans-)location data.

                new_cell (callable): cell constructor; default is :class:`Locations` or 
                        :class:`Translocations` depending on the data in `cells`.

                new_group (callable): constructor for groups of cell; default is :class:`Distributed`.

                fuzzy (callable): (trans-)location-cell weighting function.

                new_cell_kwargs (dict): keyword arguments for `new_cell`.

                new_group_kwargs (dict): keyword arguments for `new_group`.

                fuzzy_kwargs (dict): keyword arguments for `fuzzy`.

                new (callable): legacy argument; use `new_group` instead.

        """
        if new is not None:
                # `new` is for backward compatibility
                warn('`new` is deprecated in favor of `new_group`', DeprecationWarning)
                new_group = new
        if not isinstance(cells, tessellation.CellStats):
                raise TypeError('`cells` is not a `CellStats`')

        # format (trans-)locations
        precomputed = ()
        if isinstance(new_cell, Locations):
                are_translocations = False
        elif isinstance(new_cell, Translocations):
                are_translocations = True
        else:
                coord_cols, trajectory_col, get_var, get_point = identify_columns(cells.points)
                are_translocations = trajectory_col is not None
                if are_translocations:
                        precomputed = (coord_cols, trajectory_col, get_var, get_point)
                else:
                        precomputed = (coord_cols, get_var, get_point)
        if are_translocations:
                initial_point, final_point, initial_cell, final_cell, get_point = \
                        get_translocations(cells.points, cells.cell_index, *precomputed)
                if new_cell is None:
                        new_cell = Translocations
                fuzzy_args = (final_point, final_cell, initial_point, initial_cell, get_point)
        else:
                locations, location_index, get_point = \
                        get_locations(cells.points, cells.cell_index, *precomputed)
                fuzzy_args = (locations, location_index, get_point)
                if new_cell is None:
                        new_cell = Locations

        # assign/weight (trans-)locations to cells
        if fuzzy is None:
                if are_translocations:
                        def f(tessellation, cell, final_point, final_cell,
                                initial_point, initial_cell, get_point):
                                ## example handling:
                                #j = np.logical_or(initial_cell == cell, final_cell == cell)
                                #initial_point = get_point(initial_point, j)
                                #final_point = get_point(final_point, j)
                                #initial_cell = initial_cell[j]
                                #final_cell = final_cell[j]
                                #i = final_cell == cell # criterion i could be more sophisticated
                                #j[j] = i
                                #return j
                                return initial_cell == cell
                else:
                        def f(tesselation, cell, locations, location_cell, get_point):
                                return location_cell == cell
                fuzzy = f

        # time and space columns in (trans-)locations array
        if isstructured(cells.points):
                time_col = 't'
                space_cols = columns(cells.points)
                if 'n' in space_cols:
                        not_space = ['n', time_col]
                else:
                        not_space = [time_col]
                if isinstance(space_cols, pd.Index):
                        space_cols = space_cols.drop(not_space)
                else:
                        space_cols = [ c for c in space_cols if c not in not_space ]
        else:
                time_col = cells.points.shape[1] - 1
                if time_col == cells.points.shape[1] - 1:
                        space_cols = np.arange(time_col)
                else:
                        space_cols = np.ones(cells.points.shape[1], dtype=bool)
                        space_cols[time_col] = False
                        space_cols, = space_cols.nonzero()

        # pre-select cells
        if cells.tessellation.cell_label is None:
                J = 0 < cells.location_count
        else:
                J = np.logical_and(0 < cells.location_count, 0 < cells.tessellation.cell_label)

        # select (with the fuzzy filter) and pre-build cells
        _fuzzy, data, hull = {}, {}, {}
        if are_translocations:
                extra = {}
        for j, ok in enumerate(J): # for each cell
                if not ok:
                        continue

                # find (trans-)locations for cell j
                i = fuzzy(cells.tessellation, j, *fuzzy_args, **fuzzy_kwargs)
                if i.dtype in (bool, np.bool, np.bool8, np.bool_):
                        _fuzzy[j] = None
                else:
                        _fuzzy[j] = i[i != 0]
                        i = i != 0

                if are_translocations:
                        _origin = get_point(initial_point, i)
                        _destination = get_point(final_point, i)
                        __origin = _origin.copy() # make copy
                        __origin.index += 1
                        _points = _origin # for convex hull; ideally not only origins
                        points = _destination - __origin # translocations
                else:
                        points = _points = get_point(locations, i) # locations

                # convex hull
                _points = np.asarray(get_var(_points, space_cols))
                if _points.shape[1] < _points.shape[0]:
                        hull[j] = scipy.spatial.qhull.ConvexHull(_points)

                if points.size == 0:
                        J[j] = False
                else:
                        if are_translocations:
                                extra[j] = (_origin, _destination)
                        data[j] = points

        # simplify the adjacency matrix
        try:
                _adjacency = cells.tessellation.diagonal_adjacency
        except AttributeError:
                _adjacency = None
        _adjacency = cells.tessellation.simplified_adjacency(adjacency=_adjacency, label=J, format='csr')
        # reweight each row i as 1/n_i where n_i is the degree of cell i
        n = np.diff(_adjacency.indptr)
        _adjacency.data[...] = np.repeat(1.0 / np.maximum(1, n), n)

        # build every cells
        J, = np.nonzero(J)
        _cells = OrderedDict()
        for j in J: # for each cell
                try:
                        center = cells.tessellation.cell_centers[j]
                except AttributeError:
                        center = span = None
                else:
                        adj = _adjacency[j].indices
                        span = cells.tessellation.cell_centers[adj] - center

                # make cell object
                _cells[j] = new_cell(j, data[j], center, span, hull.get(j, None), **new_cell_kwargs)
                _cells[j].time_col = time_col
                _cells[j].space_cols = space_cols
                if are_translocations:
                        try:
                                _cells[j].origins = extra[j][0]
                        except AttributeError: # `_cells` does not have the `origins` attribute
                                pass
                        try:
                                _cells[j].destinations = extra[j][1]
                        except AttributeError: # `_cells` does not have the `destinations` attribute
                                pass
                try:
                        _cells[j].fuzzy = _fuzzy[j]
                except AttributeError: # `_cells` does not have the `fuzzy` attribute
                        pass

        # perform a few last checks
        if verbose:
                if hull:
                        for j in J:
                                if j not in hull:
                                        print('cannot compute convex hull for cell {}'.format(j))
                else:
                        print('cannot compute convex hulls')

        #print(sum([ c.tcount == 0 for c in _cells.values() ]))
        self = new_group(_cells, _adjacency, **new_group_kwargs)
        self.tcount = cells.points.shape[0]
        #self.dim = cells.points.shape[1]
        self.ccount = self.adjacency.shape[0]

        return self


class Maps(Lazy):
        """
        Basic container for maps and the associated parameters used to get the maps.

        Attributes `distributed_translocations`, `partition_file`, `tessellation_program`, `version`
        are deprecated and will be removed.

        Functional dependencies:

        * setting `maps` unsets `_variables`

        """
        __lazy__ = Lazy.__lazy__ + ('_variables',)

        def __init__(self, maps, mode=None):
                Lazy.__init__(self)
                self.maps = maps
                self.mode = mode
                self.min_diffusivity = None
                self.localization_error = None
                self.diffusivity_prior = None
                self.potential_prior = None
                self.jeffreys_prior = None
                self.extra_args = None
                self.distributed_translocations = None # legacy attribute
                self.partition_file = None # legacy attribute
                self.tessellation_param = None # legacy attribute
                self.version = None # legacy attribute
                self.runtime = None

        @property
        def maps(self):
                return self._maps

        @maps.setter
        def maps(self, m):
                if m is None:
                        self._variables = None
                else:
                        if not isinstance(m, pd.DataFrame):
                                raise TypeError('DataFrame expected')
                        self._variables = splitcoord(m.columns)
                self._maps = m

        @property
        def variables(self):
                """
                `list` or ``None``

                List of different variable names without the coordinate suffix.

                For example, if `maps` has columns *diffusivity*, *force x* and *force y*, `variables`
                will return *diffusivity* and *force*.
                """
                if self._variables is None:
                        return None
                else:
                        return list(self._variables.keys())

        def __nonzero__(self):
                return self.maps is not None

        def __len__(self):
                return 0 if self._variables is None else len(self._variables)

        def __contains__(self, variable_name):
                return variable_name in self._variables

        def __getitem__(self, variable_name):
                try:
                        return self.maps[self._variables[variable_name]]
                except (TypeError, KeyError):
                        raise KeyError("no such map: '{}'".format(variable_name))


class OptimizationWarning(RuntimeWarning):
        pass


class DiffusivityWarning(RuntimeWarning):
        def __init__(self, diffusivity, lower_bound=None):
                self.diffusivity = diffusivity
                self.lower_bound = lower_bound

        def __repr__(self):
                if self.lower_bound is None:
                        return 'DiffusivityWarning({})'.format(self.diffusivity)
                else:
                        return 'DiffusivityWarning({}, {})'.format(self.diffusivity, self.lower_bound)

        def __str__(self):
                if self.lower_bound is None:
                        return 'wrong diffusivity value: {}'.format(self.diffusivity)
                else:
                        return 'diffusivity too low: {} < {}'.format(self.diffusivity, self.lower_bound)


def smooth_infer_init(cells, min_diffusivity=None, jeffreys_prior=None, **kwargs):
        """
        Initialize the diffusivity array for translocations.

        This function is a collection of checks and initial computations suitable for *infer* functions
        in inference plugins.

        Arguments:

                cells (Distributed): first input argument of the *infer* function.

                min_diffusivity (float): minimum allowed diffusivity value.

                jeffreys_prior (bool): activate Jeffreys' prior.

        Returns:

        * *index* (:class:`numpy.ndarray`) -- cell indices corresponding to arrays *n*, *dt_mean* and *D_initial*.
        * *reverse_index* (:class:`numpy.ndarray`) -- element indices in arrays *index*, *n*, *dt_mean*, *D_initial* for each cell index.
        * *n* (:class:`numpy.ndarray`) -- translocation count for each cell.
        * *dt_mean* (:class:`numpy.ndarray`) -- mean translocation duration for each cell.
        * *D_initial* (:class:`numpy.ndarray`) -- initial diffusivity array.
        * *min_diffusivity* (:class:`float` or ``None``) -- minimum diffusivity value; the returned value depends on `jeffreys_prior`.
        * *D_bounds* (:class:`list`) -- list of (lower bound, upper bound) couples; suitable as *bounds* input argument for the :func:`scipy.optimize.minimize` function.
        * *border* (:class:`numpy.ndarray`) -- MxD boolean matrix with M the number of cells and D the number of space dimensions

        """
        if min_diffusivity is None:
                if jeffreys_prior:
                        min_diffusivity = 0.01
                else:
                        min_diffusivity = 0
        elif min_diffusivity is False:
                min_diffusivity = None

        # initial values and sanity checks
        index, n, dt_mean, D_initial, border = [], [], [], [], []
        reverse_index = np.full(cells.adjacency.shape[0], -1, dtype=int)

        j = 0
        for i in cells:
                cell = cells[i]

                assert i == cell.index
                if not bool(cell):
                        raise ValueError('empty cells')

                # ensure that translocations are properly oriented in time
                if not np.all(0 < cell.dt):
                        warn('translocation dts are not all positive', RuntimeWarning)
                        cell.dr[cell.dt < 0] *= -1.
                        cell.dt[cell.dt < 0] *= -1.

                # check cell i has neighbours
                try:
                        adjacent = np.vstack([ cells[c].center for c in cells.adjacency.indices[cells.adjacency.indptr[i]:cells.adjacency.indptr[i+1]] if cells[c] ])
                except ValueError:
                        continue

                # initialize the local diffusivity parameter
                dt_mean_i = np.mean(cell.dt)
                D_initial_i = np.mean(cell.dr * cell.dr) / (2. * dt_mean_i)
                #

                index.append(i)
                reverse_index[i] = j
                j += 1
                n.append(float(len(cell)))
                dt_mean.append(dt_mean_i)
                D_initial.append(D_initial_i)

                # border
                border.append(np.logical_or( np.max(adjacent) <= cell.center, cell.center <= np.min(adjacent) ))

        n, dt_mean, D_initial = np.array(n), np.array(dt_mean), np.array(D_initial)
        D_bounds = [(min_diffusivity, None)] * D_initial.size
        border = np.vstack(border)

        return index, reverse_index, n, dt_mean, D_initial, min_diffusivity, D_bounds, border


def gradn(cells, i, X, index_map=None):
        """
        Local gradient by multi-dimensional 2 degree polynomial interpolation.

        Arguments:

                cells (Distributed):
                        distributed cells.

                i (int):
                        cell index at which the gradient is evaluated.

                X (array):
                        vector of a scalar measurement at every cell.

                index_map (array):
                        index map that converts cell indices to indices in X.

        Returns:

                array:
                        local gradient vector with as many elements as there are dimensions
                        in the (trans-)location data.
        """
        cell = cells[i]

        # below, the measurement is renamed Y and the coordinates are X
        Y = X
        X0 = cell.center

        # cache the components related to X
        if not isinstance(cell.cache, dict):
                cell.cache = {}
        try:
                poly = cell.cache['poly']

        except KeyError: # fill in the cache
                # find neighbours
                adjacent = _adjacent = cells.neighbours(i)
                if index_map is not None:
                        i = index_map[i]
                        adjacent = index_map[_adjacent]
                        ok = 0 <= adjacent
                        assert np.all(ok)
                        adjacent, _adjacent = adjacent[ok], _adjacent[ok]
                if adjacent.size:
                        X0_dup = _adjacent.size
                        local = np.r_[np.full(X0_dup, i), adjacent]
                        # neighbour locations
                        X = np.vstack([ X0 ] * X0_dup + [ cells[j].center for j in _adjacent ])
                        # cache X-related terms and indices of the neighbours
                        poly = cell.cache['poly'] = (_poly2(X), local)
                else:
                        poly = cell.cache['poly'] = None

        if poly is None:
                return

        V, local = poly
        Y = Y[local]

        # interpolate
        W, _, _, _ = np.linalg.lstsq(V, Y, rcond=None)
        gradX = _poly2_deriv_eval(W, X0)

        return gradX


def _poly2(X):
        n, d = X.shape
        x = X.T
        return np.hstack((
                np.vstack([ x[u] * x[v] for v in range(d) for u in range(d) ]).T,
                X,
                np.ones((n, 1), dtype=X.dtype),
                ))

def _memoize1(f):
        results = {}
        def helper(x):
                try:
                        y = results[x]
                except KeyError:
                        y = results[x] = f(x)
                return y
        return helper

@_memoize1
def _poly2_meta_deriv(dim):
        poly = []
        for d in range(dim):
                p = []
                for v in range(dim):
                        for u in range(dim):
                                if u == d:
                                        if v == d:
                                                p.append((2., u))
                                        else:
                                                p.append((1., v))
                                else:
                                        if v == d:
                                                p.append((1., u))
                                        else:
                                                p.append((0., None))
                for u in range(dim):
                        if u == d:
                                p.append((1., None))
                        else:
                                p.append((0., None))
                p.append((0., None))
                poly.append(p)
        return poly

def _poly2_deriv_eval(W, X):
        if not X.shape[1:]:
                X = X[np.newaxis, :]
        dim = X.shape[1]
        dP = _poly2_meta_deriv(dim)
        Q = []
        for Pd in dP:
                Qd = 0.
                i = 0
                for q, u in Pd:
                        if q == 0:
                                continue
                        if u is None:
                                Qd += q * W[i]
                        else:
                                Qd += q * W[i] * X[:,u]
                        i += 1
                Q.append(Qd)
        return np.hstack(Q)


def neighbours_per_axis(i, cells, eps=None, centers=None):
        """
        """
        if centers is None:
                centers = np.vstack([ cells[j].center for j in cells.neighbours(i) ])

        cell = cells[i]
        center = cell.center
        below = np.zeros((centers.shape[1], centers.shape[0]), dtype=bool)
        above = np.zeros_like(below)

        if eps is None:
                # InferenceMAP-compatible gradient

                if cells.adjacency.dtype == bool:
                        # divide the space in non-overlapping region and
                        # assign each neighbour to a single region;
                        # assign each pair of symmetric regions to a single
                        # dimension (gradient calculation)
                        proj = centers - center[np.newaxis, :]
                        assigned_dim = np.argmax(np.abs(proj), axis=1)
                        proj_dist = proj[ np.arange(proj.shape[0]), assigned_dim ]
                        for j in range(cell.dim):
                                below[ j, (assigned_dim == j) & (proj_dist < 0) ] = True
                                above[ j, (assigned_dim == j) & (0 < proj_dist) ] = True

                elif cells.adjacency.dtype == int:
                        # neighbour cells are already classified
                        # as left (-1), right (1), top (2) or bottom (-2), etc
                        # in the adjacency matrix
                        code = cells.adjacency.data[cells.adjacency.indptr[i]:cells.adjacency.indptr[i+1]] - 1
                        for j in range(cell.dim):
                                below[ j, code == -j ] = True
                                above[ j, code ==  j ] = True

                else:
                        raise TypeError('{} adjacency labels are not supported'.format(cells.adjacency.dtype))

        else:
                # along each dimension, divide the space in half-spaces
                # minus margin `eps` (supposed to be positive)
                for j in range(cell.dim):
                        x, x0 = centers[:,j], center[j]
                        below[ j, x < x0 - eps ] = True
                        above[ j, x0 + eps < x ] = True

        return below, above


def grad1(cells, i, X, index_map=None, eps=None):
        """
        Local gradient by 2 degree polynomial interpolation along each dimension independently.

        Claims cache variable *grad1*.

        Supports the InferenceMAP way of dividing the space in non-overlapping regions (quadrants in 2D)
        so that each neighbour is involved in the calculation of a single component of the gradient.

        This is the default behaviour in cases where the `adjacent` attribute contains adequate
        labels, i.e. ``-j`` and ``j`` for each dimension ``j``, representing one side and the other
        side of cell-i's center.
        This is also the default behaviour otherwise, when `eps` is ``None``.

        If `eps` is defined, the calculation of a gradient component may recruit all the points
        minus those at a projected distance smaller than this value.

        See also:

                `neighbours_per_axis`.

        Arguments:

                cells (Distributed):
                        distributed cells.

                i (int):
                        cell index at which the gradient is evaluated.

                X (array):
                        vector of a scalar measurement at every cell.

                index_map (array):
                        index map that converts cell indices to indices in X;
                        for a given cell, should be constant.

                eps (float):
                        minimum projected distance from cell-i's center for neighbours to be considered
                        in the calculation of the gradient;
                        if `eps` is ``None``, the space is instead divided in twice as many
                        non-overlapping regions as dimensions, and each neighbour is assigned to
                        a single region (counts against a single dimension)

        Returns:

                array:
                        local gradient vector with as many elements as there are dimensions
                        in the (trans-)location data.
        """
        cell = cells[i]
        # below, the measurement is renamed y and the coordinates are X
        y = X
        X0 = cell.center

        # cache neighbours (indices and center locations)
        if not isinstance(cell.cache, dict):
                cell.cache = {}
        try:
                i, adjacent, X = cell.cache['grad1']
        except KeyError:
                adjacent = _adjacent = cells.neighbours(i)
                if index_map is not None:
                        adjacent = index_map[_adjacent]
                        ok = 0 <= adjacent
                        assert np.all(ok)
                        #adjacent, _adjacent = adjacent[ok], _adjacent[ok]
                if _adjacent.size:
                        X = np.vstack([ cells[j].center for j in _adjacent ])
                        below, above = neighbours_per_axis(i, cells, eps, X)

                        # pre-compute the X terms for each dimension
                        X_neighbours = []
                        for j in range(cell.dim):
                                u, v = below[j], above[j]
                                if not np.any(u):
                                        u = None
                                if not np.any(v):
                                        v = None

                                if u is None:
                                        if v is None:
                                                Xj = None
                                        else:
                                                Xj = 1. / (X0[j] - np.mean(X[v,j]))
                                elif v is None:
                                        Xj = 1. / (X0[j] - np.mean(X[u,j]))
                                else:
                                        Xj = np.r_[X0[j], np.mean(X[u,j]), np.mean(X[v,j])]

                                X_neighbours.append((u, v, Xj))

                        X = X_neighbours
                else:
                        X = []

                if index_map is not None:
                        i = index_map[i]
                cell.cache['grad1'] = (i, adjacent, X)

        if not X:
                return None

        y0, y = y[i], y[adjacent]

        # compute the gradient for each dimension separately
        grad = []
        for u, v, Xj in X: # j= dimension index
                #u, v, Xj= below, above, X term
                if u is None:
                        if v is None:
                                grad_j = 0.
                        else:
                                # 1./Xj = X0[j] - np.mean(X[v,j])
                                grad_j = (y0 - np.mean(y[v])) * Xj
                elif v is None:
                        # 1./Xj = X0[j] - np.mean(X[u,j])
                        grad_j = (y0 - np.mean(y[u])) * Xj
                else:
                        # Xj = np.r_[X0[j], np.mean(X[u,j]), np.mean(X[v,j])]
                        grad_j = _vander(Xj, np.r_[y0, np.mean(y[u]), np.mean(y[v])])
                grad.append(grad_j)

        return np.asarray(grad)


def _vander(x, y):
        #P = poly.polyfit(x, y, 2)
        #dP = poly.polyder(P)
        #return poly.polyval(x[0], dP)
        _, b, a = poly.polyfit(x, y, 2)
        return b + 2. * a * x[0]



__all__ = ['Local', 'Distributed', 'Cell', 'Locations', 'Translocations', 'Maps',
        'identify_columns', 'get_locations', 'get_translocations', 'distributed',
        'DiffusivityWarning', 'OptimizationWarning', 'smooth_infer_init',
        'neighbours_per_axis', 'grad1', 'gradn']

