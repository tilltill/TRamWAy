# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

"""This module implements the :class:`~rwa.storable.Storable` class for TRamWAy datatypes."""

import rwa
from rwa import HDF5Store, hdf5_storable, hdf5_not_storable
import sys
from .store import *

rwa.hdf5_agnostic_modules += ['tramway.core.analyses', 'tramway.core.scaler', \
	'tramway.tessellation', 'tramway.inference', 'tramway.feature']

if sys.version_info[0] < 3:
	from .rules import *


# backward compatibility trick for `CellStats`
try:
	from tramway.tessellation.base import CellStats
except:
	pass
else:
	from ..lazy import Lazy
	cell_stats_expose = Lazy.__slots__ + CellStats.__slots__ + ('_tesselation',)
	rwa.hdf5_storable(rwa.default_storable(CellStats, exposes=cell_stats_expose), agnostic=True)

try:
	from tramway.tessellation.kdtree.dichotomy import Dichotomy, ConnectedDichotomy
except:
	pass
else:
	dichotomy_exposes = ['base_edge', 'min_depth', 'max_depth', \
		'origin', 'lower_bound', 'upper_bound', \
		'min_count', 'max_count', 'subset', 'cell']
	rwa.hdf5_storable(rwa.generic.kwarg_storable(Dichotomy, dichotomy_exposes), agnostic=True)
	connected_dichotomy_exposes = dichotomy_exposes + ['adjacency']
	rwa.hdf5_storable(rwa.generic.kwarg_storable(ConnectedDichotomy, connected_dichotomy_exposes), agnostic=True)


try:
	from tramway.inference.base import Maps
except:
	pass
else:

	def poke_maps(store, objname, self, container, visited=None, legacy=False):
		#print('poke_maps')
		sub_container = store.newContainer(objname, self, container)
		attrs = dict(self.__dict__) # dict
		#print(list(attrs.keys()))
		if legacy:
			# legacy format
			if callable(self.mode):
				store.poke('mode', '(callable)', sub_container)
				store.poke('result', self.maps, sub_container)
			else:
				store.poke('mode', self.mode, sub_container)
				store.poke(self.mode, self.maps, sub_container)
		else:
			#print("poke 'maps'")
			store.poke('maps', self.maps, sub_container, visited=visited)
			del attrs['maps']
		deprecated = {}
		for a in ('distributed_translocations','partition_file','tessellation_param','version'):
			deprecated[a] = attrs.pop(a, None)
		for a in attrs:
			if attrs[a] is not None:
				if attrs[a] or attrs[a] == 0:
					#print("poke '{}'".format(a))
					store.poke(a, attrs[a], sub_container, visited=visited)
		for a in deprecated:
			if deprecated[a]:
				warn('`{}` is deprecated'.format(a), DeprecationWarning)
				#print("poke '{}'".format(a))
				store.poke(a, deprecated[a], sub_container, visited=visited)

	def peek_maps(store, container):
		#print('peek_maps')
		read = []
		mode = store.peek('mode', container)
		read.append('mode')
		try:
			maps = store.peek('maps', container)
			read.append('maps')
		except KeyError:
			# former standalone files
			if mode == '(callable)':
				maps = store.peek('result', container)
				read.append('result')
				mode = None
			else:
				maps = store.peek(mode, container)
				read.append(mode)
		maps = Maps(maps, mode=mode)
		for r in container:
			if r not in read:
				setattr(maps, r, store.peek(r, container))
		return maps

	rwa.hdf5_storable(rwa.Storable(Maps, \
		handlers=rwa.StorableHandler(poke=poke_maps, peek=peek_maps)), agnostic=True)

from ..analyses import Analyses, base
_analyses_storable = rwa.default_storable(Analyses, exposes=base.Analyses.__slots__)
_analyses_storable.storable_type = 'tramway.core.analyses.Analyses'
rwa.hdf5_storable(_analyses_storable)
