# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import os
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from ..core import *
from ..tesselation import *
from ..spatial.scaler import *
from ..spatial.translocation import *
from ..plot.mesh import *
from ..io import *
from warnings import warn
import six


hdf_extensions = ['.rwa', '.h5', '.hdf', '.hdf5']
imt_extensions = [ '.imt' + ext for ext in hdf_extensions ]
imt_extensions = [ hdf_extensions[0] ] + imt_extensions[1:]
fig_formats = ['png', 'pdf', 'ps', 'eps', 'svg']
sub_extensions = dict([ (a, a) for a in ['imt', 'vor', 'hpc', 'hcd', 'hpd'] ])


class UseCaseWarning(UserWarning):
	pass
class IgnoredInputWarning(UserWarning):
	pass


def tesselate(xyt_data, method='gwr', output_file=None, verbose=False, \
	scaling=False, time_scale=None, \
	knn=None, \
	ref_distance=None, rel_min_distance=0.8, rel_avg_distance=2.0, rel_max_distance=None, \
	min_cell_count=20, avg_cell_count=None, max_cell_count=None, strict_min_cell_size=None, \
	compress=False, \
	**kwargs):
	"""
	Tesselation from points series and partitioning.

	This helper routine is a high-level interface to the various tesselation techniques 
	implemented in TRamWAy.

	Arguments:
		xyt_data (str or matrix):
			Path to a `.trxyt` file or raw data in the shape of :class:`pandas.DataFrame` 
			(or any data format documented in :mod:`tramway.spatial.descriptor`).


		method ({'grid', 'kdtree', 'kmeans', 'gwr'}, optional):
			Tesselation method.
			See respectively 
			:class:`~tramway.tesselation.RegularMesh`, 
			:class:`~tramway.tesselation.KDTreeMesh`, 
			:class:`~tramway.tesselation.KMeansMesh` and 
			:class:`~tramway.tesselation.GasMesh`.

		output_file (str, optional):
			Path to a `.rwa` file. The resulting tesselation and data partition will be 
			stored in this file. If `xyt_data` is a path to a file and `output_file` is not 
			defined, then `output_file` will be adapted from `xyt_data` with extension 
			`.rwa`.

		verbose (bool, optional): Verbose output.

		scaling (bool or str, optional):
			Normalization of the data.
			Any of 'unitrange', 'whiten' or other methods defined in 
			:mod:`tramway.spatial.scaler`.

		time_scale (bool or float, optional): 
			If this argument is defined and intepretable as ``True``, the time axis is 
			scaled by this factor and used as a space variable for the tesselation (2D+T or 
			3D+T, for example).
			This is equivalent to manually scaling the ``t`` column and passing 
			``scaling=True``.

		knn (pair of ints, optional):
			After growing the tesselation, a minimum and maximum numbers of nearest 
			neighbors of each cell center can be used instead of the entire cell 
			population. Let's denote ``min_nn, max_nn = knn``. Any of ``min_nn`` and 
			``max_nn`` can be ``None``.
			If a single `int` is supplied instead of a pair, then `knn` becomes ``min_nn``.
			``min_nn`` enables cell overlap and any point may be associated with several
			cells.

		ref_distance (float, optional):
			Supposed to be the average translocation distance. Can be modified so that the 
			cells are smaller or larger.

		rel_min_distance (float, optional):
			Multiplies with `ref_distance` to define the minimum inter-cell distance.

		rel_avg_distance (float, optional):
			Multiplies with `ref_distance` to define an upper on the average inter-cell 
			distance.

		rel_max_distance (float, optional):
			Multiplies with `ref_distance` to define the maximum inter-cell distance.

		min_cell_count (int, optional):
			Minimum number of points per cell. Depending on the method, can be strictly
			enforced or interpreted as a hint.

		avg_cell_count (int, optional):
			Hint of the average number of points per cell. Per default set to four times
			`min_cell_count`.

		max_cell_count (int, optional):
			Maximum number of points per cell. This is used only by `kdtree`.


	Returns:
		tramway.tesselation.CellStats: A partition of the data with 
			:attr:`~tramway.tesselation.CellStats.tesselation` attribute set.


	Apart from the parameters defined above, extra input arguments are admitted and passed to the
	initializer of the selected tesselation method. See the individual documentation of these 
	methods for more information.

	"""
	if isinstance(xyt_data, six.string_types) or isinstance(xyt_data, list):
		xyt_data, xyt_path = load_xyt(xyt_data, return_paths=True, verbose=verbose)
	else:
		xyt_path = []
		warn('TODO: test direct data input', UseCaseWarning)
	
	if ref_distance:
		transloc_length = None
	else:
		transloc_xy = np.asarray(translocations(xyt_data))
		if transloc_xy.shape[0] == 0:
			raise ValueError('no translocation found')
		transloc_length = np.nanmean(np.sqrt(np.sum(transloc_xy * transloc_xy, axis=1)))
		if verbose:
			print('average translocation distance: {}'.format(transloc_length))
		ref_distance = transloc_length
	min_distance = rel_min_distance * ref_distance
	avg_distance = rel_avg_distance * ref_distance
	if rel_max_distance:
		# applies only to KDTreeMesh
		max_distance = rel_max_distance * ref_distance
		if method != 'kdtree':
			warn('`rel_max_distance` is relevant only with `kdtree`', IgnoredInputWarning)
	else:
		max_distance = None

	methods = dict(grid=RegularMesh, kdtree=KDTreeMesh, kmeans=KMeansMesh, gwr=GasMesh)
	constructor = methods[method]
	if not scaling:
		scaling = 'none'
	elif scaling is True:
		scaling = 'whiten'
	scalers = dict(none=Scaler, whiten=whiten, unit=unitrange)
	scaler = scalers[scaling]()

	n_pts = float(xyt_data.shape[0])
	if min_cell_count:
		min_probability = float(min_cell_count) / n_pts
	else:
		min_probability = None
		warn('undefined `min_cell_count`; not tested', UseCaseWarning)
	if not avg_cell_count:
		avg_cell_count = 4 * min_cell_count
	if avg_cell_count:
		avg_probability = float(avg_cell_count) / n_pts
	else:
		avg_probability = None
		warn('undefined `avg_cell_count`; not tested', UseCaseWarning)
	if max_cell_count:
		# applies only to KDTreeMesh
		max_probability = float(max_cell_count) / n_pts
		if method != 'kdtree':
			warn('`max_cell_count` is relevant only with `kdtree`', IgnoredInputWarning)
	else:
		max_probability = None

	colnames = ['x', 'y']
	if 'z' in xyt_data:
		colnames.append('z')
	if time_scale:
		colnames.append('t')
		scaler.factor = [('t', time_scale)]

	# initialize a Tesselation object
	tess = constructor(scaler, \
		min_distance=min_distance, \
		avg_distance=avg_distance, \
		max_distance=max_distance, \
		min_probability=min_probability, \
		avg_probability=avg_probability, \
		max_probability=max_probability, \
		**kwargs)

	# grow the tesselation
	tess.tesselate(xyt_data[colnames], verbose=verbose, **kwargs)

	# partition the dataset into the cells of the tesselation
	if knn is None:
		cell_index = tess.cellIndex(xyt_data, min_cell_size=strict_min_cell_size)
	else:
		if strict_min_cell_size is None:
			strict_min_cell_size = min_cell_count
		cell_index = tess.cellIndex(xyt_data, knn=knn, min_cell_size=strict_min_cell_size, \
			metric='euclidean')

	stats = CellStats(cell_index, points=xyt_data, tesselation=tess)

	stats.param['method'] = method
	if transloc_length:
		stats.param['transloc_length'] = transloc_length
	else:
		stats.param['ref_distance'] = ref_distance
	if min_distance:
		stats.param['min_distance'] = min_distance
	if avg_distance:
		stats.param['avg_distance'] = avg_distance
	if max_distance:
		stats.param['max_distance'] = max_distance
	if min_cell_count:
		stats.param['min_cell_count'] = min_cell_count
	if avg_cell_count:
		stats.param['avg_cell_count'] = avg_cell_count
	if max_cell_count:
		stats.param['max_cell_count'] = min_cell_count
	if knn:
		stats.param['knn'] = knn
	#if spatial_overlap: # deprecated
	#	stats.param['spatial_overlap'] = spatial_overlap
	if method == 'kdtree':
		if 'max_level' in kwargs:
			stats.param['max_level'] = kwargs['max_level']

	# save `stats`
	if output_file or xyt_path:
		if output_file is None:
			xyt_file, _ = os.path.splitext(xyt_path[0])
			imt_path = xyt_file + imt_extensions[0]
		else:
			imt_path, imt_ext = os.path.splitext(output_file)
			if imt_ext in hdf_extensions:
				imt_path = output_file
			else:
				imt_path += imt_extensions[0]
		if compress:
			stats_ = lightcopy(stats)
		else:
			stats_ = stats

		try:
			store = HDF5Store(imt_path, 'w', verbose and 1 < verbose)
			if verbose:
				print('writing file: {}'.format(imt_path))
			store.poke('cells', stats_)
			store.close()
		except:
			warn('HDF5 libraries may not be installed', ImportWarning)
			try:
				os.unlink(imt_path)
			except:
				pass

	return stats




def cell_plot(cells, xy_layer='voronoi', output_file=None, fig_format=None, \
	show=False, verbose=False, figsize=(24.0, 18.0), dpi=None, \
	point_count_hist=False, cell_dist_hist=False, point_dist_hist=False, \
	aspect=None):
	"""
	Partition plots.

	Plots a spatial representation of the tesselation and partition if data are 2D, and optionally
	histograms.

	Arguments:
		cells (str or CellStats):
			Path to a `.imt.rwa` file or :class:`~tramway.tesselation.CellStats` 
			instance.

		xy_layer ({None, 'delaunay', 'voronoi'}, optional):
			Overlay Delaunay or Voronoi graph over the data points. For 2D data only.

		output_file (str, optional):
			Path to a file in which the figure will be saved. If `cells` is a path and 
			`fig_format` is defined, `output_file` is automatically set.

		fig_format (str, optional):
			Any image format supported by :func:`matplotlib.pyplot.savefig`.

		show (bool, optional):
			Makes `cell_plot` show the figure(s) which is the default behavior if and only 
			if the figures are not saved.

		verbose (bool, optional): Verbose output.

		figsize (pair of floats, optional):
			Passed to :func:`matplotlib.pyplot.figure`. Applies only to the spatial 
			representation figure.

		dpi (int, optional):
			Passed to :func:`matplotlib.pyplot.savefig`. Applies only to the spatial 
			representation figure.

		point_count_hist (bool, optional):
			Plot a histogram of point counts (per cell). If the figure is saved, the 
			corresponding file will have sub-extension `.hpc`.

		cell_dist_hist (bool, optional):
			Plot a histogram of distances between neighbor centroids. If the figure is 
			saved, the corresponding file will have sub-extension `.hcd`.

		point_dist_hist (bool, optional):
			Plot a histogram of distances between points from neighbor cells. If the figure 
			is saved, the corresponding file will have sub-extension `.hpd`.

		aspect (str, optional):
			Aspect ratio. Can be 'equal'.

	Notes:
		See also :mod:`tramway.plot.mesh`.

	"""
	if isinstance(cells, CellStats):
		input_file = ''
	else:
		input_file = cells
		if isinstance(input_file, list):
			input_file = input_file[0]
		imt_path = input_file
		# copy-paste
		if os.path.isdir(imt_path):
			imt_path = os.listdir(imt_path)
			files, exts = zip(*os.path.splitext(imt_path))
			for ext in imt_extensions:
				if ext in exts:
					imt_path = imt_path[exts.index(ext)]
					break
			if isinstance(imt_path, list):
				imt_path = imt_path[0]
			auto_select = True
		elif os.path.isfile(imt_path):
			auto_select = False
		else:
			candidates = [ imt_path + ext for ext in imt_extensions ]
			candidates = [ f for f in candidates if os.path.isfile(f) ]
			if candidates:
				imt_path = candidates[0]
			else:
				raise IOError('no tesselation file found in {}'.format(imt_path))
			auto_select = True
		if auto_select and verbose:
			print('selecting {} as a tesselation file'.format(imt_path))

		# load the data
		input_file = imt_path
		try:
			hdf = HDF5Store(input_file, 'r')
			cells = hdf.peek('cells')
			hdf.close()
		except:
			warn('HDF5 libraries may not be installed', ImportWarning)

	# guess back some input parameters
	method_name = {RegularMesh: ('grid', 'grid', 'regular grid'), \
		KDTreeMesh: ('kdtree', 'k-d tree', 'k-d tree based tesselation'), \
		KMeansMesh: ('kmeans', 'k-means', 'k-means based tesselation'), \
		GasMesh: ('gwr', 'GWR', 'GWR based tesselation')}
	method_name, pp_method_name, method_title = method_name[type(cells.tesselation)]
	min_distance = cells.param.get('min_distance', 0)
	avg_distance = cells.param.get('avg_distance', None)
	min_cell_count = cells.param.get('min_cell_count', 0)

	# plot the data points together with the tesselation
	figs = []
	dim = cells.tesselation.cell_centers.shape[1]
	if dim == 2:
		fig = plt.figure(figsize=figsize)
		figs.append(fig)
		if 'knn' in cells.param: # if knn <= min_count, min_count is actually ignored
			plot_points(cells)
		else:
			plot_points(cells, min_count=min_cell_count)
		if aspect is not None:
			plt.gca().set_aspect(aspect)
		if xy_layer == 'delaunay':
			plot_delaunay(cells)
			plt.title(pp_method_name + ' based delaunay')
		elif xy_layer == 'voronoi':
			plot_voronoi(cells)
			plt.title(pp_method_name + ' based voronoi')
		else:
			plt.title(pp_method_name)


	print_figs = output_file or (input_file and fig_format)

	if print_figs:
		if output_file:
			filename, figext = os.path.splitext(output_file)
			if fig_format:
				figext = fig_format
			elif figext and figext[1:] in fig_formats:
				figext = figext[1:]
			else:
				figext = fig_formats[0]
		else:
			figext = fig_format
			filename, _ = os.path.splitext(input_file)
		subname, subext = os.path.splitext(filename)
		if subext and subext[1:] in sub_extensions.values():
			filename = subname
		if dim == 2:
			vor_file = '{}.{}.{}'.format(filename, sub_extensions['vor'], figext)
			if verbose:
				print('writing file: {}'.format(vor_file))
			fig.savefig(vor_file, dpi=dpi)


	if point_count_hist:
		# plot a histogram of the number of points per cell
		fig = plt.figure()
		figs.append(fig)
		plt.hist(cells.cell_count, bins=np.arange(0, min_cell_count*20, min_cell_count))
		plt.plot((min_cell_count, min_cell_count), plt.ylim(), 'r-')
		plt.title(method_title)
		plt.xlabel('point count (per cell)')
		if print_figs:
			hpc_file = '{}.{}.{}'.format(filename, 'hpc', figext)
			if verbose:
				print('writing file: {}'.format(hpc_file))
			fig.savefig(hpc_file)

	if cell_dist_hist:
		# plot a histogram of the distance between adjacent cell centers
		A = sparse.triu(cells.tesselation.cell_adjacency, format='coo')
		i, j, k = A.row, A.col, A.data
		label = cells.tesselation.adjacency_label
		if label is not None:
			i = i[0 < label[k]]
			j = j[0 < label[k]]
		pts = cells.tesselation.cell_centers
		dist = la.norm(pts[i,:] - pts[j,:], axis=1)
		fig = plt.figure()
		figs.append(fig)
		plt.hist(np.log(dist), bins=50)
		if avg_distance:
			dmin = np.log(min_distance)
			dmax = np.log(avg_distance)
			plt.plot((dmin, dmin), plt.ylim(), 'r-')
			plt.plot((dmax, dmax), plt.ylim(), 'r-')
		plt.title(method_title)
		plt.xlabel('inter-centroid distance (log)')
		if print_figs:
			hcd_file = '{}.{}.{}'.format(filename, sub_extensions['hcd'], figext)
			if verbose:
				print('writing file: {}'.format(hcd_file))
			fig.savefig(hcd_file)

	if point_dist_hist:
		adj = point_adjacency_matrix(cells, symetric=False)
		dist = adj.data
		fig = plt.figure()
		figs.append(fig)
		plt.hist(np.log(dist), bins=100)
		if avg_distance:
			dmin = np.log(min_distance)
			dmax = np.log(avg_distance)
			plt.plot((dmin, dmin), plt.ylim(), 'r-')
			plt.plot((dmax, dmax), plt.ylim(), 'r-')
		plt.title(method_title)
		plt.xlabel('inter-point distance (log)')
		if print_figs:
			hpd_file = '{}.{}.{}'.format(filename, sub_extensions['hpd'], figext)
			if verbose:
				print('writing file: {}'.format(hpd_file))
			fig.savefig(hpd_file)

	if show or not print_figs:
		plt.show()
	else:
		for fig in figs:
			plt.close(fig)


def find_imt(path, method=None, full_list=False):
	if not isinstance(path, list):
		path = [path]
	paths = []
	for p in path:
		if os.path.isdir(p):
			paths.append([ os.path.join(p, f) for f in os.listdir(p) if f.endswith('.rwa') ])
		else:
			if p.endswith('.rwa'):
				ps = [p]
			else:
				d, p = os.path.split(p)
				p, _ = os.path.splitext(p)
				if d:
					ps = [ os.path.join(d, f) for f in os.listdir(d) \
						if f.startswith(p) and f.endswith('.rwa') ]
				else:
					ps = [ f for f in os.listdir('.') \
						if f.startswith(p) and f.endswith('.rwa') ]
			paths.append(ps)
	paths = list(itertools.chain(*paths))
	found = False
	for path in paths:
		try:
			hdf = HDF5Store(path, 'r')
			try:
				cells = hdf.peek('cells')
				if isinstance(cells, CellStats) and \
					(method is None or cells.param['method'] == method):
					found = True
			except:
				pass
			hdf.close()
		except:
			pass
		if found: break
	if found:
		if full_list:
			path = paths
		return (path, cells)
	else:
		return (paths, None)
