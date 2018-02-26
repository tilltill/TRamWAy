# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
import pandas as pd
from ..core.namedcolumns import *


class Scaler(object):
	""":class:`Scaler` scales data points, point differences (vectors) or distances.

	It initializes itself with the first provided sample (in :meth:`scale_point`), and then applies
	the same transformation to the next samples.

	A default `Scaler()` instance does not scale. However, initialization still takes place so that 
	:meth:`scaled` properly works.

	It manages a constraint in the calculation of the scaling parameters, forcing a common factors
	over a subset of dimensions. Attribute :attr:`euclidean` controls the selection of this subset.
        Distances are scaled and unscaled only in this subspace, if it is defined.

	Beware that when possible data are scaled in place, but `scaledonly` optional argument, when
	available, never operates in place.

	Attributes:
		init (bool):
			``True`` as long as `Scaler` has not been initialized.
		center (array or pandas.Series):
			Vector that is substracted to each row of the data matrix to be scaled.
		factor (array or pandas.Series):
			Vector by which each row of the data matrix to be scaled is divided.
			Applies after `center`.
		columns (list or pandas.Index):
			Sequence of column names along which scaling applies. This applies only to
			structured data. `columns` is determined even if `Scaler` is set to do nothing,
			so that :meth:`scaled` can still apply.
			`columns` can be manually set after the first call to :meth:`scale_point` if data
			are not structured (do not have named columns).
		function (callable):
			A function that takes a data matrix as input and returns `center` and `factor`.
			`function` is called once during the first call to :meth:`scale_point`.
		euclidian (list):
			Sequence of names or indices of the columns to be scaled by a common factor.
	"""
	__slots__ = ['init', 'center', 'factor', 'columns', 'function', 'euclidean']

	def __init__(self, scale=None, euclidean=None):
		"""
		Arguments:
			scale (callable):
				A function that takes a data matrix as input and returns `center` and 
				`factor`. `scale` becomes the :attr:`function` attribute.
			euclidian (list):
				Sequence of names or indices of the columns to be scaled by a common 
				factor.
		"""
		self.init   = True
		self.center = None
		self.factor = None
		self.columns = []
		self.function = scale
		if euclidean and not \
			(isinstance(euclidean, list) and euclidean[1:]):
			raise TypeError('`euclidean` should be a multi-element list')
		self.euclidean = euclidean

	@property
	def ready(self):
		"""Returns `True` if scaler is initialized."""
		return not self.init

	def scaled(self, points, asarray=False):
		"""Discard columns that are not recognized by the initialized scaler. 

		Applies to points and vectors, not distances, surface areas or volumes."""
		if (isinstance(self.columns, list) and self.columns) or self.columns.size:
			if isstructured(points):
				points = points[self.columns]
			else:
				if self.center is not None and isstructured(self.center):
					raise TypeError("input data are not structured whereas scaler' is")
				points = points[:, self.columns] 
		elif isstructured(points):
			raise ValueError("input data are structured whereas scaler' is not")
		else:
			scaler_data = self.center
			if scaler_data is None:
				scaler_data = self.factor
			if scaler_data is None:
				if self.function:
					raise RuntimeError('scaler has not been initialized')
			elif scaler_data.shape[1] != points.shape[1]:
				raise ValueError('number of columns does not match')
		if asarray:
			points = np.asarray(points)
		return points

	def scale_point(self, points, inplace=True, scaledonly=False, asarray=False):
		"""
		Scale data.

		When this method is called for the first time, the `Scaler` instance initializes itself 
		for further call of any of its methods.

		Arguments:
			points (array-like):
				Data matrix to be scaled. When :meth:`scale_point` is called for the
				first time, `points` can be structured or not, without the unnecessary 
				columns, if any.
				At further calls of any (un-)scaling method, data should be in the same
				format but may feature extra columns.
			inplace (bool):
				Per default, scaling is performed in-place. With ``inplace=False``, 
				`points` are first copied.
			scaledonly (bool):
				If ``True``, undeclared columns are stripped away out of the returned 
				data.
			asarray (bool):
				If ``True``, the returned data is formatted as a :class:`numpy.array`.

		Returns:
			array-like: With default optional input arguments, the returned variable will be
				a pointer to `points`, not otherwise.
		"""
		if self.init:
			# define named columns
			if self.columns:
				raise AttributeError('remove data columns at initialization instead of defining `columns`')
			try:
				self.columns = columns(points)
			except:
				pass
			# backup predefined values
			if self.center is None:
				predefined_centers = []
			elif isinstance(self.center, list):
				predefined_centers = self.center
			if self.factor is None:
				predefined_factors = []
			elif isinstance(self.factor, list):
				predefined_factors = self.factor
			if self.function:
				# calculate centers and factors
				self.center, self.factor = self.function(points)
				# equalize factor for euclidian variables
				if self.euclidean:
					if isinstance(points, pd.DataFrame):
						xyz = points[self.euclidean].values
					elif points.dtype.names:
						xyz = np.asarray(points[self.euclidian])
					else:
						xyz = points[:,self.euclidean]
					_, self.factor[self.euclidean] = self.function(xyz.flatten())
			# overwrite the coordinates that were actually predefined
			if predefined_centers:
				if self.center is None:
					self.center = __get_row(points, 0.0)
				for col, val in predefined_centers:
					self.center[col] = val
			if predefined_factors:
				if self.factor is None:
					self.factor = __get_row(points, 1.0)
				for col, val in predefined_factors:
					self.factor[col] = val
			self.init = False
		if not (self.center is None and self.factor is None):
			if not inplace:
				points = points.copy()
			if self.center is not None:
				points -= self.center
			if self.factor is not None:
				points /= self.factor
		if scaledonly:
			points = self.scaled(points, asarray)
		elif asarray:
			points = np.asarray(points)
		return points

	def unscale_point(self, points, inplace=True):
		"""
		Scale data back to original domain.

		The calling `Scaler` instance must have been initialized.

		Arguments:
			points (array-like):
				Scaled data matrix to be unscaled.
			inplace (bool):
				Per default, scaling is performed in-place. With ``inplace=False``, 
				`points` are first copied.

		Returns:
			array-like: unscaled data matrix.
		"""
		if self.init:
			raise AttributeError('scaler has not been initialized')
		if not (self.center is None and self.factor is None):
			if not inplace:
				points = points.copy(deep=False)
			if self.factor is not None:
				points *= self.factor
			if self.center is not None:
				points += self.center
		return points


	def scale_vector(self, vect, inplace=True, scaledonly=False, asarray=False):
		"""
		Scale vectors.

		The calling `Scaler` instance must have been initialized.

		Arguments:
			vect (array-like):
				Data matrix to be scaled.
			inplace (bool):
				Per default, scaling is performed in-place. With ``inplace=False``, 
				`vect` is first copied.
			scaledonly (bool):
				If ``True``, undeclared columns are stripped away out of the returned 
				data.
			asarray (bool):
				If ``True``, the returned data is formatted as a :class:`numpy.array`.

		Returns:
			array-like: scaled data matrix.
		"""
		if self.init:
			raise AttributeError('scaler has not been initialized')
		if self.factor is not None:
			if not inplace:
				vect = vect.copy(deep=False)
			vect /= self.factor
		if scaledonly:
			vect = self.scaled(vect, asarray)
		elif asarray:
			vect = np.asarray(vect)
		return vect

	def unscale_vector(self, vect, inplace=True):
		"""
		Scale vectors back to original range.

		The calling `Scaler` instance must have been initialized.

		Arguments:
			vect (array-like):
				Scaled data matrix to be unscaled.
			inplace (bool):
				Per default, scaling is performed in-place. With ``inplace=False``, 
				`points` are first copied.

		Returns:
			array-like: unscaled data matrix.
		"""
		if self.init:
			raise AttributeError('scaler has not been initialized')
		if self.factor is not None:
			if not inplace:
				vect = vect.copy(deep=False)
			vect *= self.factor
		return vect

	def scale_size(self, size, dim=None, inplace=True, _unscale=False):
		"""
		Scale/unscale lengths, surface areas, volumes and other scalar sizes.

		The calling `Scaler` instance must have been initialized.

		Arguments:
			size (array-like):
				Values to be scaled, per element.
			dim (int):
				Number of characteristic dimensions, with 0 referring to all the
				euclidean dimensions (e.g. lengths: 1, areas: 2, volumes: 0).
			inplace (bool):
				Per default, scaling is performed in-place. With ``inplace=False``, 
				`size` is first copied.
			_unscale (bool):
				If ``True``, unscales instead.

		Returns:
			array-like: scaled values.
		"""
		if self.init:
			raise AttributeError('scaler has not been initialized')
		if self.factor is not None:
			_dim = len(self.euclidean)
			if not dim:
				dim = _dim
			if _dim < min(1, dim):
				raise ValueError('not enough euclidean dimensions')
			factor = self.factor[self.euclidean[0]]
			if self.euclidean[1:] and not np.all(self.factor[self.euclidean[1:]] == factor):
				raise ValueError('the scaling factors for the euclidean variables are not all equal')
			if not inplace:
				size = size.copy(deep=False)
			if 1 < dim:
				factor **= dim
			if _unscale:
				size *= factor
			else:
				size /= factor
		return size

	def scale_distance(self, dist, inplace=True):
		return self.scale_size(dist, 1, inplace)

	def unscale_distance(self, dist, inplace=True):
		return self.scale_size(dist, 1, inplace, True)

	def scale_length(self, dist, inplace=True):
		return self.scale_size(dist, 1, inplace)

	def unscale_length(self, dist, inplace=True):
		return self.scale_size(dist, 1, inplace, True)

	def scale_surface_area(self, area, inplace=True):
		return self.scale_size(area, 2, inplace)

	def unscale_surface_area(self, area, inplace=True):
		return self.scale_size(area, 2, inplace, True)

	def scale_volume(self, vol, inplace=True):
		return self.scale_size(vol, 0, inplace)

	def unscale_volume(self, vol, inplace=True):
		return self.scale_size(vol, 0, inplace, True)


def _whiten(x):
	'''Scaling function for :class:`Scaler`. Performs ``(x - mean(x)) / std(x)``. Consider using
	:func:`whiten` instead.'''
	scaling_center = x.mean(axis=0)
	scaling_factor = x.std(axis=0)
	return (scaling_center, scaling_factor)

def whiten(): # should be a function so that each new instance is a distinct one
	"""Returns a :class:`Scaler` that scales data ``x`` following: ``(x - mean(x)) / std(x)``."""
	return Scaler(_whiten)


def _unitrange(x):
	'''Scaling function for :class:`Scaler`. Performs ``(x - min(x)) / (max(x) - min(x))``. 
	Consider using :func:`unitrange` instead.'''
	scaling_center = x.min(axis=0)
	scaling_factor = x.max(axis=0) - scaling_center
	return (scaling_center, scaling_factor)

def unitrange():
	"""Returns a :class:`Scaler` that scales data ``x`` following: 
	``(x - min(x)) / (max(x) - min(x))``."""
	return Scaler(_unitrange)


def __get_row(points, fill=None):
	if isinstance(points, pd.DataFrame):
		row = points.iloc[0]
	else:
		row = points[0]
	if fill is not None:
		row.fill(fill)
	return row
