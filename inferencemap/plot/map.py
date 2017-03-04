
from math import tan, atan2, degrees, radians
import numpy as np
import pandas as pd
import numpy.ma as ma
from inferencemap.tesselation import *
from inferencemap.inference import Distributed
from matplotlib.patches import Polygon, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.linalg as la
import scipy.sparse as sparse
from warnings import warn


def scalar_map_2d(cells, values):
	if isinstance(values, pd.DataFrame):
		if values.shape[1] != 1:
			warn('multiple parameters available; mapping first one only', UserWarning)
		values = values.iloc[:,0] # to Series
	#values = pd.to_numeric(values, errors='coerce')

	polygons = []
	if isinstance(cells, Distributed):
		ix, xy, ok = zip(*[ (i, c.center, 0 < c.tcount) for i, c in cells.cells.items() ])
		ix, xy, ok = np.array(ix), np.array(xy), np.array(ok)
		voronoi = scipy.spatial.Voronoi(xy)
		for c, r in enumerate(voronoi.point_region):
			if ok[c]:
				region = [ v for v in voronoi.regions[r] if 0 <= v ]
				if region[1:]:
					vertices = voronoi.vertices[region]
					polygons.append(Polygon(vertices, True))
	elif isinstance(cells, CellStats) and isinstance(cells.tesselation, Voronoi):
		xy = cells.tesselation.cell_centers
		ix = np.arange(xy.shape[0])
		ok = 0 < cells.cell_count
		for i in ix:
			vertices = cells.tesselation.boundaries(i)
			if vertices is not None:
				polygons.append(Polygon(vertices, True))
	else:
		raise TypeError('wrong type for `cells`: {}'.format(type(cells)))

	xy_min, xy_max = xy.min(axis=0), xy.max(axis=0)

	scalar_map = values.loc[ix[ok]].values

	#print(np.nonzero(~ok)[0])

	try:
		if np.any(np.isnan(scalar_map)):
			#print(np.nonzero(np.isnan(scalar_map)))
			warn('NaNs ; changing them into 0s', RuntimeWarning)
			scalar_map[np.isnan(scalar_map)] = 0
	except TypeError as e: # help debug
		print(scalar_map)
		print(scalar_map.dtype)
		raise e

	fig, ax = plt.gcf(), plt.gca() # before PatchCollection
	patches = PatchCollection(polygons, alpha=0.9)
	patches.set_array(scalar_map)
	ax.add_collection(patches)

	ax.set_xlim(xy_min[0], xy_max[0])
	ax.set_ylim(xy_min[1], xy_max[1])

	try:
		fig.colorbar(patches, ax=ax)
	except AttributeError as e:
		warn(e.args[0], RuntimeWarning)



def field_map_2d(cells, values, angular_width=30.0, overlay=False):
	force_amplitude = pd.Series(data=np.linalg.norm(np.asarray(values), axis=1), index=values.index)
	if not overlay:
		scalar_map_2d(cells, force_amplitude)
	ax = plt.gca()
	xmin, xmax = ax.get_xlim()
	ymin, ymax = ax.get_ylim()
	if ax.get_aspect() == 'equal':
		aspect_ratio = 1
	else:
		aspect_ratio = (xmax - xmin) / (ymax - ymin)
	# compute the distance between adjacent cell centers
	if isinstance(cells, Distributed):
		A = cells.adjacency
	elif isinstance(cells, CellStats) and isinstance(cells.tesselation, Delaunay):
		A = cells.tesselation.cell_adjacency
	A = sparse.triu(A, format='coo')
	I, J = A.row, A.col
	if isinstance(cells, Distributed):
		pts_i = np.stack([ cells.cells[i].center for i in I ])
		pts_j = np.stack([ cells.cells[j].center for j in J ])
	elif isinstance(cells, CellStats):
		assert isinstance(cells.tesselation, Tesselation)
		pts_i = cells.tesselation.cell_centers[I]
		pts_j = cells.tesselation.cell_centers[J]
	dist = la.norm(pts_i - pts_j, axis=1)
	# scale force amplitude
	scale = np.median(dist) / 2.0 / force_amplitude.median()
	# 
	dw = float(angular_width) / 2.0
	t = tan(radians(dw))
	t = np.array([[0.0, -t], [t, 0.0]])
	markers = []
	for i in values.index:
		try:
			center = cells.tesselation.cell_centers[i]
		except AttributeError:
			center = cells.cells[i].center
		radius = force_amplitude[i]
		f = np.asarray(values.loc[i]) * scale
		#fx, fy = f
		#angle = degrees(atan2(fy, fx))
		#markers.append(Wedge(center, radius, angle - dw, angle + dw))
		base, vertex = center + np.outer([-0.5, 0.5], f)
		ortho = np.dot(t, f)
		vertices = np.stack((vertex, base + ortho, base - ortho), axis=0)
		#vertices[:,0] = center[0] + aspect_ratio * (vertices[:,0] - center[0])
		markers.append(Polygon(vertices, True))

	patches = PatchCollection(markers, facecolor='y', edgecolor='k', alpha=0.9)
	ax.add_collection(patches)

