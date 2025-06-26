# -*- coding: utf-8 -*-
"""
RingsPy utilities and functions

author: king_yin3613
email: haoyin2022@u.northwestern.edu
"""

import math
import bisect
import numpy as np
from scipy.spatial.distance import cdist
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpltPath
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d
import triangle as tr
from pathlib import Path
import datetime
import pkg_resources
import time
import shapely.plotting
import shapely.ops
import shapely as shp
from bisect import bisect_left, bisect_right
import FreeCAD as App # type: ignore
# import cProfile
# import pstats
import os
# from line_profiler import LineProfiler



def calc_knotflow(y,z,m1,m2,a1,a2,Uinf,yc):

    dy1 = -m1/2/np.pi*(y-yc)/((z-a1)**2+(y-yc)**2)
    dz1 = -m1/2/np.pi*(z-a1)/((z-a1)**2+(y-yc)**2)    
    dy2 = m2/2/np.pi*(y-yc)/((z-a2)**2+(y-yc)**2)
    dz2 = m2/2/np.pi*(z-a2)/((z-a2)**2+(y-yc)**2)

    dydz = (dy1+dy2)/(Uinf + dz1 + dz2)
    return dydz

def calc_knotstream(y,z,m1,m2,a1,a2,Uinf):

    psi_frstr = Uinf*y
    psi_snk = -m1/(2*np.pi)*np.arctan2((y),(z-a1))
    psi_src = m2/(2*np.pi)*np.arctan2((y),(z-a2))

    psi = psi_frstr + psi_src + psi_snk

    return psi

def sort_coordinates(coords):
    # from https://pavcreations.com/clockwise-and-counterclockwise-sorting-of-coordinates/
    x = coords[:,0]
    y = coords[:,1]
    cx = np.mean(x)
    cy = np.mean(y)
    angles = np.arctan2(x-cx, y-cy)
    indices = np.argsort(angles)
    return coords[indices,:]

def check_iscollinear(p1,p2,boundaries):
    """ Make sure the line segment x1-x2 is collinear with 
        any line segment of the boundary polygon
    """
    collinear = 0
    if (p2[0]-p1[0]) == 0: # inf slope
        k1 = 99999999 
    else:
        k1 = (p2[1]-p1[1])/(p2[0]-p1[0]) # slope of the new line segment
        
    for boundary in boundaries: # loop over boundary lines
        bp1 = boundary[0]
        bp2 = boundary[1]
        
        if (bp2[0]-bp1[0]) == 0: # inf slope
            k2 = 99999999
        else:
            k2 = (bp2[1]-bp1[1])/(bp2[0]-bp1[0]) # slope of the boundary line segment
            
        if math.isclose(k1,k2,rel_tol=1e-05, abs_tol=1e-10): # check if slopes are equal (with a tolerance)
            p3 = (p1+p2)/2 # mid point of the new line
            if k2 == 99999999: # inf slope
                p3_on = (p3[0] == bp1[0])
            else:
                p3_on = math.isclose((p3[1] - bp1[1]),k2*(p3[0] - bp1[0]),rel_tol=1e-05, abs_tol=1e-10)
            p3_between = (min(bp1[0], bp2[0]) <= p3[0] <= max(bp1[0], bp2[0])) and (min(bp1[1], bp2[1]) <= p3[1] <= max(bp1[1], bp2[1]))
            if (p3_on and p3_between): # check if mid point of the new line is on the boundary line segment
                collinear += 1
                
    return collinear

def rotate_around_point_highperf(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point.
    
    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy

def relax_points(vor,omega):

    filtered_regions = []
    
    nonempty_regions = list(filter(None, vor.regions))
    filtered_regions = [region for region in nonempty_regions if not any(x < 0 for x in region)]
    # filtered_regions = non boundary regions
    
    centroids = np.zeros((len(filtered_regions),2))
    for i in range(0,len(filtered_regions)):   
        vertices = vor.vertices[filtered_regions[i] + [filtered_regions[i][0]], :]
        centroid = find_centroid(vertices,omega) # get the centroid of these verts
        centroids[i,:] = centroid

    return centroids # store the centroids as the new site positions

def find_centroid(vertices,omega):
    # https://en.wikipedia.org/wiki/Centroid#Of_a_polygon

    area = 0
    centroid_x = 0
    centroid_y = 0

    # Vectorized calculation of step
    steps = (vertices[:-1, 0] * vertices[1:, 1]) - (vertices[1:, 0] * vertices[:-1, 1])
    steps *= omega  # Apply relaxation factor

    # Sum up the area and centroid contributions
    area = np.sum(steps)
    centroid_x = np.sum((vertices[:-1, 0] + vertices[1:, 0]) * steps)
    centroid_y = np.sum((vertices[:-1, 1] + vertices[1:, 1]) * steps)

    area /= 2
    centroid_x /= (6.0 * area)
    centroid_y /= (6.0 * area)
    
    return np.array([[centroid_x, centroid_y]])

def find_intersect(p,normal,boundaries):

    # Find the intersect point of infinite ridges (rays) with the boundary lines
    # Ref: https://stackoverflow.com/questions/14307158/how-do-you-check-for-
    # intersection-between-a-line-segment-and-a-line-ray-emanatin#:~:text=
    # Let%20r%20%3D%20(cos%20%CE%B8%2C,0%20%E2%89%A4%20u%20%E2%89%A4%201).&
    # text=Then%20your%20line%20segment%20intersects,0%20%E2%89%A4%20u%20%E2%89%A4%201.

    intersect_points = []
    for boundary in boundaries: # loop over boundary lines
        q = np.asarray(boundary[0])
        s = np.asarray(boundary[1]) - np.asarray(boundary[0])
        if np.cross(normal,s) == 0: # parallel
            t = np.inf
            u = np.inf
        else:
            t = np.cross((q-p),s)/np.cross(normal,s)
            u = np.cross((q-p),normal)/np.cross(normal,s)
        if (u >= 0) and (u <= 1):
            if (t >= 0) and math.isfinite(t):
                t_final = t
                intersect_point = p + normal * t_final
                intersect_points.append(intersect_point)
    if not intersect_points:
        return np.empty([0,2])
    elif np.shape(intersect_points)[0] > 1:
        # vector crosses multiple boundaries, get the closest one as adhoc solution
        intersect_points = np.reshape(intersect_points,(np.shape(intersect_points)[0],2))
        dists = ( (intersect_points.T[0] - p[0][0])**2 + (intersect_points.T[1] - p[0][1])**2)**0.5
        close_ind = np.argmin(dists)
        close_intersect = np.array([intersect_points[close_ind]])
        return np.reshape(close_intersect,(1,2))
    else:
        # print(np.asarray(intersect_points))
        return np.reshape(intersect_points,(1,2))

def check_isinside(points,boundary_points,buf):

    """
    checks if points are inside poly path created by boundaries
    """
    
    boundary = shp.Polygon((boundary_points))
    boundary = boundary.buffer(buf)
    points = shp.points(points)
    points_in =  shp.within(points,boundary)
    
    return np.array(points_in)



def Clipping_Box(box_shape,box_center,box_size,box_width,box_depth,x_notch_size,y_notch_size,form):
    """
    clipping box for the delauney points (generated cell points)
    TBD: adjust to general shape using polygon or similar
    """
    ax = plt.gca()
    if box_shape == "cube": # cube box
        x_min = box_center[0] - box_size/2
        x_max = box_center[0] + box_size/2 
        y_min = box_center[1] - box_size/2
        y_max = box_center[1] + box_size/2
        boundary_points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        
        points = sort_coordinates(boundary_points)
        pgon = shp.Polygon(points)
        seg_pgon = shp.segmentize(pgon,0.1)
        boundary_points = np.array(seg_pgon.exterior.coords)
        # print(boundary_points)
        
    elif box_shape =="rectangle": # rectangular box
        x_min = box_center[0] - box_width/2
        x_max = box_center[0] + box_width/2 
        y_min = box_center[1] - box_depth/2
        y_max = box_center[1] + box_depth/2
        boundary_points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

    elif box_shape == "notchedsquare": # notched square box (can be rectangular)
        x_min = box_center[0] - box_width/2
        x_max = box_center[0] + box_width/2 
        y_min = box_center[1] - box_depth/2
        y_max = box_center[1] + box_depth/2
        x_notch = x_min + x_notch_size
        y_notch_min = box_center[1] - y_notch_size/2
        y_notch_max = box_center[1] + y_notch_size/2
        boundary_points = np.array([[x_min, y_min],[x_max, y_min],[x_max, y_max],[x_min, y_max],[x_min, y_notch_max],[x_notch, y_notch_max],\
                            [x_notch,y_notch_min],[x_min,y_notch_min]])
                   
    elif box_shape == "input":
        boundary_points = []
        check = False
        with open(Path(form[1].geoFile.text()), "r") as geofil:
            for line in geofil:
                if check and "P(" in line:
                    coord = np.round(np.array(line.split("P")[1].strip(' ').strip('(').strip('\n').strip(')').split(' '),dtype='float'),5)
                    # print(coord)
                    boundary_points.append(coord[0:2])
                if "edges" in line:
                    check = True
                if "faces" in line:
                    check = False
        boundary_points = np.array(boundary_points)
        x_min = min(boundary_points[:,0])
        x_max = max(boundary_points[:,0])
        y_min = min(boundary_points[:,1])
        y_max = max(boundary_points[:,1])

    else:
        print('box_shape: {:s} is not supported for current version, please check README for more details.'.format(box_shape))
        # exit()
    
    ax.plot(boundary_points[:,0],boundary_points[:,1],'ro',markersize=3.0)

    points = sort_coordinates(boundary_points)
    pgon = shp.Polygon(points)
    # print('cross sectional area',pgon.area)
    bound_area = pgon.area

    # Relies on boundary points to be defined in order (i.e. cannot only be clockwise for notch)
    boundaries = np.concatenate((np.expand_dims(boundary_points,axis=1),np.roll(np.expand_dims(boundary_points,axis=1),-1,axis=0)),axis=1)

    return x_min,x_max,y_min,y_max,boundaries,boundary_points, bound_area


def CellPlacement_Binary_Lloyd(nrings,width_heart,width_sparse,width_dense,\
                               cellsize_sparse,cellsize_dense,iter_max,\
                               mergeFlag,boundary_points,omega):
    """
    packing cells by firstly placing new cells and then performing Lloyd's relaxation in the generation rings
    """

    # generate radii for rings
    radii = np.concatenate(([width_heart],np.tile([width_sparse,width_dense],nrings)))
    # noise = np.random.normal(1,0.0,len(radii))
    # noise = np.ones(len(radii))
    # radii = np.multiply(radii,noise)
    radii = np.concatenate(([0],np.cumsum(radii)))

    # generate perimeter points for heart region
    PerimeterPoints = []
    npoint = int(np.ceil(2*np.pi*radii[1]/(2*cellsize_dense)))
    t = np.linspace(0, 2*np.pi, npoint, endpoint=False)
    x = radii[1] * np.cos(t)
    y = radii[1] * np.sin(t)
    PerimeterPoints.append(np.c_[x, y])
    PerimeterPointsSites = np.concatenate(PerimeterPoints, axis=0)
    
    #generate internal points for heart region
    n_nonoverlapped_cells = int(2*np.floor((radii[1]/cellsize_dense)**2))
    inside_cells = 1e-1*(np.random.rand(n_nonoverlapped_cells,2) - [0.5,0.5]) # add cell points in a very small area
    # not too small or else flow mesh is too extreme at center, changed 1e-3 to 1e-1 -SA
    
    sites = np.vstack((PerimeterPointsSites,inside_cells))

    lloyd_iter = 0
    while lloyd_iter < iter_max:
        vor = Voronoi(sites)
        sites = relax_points(vor,omega) # returns new sites which are relaxed centroids of vor based on old sites
        sites = np.vstack((PerimeterPointsSites,sites))
        lloyd_iter += 1

    existing_sites = np.copy(sites[npoint:,:])
    
    # generate sites for each ring    
    for i in range(1,len(radii)-1):
        
        OuterPerimeterPoints = []
        OuterPerimeter_radii = radii[i+1]
        
        if (i % 2) == 0: # if even, dense cells
            cellradius = cellsize_dense/2
            aspect = 1
        else:
            cellradius = cellsize_sparse/2
            aspect = 1

        OuterPerimeter_npoint = int(np.ceil(2*np.pi*OuterPerimeter_radii/(2*cellradius)))
        
        t = np.linspace(0, 2*np.pi, OuterPerimeter_npoint, endpoint=False)
        x = OuterPerimeter_radii * np.cos(t)
        y = OuterPerimeter_radii * np.sin(t)
        OuterPerimeterPoints.append(np.c_[x, y])
        OuterPerimeterPointsSites = np.concatenate(OuterPerimeterPoints, axis=0)
        
        # generate internal points
        nsubrings = int(0.5*np.ceil((radii[i+1]-radii[i])/cellradius))
        subradii = np.linspace(radii[i], radii[i+1], int(aspect*nsubrings), endpoint=False)
        
        subnpoints = np.ceil(2*np.pi*subradii/(2*cellradius)).astype(int)
        # take minimum number of points for all subradii
        subpoints = int(np.min(subnpoints))
        
        circles = []
        radialwidthnoise = np.squeeze((np.random.rand(subpoints,1) - [0.5]))
        for subr in subradii:
            t = np.linspace(0, 2*np.pi, subpoints, endpoint=False)
            dt = t[1] - t[0]
            t = t + radialwidthnoise*dt
            x = subr * np.cos(t)
            y = subr * np.sin(t)
            circles.append(np.c_[x, y])
            
        inside_cells = np.concatenate(circles, axis=0)
        
        # radial randomness 
        noise = (np.random.rand(len(inside_cells),1) - [0.5])*cellradius*2
        noise = np.squeeze(noise)
        r = np.sqrt(inside_cells[:,0]**2 + inside_cells[:,1]**2) + noise
        theta = np.arctan2(inside_cells[:,1],inside_cells[:,0])
        inside_cells[:,0] = r*np.cos(theta)
        inside_cells[:,1] = r*np.sin(theta)

        sites = np.vstack((OuterPerimeterPointsSites,inside_cells))
                    
        existing_sites = np.vstack((sites,existing_sites))
        PerimeterPointsSites = np.copy(OuterPerimeterPointsSites)

    # cut sites to buffered boundary to make lloyd relaxation more efficient
    existing_sites = existing_sites[check_isinside(existing_sites,boundary_points,0.2)]

    # try doing the full triangulation and getting the vertices back
    num_bound = np.shape(boundary_points)[0]  # create boundary segements to enforce boundaries 
    boundary_segments = np.array([np.linspace(0,num_bound-1,num_bound),np.concatenate((np.linspace(1,num_bound-1,num_bound-1),np.array([0])))]).transpose()
    boundary_region = np.array([[5.75,0,1,0]])

    lloyd_iter = 0
    for i in range(0,iter_max):
        print('iter',i)
        vor = Voronoi(existing_sites)
        existing_sites = relax_points(vor,omega) # returns new sites which are relaxed centroids of vor based on old sites
        # try triangulating after relax
        delaunay_vertices = np.concatenate((boundary_points,np.array(existing_sites))) 
        tri_inp = {'vertices': delaunay_vertices,'segments':boundary_segments,'regions':boundary_region}
        conforming_delaunay = tr.triangulate(tri_inp, 'peq3D') 
        existing_sites = np.array(conforming_delaunay['vertices']) # flow point coords

    print('pass lloyd')
    # # check if sites are too close the the boundary 
    # if mergeFlag == 'On':
    #     num_bound = np.shape(boundary_points)[0]  # create boundary segements to enforce boundaries 
    #     boundary_segments = np.array([np.linspace(0,num_bound-1,num_bound),np.concatenate((np.linspace(1,num_bound-1,num_bound-1),np.array([0])))]).transpose()
    #     boundary_region = np.array([[0,0,1,0]])
    #     tri_inp = {'vertices': existing_sites,'segments':boundary_segments,'regions':boundary_region}
    #     conform = tr.triangulate(tri_inp, 'peAq0D')
    #     conform_sites = conform['vertices']
    #     existing_sites[check_isinside(conform_sites,boundary_points,-1e-2)]
    #     existing_sites = np.squeeze(existing_sites)

    # Clip again sites to be inside boundary
    path_in = check_isinside(existing_sites,boundary_points,0) # checks sites inside boundary using path
    sites = existing_sites[path_in]


    return sites, radii


def CellPlacement_Debug(nrings,width_heart,width_sparse,width_dense):
    """
    place single cell for debug
    """
    # generate radii for rings
    radii = np.concatenate(([width_heart],np.tile([width_sparse,width_dense],nrings)))
    noise = np.random.normal(1,0.25,len(radii))
    radii = np.multiply(radii,noise)
    radii = np.concatenate(([0],np.cumsum(radii)))

    # generate point at center    
    # sites = np.array([[.012,0.0125],[-0.0125,-0.012],[0.012,-0.012],[-0.0125,0.0125]])
    # sites = np.array([[-0.01,0.01],[0.01,-0.01]])
    sites = np.array([[0.001,0.015],[0.001,-0.01],[0.011,0.001],[-0.013,-0.001]])
    # sites = np.array([[0.1,0.1]])

    return sites, radii


def BuildFlowMesh(outDir, geoName,nsegments,long_connector_ratio,z_min,z_max,boundaries,conforming_delaunay):

    
    ax = plt.gca()
    ''' 
    NOTE: There are nsegment number of transverse layers, shifted to occur at 
    the mid-segment location, and there are nsegment+1 longitudinal layers.
    There are 2*nsegments+1 number of node layers.
    '''
    #***************************************************************************

    delaunay_pts = np.array(conforming_delaunay['vertices']) # flow point coords
    npts = len(delaunay_pts)

    # different package to calculate voronoi region info because it produces output in different form
    vorsci = Voronoi(conforming_delaunay.get('vertices')) # location of vor sites
    vortri = tr.voronoi(conforming_delaunay.get('vertices'))

    vortri_rayinds = vortri[2]
    vortri_vertices = vortri[0]
    vortri_raycoords = vortri_vertices[vortri_rayinds]
    vortri_raydirs = vortri[3]

    # ax.plot(vorsci.vertices[:,0],vorsci.vertices[:,1],'go',markersize=1.)
    # # ax.plot(vortri_vertices[:,0],vortri_vertices[:,1],'g^',markersize=3.)
    # plt.show()

    nrdgs = len(vorsci.ridge_points) # number of ridges and thus number of flow elements
    nlayers = nsegments + 2 # number of node layers (on per segment plus top and bottom layers)
    nnodes = npts*nlayers 
    nels_long = npts*(nsegments+1) # in between layers plus top and bottom
    nels_trans = nrdgs*nsegments 
    nels = nels_long + nels_trans
    delaun_elems_long = np.zeros((nels_long,20)) 
    delaun_elems_trans = np.zeros((nels_trans,20)) # empty arrays for flow info
    # n1, n2, l1, l2, area, volume, element type, v1, v2 ... vn

    segment_length = (z_max - z_min) / (nsegments + (nsegments-1)*long_connector_ratio) # beam segment length
    connector_l_length = segment_length*long_connector_ratio

    # define z coordinate for each node layer
    z_coord = z_min    
    coordsz = np.zeros((nnodes,1))
    for l in range(0,nlayers):
        for p in range(0,npts):
            coordsz[p+l*npts] = z_coord
        if (l == 0) or (l == nlayers-2): # the first or last layer
            z_coord += segment_length/2
        else: # for middle elements include connector distance
            z_coord += segment_length + connector_l_length

    # nodes
    delaun_num = np.zeros(((nnodes),1)) # delaun_num is hacky because of np dimension nonsense -SA
    delaun_num[:,0] = np.linspace(0,nnodes-1,nnodes,dtype='int64')
    delaun_verts_layers = np.tile(delaunay_pts,(nlayers,1)) # tile node xy coords for each layer
    delaun_nodes = np.concatenate((delaun_verts_layers,coordsz),axis=1)

    # longitudinal elements
    for l in range(0,nsegments+1):
        long_area = 0
        # get layer properties
        if (l == 0) or (l == nsegments): # first or last element layer
            typeFlag = 1
            el_len1 = segment_length/2
            el_len2 = 0
        else:
            typeFlag = 0
            el_len1 = (segment_length + connector_l_length)/2
            el_len2 = (segment_length + connector_l_length)/2 
            # for mid longitudinal elements they are always half and half, won't work for curved beams though -SA
        # for each element in layer
        el_len = el_len1 + el_len2 # in long direction, total length is just sum of lengths due to purely vertical direction
        for p in range(0,npts):
            nel_l = p + l*npts # element index
            nd = [p + l*npts, p + (l+1)*npts] # manual calculation of 3D node numbers

            pr = vorsci.point_region[p] # index of voronoi region
            pv = vorsci.regions[pr] # vertices of voronoi region
            refpt = (delaun_nodes[nd[0],0:2] + delaun_nodes[nd[1],0:2])/2 # average xy of top/bot of long element to get average xy location

            if (-1) in pv: # check if boundary cell, as noted by -1 for vertice index
                # this part gets the intersection of the infinite ray and a perpindicular line 
                # through the delaunay point (flow node), since for boundary cells the flow node is on the boundary,
                # then all intersection points are added to the region vertices to calculate the region area 
                # it could probably be done more efficiently -SA

                gen = (v for v in pv if v != -1)
                coords_in = np.empty((0,2)) # coordinates of voronoi vertices
                coords_out = np.empty((0,2)) # coordinates of new intersection vertices
                corner = False
                if np.shape(pv)[0] < 3: # check if a corner element with two rays from vertex (only case with only one real vertex)
                    corner = True
                for v in gen: # for every actual vertex of the cell
                    coord = vorsci.vertices[v]
                    path_in = check_isinside([coord],boundaries[:,1],0) # checks coordinate is inside
                    if path_in:
                        # for each vertex add to list of coordinates
                        coords_in = np.vstack([coords_in,coord])
                        # check if vertex is start of infinite ray
                        inds = np.where((np.isclose(vortri_raycoords,coord,rtol=1e-05, atol=1e-5)).all(axis=1)) 
                        if np.any(inds): # if the vertex is start of ray
                            int_pts = np.empty((np.size(inds),2))  # could be multiple boundary rays for single vertex (i.e. a corner)
                            # if infinite ray
                            if corner: # if corner with two rays from vertex
                                for i in range(0, np.shape(inds)[1]): # take both rays      
                                    # Find the intersect point of infinite ridges (rays) with the boundary lines
                                    ray_dir = vortri_raydirs[inds[0][i]]
                                    normal = ray_dir[:]/np.linalg.norm(ray_dir[:]) # normal
                                    intpts = find_intersect(coord,normal,boundaries)
                                    if intpts.any(): # double check there was an interesection
                                        int_pts[i,:] = intpts # add intersection point to coordinates
                                    else:
                                        print('(flowa)',coord)
                                        int_pts[i,:] = coord
                            else:
                                # only take first ray? *** how to pick which ray to take -SA coordsd as in transverse?
                                # Find the intersect point of infinite ridges (rays) with the boundary lines
                                ray_dir = vortri_raydirs[inds[0][0]]
                                normal = ray_dir[:]/np.linalg.norm(ray_dir[:]) # normal
                                intpts = find_intersect(coord,normal,boundaries)
                                if intpts.any(): # double check there was an interesection
                                    int_pts = intpts
                                else:
                                    print('(flowb)',coord)
                                    int_pts = coord
                            coords_out = np.vstack((coords_out,int_pts))
                    else: # if the point is outside, treat like a new infinite ridge ***************
                        print('bound cell')
                        # boundary cell for which one real vertex is outside, is that vertex needed for correct area?
                coords = np.vstack((coords_in,coords_out,refpt)) # combine interior with exterior nodes and boundary flow point for correct area 
                if np.shape(coords)[0] > 2:
                    # cells must be convex and greater than only two points
                    coords_sort = sort_coordinates(coords)
                    pgon = shp.Polygon(coords_sort)
                    flow_area = pgon.area
                    # shapely.plotting.plot_polygon(pgon)
                else:
                    flow_area = 0
                    print('too small?')

            else: # not infinite
                path_in = check_isinside(vorsci.vertices[pv],boundaries[:,1],0) # checks sites inside boundary using path
                if not path_in.all(): # if any coordinates are not in, cut cell to get actual area
                    # cut cell geometry by line of boundary points
                    bound = shp.LineString(boundaries[:,1])
                    coords = vorsci.vertices[pv]
                    coords_sort = sort_coordinates(coords) 
                    cell = shp.Polygon(coords_sort)
                    outcome = shapely.ops.split(cell, bound)
                    # take the resulting shape with maximum area (since that will always be the remaining cell? seems not true)
                    # should compare coordinates - SA
                    # areas = [pgon.area for pgon in outcome.geoms]
                    pgons = outcome.geoms
                    for pgon in pgons:
                        # get coordinates, including new coordinate of intersection point
                        coords = shapely.get_coordinates(pgon)
                        pgon_in = check_isinside(coords,boundaries[:,1],0)
                        if pgon_in.all():
                            flow_area = pgon.area
                            # print('in')
                            # shapely.plotting.plot_polygon(pgon)
                            pv = np.array(pv)[path_in] # only inside points for index connectivity
                        else:
                            flow_area = 0
                            # print('out')
                else: # if all the coordinates are in
                    coords = vorsci.vertices[pv] # coordinates of voronoi vertices
                    # calculate the flux area of voronoi region of arbitrary shape
                    coords_sort = sort_coordinates(coords) # cells must be convex
                    pgon = shp.Polygon(coords_sort)
                    flow_area = pgon.area
                    # shapely.plotting.plot_polygon(pgon)

            el_vol = el_len*flow_area/3 # element volume
            if el_vol != 0:
                delaun_elems_long[nel_l,0:2] = nd # element connectivity
                delaun_elems_long[nel_l,2] =  el_len1
                delaun_elems_long[nel_l,3] =  el_len2
                delaun_elems_long[nel_l,4] = flow_area
                delaun_elems_long[nel_l,5] =  el_vol
                delaun_elems_long[nel_l,6] = typeFlag
                delaun_elems_long[nel_l,7:7+len(pv)] = [v for v in pv] # voronoi indices of related region
                long_area += flow_area
                # if l == 1:
                #     ax.annotate('p{:d}'.format(p),refpt,size=5,color='b')
        # print('longitudinal area',long_area)
    # plt.show()

    # transverse elements
    for l in range(0,nsegments): # each segment
        tran_area = 0
        el_h = segment_length + connector_l_length # element height 
        typeFlag = 2
        for r in range(0,nrdgs):
            nel_t = r + l*nrdgs # element index

            pd = vorsci.ridge_points[r] # 2D flow index for ridge
            coordsd = delaunay_pts[pd] # 2D flow coords
            nd = pd + (l+1)*npts # translate 2D flow index to 3D flow index
            el_len = np.linalg.norm((coordsd[0,:]-coordsd[1,:])) # distance bewteen flow nodes (element length)

            pv = vorsci.ridge_vertices[r] # 2D vor index for ridge
            if (-1) in pv: # check if inifite ridge, as noted by -1 for vertice index
                v = pv[bisect_right(pv,-1)] # get non-negative index (finite ray)                
                coord = vorsci.vertices[v]
                path_in = check_isinside(np.array([coord]),boundaries[:,1],0) # checks if ridge inside boundary using path
                if path_in:
                    # get intercept of infinite ray by checking which infinite point the vertex is (comparing vortri and vorsci)
                    inds = np.where((np.isclose(vortri_raycoords,coord,rtol=1e-05, atol=1e-5)).all(axis=1)) 
                    # if infinite ray
                    for i in range(0, np.size(inds)):           
                        # Find the intersect point of infinite ridges (rays) with the boundary lines
                        ray_dir = vortri_raydirs[inds[0][i]]
                        normal = ray_dir[:]/np.linalg.norm(ray_dir[:]) # normal
                        # all infinite rays cross boundary at delauney line of ridge? should be -SA
                        intpts = find_intersect(coord,normal,[coordsd]) # 
                        # if not intpts.any():
                            # print('(flow2)',coord) # means second ray of two ray vertex which wasn't in element
                        if intpts.any():
                            coordsv = np.vstack((coord,intpts)) # combine finite and infinite points
                            vor_len = np.linalg.norm((coordsv[0,:]-coordsv[1,:])) # distance between vor coords (for area)

                    coords_pgon = np.vstack((coordsd[0,:],coordsv[0,:],coordsd[1,:],coordsv[1,:]))
                    pgon = shp.Polygon(coords_pgon)
                    # shapely.plotting.plot_polygon(pgon)
                    # tran_area += pgon.area
                else:
                    vor_len = 0

            else: # if finite ridge
                coords = vorsci.vertices[pv]
                path_in = check_isinside(coords,boundaries[:,1],0) # checks if ridge fully inside boundary using path
                if path_in.all(): # if both vertices are inside the boundary
                    coordsv = coords # coordinates of voronoi vertices
                    vor_len = np.linalg.norm((coordsv[0,:]-coordsv[1,:])) # distance between vor coords (for area)

                    coords_pgon = np.vstack((coordsd[0,:],coordsv[0,:],coordsd[1,:],coordsv[1,:]))
                    pgon = shp.Polygon(coords_pgon)
                    # shapely.plotting.plot_polygon(pgon)
                    # tran_area += pgon.area

                elif path_in.any(): # if only one vertice is inside the boundary, treat as inifite ridge
                    v = [v for v in pv if check_isinside(np.array([vorsci.vertices[v]]),boundaries[:,1],0)] # index of inside vertex
                    d = [v for v in pv if not check_isinside(np.array([vorsci.vertices[v]]),boundaries[:,1],0)] # index of outside vertex
                    coord = vorsci.vertices[v] # coordinates of inside vertex
                    # create direction of ridge
                    ray_dir = vorsci.vertices[d] - vorsci.vertices[v]     # outside coords minus inside coords
                    normal = ray_dir/np.linalg.norm(ray_dir) # normal
                    intpts = find_intersect(coord,normal,[coordsd]) # 
                    coordsv = np.vstack((coord,intpts)) # combine finite and infinite points 
                    vor_len = np.linalg.norm((coordsv[0,:]-coordsv[1,:])) # distance between vor coords (for area)

                    coords_pgon = np.vstack((coordsd[0,:],coordsv[0,:],coordsd[1,:],coordsv[1,:]))
                    pgon = shp.Polygon(coords_pgon)
                    # shapely.plotting.plot_polygon(pgon)
                    # tran_area += pgon.area
                else: # finite ridge totally outside
                    vor_len = 0

            # calculate each length of the element
            coordc = (coordsv[0,:]+coordsv[1,:])/2 # average vor coords to get center of ridge
            el_len1 = np.linalg.norm((coordsd[0,:]-coordc)) # distance between flow node and center of ridge
            el_len2 = np.linalg.norm((coordsd[1,:]-coordc))
            flow_area = vor_len*el_h # flux area
            el_vol = el_len*flow_area/3 # element volume (2D area*length*1/3*2 for two pyramids)
            if el_vol != 0:
                delaun_elems_trans[nel_t,0:2] = nd # element connectivity
                delaun_elems_trans[nel_t,2] =  el_len1
                delaun_elems_trans[nel_t,3] =  el_len2
                delaun_elems_trans[nel_t,4] = flow_area
                delaun_elems_trans[nel_t,5] = el_vol
                delaun_elems_trans[nel_t,6] = typeFlag # element type
                delaun_elems_trans[nel_t,7] = pv[0] # voronoi indices of related ridge
                delaun_elems_trans[nel_t,8] = pv[1] #
                tran_area += 0.5*(el_len)*vor_len
                # if l == 0:
                #     ax.plot(coordsv[:,0],coordsv[:,1],'bs-',markersize=3)
                #     ax.annotate('r{:d}'.format(r),coordc,size=5,color='g')
        # print('transverse area',tran_area) # error using formulated, correct using shapely
    
    # remove zero volume rows
    delaun_elems_long = delaun_elems_long[~np.all(delaun_elems_long == 0, axis=1)]
    delaun_elems_trans = delaun_elems_trans[~np.all(delaun_elems_trans == 0, axis=1)]
    delaun_elems = np.concatenate((delaun_elems_long,delaun_elems_trans))
    delaun_elems[:,0:2] += 1 # use 1-base for meshing
    nels = np.shape(delaun_elems)[0]

    # plt.show()
    total_vol = sum(delaun_elems[:,5])
    print('total flow volume','{:.2e}'.format(total_vol))
    

    with open(Path(outDir + '/' + geoName + '/' + geoName +'-flowMesh.inp'), 'w') as meshfile:
        meshfile.write('*Heading\n*Preprint, echo=NO, model=NO, history=NO, contact=NO\n*Part, name=')
        meshfile.write(geoName)
        meshfile.write('\n*Node, nset=all\n')
        i = 0
        for row in delaun_nodes:
            i += 1
            meshfile.write(str(int(i)) + ', ' + ', '.join([str(np.format_float_positional(a, precision=10, unique=False, fractional=False, trim='k')) for a in row]) + '\n')
        meshfile.write('*Element, type=U1, elset=all\n')
        i = 0
        for row in delaun_elems:
            i += 1
            meshfile.write(str(int(i)) + ', ' + ', '.join([str(int(a)) for a in row[0:2]]) + '\n')
        meshfile.write('*End Part\n')
    
    # add node index for visualization
    ncount = np.arange(0,nnodes,1,dtype=int)
    delaun_nodes = np.hstack((np.expand_dims(ncount.transpose(),axis=1),delaun_nodes))
    
    # np.save(Path(outDir + '/' + geoName + '/' + geoName +'-flowElements.npy'), delaun_elems)
    np.savetxt(Path(outDir + '/' + geoName + '/' + geoName +'-flowElements.dat'), delaun_elems, fmt = \
               ['%d','%d','%0.8f','%0.8f','%0.8f','%0.8f','%d','%d','%d','%d','%d','%d','%d','%d','%d','%d','%d','%d','%d','%d']\
        ,header='Flow Mesh Element Information\nn = '+ str(nels) + '\n[n1 n2 l1 l2 A V type v1 v2 ... vn]')
    
    delaun_elems[:,0:2] -= 1 # convert back to 0 base for later indexing

    return delaun_nodes, delaun_elems


def RebuildVoronoi_ConformingDelaunay_New(ttvertices,ttedges,ttray_origins,ttray_directions,\
                                      boundaries,boundaryFlag,boundary_points_original):
    """Clip Voronoi mesh by the boundaries, rebuild the new Voronoi mesh
        Modified to fix optional addition of boundary corner points - SA
        NOTE: 'tt' prefix to refer to original tessellation, while w/o prefix refers to boundary conforming tesselation)
    """

    # Store indices of Voronoi vertices for each finite ridge
    finite_ridges = [] 
    voronoi_vertices_in = [] 
    infinite_ridges = [] 
    ray_origins = []
    ray_directions = []
    boundary_points = []

    # Remove points outside the boundary
    vertices_in = ttvertices[check_isinside(ttvertices,boundary_points_original,0)] # list of vertices inside boundaries

    # Construct infinite ridges based on ridges that were cut by the boundary
    for ridge in ttedges: # for finite ridges
        points = ttvertices[ridge] # coordinates of ridge vertices
        points_check = check_isinside(points,boundary_points_original,0) # check check if fully in
        if all(points_check): # if fully in
            # get new index for vertices
            inds = [np.where(np.all(vertices_in == points[0],axis=1))[0],np.where(np.all(vertices_in == points[1],axis=1))[0]]
            finite_ridges.append([inds[0][0],inds[1][0]]) # add to finite ridges with new vertex index
        # possibly can make this next part better but for now just checking the two cases uniquely 
        # checking finite ridges which have become infinite due to boundary
        elif points_check[0] and not points_check[1]: # if only the first point is inside
            # now becomes an infinite ray
            ind = np.where(np.all(vertices_in == points[0],axis=1))[0]
            infinite_ridges.append((ind[0],-1)) # add to infinite ridges
            # get origin and define direction
            ray_origins.append(ind) # append vertex index to ray_origins list
            direction = points[1] - points[0]
            ray_directions.append(direction)
        elif points_check[1] and not points_check[0]: # if the second point is in
            ind = np.where(np.all(vertices_in == points[1],axis=1))[0]
            infinite_ridges.append((ind[0],-1)) # add to infinite ridges
            # get origin and define direction
            ray_origins.append(ind) # append vertex index to ray_origins list
            direction = points[0] - points[1]
            ray_directions.append(direction)
    for origin in range(0,len(ttray_origins)): # for infinite ridges
        point_ind = ttray_origins[origin] # old index of originating point
        points_check = check_isinside([ttvertices[point_ind]],boundary_points_original,0) # check coords of orginating vertex
        if all(points_check): # if infinite ridge originates within the boundaries (i.e. original infinite ridge which is still an infinite ridge)
            ind = np.where((vertices_in == ttvertices[point_ind]).all(axis=1))[0]
            ind = ind[0]
            infinite_ridges.append((ind,-1))
            ray_origins.append([ind]) # append new vertex index to ray_origins list
            ray_directions.append(ttray_directions[origin]) # get original ray_direction and append

    # Store and update arrays
    finite_ridges_new = np.asarray(finite_ridges)
    nfinite_ridge = finite_ridges_new.shape[0]

    voronoi_vertices_in = vertices_in
    nvertices_in = voronoi_vertices_in.shape[0]

    infinite_ridges_new = np.asarray(infinite_ridges)
    ninfinite_ridge = infinite_ridges_new.shape[0]

    ray_directions = np.asarray(ray_directions) 
    ray_origins = np.asarray(ray_origins) 

    # shouldn't be necessary with new method, need to figure out why - SA
    # Reconstruct the connectivity for ridges since the unique operation rearrange the order of vertices 
    for i in range(0,ninfinite_ridge): # loop over voronoi ridges in original infinite_ridge list
        infinite_ridges_new[i,1] = nvertices_in + i
        # ax.plot(voronoi_vertices_in[i,0],voronoi_vertices_in[i,1],'go',linewidth=2.)

    # Now find the intersect point of infinite ridges (rays) with the boundary lines
    # checks for points too close to boundary as well
    n = 0
    for i in range(0,len(ray_origins)):
        p = vertices_in[ray_origins[i]]
        normal = ray_directions[i,:]/np.linalg.norm(ray_directions[i,:]) # normal                 
        intpt = find_intersect(p,normal,boundaries)
        if intpt.any(): # if an intersection is found (should be the case but some bugs rn)
            # if mergeFlag == 'On': # if the merge operation is selected 
            #     dist = np.sum((p - intpt)**2,axis=1) # distance between ray origin and boundary
            #     if dist < merge_tol: # if the origin is too close to the boundary
            #         n += 1
            #         voronoi_vertices_in[ray_origins[i]] = intpt # replace it with the boundary
            #         # this keeps the total number of voronoi points the same for connectivity
            #         boundary_points.append(intpt) # also add to array for boundary ridges
            #         # basically stretching cell to boundary
            #     else: # add intersection point to boundary
            #         boundary_points.append(intpt)
            # else:
            boundary_points.append(intpt)
            # plt.quiver(p[0,0],p[0,1],normal[0],normal[1],color='b')
        else:
            print('(voronoi)')
    # if mergeFlag == 'On':
    #     print(n,'points merged with the boundary')
    nboundary_pts = len(boundary_points)
    boundary_points = np.reshape(boundary_points,(nboundary_pts,2)) # need to reshape because extra second dimension for some reason
    

    if boundaryFlag == 'On': # add the boundary points for homogenization 
        boundary_points = np.vstack((boundary_points,boundary_points_original)) # have to be at end for indexing
        nboundary_pts = np.shape(boundary_points)[0]
        # sometimes there is a bug in the corner when using boundaries, not sure why, just remesh for now -SA
    # ax.plot(boundary_points[:,0],boundary_points[:,1],'b^',markersize=5)

    # print(boundary_points)
    points = sort_coordinates(boundary_points)
    # print(points)
    pgon = shp.Polygon(points)
    # print('cross sectional area',pgon.area)
    # cell_area = pgon.area
    # ax = plt.gca()
    # shapely.plotting.plot_polygon(pgon)

    # Construct the connectivity for a line path consisting of the boundary points to get beams along boundary
    # create boundary array with boundary points and false points along boundary for correct ridges
    boundary_ridges_new = np.zeros((nboundary_pts,2))
    boundary_points_new = np.copy(boundary_points) 
    next_point = np.copy(boundary_points_new[0])  # get first point
    boundary_points_new[0] = [np.inf,np.inf]
    # plt.text(next_point[0],next_point[1],str(0))
    
    # Conform line path to boundary
    if boundaryFlag == 'On': 
        for i in range(0,boundary_points_new.shape[0]-1):
            next_point_id = cdist([next_point], boundary_points_new).argmin()
            if check_iscollinear(next_point,boundary_points_new[next_point_id],boundaries) > 0: # check if the new line segment is collinear with any boundary line segment
                boundary_ridges_new[i,1] = next_point_id
                boundary_ridges_new[i+1,0] = next_point_id
                next_point = np.copy(boundary_points_new[next_point_id])
                boundary_points_new[next_point_id] = [np.inf,np.inf]
                # plt.text(next_point[0],next_point[1],str(i+1))
            else:
                boundary_points_new_check = np.copy(boundary_points_new)
                while check_iscollinear(next_point,boundary_points_new_check[next_point_id],boundaries) == 0:
                    boundary_points_new_check[next_point_id] = [np.inf,np.inf]
                    next_point_id = cdist([next_point], boundary_points_new_check).argmin()
                    
                boundary_ridges_new[i,1] = next_point_id
                boundary_ridges_new[i+1,0] = next_point_id
                next_point = np.copy(boundary_points_new[next_point_id])
                boundary_points_new[next_point_id] = [np.inf,np.inf]
                # plt.text(next_point[0],next_point[1],str(i+1))
    else:
        for i in range(0,boundary_points_new.shape[0]-1):
            next_point_id = cdist([next_point], boundary_points_new).argmin()
            boundary_ridges_new[i,1] = next_point_id
            boundary_ridges_new[i+1,0] = next_point_id
            next_point = np.copy(boundary_points_new[next_point_id])
            boundary_points_new[next_point_id] = [np.inf,np.inf]
            # plt.text(next_point[0],next_point[1],str(i+1))
            
    # Update arrays and construct output
    boundary_ridges_new = (boundary_ridges_new + nvertices_in).astype(int) # shift the indices with "nvertices_in"
    voronoi_vertices = np.vstack((voronoi_vertices_in,boundary_points)) # vertical stack "in" vertices and "cross" boundary points
    nvertices = voronoi_vertices.shape[0]
    boundary_ridges_new = np.vstack((infinite_ridges_new,boundary_ridges_new))

    nboundary_ridge = boundary_ridges_new.shape[0]
    voronoi_ridges = np.vstack((finite_ridges_new,boundary_ridges_new))
    nridge = voronoi_ridges.shape[0]

    # Calculate minimum ridge length for reference
    ridge_lengths = voronoi_vertices[voronoi_ridges[:,0]] - voronoi_vertices[voronoi_ridges[:,1]]
    ridge_lengths = np.linalg.norm(ridge_lengths,axis=1)
    # get ridge lengths not equal to zero
    ridge_lengths_red = ridge_lengths[ridge_lengths >= 1e-15]
    # ridge_lengths = ridge_lengths[1e-2 >= ridge_lengths]
    print('The minimum ridge distance is', float('{:.2E}'.format(min(ridge_lengths_red))))

    # plt.hist(ridge_lengths,bins=100)
    

    
    return voronoi_vertices,finite_ridges_new,\
        boundary_ridges_new,nvertices,nvertices_in,nfinite_ridge,nboundary_ridge,\
        nboundary_pts,voronoi_ridges,nridge


def LayerOperation(NURBS_degree,nsegments,theta_min,theta_max,finite_ridges_new,boundary_ridges_new,nfinite_ridge,nboundary_ridge,\
                   z_min,z_max,long_connector_ratio,voronoi_vertices,nvertex,generation_center,knotFlag, knotParams,box_center,box_depth):
    
    m1 = knotParams.get('m1')
    m2 = knotParams.get('m2')
    a1 = knotParams.get('a1')
    a2 = knotParams.get('a2')
    Uinf = knotParams.get('Uinf')
    # Number of points per layer
    npt_per_layer = nvertex

    # Number of control points per beam
    nctrlpt_per_elem = NURBS_degree + 1
    nctrlpt_per_beam = 2*NURBS_degree + 1

    # Number of layers
    nlayers = nctrlpt_per_beam*nsegments
    
    # Number of connectors per beam
    nconnector_t_per_beam = int((nctrlpt_per_beam-1)/NURBS_degree+1)
    nconnector_t_per_grain = int(nconnector_t_per_beam*nsegments)
    
    # Length of layer
    segment_length = (z_max - z_min) / (nsegments + (nsegments-1)*long_connector_ratio) # beam segment length
    segment_angle = (theta_max-theta_min) / (nsegments + (nsegments-1)*long_connector_ratio)

    # Size of connectors
    connector_l_angle = segment_angle*long_connector_ratio
    connector_l_length = segment_length*long_connector_ratio

    # Rotation angle and layer z-coordinates
    theta = np.linspace(theta_min,theta_max-connector_l_angle*(nsegments-1),nlayers-(nsegments-1))
    z_coord = np.linspace(z_min,z_max-connector_l_length*(nsegments-1),nlayers-(nsegments-1))
    
    # Insert repeated layers for the longitudinal connectors
    for i in range(nlayers-(nsegments-1)-(nctrlpt_per_beam-1),1,-(nctrlpt_per_beam-1)): 
        theta = np.insert(theta,i,theta[i-1])
        theta[i:] += connector_l_angle
        z_coord = np.insert(z_coord,i,z_coord[i-1])
        z_coord[i:] += connector_l_length
    
    # Matricies for adding new coordinates due to layering, rotation, and knot flow
    vertices_new = np.zeros((nlayers,npt_per_layer,3))   
    vertices_rep = np.tile(np.transpose(voronoi_vertices[:,1]),(nlayers,1))   
    vertices_flow = np.tile(np.transpose(voronoi_vertices[:,1]),(nlayers,1)) 
    knot_dy = np.zeros((nlayers,npt_per_layer))
    # knot_radii = np.zeros((nlayers,1))
    # vertices_knot = np.array([])

    # Defining edge to knot influence area for clean mesh boundary
    knot_bound = 0.95*box_depth/2 + box_center[1]
    
    # Calculate bend around knot
    if knotFlag == 'On':
        vertices_flow[:,:] = odeint(calc_knotflow,voronoi_vertices[:,1],z_coord[:],args=(m1,m2,a1,a2,Uinf,box_center[1]))
        knot_dy = vertices_flow - vertices_rep
    # knot_indices = [] # empty list of indices which are in knot
        
        
    # Extrude layers
    for i in range(0,nlayers):
        # Define z coord based on plane
        vertices_new[i,:,2] = z_coord[i]

        for j in range(0,npt_per_layer):
            
            # Rotate xy plane for trunk rotation
            vertices_new[i,j,:2] = rotate_around_point_highperf(voronoi_vertices[j,:], theta[i], generation_center)
            #
            # Straighten out edges of box in case knot flow bulges mesh
            if abs(vertices_new[i,j,1]) > knot_bound: 
                knot_dy[i,j] = 0

            # added randomness to morphology in the L plane - not calibrated yet -SA
            # vertices_new[i,j,0] = vertices_new[i,j,0]*(np.random.random()*random_noise+1)# + dx # sin(z_coord[i]/z_max)
            # vertices_new[i,j,1] = vertices_new[i,j,1]*(np.random.random()*random_noise+1) + dy #(np.random.random()*random_noise/10+1)

    # Adjust rotation due to knot flow
    vertices_new[:,:,1] = vertices_new[:,:,1] + knot_dy[:,:]
    
    # Vertex data in 3D
    voronoi_vertices_3D = np.reshape(vertices_new,(-1,3))
    nvertices_3D = voronoi_vertices_3D.shape[0]
    voronoi_vertices_2D = voronoi_vertices_3D[0:npt_per_layer,0:2] # adhoc array for later use with random field
    
    # Voronoi ridge data in 3D 
    finite_ridges_3D = np.tile(finite_ridges_new, (nlayers,1)) # temporarily assume each layer has the same number of finite_ridges and same finite ridge connectivities
    # calculate offset 
    x = np.arange(0,nlayers)*npt_per_layer
    finite_ridges_3D_offset = np.repeat(x, nfinite_ridge)
    finite_ridges_3D_offset = np.tile(finite_ridges_3D_offset, (2,1))
    finite_ridges_3D = finite_ridges_3D + finite_ridges_3D_offset.T 

    boundary_ridges_3D = np.tile(boundary_ridges_new, (nlayers,1)) # temporarily assume each layer has the same number of finite_ridges and same finite ridge connectivities
    # calculate offset 
    x = np.arange(0,nlayers)*npt_per_layer
    boundary_ridges_3D_offset = np.repeat(x, nboundary_ridge)
    boundary_ridges_3D_offset = np.tile(boundary_ridges_3D_offset, (2,1))
    boundary_ridges_3D = boundary_ridges_3D + boundary_ridges_3D_offset.T
    
    
    return voronoi_vertices_3D,nvertices_3D,nlayers,segment_length,nctrlpt_per_elem,nctrlpt_per_beam,nconnector_t_per_beam,\
           nconnector_t_per_grain,theta,z_coord,npt_per_layer,finite_ridges_3D,boundary_ridges_3D, voronoi_vertices_2D


def RidgeMidQuarterPts(voronoi_vertices_3D,nvertex,nvertices_in,voronoi_ridges,\
                       finite_ridges_new,boundary_ridges_new,finite_ridges_3D,boundary_ridges_3D,nfinite_ridge,\
                       nboundary_ridge,nboundary_pts,nlayers,voronoi_vertices):
    
    count = 0 # count for vertex (point) index
    
    for i in range(0,nlayers):
        ######################### For finite Voronoi ridges ###########################
        # Form a list of middle points of finite Voronoi edges
        finite_ridge_mid = []
        finite_midpt_indices = []
        count += nvertex
        
        for vpair in finite_ridges_3D[i*nfinite_ridge+0:i*nfinite_ridge+nfinite_ridge,:]: # for vpair in finite_ridges_new:
            midpoint = (voronoi_vertices_3D[vpair[0]] + voronoi_vertices_3D[vpair[1]])/2
            finite_ridge_mid.append(midpoint)
            finite_midpt_indices.append(count)
            count += 1
            
        finite_ridge_mid = np.tile(finite_ridge_mid, (2,1)) # duplicate the mid point list
        finite_second_midpt_indices = [x+nfinite_ridge for x in finite_midpt_indices]
        finite_midpt_indices = np.concatenate((finite_midpt_indices,finite_second_midpt_indices))
        count += nfinite_ridge
        nfinite_midpt = nfinite_ridge*2
        
        # Form a list of quarter points of Voronoi edges
        finite_ridge_quarter = []
        finite_quarterpt_indices = []
        for vpair in finite_ridges_3D[i*nfinite_ridge+0:i*nfinite_ridge+nfinite_ridge,:]: # for vpair in finite_ridges_new:
            quarterpoint = 3./4*voronoi_vertices_3D[vpair[0]] + 1./4*voronoi_vertices_3D[vpair[1]]
            finite_ridge_quarter.append(quarterpoint)
            finite_quarterpt_indices.append(count)
            count += 1
        
        for vpair in finite_ridges_3D[i*nfinite_ridge+0:i*nfinite_ridge+nfinite_ridge,:]: # for vpair in finite_ridges_new:
            quarterpoint = 1./4*voronoi_vertices_3D[vpair[0]] + 3./4*voronoi_vertices_3D[vpair[1]]
            finite_ridge_quarter.append(quarterpoint)
            finite_quarterpt_indices.append(count)
            count += 1
        finite_quarterpt_indices = np.asarray(finite_quarterpt_indices)
        
        nfinite_quarterpt = nfinite_ridge*2
        
        nfinitept_per_layer = nvertices_in + nfinite_midpt + nfinite_quarterpt
        
        ######################### For boundary Voronoi ridges #########################
        # Form a list of mid-points of boundary Voronoi edges
        boundary_ridge_mid = []
        boundary_midpt_indices = []
        boundary_ridges_3D = boundary_ridges_3D.astype(int)
        for vpair in boundary_ridges_3D[i*nboundary_ridge+0:i*nboundary_ridge+nboundary_ridge,:]: # for vpair in boundary_ridges_new:
            boundary_midpoint = (voronoi_vertices_3D[vpair[0]] + voronoi_vertices_3D[vpair[1]])/2
            boundary_ridge_mid.append(boundary_midpoint)
            boundary_midpt_indices.append(count)
            count += 1
        
        boundary_ridge_mid = np.tile(boundary_ridge_mid, (2,1))
        boundary_second_midpt_indices = [x+nboundary_ridge for x in boundary_midpt_indices]
        boundary_midpt_indices = np.concatenate((boundary_midpt_indices,boundary_second_midpt_indices))
        count += nboundary_ridge
        nboundary_midpt = nboundary_ridge*2
        
        # Form a list of quarter-points of boundary Voronoi edges
        boundary_ridge_quarter = []
        boundary_quarterpt_indices = []
        for vpair in boundary_ridges_3D[i*nboundary_ridge+0:i*nboundary_ridge+nboundary_ridge,:]: # for vpair in boundary_ridges_new:
            boundary_quarterpoint = 3./4*voronoi_vertices_3D[vpair[0]] + 1./4*voronoi_vertices_3D[vpair[1]]
            boundary_ridge_quarter.append(boundary_quarterpoint)
            boundary_quarterpt_indices.append(count)
            count += 1
        
        for vpair in boundary_ridges_3D[i*nboundary_ridge+0:i*nboundary_ridge+nboundary_ridge,:]: # for vpair in boundary_ridges_new:
            boundary_quarterpoint = 1./4*voronoi_vertices_3D[vpair[0]] + 3./4*voronoi_vertices_3D[vpair[1]]
            boundary_ridge_quarter.append(boundary_quarterpoint)
            boundary_quarterpt_indices.append(count)
            count += 1
        boundary_quarterpt_indices = np.asarray(boundary_quarterpt_indices)
            
            
        nboundary_quarterpt = nboundary_ridge*2
        
        nboundary_pt_per_layer = nboundary_pts + nboundary_midpt + nboundary_quarterpt
        
        npt_per_layer_vtk = nfinitept_per_layer + nboundary_pt_per_layer
        
        voronoi_ridges_2D = voronoi_ridges + np.ones(voronoi_ridges.shape)*(i*npt_per_layer_vtk)
    
        all_midpt_indices = np.vstack((np.reshape(finite_midpt_indices,(2,-1)).T,np.reshape(boundary_midpt_indices,(2,-1)).T))
        all_quarterpt_indices = np.vstack((np.reshape(finite_quarterpt_indices,(2,-1)).T,np.reshape(boundary_quarterpt_indices,(2,-1)).T))
    
        all_pts_2D = np.vstack((voronoi_vertices_3D[i*nvertex+0:i*nvertex+nvertex],finite_ridge_mid,finite_ridge_quarter,boundary_ridge_mid,boundary_ridge_quarter))
        
        # voronoi_ridges_2D = np.vstack((finite_ridges_3D[i*nfinite_ridge+0:i*nfinite_ridge+nfinite_ridge,:],boundary_ridges_3D[i*nboundary_ridge+0:i*nboundary_ridge+nboundary_ridge,:]))
        all_ridges_2D = np.hstack((voronoi_ridges_2D,all_midpt_indices,all_quarterpt_indices))
        
        
        if i == 0:
            all_pts_3D = all_pts_2D
            all_ridges_3D = all_ridges_2D
        else:
            all_pts_3D = np.vstack((all_pts_3D,all_pts_2D))
            all_ridges_3D = np.vstack((all_ridges_3D,all_ridges_2D))

    
    ### Old code

    ######################### For finite Voronoi ridges ###########################
    # Form a list of middle points of finite Voronoi edges
    finite_ridge_mid = []
    finite_midpt_indices = []
    count = nvertex

    for vpair in finite_ridges_new:
        midpoint = (voronoi_vertices[vpair[0]] + voronoi_vertices[vpair[1]])/2
        finite_ridge_mid.append(midpoint)
        finite_midpt_indices.append(count)
        count += 1
        
    finite_ridge_mid = np.tile(finite_ridge_mid, (2,1)) # duplicate the mid point list
    finite_second_midpt_indices = [x+nfinite_ridge for x in finite_midpt_indices]
    finite_midpt_indices = np.concatenate((finite_midpt_indices,finite_second_midpt_indices))
    count += nfinite_ridge
    nfinite_midpt = nfinite_ridge*2
    
    # Form a list of quarter points of Voronoi edges
    finite_ridge_quarter = []
    finite_quarterpt_indices = []
    for vpair in finite_ridges_new:
        quarterpoint = 3./4*voronoi_vertices[vpair[0]] + 1./4*voronoi_vertices[vpair[1]]
        finite_ridge_quarter.append(quarterpoint)
        finite_quarterpt_indices.append(count)
        count += 1
    
    for vpair in finite_ridges_new:
        quarterpoint = 1./4*voronoi_vertices[vpair[0]] + 3./4*voronoi_vertices[vpair[1]]
        finite_ridge_quarter.append(quarterpoint)
        finite_quarterpt_indices.append(count)
        count += 1
    finite_quarterpt_indices = np.asarray(finite_quarterpt_indices)
    
    nfinite_quarterpt = nfinite_ridge*2
    
    nfinitept_per_layer = nvertices_in + nfinite_midpt + nfinite_quarterpt
    
    ######################### For boundary Voronoi ridges #########################
    # Form a list of mid-points of boundary Voronoi edges
    boundary_ridge_mid = []
    boundary_midpt_indices = []
    boundary_ridges_new = boundary_ridges_new.astype(int)
    for vpair in boundary_ridges_new:
        boundary_midpoint = (voronoi_vertices[vpair[0]] + voronoi_vertices[vpair[1]])/2
        boundary_ridge_mid.append(boundary_midpoint)
        boundary_midpt_indices.append(count)
        count += 1
    
    boundary_ridge_mid = np.tile(boundary_ridge_mid, (2,1))
    boundary_second_midpt_indices = [x+nboundary_ridge for x in boundary_midpt_indices]
    boundary_midpt_indices = np.concatenate((boundary_midpt_indices,boundary_second_midpt_indices))
    count += nboundary_ridge
    nboundary_midpt = nboundary_ridge*2
    
    # Form a list of quarter-points of boundary Voronoi edges
    boundary_ridge_quarter = []
    boundary_quarterpt_indices = []
    for vpair in boundary_ridges_new:
        boundary_quarterpoint = 3./4*voronoi_vertices[vpair[0]] + 1./4*voronoi_vertices[vpair[1]]
        boundary_ridge_quarter.append(boundary_quarterpoint)
        boundary_quarterpt_indices.append(count)
        count += 1
    
    for vpair in boundary_ridges_new:
        boundary_quarterpoint = 1./4*voronoi_vertices[vpair[0]] + 3./4*voronoi_vertices[vpair[1]]
        boundary_ridge_quarter.append(boundary_quarterpoint)
        boundary_quarterpt_indices.append(count)
        count += 1
    boundary_quarterpt_indices = np.asarray(boundary_quarterpt_indices)
        
        
    nboundary_quarterpt = nboundary_ridge*2
    
    nboundary_pt_per_layer = nboundary_pts + nboundary_midpt + nboundary_quarterpt
    
    
    npt_per_layer = nvertices_in + nboundary_pts
    npt_per_layer_vtk = nfinitept_per_layer + nboundary_pt_per_layer
    
    all_midpt_indices = np.vstack((np.reshape(finite_midpt_indices,(2,-1)).T,np.reshape(boundary_midpt_indices,(2,-1)).T))
    all_quarterpt_indices = np.vstack((np.reshape(finite_quarterpt_indices,(2,-1)).T,np.reshape(boundary_quarterpt_indices,(2,-1)).T))
    
    all_pts_2D = np.vstack((voronoi_vertices,finite_ridge_mid,finite_ridge_quarter,boundary_ridge_mid,boundary_ridge_quarter))
    all_ridges = np.hstack((voronoi_ridges,all_midpt_indices,all_quarterpt_indices)).astype(int)

    return all_pts_3D,npt_per_layer_vtk,all_pts_2D,all_ridges


def VertexandRidgeinfo(all_pts_2D,all_ridges,npt_per_layer,\
                       geoName,radii,generation_center,\
                       cellwallthickness_sparse,cellwallthickness_dense):
    
    """Generate the vertex and ridge info 
       Vertex info includes: coordinates, ridge indices, indices of another vertex for each ridge, ridge lengths, ridge angles, ridge width
       Ridge info includes: 2 vertex indices, 2 mid point indices, 2 quarter point indices """

    
    # Calculate lengths of all Voronoi ridges
    vector = all_pts_2D[all_ridges[:,1],:] - all_pts_2D[all_ridges[:,0],:]
    all_ridge_lengths = np.linalg.norm(vector, axis=1)
    
    # Calculate angles of all Voronoi ridges (angles measured counter-clock wise, x-axis --> (1,0), y-axis --> (0,1))
    all_ridge_angles = np.arctan2(vector[:,1],vector[:,0]) # np.arctan2(y, x) * 180 / np.pi = the angle

    # thicknesses of all Voronoi vertices (here identical thickness for all wings belonging to the same vertex is assumed)
    vertex_cellwallthickness_2D = np.zeros(npt_per_layer)
    
    vertex_distance2logcenter = np.linalg.norm(all_pts_2D - generation_center, axis=1) # calc distance between vertex and logcenter
    for i in range(0,npt_per_layer):
        if (bisect.bisect(radii,vertex_distance2logcenter[i]) % 2) == 0: # if even
            vertex_cellwallthickness_2D[i] = cellwallthickness_sparse # actually wrong, compared to BL placement. flipped as patch
        else: # if odd
            vertex_cellwallthickness_2D[i] = cellwallthickness_dense # actually wrong, compared to BL placement. flipped as patch
    #==================================================================================================
     
    # Generate a list containing info for each vertex
    all_vertices_info_2D = []
    all_direction_info_2D = []
    vert_area = 0
    for i in range(0,npt_per_layer): # loop over all vertices
        
        allrows = np.where(all_ridges == i)[0] # find all Voronoi ridges containing the ith vertex
        vertex_info = []
        vertex_info.append(len(allrows)) # append total number of wings for the ith vertex
        vertex_info.append(allrows) # append ridge indices
        vertex_info.append(all_ridges[allrows,0:2][np.nonzero(all_ridges[allrows,0:2]-i)]) # append the indices of vetices at the other end of each ridge
        direction_info = []
        direction = np.where(all_ridges[allrows,0:2] == i)[1] # the position of current vertex in each ridge, p1 --> 0, p2 --> 1 
        direction = -np.power(-1,direction+1) # convert the way to express the direction of each ridge to: 1/-1 --> pointing from/to the current vertex
        direction_info.append(direction) # append ridge direction
        vertex_info.append(all_ridge_lengths[allrows]/2) # append wing lengths
        vertex_info.append(np.ones(len(allrows))*vertex_cellwallthickness_2D[i]) # append wing widths
        # if len(allrows) >3:
        #     print(all_pts_2D[allrows,0:2])
        vertex_info.append(all_ridge_angles[allrows]+math.pi*(-direction.clip(max=0))) # append wing angles
        
        all_vertices_info_2D.append(vertex_info)
        all_direction_info_2D.append(direction_info)
        # ==================================================================================================
        # calculate each beam cross-sectional area and add to total for net area for later porosity calcs
        v_coord = all_pts_2D[i,:]
        p_coords = all_pts_2D[all_ridges[allrows,0:2][np.nonzero(all_ridges[allrows,0:2]-i)],:]
        v_points = []
        for p in p_coords:
            r = p - v_coord
            perp = (r[1],-r[0])
            orth = np.linalg.norm(perp)
            width = vertex_cellwallthickness_2D[i]
            ph = (p + v_coord)/2
            p1 = ph + width/2*orth
            p2 = ph - width/2*orth
            v_points.append(p1)
            v_points.append(p2)
        area_points = sort_coordinates(np.array(v_points))
        pgon = shp.Polygon(area_points)
        vert_area += pgon.area

    # flattened 1D numpy array of lengths, widths and angles of wings
    # ref:https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
    ridge_inices = [vertex_info[1] for vertex_info in all_vertices_info_2D]
    ridge_inices = np.array([item for sublist in ridge_inices for item in sublist])
    another_vertex_inices = [vertex_info[2] for vertex_info in all_vertices_info_2D]
    another_vertex_inices = np.array([item for sublist in another_vertex_inices for item in sublist])
    ridge_lengths = [vertex_info[3] for vertex_info in all_vertices_info_2D]
    ridge_lengths = np.array([item for sublist in ridge_lengths for item in sublist])
    ridge_widths = [vertex_info[4] for vertex_info in all_vertices_info_2D]
    ridge_widths = np.array([item for sublist in ridge_widths for item in sublist])
    ridge_angles = [vertex_info[5] for vertex_info in all_vertices_info_2D]
    ridge_angles = np.array([item for sublist in ridge_angles for item in sublist])
    ridge_directions = [direction_info[0] for direction_info in all_direction_info_2D]
    ridge_directions = np.array([item for sublist in ridge_directions for item in sublist])
    
    flattened_all_vertices_2D = np.vstack((ridge_inices,another_vertex_inices,ridge_lengths,ridge_widths,ridge_angles,ridge_directions))
 
    # Convert ragged info list to a rectangular shape numpy array
    max_wings = max([vertex_info[0] for vertex_info in all_vertices_info_2D])
    
    all_vertices_info_2D_nparray = np.zeros((npt_per_layer,1+max_wings*5))
    all_vertices_info_2D_nparray[:,0] = [vertex_info[0] for vertex_info in all_vertices_info_2D]
    
    for i in range(0,npt_per_layer):
            nwings = all_vertices_info_2D[i][0]
            all_vertices_info_2D_nparray[i,1:nwings+1] = all_vertices_info_2D[i][1]
            all_vertices_info_2D_nparray[i,max_wings+1:max_wings+nwings+1] = all_vertices_info_2D[i][2]
            all_vertices_info_2D_nparray[i,2*max_wings+1:2*max_wings+nwings+1] = all_vertices_info_2D[i][3]
            all_vertices_info_2D_nparray[i,3*max_wings+1:3*max_wings+nwings+1] = all_vertices_info_2D[i][4]
            all_vertices_info_2D_nparray[i,4*max_wings+1:4*max_wings+nwings+1] = all_vertices_info_2D[i][5]
        # Save info to txt files
    all_vertices_2D = np.hstack((all_pts_2D[0:npt_per_layer,:],all_vertices_info_2D_nparray))

    # write vertex info
    for i in range(0,npt_per_layer):
        nwings = all_vertices_info_2D[i][0]
        for j in range(0,nwings):
            all_vertices_info_2D_nparray[i,1+j*5] = all_vertices_info_2D[i][1][j]
            all_vertices_info_2D_nparray[i,2+j*5] = all_vertices_info_2D[i][2][j]
            all_vertices_info_2D_nparray[i,3+j*5] = all_vertices_info_2D[i][3][j]
            all_vertices_info_2D_nparray[i,4+j*5] = all_vertices_info_2D[i][4][j]
            all_vertices_info_2D_nparray[i,5+j*5] = all_vertices_info_2D[i][5][j]
    # Save info to txt files
    all_vertices_2Dchrono = np.hstack((all_pts_2D[0:npt_per_layer,:],all_vertices_info_2D_nparray))
    # np.save(Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName +'-vertex.mesh'), all_vertices_2Dchrono)
    np.savetxt(Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName +'-vertex.mesh'), all_vertices_2Dchrono, fmt='%.16g', delimiter=' '\
        ,header='Vertex Data Generated with RingsPy Mesh Generation Tool\n\
        Number of vertices\n'+ str(npt_per_layer) +  '\nMax number of wings for one vertex\n'+ str(max_wings) + '\n\
        [xcoord ycoord nwings ridge1 farvertex1 length1 width1 angle1 ... ridgen farvertexn lengthn widthn anglen]', comments='')
    
    # not used - SA
    #     np.savetxt(Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName +'-ridge.mesh'), all_ridges, fmt='%d', delimiter=' '\
    #         ,header='Ridge Data Generated with RingsPy Mesh Generation Tool\n\
    # Number of ridges\n'+ str(nridge) +
    #     '\n\
    # [vertex1 vertex2 midpt1 midpt2 qrtrpt1 qrtrpt2]', comments='')
    
    return all_vertices_2D, max_wings, flattened_all_vertices_2D, all_ridges, vert_area


def GenerateBeamElement(voronoi_vertices_3D,nvertices_3D,NURBS_degree,nctrlpt_per_beam,nctrlpt_per_elem,nsegments,\
                        npt_per_layer,nvertex,voronoi_ridges,nridge,all_vertices_2D,\
                        nconnector_t_per_beam,nconnector_t_per_grain):
    
    IGAvertices = np.copy(voronoi_vertices_3D)
    # Connectivity for IGA Control Points (Vertices)
    
    # Beams
    ngrain = nvertex
    nbeam_total = ngrain*nsegments
    
    beam_connectivity_original = np.zeros((nbeam_total,nctrlpt_per_beam))
    for i in range(0,nsegments):
        for j in range(0, ngrain):
            irow = i*ngrain + j
            ivertex = i*npt_per_layer*nctrlpt_per_beam + j
            for icol in range(0,nctrlpt_per_beam):
                beam_connectivity_original[irow,icol] = ivertex+icol*npt_per_layer
    
    # Rearange beam connectivity such that each row is corresponding to a Bezier beam element
    beam_connectivity = np.copy(beam_connectivity_original)
    for ictrlpt in range(nctrlpt_per_beam-NURBS_degree,1,-NURBS_degree):
        beam_connectivity = np.insert(beam_connectivity,ictrlpt,beam_connectivity[:,ictrlpt-1],axis=1)
    
    beam_connectivity_original = (beam_connectivity_original+1).astype(int) # +1 because of in abaqus index starts from 1
    beam_connectivity = (np.reshape(beam_connectivity,(-1,nctrlpt_per_elem))+1).astype(int) # +1 because of in abaqus index starts from 1
    
    
    nbeamElem = beam_connectivity.shape[0]
    
    
    # Transverse connectors 
    nconnector_t = nridge*nconnector_t_per_grain
    connector_t_connectivity = np.zeros((nconnector_t,2))
    for i in range(0,nsegments):
        for j in range(0,nconnector_t_per_beam):
            for k in range(0,nridge):
                irow = i*nconnector_t_per_beam*nridge + j*nridge + k
                connector_t_connectivity[irow,:] = voronoi_ridges[k]+i*npt_per_layer*nctrlpt_per_beam+j*NURBS_degree*npt_per_layer
                
    connector_t_connectivity = (connector_t_connectivity+1).astype(int)
    # Slicing block indices, https://stackoverflow.com/questions/39692769/efficient-numpy-indexing-take-first-n-rows-of-every-block-of-m-rows
    connector_t_index = np.linspace(0,nconnector_t-1,nconnector_t).astype(int)
    connector_t_bot_index = connector_t_index[np.mod(np.arange(connector_t_index.size),nridge*nconnector_t_per_beam)<nridge]
    connector_t_top_index = connector_t_index[np.mod(np.arange(connector_t_index.size),nridge*nconnector_t_per_beam)>=(nconnector_t_per_beam-1)*nridge]
    connector_t_reg_index = np.setdiff1d(np.setdiff1d(connector_t_index, connector_t_bot_index),connector_t_top_index)
    connector_t_bot_connectivity = np.copy(connector_t_connectivity)[connector_t_bot_index,:]
    connector_t_top_connectivity = np.copy(connector_t_connectivity)[connector_t_top_index,:]
    connector_t_reg_connectivity = np.copy(connector_t_connectivity)[connector_t_reg_index,:]

    
    # Longitudinal connectors
    nwings = all_vertices_2D[:,2].astype(int)
    nconnector_l_per_layer = sum(nwings)
    nconnector_l = nconnector_l_per_layer * (nsegments-1)
    connector_l_connectivity = np.zeros((nconnector_l,2))
    connector_l_vertex_dict = np.zeros(nconnector_l)
    irow_conn_l = 0
    for i in range(0,nsegments-1): # loop over layers of longitudinal connectors
        for j in range(0,ngrain): # loop over grains in each layer
            n = nwings[j]
            irow = i*ngrain + j
            for k in range(0,n): # loop over wings of each grain
                connector_l_connectivity[irow_conn_l,:] = (beam_connectivity_original[irow,-1],beam_connectivity_original[irow,-1]+npt_per_layer)
                connector_l_vertex_dict[irow_conn_l] = irow
                irow_conn_l += 1
    
    connector_l_vertex_dict = connector_l_vertex_dict.astype(int)
    connector_l_connectivity = connector_l_connectivity.astype(int)
    
    return IGAvertices,beam_connectivity_original,nbeam_total,\
    beam_connectivity,nbeamElem,\
    connector_t_bot_connectivity,connector_t_top_connectivity,\
    connector_t_reg_connectivity,connector_l_connectivity,\
    nconnector_t,nconnector_l,connector_l_vertex_dict


def InsertPrecrack(all_pts_2D,all_ridges,nridge,precrack_nodes,\
                     cellsize_early,nsegments):

    precrack_midpts = (precrack_nodes[:,0:2]+precrack_nodes[:,2:4])/2.0
    ridge_midpts = all_pts_2D[all_ridges[:,2]]
    ridge_midpts_tree = KDTree(ridge_midpts)
    near_ridges = []
    for i in range(0,len(precrack_midpts)):
        near_ridges.append(ridge_midpts_tree.query_ball_point(precrack_midpts[i,:],max(cellsize_early,1*np.linalg.norm(precrack_nodes[i,2:4]-precrack_midpts[i,:]))))
    # Find the intersect point of neighboring ridges (lines) with the precrack line
    # actually finding if the precrack just interesects? - SA
    precrack_elem = []
    for i in range(0,len(precrack_nodes)):
        p = precrack_nodes[i,0:2] # find starting point of the precrack
        normal = (precrack_nodes[i,2:4] - p)/np.linalg.norm(precrack_nodes[i,2:4] - p) # find precrack normal
        for ridge in near_ridges[i]:
            q = all_pts_2D[all_ridges[ridge,0],:]
            s = all_pts_2D[all_ridges[ridge,1],:] - q
            u = np.cross((q-p),normal)/np.cross(normal,s)
            if (u >= 0) and (u <= 1):
                precrack_elem.append(ridge)
    nprecracked_elem = len(precrack_elem) 
    
    # Apply precrack to every layer of beam segment
    precrack_elem = precrack_elem*nsegments # repeat list 
    offset = np.repeat(list(range(0,nsegments)),nprecracked_elem)
    offset = [i * nridge for i in offset]
    precrack_elem = [a + b for a, b in zip(precrack_elem, offset)]
    
    nconnector_t_precrack = len(precrack_elem)*3 # times 3 for bot,reg,top layers
    nconnector_l_precrack = 0

    # Visualize the precracks in the preview plot
    ax = plt.gca()
    ax.plot(precrack_nodes[0,0::2],precrack_nodes[0,1::2],'r-',linewidth=3)
    # plt.show()

    # print(precrack_elem)

    return precrack_elem, nconnector_t_precrack, nconnector_l_precrack


def ConnectorMeshFile(geoName,IGAvertices,connector_t_bot_connectivity,\
                      connector_t_reg_connectivity,connector_t_top_connectivity,\
                      connector_l_connectivity,all_vertices_2D,\
                      max_wings,flattened_all_vertices_2D,nsegments,segment_length,long_connector_ratio,\
                      nctrlpt_per_beam,theta,nridge,\
                      randomFlag,random_field,knotParams,knotFlag,box_center,voronoi_vertices_2D,precrack_elem,cellwallthick,radii,z_max,rayFlag):
    # pr = cProfile.Profile()
    # pr.enable()

    # Uncompress paramters for knot flow
    m1 = knotParams.get('m1')
    m2 = knotParams.get('m2')
    a1 = knotParams.get('a1')
    a2 = knotParams.get('a2')
    Uinf = knotParams.get('Uinf')

    ######### txt File stores the connector data for Abaqus analyses ##############
    nel_con_tbot = connector_t_bot_connectivity.shape[0]
    nel_con_treg = connector_t_reg_connectivity.shape[0]
    nel_con_ttop = connector_t_top_connectivity.shape[0]
    nel_con_l = connector_l_connectivity.shape[0]
    nel_con = nel_con_tbot + nel_con_treg + nel_con_ttop + nel_con_l
    
    # calculate heights of transverse connectors
    height_connector_t = segment_length*(1 - long_connector_ratio)/4
    
    # calculate longitudinal connectors
    nlayers_conn_l = nsegments - 1
    conn_l_ridge_index_2D = flattened_all_vertices_2D[0,:].astype(int)
    conn_l_lengths_2D = flattened_all_vertices_2D[2,:]
    conn_l_angles_2D = flattened_all_vertices_2D[4,:]
    
    conn_l_ridge_index = np.tile(conn_l_ridge_index_2D,nlayers_conn_l)
    conn_l_lengths = np.tile(conn_l_lengths_2D,nlayers_conn_l)
    conn_l_angles = np.tile(conn_l_angles_2D,nlayers_conn_l)
    
    # for i in range(0,nlayers_conn_l):
    #     conn_l_angles[i*2*nridge:(i+1)*2*nridge] = conn_l_angles[i*2*nridge:(i+1)*2*nridge] - theta[(i+1)*nctrlpt_per_beam]
    indices = np.arange(nlayers_conn_l)[:, None] * 2 * nridge
    conn_l_angles[indices] -= theta[(indices // (2 * nridge) + 1) * nctrlpt_per_beam]

    rot = np.array([[np.cos(conn_l_angles), -np.sin(conn_l_angles)], [np.sin(conn_l_angles), np.cos(conn_l_angles)]]).T
    rot = np.hstack((rot[:,0,:],np.zeros(len(conn_l_angles))[:, np.newaxis])) # convert to 3D rotation matrix, assumes rotation remains still in-plane
    conn_l_tangents = rot
            
    # number of random field realizations is number of x-y points (i.e. one per beam column)
    nrf = np.shape(voronoi_vertices_2D)[0]
    # Create random field realizations
    if randomFlag in ['on','On','Y','y','Yes','yes']:
        random_field.generateRandVariables(nrf, seed = 8) # generation of random numbers, critical step
        random_field.generateFieldOnGrid()                  # calculation of preparation files, can be used for any geometry
        rf_array = np.empty((nel_con_l,3,1))
    # Meshdata = [nodex1 nodey1 nodez1 nodex2 nodey2 nodez2 centerx centery centerz 
    # dx1 dy1 dz1 dx2 dy2 dz2 n1x n1y n1z n2x n2y n2z width height random_field connector_flag knot_flag precrack_flag short_flag ray_flag]   
    Meshdata = np.zeros((nel_con,29))
    
    if knotFlag == 'On':
        ktol = 0.02 # needs to be calibrated -SA
    else:
        ktol = 0.0
    
    IGAcopy = np.copy(IGAvertices) # to protect reference, not sure if necessary. removed copy for efficiency -SA
    # Add basic connector information and reset random field value to 1 for non-longitudinal connectors (just to make clear RF is only for long)
    for i in range(0,nel_con_tbot): # transverse bottom of beam
        Meshdata[i,0:3] = IGAcopy[connector_t_bot_connectivity[i,0]-1,:] # xyz coords of first half
        Meshdata[i,3:6] = IGAcopy[connector_t_bot_connectivity[i,1]-1,:] # xyz coords of second half
        Meshdata[i,22] = height_connector_t # connector length
        Meshdata[i,2] += height_connector_t/2
        Meshdata[i,5] += height_connector_t/2
        Meshdata[i,24] = 1 # connector type
        if i in precrack_elem: # 
            Meshdata[i,26] = 1 # precrack flag
    offset = nel_con_tbot
    for i in range(0,nel_con_treg): # transverse mid beam
        Meshdata[i+offset,0:3] = IGAcopy[connector_t_reg_connectivity[i,0]-1,:]
        Meshdata[i+offset,3:6] = IGAcopy[connector_t_reg_connectivity[i,1]-1,:]
        Meshdata[i+offset,22] = height_connector_t*2
        Meshdata[i+offset,24] = 2
        if i in precrack_elem:
            Meshdata[i+offset,26] = 1 # precrack flag
    offset = nel_con_tbot+nel_con_treg
    for i in range(0,nel_con_ttop): # transverse top of beam
        Meshdata[i+offset,0:3] = IGAcopy[connector_t_top_connectivity[i,0]-1,:]
        Meshdata[i+offset,3:6] = IGAcopy[connector_t_top_connectivity[i,1]-1,:]
        Meshdata[i+offset,22] = height_connector_t
        Meshdata[i+offset,2] += -height_connector_t/2
        Meshdata[i+offset,5] += -height_connector_t/2
        Meshdata[i+offset,24] = 3
        if i in precrack_elem:
            Meshdata[i+offset,26] = 1 # precrack flag
    offset = nel_con_tbot+nel_con_treg+nel_con_ttop
    for i in range(0,nel_con_l): # longitudinal
        Meshdata[i+offset,0:3] = IGAcopy[connector_l_connectivity[i,0]-1,:]
        Meshdata[i+offset,3:6] = IGAcopy[connector_l_connectivity[i,1]-1,:]
        Meshdata[i+offset,22] = conn_l_lengths[i]
        Meshdata[i+offset,24] = 4
        # calculate random field per connector
        # inefficient to loop, need to make one operation! -SA
        if randomFlag in ['on','On','Y','y','Yes','yes']:
            rf_ind = np.where((voronoi_vertices_2D==Meshdata[i+offset,0:2]).all(axis=1))[0]
            rf_array[i,:,0] = random_field.getFieldEOLE(np.array([[Meshdata[i+offset,2],0,0]]),rf_ind)
        else:
            Meshdata[i+offset,23] = 1
    if randomFlag in ['on','On','Y','y','Yes','yes'] and nsegments > 1:
        Meshdata[offset:,23] =  rf_array[:,0,0] # EOLE projection with z value
    psi_values = calc_knotstream(Meshdata[:,1]-box_center[1],Meshdata[:,2],m1,m2,a1,a2,Uinf) # check if in knot 
    inds = np.where(np.abs(psi_values) < ktol)
    Meshdata[inds,25] = 1
    

    # Calculate connector center    
    Meshdata[:,6:9] = (Meshdata[:,0:3] + Meshdata[:,3:6])/2
    # Calculate distance from center to vertex 1
    Meshdata[:,9:12] = Meshdata[:,6:9] - Meshdata[:,0:3]
    # Calculate distance from center to vertex 2
    Meshdata[:,12:15] = Meshdata[:,6:9] - Meshdata[:,3:6]
    
    # Flag short connectors
    dist = np.linalg.norm((Meshdata[:,0:3] - Meshdata[:,3:6]),axis=1)
    Meshdata[dist < cellwallthick/4, 27] = 1


    # Calculate unit t vector
    tvects = np.zeros((nel_con,3))
    diff = Meshdata[:,0:3] - Meshdata[:,3:6]
    tvects[:,0:3] = diff / np.linalg.norm(diff, axis=1, keepdims=True) #n1 - n2 = t
    # tvects[:,0:3] = (Meshdata[:,0:3] - Meshdata[:,3:6])/np.linalg.norm((Meshdata[:,0:3] - Meshdata[:,3:6])) #n1 - n2 = t
    # Calculate cross product with z axis (*** temporary until full connector rotation is added - SA)
    zvects = np.zeros((nel_con,3))
    zvects[:nel_con-nel_con_l,2] = 1
    zvects[nel_con-nel_con_l:nel_con,0] = 1
    # n vectors of beam (n1 for bot, reg, top, and n2 for long)
    nvects = np.zeros((nel_con,3))
    nvects[:nel_con-nel_con_l,:] = np.cross(tvects[:nel_con-nel_con_l,:],zvects[:nel_con-nel_con_l,:])
    nvects[nel_con-nel_con_l:nel_con,:] = np.cross(tvects[nel_con-nel_con_l:nel_con,:],zvects[nel_con-nel_con_l:nel_con,:])
    Meshdata[:,15:18] = zvects
    Meshdata[:,18:21] = nvects

    
    # Add z-variation for cellwall thicknesses/connector widths
    for i in range(0,nridge):
        Meshdata[i,21] = all_vertices_2D[connector_t_bot_connectivity[i,1]-1,3+max_wings*3] # assign widths to all bot transverse connectors
    Meshdata[:nel_con-nel_con_l,21] = np.tile(Meshdata[0:nridge,21],3*nsegments) # use the same widths for reg and top transverse connectors
    
    conn_l_widths = Meshdata[conn_l_ridge_index,21]
    Meshdata[nel_con-nel_con_l:nel_con,21] = conn_l_widths

    # Add the eccentricity to the centers for longitudinal connectors (new center is correct)
    Meshdata[nel_con-nel_con_l:nel_con,6:9] += conn_l_tangents*Meshdata[nel_con-nel_con_l:nel_con,22][:, np.newaxis]/2
    # print(conn_l_tangents)


    # Add radial rays in same manner as precrack
    if rayFlag == 'On':
        radii_rays = radii[2:]
        zstart = np.arange(0,z_max,segment_length/2)
        ds = 0.02 #arbitrary
        nthetas = np.array(radii_rays/ds,dtype='int')
        zmatrix = np.empty(len(radii_rays))
        # get an array of rays and corresponding height for each ring, with randomness
        rands =  [np.random.random((th,1)) - 0.5 for th in nthetas]
        zmatrix = [(np.tile(zstart,(th,1)) + rands[r]) for r, th in enumerate(nthetas)]
        thmatrix = [np.linspace(np.random.random()*(np.pi/16),2*np.pi,th) for th in nthetas]
        # if close to a theta and z value picked by radii index

        # segment_length
        widthvals = Meshdata[0:offset,21]
        latewidth = max(widthvals)
        rvecs = Meshdata[0:offset,0:2]
        dvecs = Meshdata[0:offset,3:5] - Meshdata[0:offset,0:2]
        rvals = np.linalg.norm(rvecs,axis=1)
        zvals = Meshdata[0:offset,2]
        thetas = np.acos(np.divide(rvecs[:,0],rvals)) # assume evec = <1,0> to take angle from x axis
        psis = np.acos(np.divide(np.sum(rvecs*dvecs,axis=1),np.multiply(rvals,np.linalg.norm(dvecs,axis=1))))

        total_info = np.column_stack((rvals,zvals,thetas,psis,widthvals))

        for i in range(0,len(rvals)):
            r,z,theta,psi,width = total_info[i,:]
            if width == latewidth:
                thtol = 2e-3
            else:
                thtol = 4e-3
            r_idx = np.abs((radii_rays - r)).argmin()
            thetavec = thmatrix[r_idx]
            # if close to one of the thetas
            th_idx = np.isclose(theta,thetavec,atol=thtol)
            if th_idx.any(): #
                zvec = zmatrix[r_idx][th_idx] # for whichever thetas its close to
                z_idx = np.isclose(z,zvec,atol=1e-1) # check if its close to the corresponding z value
                if z_idx.any(): # then if its in a marked location
                    # tan or rad direction
                    if (np.pi/4 < psi < 3*np.pi/4) or (5*np.pi/4 < psi < 7*np.pi/4):
                        Meshdata[i,28] = 1  # tangential 
                        Meshdata[i,24] += 10  # modify connector type 
                    else:
                        Meshdata[i,28] = 2 # radial
                        Meshdata[i,24] += 20 

    # Replace nodal coordinates with nodal indices
    Meshdata[:,0:2] = np.concatenate((connector_t_bot_connectivity,connector_t_reg_connectivity,connector_t_top_connectivity,connector_l_connectivity))
    Meshdata = np.delete(Meshdata,[2,3,4,5],1)
    
    # np.save(Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName+'-mesh.txt'), Meshdata)

    np.savetxt(Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName+'-mesh.txt'), Meshdata, fmt='%.16g', delimiter=' '\
    ,header='# Connector Data Generated with RingsPy Mesh Generation Tool\n\
    Number of bot connectors\n'+ str(nel_con_tbot) +
    '\n\
    Number of reg connectors\n'+ str(nel_con_treg) +
    '\n\
    Number of top connectors\n'+ str(nel_con_ttop) +
    '\n\
    Number of long connectors\n'+ str(nel_con_l) +
    '\n\
    [inode jnode centerx centery centerz dx1 dy1 dz1 dx2 dy2 dz2 n1x n1y n1z n2x n2y n2z width height random_field connector_flag knot_flag precrack_flag short_flag ray_flag]', comments='')  

    # pr.disable()
    # loc = os.path.join(r"C:\Users\SusanAlexisBrown\woodWorkbench", geoName, 'connProfile.cProf')
    # pr.dump_stats(loc)
    # p = pstats.Stats(loc)
    # p.strip_dirs().sort_stats('cumulative').print_stats(10)

    return Meshdata,conn_l_tangents,height_connector_t


def VisualizationFiles(geoName,NURBS_degree,nlayers,npt_per_layer_vtk,all_pts_3D,\
                       nsegments,nridge,voronoi_ridges,all_ridges,nvertex,\
                       nconnector_t,nconnector_l,nctrlpt_per_beam,ConnMeshData,\
                       conn_l_tangents,all_vertices_2D, delaun_nodes, delaun_elems):
    # pr = cProfile.Profile()
    # pr.enable()

    # Calculate model parameters
    ngrain = nvertex
    ninterval_per_beam_vtk = int((nctrlpt_per_beam-1)/2) # 2 layers ----> 1 interval
    nconnector_t_per_beam = int((nctrlpt_per_beam-1)/NURBS_degree+1)

    # Vertex Data for VTK use
    VTKvertices = np.copy(all_pts_3D)
    
    # Connectivity for VTK use
    # Vertex
    npt_total_vtk = nlayers*npt_per_layer_vtk
    vertex_connectivity_vtk = np.linspace(0,npt_total_vtk-1,npt_total_vtk)
    
    # Beam
    nbeam_total_vtk = ngrain*nsegments*ninterval_per_beam_vtk
    beam_connectivity_vtk = np.zeros((nbeam_total_vtk,3)) # 3 is the number of points per beam in Paraview
    for i in range(0,nsegments):
        for j in range(0,ninterval_per_beam_vtk):
            for k in range(0,ngrain):
                irow = i*ninterval_per_beam_vtk*ngrain + j*ngrain + k
                ivertex = (i + (i*ninterval_per_beam_vtk+j)*2)*npt_per_layer_vtk + k
                beam_connectivity_vtk[irow,:] = (ivertex,ivertex+2*npt_per_layer_vtk,ivertex+npt_per_layer_vtk)
    # Transverse connectors 
    connector_t_connectivity_vtk = np.zeros((nconnector_t,2))
    for i in range(0,nsegments):
        for j in range(0,nconnector_t_per_beam):
            for k in range(0,nridge):
                irow = i*nconnector_t_per_beam*nridge + j*nridge + k
                connector_t_connectivity_vtk[irow,:] = (voronoi_ridges[k][0]+i*npt_per_layer_vtk*nctrlpt_per_beam+j*NURBS_degree*npt_per_layer_vtk,voronoi_ridges[k][1]+i*npt_per_layer_vtk*nctrlpt_per_beam+j*NURBS_degree*npt_per_layer_vtk)
    
    # Longitudinal connectors
    connector_l_connectivity_vtk = np.zeros((nconnector_l,2))
    nwings = all_vertices_2D[:,2].astype(int)
    irow_conn_l = 0
    for i in range(0,nsegments-1): # loop over layers of longitudinal connectors
        for j in range(0,ngrain): # loop over grains in each layer
            nw = nwings[j]
            irow = i*ngrain + j
            for k in range(0,nw): # loop over wings of each grain
                connector_l_connectivity_vtk[irow_conn_l,:] = (beam_connectivity_vtk[irow+(ninterval_per_beam_vtk-1)*ngrain,1],beam_connectivity_vtk[irow+(ninterval_per_beam_vtk-1)*ngrain,1]+npt_per_layer_vtk)
                irow_conn_l += 1
    
    # Quad - Beam Wings
    nquad = nridge*2
    nquad_total = nquad*nsegments*ninterval_per_beam_vtk
    quad_connectivity = np.zeros((nquad_total,8))
    for i in range(0,nsegments):
        for j in range(0,ninterval_per_beam_vtk):
            for k in range(0,nridge):
                for l in range(0,2):
                    irow = i*ninterval_per_beam_vtk*nquad + j*nquad + l*nridge + k
                    ivertex = i + (i*ninterval_per_beam_vtk+j)*2
                    quad_connectivity[irow,:] = (all_ridges[k][l]+ivertex*npt_per_layer_vtk,all_ridges[k][l]+(ivertex+2)*npt_per_layer_vtk, \
                                                 all_ridges[k][l+2]+(ivertex+2)*npt_per_layer_vtk,all_ridges[k][l+2]+ivertex*npt_per_layer_vtk, \
                                                 all_ridges[k][l]+(ivertex+1)*npt_per_layer_vtk,all_ridges[k][l+4]+(ivertex+2)*npt_per_layer_vtk, \
                                                 all_ridges[k][l+2]+(ivertex+1)*npt_per_layer_vtk,all_ridges[k][l+4]+ivertex*npt_per_layer_vtk)

    # Quad - Connector Cross-sections
    nquad_conn = nconnector_t + nconnector_l
    nquad_conn_total = nquad_conn
    
    Quad_center = np.copy(ConnMeshData[:,2:5])
    
    ##### Set normals of longitudinal connectors to be vertical normal = (0,0,1)
    ConnMeshData[nconnector_t:nquad_conn_total,5:7] = 0
    ConnMeshData[nconnector_t:nquad_conn_total,8:10] = 0
    #### (Need to changed for inclined beam axis in the future)
    
    Quad_normal1 = np.copy(ConnMeshData[:,5:8])
    Quad_normal2 = np.copy(ConnMeshData[:,8:11])
    Quad_length1 = np.linalg.norm(Quad_normal1, axis=1)
    Quad_length2 = np.linalg.norm(Quad_normal2, axis=1)
    Quad_width = np.copy(ConnMeshData[:,17])
    Quad_height = np.copy(ConnMeshData[:,18])
    Quad_normal = Quad_normal1/Quad_length1[:, np.newaxis]
    Quad_tangent = np.zeros((nquad_conn_total,3))
    Quad_tangent[0:nconnector_t] = np.tile(np.array([0.0,0.0,1.0]),(nconnector_t,1)) # assume a tangent of (0,0,1) for all conn_t

    Quad_tangent[nconnector_t:nquad_conn] = conn_l_tangents # conn_l
    Quad_bitangent = np.cross(Quad_normal,Quad_tangent)
    
    nconnector_t_bot = int(nconnector_t/3)
    nconnector_t_top = int(nconnector_t/3)
    # Add eccentricities to bot/top transverse connectors
    Quad_center[0:nconnector_t_bot,:] = Quad_center[0:nconnector_t_bot,:] + Quad_tangent[0:nconnector_t_bot,:]*Quad_height[0:nconnector_t_bot,np.newaxis]/2 # bot connectors
    Quad_center[(nconnector_t-nconnector_t_top):nconnector_t,:] = Quad_center[(nconnector_t-nconnector_t_top):nconnector_t,:] - Quad_tangent[(nconnector_t-nconnector_t_top):nconnector_t,:]*Quad_height[(nconnector_t-nconnector_t_top):nconnector_t,np.newaxis]/2
    
    quad_conn_vertices = np.zeros((nquad_conn_total*4,3))
    quad_conn_vertices[0:nquad_conn_total,:] = Quad_center - Quad_tangent*Quad_height[:, np.newaxis]/2 - Quad_bitangent*Quad_width[:, np.newaxis]/2
    quad_conn_vertices[nquad_conn_total:2*nquad_conn_total,:] = Quad_center - Quad_tangent*Quad_height[:, np.newaxis]/2 + Quad_bitangent*Quad_width[:, np.newaxis]/2
    quad_conn_vertices[2*nquad_conn_total:3*nquad_conn_total,:] = Quad_center + Quad_tangent*Quad_height[:, np.newaxis]/2 + Quad_bitangent*Quad_width[:, np.newaxis]/2
    quad_conn_vertices[3*nquad_conn_total:4*nquad_conn_total,:] = Quad_center + Quad_tangent*Quad_height[:, np.newaxis]/2 - Quad_bitangent*Quad_width[:, np.newaxis]/2
    
    quad_conn_connectivity = np.linspace(0,nquad_conn_total*4-1,nquad_conn_total*4)
    quad_conn_connectivity = np.reshape(quad_conn_connectivity, [4,-1]).T
    quad_conn_connectivity = quad_conn_connectivity + npt_total_vtk

    # Modify Vertex Data by adding quad_conn_vertices
    VTKvertices_quad_conn = np.vstack((VTKvertices,quad_conn_vertices))
    npt_total_vtk_quad_conn = VTKvertices_quad_conn.shape[0]

    # Hex - Connector Volumes
    
    hex_conn_vertices = np.zeros((nquad_conn_total*4*3,3))
    hex_conn_vertices[0:nquad_conn_total,:] = Quad_center - Quad_tangent*Quad_height[:, np.newaxis]/2 - Quad_bitangent*Quad_width[:, np.newaxis]/2
    hex_conn_vertices[nquad_conn_total:2*nquad_conn_total,:] = Quad_center - Quad_tangent*Quad_height[:, np.newaxis]/2 + Quad_bitangent*Quad_width[:, np.newaxis]/2
    hex_conn_vertices[2*nquad_conn_total:3*nquad_conn_total,:] = Quad_center + Quad_tangent*Quad_height[:, np.newaxis]/2 + Quad_bitangent*Quad_width[:, np.newaxis]/2
    hex_conn_vertices[3*nquad_conn_total:4*nquad_conn_total,:] = Quad_center + Quad_tangent*Quad_height[:, np.newaxis]/2 - Quad_bitangent*Quad_width[:, np.newaxis]/2
    
    hex_conn_vertices[4*nquad_conn_total:5*nquad_conn_total,:] = Quad_center - Quad_tangent*Quad_height[:, np.newaxis]/2 - Quad_bitangent*Quad_width[:, np.newaxis]/2 - Quad_normal*Quad_length1[:, np.newaxis]
    hex_conn_vertices[5*nquad_conn_total:6*nquad_conn_total,:] = Quad_center - Quad_tangent*Quad_height[:, np.newaxis]/2 + Quad_bitangent*Quad_width[:, np.newaxis]/2 - Quad_normal*Quad_length1[:, np.newaxis]
    hex_conn_vertices[6*nquad_conn_total:7*nquad_conn_total,:] = Quad_center + Quad_tangent*Quad_height[:, np.newaxis]/2 + Quad_bitangent*Quad_width[:, np.newaxis]/2 - Quad_normal*Quad_length1[:, np.newaxis]
    hex_conn_vertices[7*nquad_conn_total:8*nquad_conn_total,:] = Quad_center + Quad_tangent*Quad_height[:, np.newaxis]/2 - Quad_bitangent*Quad_width[:, np.newaxis]/2 - Quad_normal*Quad_length1[:, np.newaxis]

    hex_conn_vertices[8*nquad_conn_total:9*nquad_conn_total,:] = Quad_center - Quad_tangent*Quad_height[:, np.newaxis]/2 - Quad_bitangent*Quad_width[:, np.newaxis]/2 + Quad_normal*Quad_length2[:, np.newaxis]
    hex_conn_vertices[9*nquad_conn_total:10*nquad_conn_total,:] = Quad_center - Quad_tangent*Quad_height[:, np.newaxis]/2 + Quad_bitangent*Quad_width[:, np.newaxis]/2 + Quad_normal*Quad_length2[:, np.newaxis]
    hex_conn_vertices[10*nquad_conn_total:11*nquad_conn_total,:] = Quad_center + Quad_tangent*Quad_height[:, np.newaxis]/2 + Quad_bitangent*Quad_width[:, np.newaxis]/2 + Quad_normal*Quad_length2[:, np.newaxis]
    hex_conn_vertices[11*nquad_conn_total:12*nquad_conn_total,:] = Quad_center + Quad_tangent*Quad_height[:, np.newaxis]/2 - Quad_bitangent*Quad_width[:, np.newaxis]/2 + Quad_normal*Quad_length2[:, np.newaxis]
    
    hex_conn1_connectivity = np.linspace(0,nquad_conn_total*4*2-1,nquad_conn_total*4*2)
    hex_conn1_connectivity = np.reshape(hex_conn1_connectivity, [8,-1]).T
    hex_conn2_connectivity = np.linspace(0,nquad_conn_total*4*2-1,nquad_conn_total*4*2)
    hex_conn2_connectivity = np.reshape(hex_conn2_connectivity, [8,-1]).T
    hex_conn2_connectivity[:,4:8] = hex_conn2_connectivity[:,4:8] + nquad_conn_total*4
    hex_conn_connectivity = np.vstack((hex_conn1_connectivity,hex_conn2_connectivity))
    hex_conn_connectivity = hex_conn_connectivity + npt_total_vtk

    # Modify Vertex Data by adding hex_conn_vertices
    VTKvertices_hex_conn = np.vstack((VTKvertices,hex_conn_vertices))
    npt_total_vtk_hex_conn = VTKvertices_hex_conn.shape[0]
  
    # =============================================================================
    # Paraview Visualization File
    collocation_flag_vtk = np.concatenate((np.ones(ngrain),np.zeros(npt_per_layer_vtk*NURBS_degree-ngrain)))
    collocation_flag_vtk = np.concatenate((np.tile(collocation_flag_vtk, nconnector_t_per_beam-1),np.concatenate((np.ones(ngrain),np.zeros(npt_per_layer_vtk-ngrain)))))
    collocation_flag_vtk = np.tile(collocation_flag_vtk, nsegments)

    # =============================================================================
    # Paraview Flow File
    
    delaun_nodes_sorted = delaun_nodes[np.argsort(delaun_nodes[:,2]),:]
    delaun_nodes_sorted = delaun_nodes_sorted[np.argsort(delaun_nodes_sorted[:, 1], kind='stable'),:]

    npts_flow_vtk = np.size(delaun_nodes,axis=0)
    ncell_flow = np.size(delaun_elems,axis=0)

    vtkfile_flow = open (Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName + '_flow'+'.vtu'),'w')
    
    vtkfile_flow.write('<VTKFile type="UnstructuredGrid" version="2.0" byte_order="LittleEndian">'+'\n')
    vtkfile_flow.write('<UnstructuredGrid>'+'\n')
    vtkfile_flow.write('<Piece NumberOfPoints="'+str(npts_flow_vtk)+'"'+' '+'NumberOfCells="'+str(ncell_flow)+'">'+'\n')
    
    # <Points>
    vtkfile_flow.write('<Points>'+'\n')
    vtkfile_flow.write('<DataArray type="Float64" NumberOfComponents="3" format="ascii">'+'\n')
    for i in range(0,npts_flow_vtk):
        X,Y,Z = delaun_nodes_sorted[i,1:5]
        vtkfile_flow.write(' '+'%11.8e'%X+'  '+'%11.8e'%Y+'  '+'%11.8e'%Z+'\n')
    vtkfile_flow.write('</DataArray>'+'\n')
    vtkfile_flow.write('</Points>'+'\n')
    # </Points>
    
    # <PointData> 
    # </PointData> 
      
    # <Cells>
    vtkfile_flow.write('<Cells>'+'\n')
    
    # Cell connectivity
    vtkfile_flow.write('<DataArray type="Int32" Name="connectivity" format="ascii">'+'\n')
    for i in range(0,ncell_flow):
        n1,n2 = delaun_elems[i,0:2]
        s1 = np.where(delaun_nodes_sorted[:,0] == n1)[0]
        s2 = np.where(delaun_nodes_sorted[:,0] == n2)[0]
        vtkfile_flow.write(' '+'%i'%s1+'  '+'%i'%s2+'\n')
    vtkfile_flow.write('</DataArray>'+'\n')
    
    # Cell offsets
    vtkfile_flow.write('<DataArray type="Int32" Name="offsets" format="ascii">'+'\n')
    offset = 2
    for n in range(0,ncell_flow):
        vtkfile_flow.write(' '+'%i'%offset+'\n')
        offset += 2
    vtkfile_flow.write('</DataArray>'+'\n')
    
    # Cell type
    vtkfile_flow.write('<DataArray type="UInt8" Name="types" format="ascii">'+'\n')
    for i in range(0,ncell_flow):
        vtkfile_flow.write(str(3)+'\n')

    vtkfile_flow.write('</DataArray>'+'\n')
    
    vtkfile_flow.write('</Cells>'+'\n')
    # </Cells>
    
    vtkfile_flow.write('</Piece>'+'\n')
    #</Piece>
    
    vtkfile_flow.write('</UnstructuredGrid>'+'\n')
    #</UnstructuredGrid>
    
    vtkfile_flow.write('</VTKFile>'+'\n')
    #</VTKFile>
    
    vtkfile_flow.close()
    

    # =============================================================================
    # Paraview Vertices File
    VTKcell_types_vertices = (np.ones(npt_total_vtk)).astype(int)

    ncell_vertices = VTKcell_types_vertices.shape[0]

    vtkfile_vertices = open (Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName + '_vertices'+'.vtu'),'w')
    
    vtkfile_vertices.write('<VTKFile type="UnstructuredGrid" version="2.0" byte_order="LittleEndian">'+'\n')
    vtkfile_vertices.write('<UnstructuredGrid>'+'\n')
    vtkfile_vertices.write('<Piece NumberOfPoints="'+str(npt_total_vtk)+'"'+' '+'NumberOfCells="'+str(ncell_vertices)+'">'+'\n')
    
    # <Points>
    vtkfile_vertices.write('<Points>'+'\n')
    vtkfile_vertices.write('<DataArray type="Float64" NumberOfComponents="3" format="ascii">'+'\n')
    for i in range(0,npt_total_vtk):
        X,Y,Z = VTKvertices[i]
        vtkfile_vertices.write(' '+'%11.8e'%X+'  '+'%11.8e'%Y+'  '+'%11.8e'%Z+'\n')
    vtkfile_vertices.write('</DataArray>'+'\n')
    vtkfile_vertices.write('</Points>'+'\n')
    # </Points>
    
    # <PointData> 
    vtkfile_vertices.write("<"+"PointData"\
        +" "+"Tensors="+'"'+""+'"'\
        +" "+"Vevtors="+'"'+""+'"'\
        +" "+"Scalars="+'"'+"IGAcollocation_flag"+'"'+">"+'\n')
    
    # Point Data
    vtkfile_vertices.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"IGAcollocation_flag" format='+'"'+"ascii"+'"'+">"+'\n')
    for i in range(0,npt_total_vtk):
        X = collocation_flag_vtk[i]
        vtkfile_vertices.write('%11.8e'%X+'\n')
    vtkfile_vertices.write("</DataArray>"+'\n')
    
    vtkfile_vertices.write('</PointData>'+'\n')
    # </PointData> 
    
    
    # <Cells>
    vtkfile_vertices.write('<Cells>'+'\n')
    
    # Cell connectivity
    vtkfile_vertices.write('<DataArray type="Int32" Name="connectivity" format="ascii">'+'\n')
    for x in vertex_connectivity_vtk.astype(int):
        vtkfile_vertices.write("%s\n" % x)
    vtkfile_vertices.write('\n')  
    vtkfile_vertices.write('</DataArray>'+'\n')
    
    # Cell offsets
    vtkfile_vertices.write('<DataArray type="Int32" Name="offsets" format="ascii">'+'\n')
    current_offset = 0
    for element in vertex_connectivity_vtk:
        element_offset = 1
        current_offset += element_offset
        vtkfile_vertices.write(str(current_offset)+'\n')
    vtkfile_vertices.write('</DataArray>'+'\n')
    
    # Cell type
    vtkfile_vertices.write('<DataArray type="UInt8" Name="types" format="ascii">'+'\n')
    for i in range(0,ncell_vertices):
        element = VTKcell_types_vertices[i]
        vtkfile_vertices.write(str(element)+'\n')

    vtkfile_vertices.write('</DataArray>'+'\n')
    
    vtkfile_vertices.write('</Cells>'+'\n')
    # </Cells>
    
    vtkfile_vertices.write('</Piece>'+'\n')
    #</Piece>
    
    vtkfile_vertices.write('</UnstructuredGrid>'+'\n')
    #</UnstructuredGrid>
    
    vtkfile_vertices.write('</VTKFile>'+'\n')
    #</VTKFile>
    
    vtkfile_vertices.close()

    # =============================================================================
    # Paraview Beam File

    Beam_width = np.copy(all_vertices_2D[:,12])
    Beam_width_vtk = np.tile(Beam_width,(ninterval_per_beam_vtk*nsegments))
    Beam_width_vtk = np.concatenate((Beam_width_vtk,np.zeros(nquad_total)))

    VTKcell_types_beams = np.concatenate((21*np.ones(nbeam_total_vtk),23*np.ones(nquad_total))).astype(int)
  
    ncell_beams = VTKcell_types_beams.shape[0]
    
    vtkfile_beams = open (Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName + '_beams'+'.vtu'),'w')
    
    vtkfile_beams.write('<VTKFile type="UnstructuredGrid" version="2.0" byte_order="LittleEndian">'+'\n')
    vtkfile_beams.write('<UnstructuredGrid>'+'\n')
    vtkfile_beams.write('<Piece NumberOfPoints="'+str(npt_total_vtk)+'"'+' '+'NumberOfCells="'+str(ncell_beams)+'">'+'\n')
    
    # <Points>
    vtkfile_beams.write('<Points>'+'\n')
    vtkfile_beams.write('<DataArray type="Float64" NumberOfComponents="3" format="ascii">'+'\n')
    for i in range(0,npt_total_vtk):
        X,Y,Z = VTKvertices[i]
        vtkfile_beams.write(' '+'%11.8e'%X+'  '+'%11.8e'%Y+'  '+'%11.8e'%Z+'\n')
    vtkfile_beams.write('</DataArray>'+'\n')
    vtkfile_beams.write('</Points>'+'\n')
    # </Points>
    
    # <PointData> 
    # </PointData> 
      
    # <Cells>
    vtkfile_beams.write('<Cells>'+'\n')
    
    # Cell connectivity
    vtkfile_beams.write('<DataArray type="Int32" Name="connectivity" format="ascii">'+'\n')
    vtkfile_beams.write("\n".join(" ".join(map(str, x)) for x in beam_connectivity_vtk.astype(int)))
    vtkfile_beams.write('\n')
    vtkfile_beams.write("\n".join(" ".join(map(str, x)) for x in quad_connectivity.astype(int)))
    vtkfile_beams.write('\n') 
    vtkfile_beams.write('</DataArray>'+'\n')
    
    # Cell offsets
    vtkfile_beams.write('<DataArray type="Int32" Name="offsets" format="ascii">'+'\n')
    current_offset = 0
    for element in beam_connectivity_vtk:
        element_offset = len(element)
        current_offset += element_offset
        vtkfile_beams.write(str(current_offset)+'\n')
    for element in quad_connectivity:
        element_offset = len(element)
        current_offset += element_offset
        vtkfile_beams.write(str(current_offset)+'\n')
    vtkfile_beams.write('</DataArray>'+'\n')
    
    # Cell type
    vtkfile_beams.write('<DataArray type="UInt8" Name="types" format="ascii">'+'\n')
    for i in range(0,ncell_beams):
        element = VTKcell_types_beams[i]
        vtkfile_beams.write(str(element)+'\n')

    vtkfile_beams.write('</DataArray>'+'\n')
    
    vtkfile_beams.write('</Cells>'+'\n')
    # </Cells>

    # <CellData>
    vtkfile_beams.write("<"+"CellData"\
            +" "+"Tensors="+'"'+""+'"'\
            +" "+"Vectors="+'"'+""+'"'\
            +" "+"Scalars="+'"'+"beam_width"+'"'+">"+'\n')
    
    vtkfile_beams.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"beam_width"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
    for i in range(0,ncell_beams):
        X = Beam_width_vtk[i]
        vtkfile_beams.write('%11.8e'%X+'\n')
    vtkfile_beams.write('</DataArray>'+'\n')
    
    vtkfile_beams.write('</CellData>'+'\n')

    vtkfile_beams.write('</Piece>'+'\n')
    #</Piece>
    
    vtkfile_beams.write('</UnstructuredGrid>'+'\n')
    #</UnstructuredGrid>
    
    vtkfile_beams.write('</VTKFile>'+'\n')
    #</VTKFile>
    
    vtkfile_beams.close()
    
    # =============================================================================
    # Paraview Connector (Axis + Center section) File
    VTKcell_types_conns = np.concatenate((3*np.ones(nconnector_t),3*np.ones(nconnector_l),9*np.ones(nconnector_t),9*np.ones(nconnector_l))).astype(int)
    ncell_conns = VTKcell_types_conns.shape[0]
    
    Quad_width_vtk = np.tile(Quad_width,(2))

    vtkfile_conns = open (Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName + '_conns'+'.vtu'),'w')
    
    vtkfile_conns.write('<VTKFile type="UnstructuredGrid" version="2.0" byte_order="LittleEndian">'+'\n')
    vtkfile_conns.write('<UnstructuredGrid>'+'\n')
    vtkfile_conns.write('<Piece NumberOfPoints="'+str(npt_total_vtk_quad_conn)+'"'+' '+'NumberOfCells="'+str(ncell_conns)+'">'+'\n')
    
    # <Points>
    vtkfile_conns.write('<Points>'+'\n')
    vtkfile_conns.write('<DataArray type="Float64" NumberOfComponents="3" format="ascii">'+'\n')
    for i in range(0,npt_total_vtk_quad_conn):
        X,Y,Z = VTKvertices_quad_conn[i]
        vtkfile_conns.write(' '+'%11.8e'%X+'  '+'%11.8e'%Y+'  '+'%11.8e'%Z+'\n')
    vtkfile_conns.write('</DataArray>'+'\n')
    vtkfile_conns.write('</Points>'+'\n')
    # </Points>
    
    # <PointData> 
    # </PointData> 
      
    # <Cells>
    vtkfile_conns.write('<Cells>'+'\n')
    
    # Cell connectivity
    vtkfile_conns.write('<DataArray type="Int32" Name="connectivity" format="ascii">'+'\n')
    vtkfile_conns.write("\n".join(" ".join(map(str, x)) for x in connector_t_connectivity_vtk.astype(int)))
    vtkfile_conns.write('\n')
    vtkfile_conns.write("\n".join(" ".join(map(str, x)) for x in connector_l_connectivity_vtk.astype(int)))
    vtkfile_conns.write('\n')
    vtkfile_conns.write("\n".join(" ".join(map(str, x)) for x in quad_conn_connectivity.astype(int)))
    vtkfile_conns.write('\n')   
    vtkfile_conns.write('</DataArray>'+'\n')
    
    # Cell offsets
    vtkfile_conns.write('<DataArray type="Int32" Name="offsets" format="ascii">'+'\n')
    current_offset = 0
    for element in connector_t_connectivity_vtk:
        element_offset = len(element)
        current_offset += element_offset
        vtkfile_conns.write(str(current_offset)+'\n')
    for element in connector_l_connectivity_vtk:
        element_offset = len(element)
        current_offset += element_offset
        vtkfile_conns.write(str(current_offset)+'\n')
    for element in quad_conn_connectivity:
        element_offset = len(element)
        current_offset += element_offset
        vtkfile_conns.write(str(current_offset)+'\n')
    vtkfile_conns.write('</DataArray>'+'\n')
    
    # Cell type
    vtkfile_conns.write('<DataArray type="UInt8" Name="types" format="ascii">'+'\n')
    for i in range(0,ncell_conns):
        element = VTKcell_types_conns[i]
        vtkfile_conns.write(str(element)+'\n')

    vtkfile_conns.write('</DataArray>'+'\n')
    
    vtkfile_conns.write('</Cells>'+'\n')
    # </Cells>

    # <CellData>
    vtkfile_conns.write("<"+"CellData"\
            +" "+"Tensors="+'"'+""+'"'\
            +" "+"Vevtors="+'"'+""+'"'\
            +" "+"Scalars="+'"'+"Connector_width"+'"'+">"+'\n')
    
    vtkfile_conns.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"Connector_width"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
    for i in range(0,ncell_conns):
        X = Quad_width_vtk[i]
        vtkfile_conns.write('%11.8e'%X+'\n')
    vtkfile_conns.write('</DataArray>'+'\n')

    vtkfile_conns.write('</CellData>'+'\n')
        
    vtkfile_conns.write('</Piece>'+'\n')
    #</Piece>
    
    vtkfile_conns.write('</UnstructuredGrid>'+'\n')
    #</UnstructuredGrid>
    
    vtkfile_conns.write('</VTKFile>'+'\n')
    #</VTKFile>
    
    vtkfile_conns.close()
    
    # =============================================================================
    # Paraview Connector (Volume) File
    VTKcell_types_conns_vol = np.concatenate((12*np.ones(nconnector_t),12*np.ones(nconnector_l),12*np.ones(nconnector_t),12*np.ones(nconnector_l))).astype(int)
    ncell_conns_vol = VTKcell_types_conns_vol.shape[0]
    
    Quad_width_vtk = np.tile(Quad_width,(2))
    knot_vtk = np.tile(np.copy(ConnMeshData[:,21]),(2))
    RF_vtk = np.tile(np.copy(ConnMeshData[:,19]),(2))
    precrack_vtk = np.tile(np.copy(ConnMeshData[:,22]),(2))
    short_vtk = np.tile(np.copy(ConnMeshData[:,23]),(2))
    rays_vtk = np.tile(np.copy(ConnMeshData[:,24]),(2))

    vtkfile_conns_vol = open (Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName + '_conns_vol'+'.vtu'),'w')
    
    vtkfile_conns_vol.write('<VTKFile type="UnstructuredGrid" version="2.0" byte_order="LittleEndian">'+'\n')
    vtkfile_conns_vol.write('<UnstructuredGrid>'+'\n')
    vtkfile_conns_vol.write('<Piece NumberOfPoints="'+str(npt_total_vtk_hex_conn)+'"'+' '+'NumberOfCells="'+str(ncell_conns_vol)+'">'+'\n')
    
    # <Points>
    vtkfile_conns_vol.write('<Points>'+'\n')
    vtkfile_conns_vol.write('<DataArray type="Float64" NumberOfComponents="3" format="ascii">'+'\n')
    for i in range(0,npt_total_vtk_hex_conn):
        X,Y,Z = VTKvertices_hex_conn[i]
        vtkfile_conns_vol.write(' '+'%11.8e'%X+'  '+'%11.8e'%Y+'  '+'%11.8e'%Z+'\n')
    vtkfile_conns_vol.write('</DataArray>'+'\n')
    vtkfile_conns_vol.write('</Points>'+'\n')
    # </Points>
    
    # <PointData> 
    # </PointData> 
      
    # <Cells>
    vtkfile_conns_vol.write('<Cells>'+'\n')
    
    # Cell connectivity
    vtkfile_conns_vol.write('<DataArray type="Int32" Name="connectivity" format="ascii">'+'\n')
    vtkfile_conns_vol.write("\n".join(" ".join(map(str, x)) for x in hex_conn_connectivity.astype(int)))
    vtkfile_conns_vol.write('\n')   
    vtkfile_conns_vol.write('</DataArray>'+'\n')
    
    # Cell offsets
    vtkfile_conns_vol.write('<DataArray type="Int32" Name="offsets" format="ascii">'+'\n')
    current_offset = 0
    for element in hex_conn_connectivity:
        element_offset = len(element)
        current_offset += element_offset
        vtkfile_conns_vol.write(str(current_offset)+'\n')
    vtkfile_conns_vol.write('</DataArray>'+'\n')
    
    # Cell type
    vtkfile_conns_vol.write('<DataArray type="UInt8" Name="types" format="ascii">'+'\n')
    for i in range(0,ncell_conns_vol):
        element = VTKcell_types_conns_vol[i]
        vtkfile_conns_vol.write(str(element)+'\n')

    vtkfile_conns_vol.write('</DataArray>'+'\n')
    
    vtkfile_conns_vol.write('</Cells>'+'\n')
    # </Cells>
    
    # <CellData>
    vtkfile_conns_vol.write("<"+"CellData"\
            +" "+"Tensors="+'"'+""+'"'\
            +" "+"Vectors="+'"'+""+'"'\
            +" "+"Scalars="+'"'+"Connector_width"+'"'+">"+'\n')
    
    vtkfile_conns_vol.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"Connector_width"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
    for i in range(0,ncell_conns_vol):
        X = Quad_width_vtk[i]
        vtkfile_conns_vol.write('%11.8e'%X+'\n')
    vtkfile_conns_vol.write('</DataArray>'+'\n')

    # add knot visualization
    vtkfile_conns_vol.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"knotFlag"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
    for i in range(0,ncell_conns_vol):
        K = knot_vtk[i]
        vtkfile_conns_vol.write('%11.8e'%K+'\n')
    vtkfile_conns_vol.write('</DataArray>'+'\n')
    # add random field visualization
    vtkfile_conns_vol.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"randomFlag"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
    for i in range(0,ncell_conns_vol):
        RF = RF_vtk[i]
        vtkfile_conns_vol.write('%11.8e'%RF+'\n')
    vtkfile_conns_vol.write('</DataArray>'+'\n')
    # add precrack visualization
    vtkfile_conns_vol.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"precrackFlag"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
    for i in range(0,ncell_conns_vol):
        PC = precrack_vtk[i]
        vtkfile_conns_vol.write('%11.8e'%PC+'\n')
    vtkfile_conns_vol.write('</DataArray>'+'\n')
    # add short connectors visualization
    vtkfile_conns_vol.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"shortFlag"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
    for i in range(0,ncell_conns_vol):
        SC = short_vtk[i]
        vtkfile_conns_vol.write('%11.8e'%SC+'\n')
    vtkfile_conns_vol.write('</DataArray>'+'\n')
    # add radial ray visualization
    vtkfile_conns_vol.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"rayFlag"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
    for i in range(0,ncell_conns_vol):
        RR = rays_vtk[i]
        vtkfile_conns_vol.write('%11.8e'%RR+'\n')
    vtkfile_conns_vol.write('</DataArray>'+'\n')

    vtkfile_conns_vol.write('</CellData>'+'\n')
    
    vtkfile_conns_vol.write('</Piece>'+'\n')
    #</Piece>
    
    vtkfile_conns_vol.write('</UnstructuredGrid>'+'\n')
    #</UnstructuredGrid>
    
    vtkfile_conns_vol.write('</VTKFile>'+'\n')
    #</VTKFile>
    
    vtkfile_conns_vol.close()
    
    # pr.disable()
    # loc = os.path.join(r"C:\Users\SusanAlexisBrown\woodWorkbench", geoName, 'connProfile.cProf')
    # pr.dump_stats(loc)
    # p = pstats.Stats(loc)
    # p.strip_dirs().sort_stats('cumulative').print_stats(10)


def BezierExtraction(NURBS_degree,nbeam_total):
    nctrlpt_per_beam = 2*NURBS_degree + 1
    knotVec_per_beam = np.concatenate((np.zeros(NURBS_degree),(np.linspace(0,1,int((nctrlpt_per_beam-1)/NURBS_degree+1))),np.ones(NURBS_degree)))
    knotVec = np.tile(knotVec_per_beam, (nbeam_total,1))
    return knotVec


def BezierBeamFile(geoName,NURBS_degree,nctrlpt_per_beam,\
                   nconnector_t_per_beam,npatch,knotVec):
############################# Abaqus txt File #################################
    txtfile = open (Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName + 'IGA.txt'),'w')
    # txtfile = open (os.path.join(path,geoName)+'IGA.txt','w')
    txtfile.write('# Dimension of beam elements \n')
    txtfile.write('1 \n')
    txtfile.write('# Order of basis function \n')
    txtfile.write('{:d} \n'.format(NURBS_degree))
    txtfile.write('# Number of control points per patch \n')
    txtfile.write('{:d} \n'.format(nctrlpt_per_beam))
    txtfile.write('# Number of elements per patch \n') 
    txtfile.write('{:d} \n'.format(nconnector_t_per_beam-1))
    txtfile.write('# Number of Patches \n') 
    txtfile.write('{:d} \n'.format(npatch))
    # Loop over patches
    for ipatch in range(0,npatch):
        txtfile.write('{:s} \n'.format('PATCH-'+str(ipatch+1)))
        txtfile.write('Size of knot vectors \n') 
        txtfile.write('{:d} \n'.format(knotVec.shape[1])) 
        txtfile.write('knot vectors \n')
        for j in range(0,knotVec.shape[1]):
            txtfile.write('{:f} '.format(knotVec[ipatch,j])) 
        txtfile.write('\n')
        # txtfile.write('Weight of points \n') 
        # for i=1:noCtrPts
        #     txtfile.write('{:d}, {:f} \n'.format()) 
        # txtfile.write('control points \n') 
        # for i in range(0,numnode):
        #     txtfile.write('{:d}, {:#.9e}, {:#.9e},  {:#.9e} \n'.format(i+1,woodIGAvertices[i,0],woodIGAvertices[i,1],woodIGAvertices[i,2]))
        # txtfile.write('element connectivity \n') 
        # for i=1:nelem
        #         txtfile.write('{:d}',(ipatch-1)*nelem+i) 
        #         for j=1:nnode
        #             txtfile.write(', {:d} ', elementB(i,j)) 
    
        #         txtfile.write('\n') 
        # txtfile.write('element Bezier extraction operator \n') 
        # for i=1:nelem
        #         txtfile.write('{:d}',(ipatch-1)*nelem+i) 
        #         for j=1:nnode
        #             for k=1:nnode
        #                 txtfile.write(', {:d} ',C(j,k,i)) 
        #         txtfile.write('\n') 
    
    txtfile.close()


def ABQParams(nconnector_l,nconnector_l_precrack,nconnector_t,nconnector_t_precrack,\
                 cellwallthickness_early,cellwallthickness_late,height_connector_t,nbeamElem):
    
    # IGA beam parameters
    ninstance = 1
    nsvars_beam = 27 # number of svars for beam
    nsecgp = 4 # number of gauss points for beam sectional integration
    nsvars_secgp = 16 # number of svars at each gauss point
    iprops_beam = np.zeros(6)
    iprops_beam[0]    = 1 # section type index
    iprops_beam[1]    = 1 # number of instances
    iprops_beam[2]    = nconnector_t-nconnector_t_precrack # number of transverse connectors
    iprops_beam[3]    = nconnector_t_precrack # number of precracked transverse connectors
    iprops_beam[4]    = nconnector_l-nconnector_l_precrack # number of longitudinal connectors
    iprops_beam[5]    = nconnector_l_precrack # number of precracked longitudinal connectors
    iprops_beam = [int(x) for x in iprops_beam] 

    props_beam = np.zeros(16)
    props_beam[0]     = 1.5E-7 # Wood substance density [tonne/mm^3]
    props_beam[1]     = 0.8E+4 # Mesoscale elastic modulus [MPa]
    props_beam[2]     = 0.3 # Beam Poisson's ratio
    props_beam[3]     = cellwallthickness_early # Cross-sectional height [mm]
    props_beam[4]     = cellwallthickness_early # Cross-sectional width [mm]
    props_beam[5]     = 100 # Tensile Strength [MPa]
    props_beam[6]     = 200 # Tensile fracture energy [mJ/mm^2]
    props_beam[7]     = 4.1 # Shear Strength Ratio
    props_beam[8]     = 0.2 # Softening Exponent
    props_beam[9]     = 0.2 # Initial Friction
    props_beam[10]    = 0.0 # Asymptotic Friction
    props_beam[11]    = 600 # Transitional Stress [MPa]
    props_beam[12]    = 0.0 # Tensile Unloading
    props_beam[13]    = 0.0 # Shear Unloading
    props_beam[14]    = 0.0 # Shear Softening
    props_beam[15]    = 1.0 # Elastic Analysis Flag

    # Transverse connector parameters
    nsvars_conn_t = 32  # number of svars for transverse connector
    iprops_connector_t_bot = np.zeros(7)
    props_connector_t_bot = np.zeros(26)

    props_connector_t_bot[0]     = 1.5E-7 # Wood substance density [tonne/mm^3]
    props_connector_t_bot[1]     = 0.8E+4 # Mesoscale elastic modulus [MPa]
    props_connector_t_bot[2]     = 0.25 # Shear-Normal coupling coefficient
    props_connector_t_bot[3]     = height_connector_t # Connector height [mm]
    props_connector_t_bot[4]     = 0 # M-Distance [mm]
    props_connector_t_bot[5]     = -height_connector_t/2 # L-Distance [mm]
    props_connector_t_bot[6]     = 30.0 # Tensile Strength [MPa]
    props_connector_t_bot[7]     = 1.0 # Tensile characteristic length [mm] will be updated to # Tensile fracture energy [mJ/mm^2]
    props_connector_t_bot[8]     = 2.6 # Shear Strength Ratio
    props_connector_t_bot[9]     = 0.2 # Softening Exponent
    props_connector_t_bot[10]    = 0.2 # Initial Friction
    props_connector_t_bot[11]    = 0.0 # Asymptotic Friction
    props_connector_t_bot[12]    = 600 # Transitional Stress [MPa]
    props_connector_t_bot[12]    = 0.0 # Tensile Unloading
    props_connector_t_bot[14]    = 0.0 # Shear Unloading
    props_connector_t_bot[15]    = 0.0 # Shear Softening
    props_connector_t_bot[16]    = 0.0 # Elastic Analysis Flag
    props_connector_t_bot[17]    = 0.2 # Compressive Yielding Strength [MPa]
    props_connector_t_bot[18]    = 600 # Initial Hardening Modulus Ratio
    props_connector_t_bot[19]    = 0.0 # Transitional Strain Ratio
    props_connector_t_bot[20]    = 0.0 # Deviatoric Strain Threshold Ratio
    props_connector_t_bot[21]    = 0.0 # Deviatoric Damage Parameter
    props_connector_t_bot[22]    = 0.0 # Final Hardening Modulus Ratio
    props_connector_t_bot[23]    = 0.0 # Densification Ratio
    props_connector_t_bot[24]    = 0.0 # Volumetric Deviatoric Coupling
    props_connector_t_bot[25]    = 0.0 # Compressive Unloading

    iprops_connector_t_bot = [3,ninstance,nbeamElem,nconnector_t-nconnector_t_precrack,nconnector_t_precrack,nconnector_l-nconnector_l_precrack,nconnector_l_precrack]
    iprops_connector_t_bot = [int(x) for x in iprops_connector_t_bot] 
    iprops_connector_t_top = [2,ninstance,nbeamElem,nconnector_t-nconnector_t_precrack,nconnector_t_precrack,nconnector_l-nconnector_l_precrack,nconnector_l_precrack]
    iprops_connector_t_top = [int(x) for x in iprops_connector_t_top] 
    iprops_connector_t_reg = [1,ninstance,nbeamElem,nconnector_t-nconnector_t_precrack,nconnector_t_precrack,nconnector_l-nconnector_l_precrack,nconnector_l_precrack]
    iprops_connector_t_reg = [int(x) for x in iprops_connector_t_reg] 

    props_connector_t_reg = np.copy(props_connector_t_bot)
    props_connector_t_reg[3] = height_connector_t*2
    props_connector_t_reg[5] = 0
    props_connector_t_top = np.copy(props_connector_t_bot)
    props_connector_t_top[5] = height_connector_t/2

    # Longitudinal connector parameters
    nsvars_conn_l = 32  # number of svars for transverse connector
    iprops_connector_l = np.zeros(7)
    props_connector_l = np.zeros(24)

    props_connector_l[0]     = 1.5E-7 # Wood substance density [tonne/mm^3]
    props_connector_l[1]     = 0.8E+4 # Mesoscale elastic modulus [MPa]
    props_connector_l[2]     = 0.18 # Shear-Normal coupling coefficient
    props_connector_l[3]     = cellwallthickness_early*cellwallthickness_late # Connector sectional area [mm^2]
    props_connector_l[4]     = 3.0E+2 # Tensile Strength [MPa]
    props_connector_l[5]     = 0.2E+1# 0.0105 # Tensile characteristic length [mm] will be updated to # Tensile fracture energy [mJ/mm^2]
    props_connector_l[6]     = 2.6 # Shear Strength Ratio
    props_connector_l[7]     = 0.2 # Softening Exponent
    props_connector_l[8]     = 0.2 # Initial Friction
    props_connector_l[9]     = 0.0 # Asymptotic Friction
    props_connector_l[10]    = 600 # Transitional Stress [MPa]
    props_connector_l[11]    = 0.0 # Tensile Unloading
    props_connector_l[12]    = 0.0 # Shear Unloading
    props_connector_l[13]    = 0.0 # Shear Softening
    props_connector_l[14]    = 1.0 # Elastic Analysis Flag
    props_connector_l[15]    = 0.2 # Compressive Yielding Strength [MPa]
    props_connector_l[16]    = 600 # Initial Hardening Modulus Ratio
    props_connector_l[17]    = 0.0 # Transitional Strain Ratio
    props_connector_l[18]    = 0.0 # Deviatoric Strain Threshold Ratio
    props_connector_l[19]    = 0.0 # Deviatoric Damage Parameter
    props_connector_l[20]    = 0.0 # Final Hardening Modulus Ratio
    props_connector_l[21]    = 0.0 # Densification Ratio
    props_connector_l[22]    = 0.0 # Volumetric Deviatoric Coupling
    props_connector_l[23]    = 0.0 # Compressive Unloading

    iprops_connector_l = [1,ninstance,nbeamElem,nconnector_t-nconnector_t_precrack,nconnector_t_precrack,nconnector_l-nconnector_l_precrack,nconnector_l_precrack]
    iprops_connector_l = [int(x) for x in iprops_connector_l] 

    # Strain rate effect parameters
    props_strainrate = np.zeros(4)
    props_strainrate[0] = 0.0 # Strain rate effect flag 
    props_strainrate[1] = 5.0 # Physical time scaling factor
    props_strainrate[2] = 1.0E-5 # Strain rate effect constant c0
    props_strainrate[3] = 5.0E-2 # Strain rate effect constant c1


    return nsvars_beam, nsecgp, nsvars_secgp, iprops_beam, props_beam, \
        nsvars_conn_t, iprops_connector_t_bot, props_connector_t_bot, iprops_connector_t_reg, props_connector_t_reg, \
        iprops_connector_t_top, props_connector_t_top, \
        nsvars_conn_l, iprops_connector_l, props_connector_l, props_strainrate


def AbaqusFile(geoName,NURBS_degree,npatch,nsegments,woodIGAvertices,beam_connectivity,\
               connector_t_bot_connectivity,connector_t_reg_connectivity,\
               connector_t_top_connectivity,connector_l_connectivity,\
               length,props_beam,iprops_beam,props_connector_t_bot,iprops_connector_t_bot,\
               props_connector_t_reg,iprops_connector_t_reg,props_connector_t_top,\
               iprops_connector_t_top,props_connector_l,iprops_connector_l,props_strainrate,\
               timestep,totaltime,boundary_conditions,BC_velo_dof,BC_velo_value,\
               x_max,x_min,y_max,y_min,z_max,z_min,boundaries,nsvars_beam,nsvars_conn_t,\
               nsvars_conn_l,nsecgp,nsvars_secgp,cellwallthickness_early,merge_operation,merge_tol,\
               precrackFlag,precrack_elem):

    # Beam
    nelnode = NURBS_degree + 1
    noGPs = NURBS_degree + 1
    if NURBS_degree == 2:
        eltype = 201
    elif NURBS_degree == 3:
        eltype = 202
    else:
        print('Current NURBS degree is not supported. Check variable NURBS_degree.')
        exit()
    nelnode_connector = 2
    eltype_connector_t = 501
    eltype_connector_l = 601
    
    numnode = woodIGAvertices.shape[0]
    nelem = beam_connectivity.shape[0]
    nnode = beam_connectivity.shape[1]
    nel_con_tbot = connector_t_bot_connectivity.shape[0]
    nel_con_treg = connector_t_reg_connectivity.shape[0]
    nel_con_ttop = connector_t_top_connectivity.shape[0]
    nel_con_l = connector_l_connectivity.shape[0]
    nconnector_t_precrack = iprops_connector_l[4]
    nconnector_l_precrack = iprops_connector_l[6]
    nel_con = nelem + nel_con_tbot + nel_con_treg + nel_con_ttop\
                + nel_con_l + nconnector_t_precrack + nconnector_l_precrack
    
    xaxis = 1
    yaxis = 2
    zaxis = 3

    # Generate a .inp file which can be directly imported and played in Abaqus
    meshinpfile = open(Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName + '-mesh.inp'),'w')
    meshinpfile.write('*HEADING'+'\n')
    meshinpfile.write('** Job name: {0} Model name: Model-{1}\n'.format(geoName,geoName))
    meshinpfile.write('** Generated by Wood Mesh Generator V11.0\n')
    # PART
    meshinpfile.write('** \n')
    meshinpfile.write('** PARTS\n') 
    meshinpfile.write('** \n') 
    meshinpfile.write('*Part, name=Part-1\n')
    meshinpfile.write('*End Part\n')
    meshinpfile.write('** \n')
    # ASSEMBLY
    meshinpfile.write('** \n')
    meshinpfile.write('** ASSEMBLY\n') 
    meshinpfile.write('** \n') 
    meshinpfile.write('*Assembly, name=Assembly\n')
    meshinpfile.write('** \n') 
    # INSTANCE
    meshinpfile.write('*Instance, name=Part-1-1, part=Part-1\n')
    # nodes
    meshinpfile.write('*Node,nset=AllNodes\n')
    for i in range(0,numnode):
        meshinpfile.write('{:d}, {:#.9e}, {:#.9e},  {:#.9e}\n'.format(i+1,woodIGAvertices[i,0],woodIGAvertices[i,1],woodIGAvertices[i,2]))
    # beam element connectivity 
    count = 1
    meshinpfile.write('*Element, type=B32,elset=AllBeams\n') 
    for i in range(0,nelem):
        meshinpfile.write('{:d}'.format(count))
        count += 1
        # meshinpfile.write('{:d}'.format(i+1)) 
        for j in range(0,nnode):
            meshinpfile.write(', {:d}'.format(beam_connectivity[i,j])) 
        meshinpfile.write('\n')
        
    if precrackFlag in ['on','On','Y','y','Yes','yes']:
        connector_t_precracked_index = []
        connector_t_precracked_connectivity = []
        connector_l_precracked_index = []
        connector_l_precracked_connectivity = []
        # bottom transverse connector element connectivity 
        meshinpfile.write('*Element, type=T3D2,elset=AllBotConns\n') 
        for i in range(0,nel_con_tbot):
            if i in precrack_elem:
                connector_t_precracked_index.append(count)
                count += 1
                connector_t_precracked_connectivity.append(connector_t_bot_connectivity[i,:])
            else:
                meshinpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    meshinpfile.write(', {:d}'.format(connector_t_bot_connectivity[i,j])) 
                meshinpfile.write('\n')
        # regular transverse connector element connectivity 
        meshinpfile.write('*Element, type=T3D2,elset=AllRegConns\n') 
        for i in range(0,nel_con_treg):
            if i in precrack_elem:
                connector_t_precracked_index.append(count)
                count += 1
                connector_t_precracked_connectivity.append(connector_t_reg_connectivity[i,:])
            else:
                meshinpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    meshinpfile.write(', {:d}'.format(connector_t_reg_connectivity[i,j])) 
                meshinpfile.write('\n')
        # top transverse connector element connectivity 
        meshinpfile.write('*Element, type=T3D2,elset=AllTopConns\n') 
        for i in range(0,nel_con_ttop):
            if i in precrack_elem:
                connector_t_precracked_index.append(count)
                count += 1
                connector_t_precracked_connectivity.append(connector_t_top_connectivity[i,:])
            else:
                meshinpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    meshinpfile.write(', {:d}'.format(connector_t_top_connectivity[i,j])) 
                meshinpfile.write('\n')
        # precracked transverse connector element connectivity 
        meshinpfile.write('*Element, type=T3D2,elset=AllPrecrackTConns\n') 
        for i in range(0,len(connector_t_precracked_connectivity)):
            meshinpfile.write('{:d}'.format(connector_t_precracked_index[i]))
            for j in range(0,nelnode_connector):
                meshinpfile.write(', {:d}'.format(connector_t_precracked_connectivity[i][j]))
            meshinpfile.write('\n')
        # longitudinal connector element connectivity 
        meshinpfile.write('*Element, type=T3D2,elset=AllLongConns\n') 
        for i in range(0,nel_con_l):
            if i in precrack_elem:
                connector_l_precracked_index.append(count)
                count += 1
                connector_l_precracked_connectivity.append(connector_l_connectivity[i,:])
            else:
                meshinpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    meshinpfile.write(', {:d}'.format(connector_l_connectivity[i,j])) 
                meshinpfile.write('\n')
        # precracked longitudinal connector element connectivity 
        meshinpfile.write('*Element, type=T3D2,elset=AllPrecrackLConns\n') 
        for i in range(0,len(connector_l_precracked_connectivity)):
            meshinpfile.write('{:d}'.format(connector_l_precracked_index[i]))
            for j in range(0,nelnode_connector):
                meshinpfile.write(', {:d}'.format(connector_l_precracked_connectivity[i][j]))
            meshinpfile.write('\n')
    else:
        # bottom transverse connector element connectivity 
        meshinpfile.write('*Element, type=T3D2,elset=AllBotConns\n') 
        for i in range(0,nel_con_tbot):
            # meshinpfile.write('{:d}'.format(i+1))
            meshinpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                meshinpfile.write(', {:d}'.format(connector_t_bot_connectivity[i,j])) 
            meshinpfile.write('\n')
        # regular transverse connector element connectivity 
        meshinpfile.write('*Element, type=T3D2,elset=AllRegConns\n') 
        for i in range(0,nel_con_treg):
            # meshinpfile.write('{:d}'.format(i+1))
            meshinpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                meshinpfile.write(', {:d}'.format(connector_t_reg_connectivity[i,j])) 
            meshinpfile.write('\n')
        # top transverse connector element connectivity 
        meshinpfile.write('*Element, type=T3D2,elset=AllTopConns\n') 
        for i in range(0,nel_con_ttop):
            # meshinpfile.write('{:d}'.format(i+1))
            meshinpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                meshinpfile.write(', {:d}'.format(connector_t_top_connectivity[i,j])) 
            meshinpfile.write('\n')
        # longitudinal connector element connectivity 
        meshinpfile.write('*Element, type=T3D2,elset=AllLongConns\n') 
        for i in range(0,nel_con_l):
            # meshinpfile.write('{:d}'.format(i+1))
            meshinpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                meshinpfile.write(', {:d}'.format(connector_l_connectivity[i,j])) 
            meshinpfile.write('\n')

    # Ghost mesh for easy visualization
    # count_offset = 10**(int(math.log10(count)))
    # count += count_offset-1 # set an offset with the same order of magnitude
    count_offset = 10**(int(math.log10(count))+1) # set an offset with the order of magnitude of the max number + 1
    count = count_offset + 1
    
    if NURBS_degree == 2:
        meshinpfile.write('*ELEMENT, TYPE=B32, ELSET=VisualBeams\n')
        for i in range(0,nelem):
            meshinpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nnode):
                meshinpfile.write(', {:d}'.format(beam_connectivity[i,j])) 
            meshinpfile.write('\n')
            
    if precrackFlag in ['on','On','Y','y','Yes','yes']:
        visual_connector_precracked_index = []
        meshinpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualBotConns\n')
        for i in range(0,nel_con_tbot):
            if i in precrack_elem:
                visual_connector_precracked_index.append(count)
                count += 1
            else:
                meshinpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    meshinpfile.write(', {:d}'.format(connector_t_bot_connectivity[i,j])) 
                meshinpfile.write('\n')
        meshinpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualRegConns\n')
        for i in range(0,nel_con_treg):
            if i in precrack_elem:
                visual_connector_precracked_index.append(count)
                count += 1
            else:
                meshinpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    meshinpfile.write(', {:d}'.format(connector_t_reg_connectivity[i,j])) 
                meshinpfile.write('\n')
        meshinpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualTopConns\n')
        for i in range(0,nel_con_ttop):
            if i in precrack_elem:
                visual_connector_precracked_index.append(count)
                count += 1
            else:
                meshinpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    meshinpfile.write(', {:d}'.format(connector_t_top_connectivity[i,j])) 
                meshinpfile.write('\n')
        # precracked transverse connector element connectivity 
        meshinpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualPrecrackTConns\n')
        for i in range(0,len(connector_t_precracked_connectivity)):
            meshinpfile.write('{:d}'.format(visual_connector_precracked_index[i]))
            for j in range(0,nelnode_connector):
                meshinpfile.write(', {:d}'.format(connector_t_precracked_connectivity[i][j]))
            meshinpfile.write('\n')
        # longitudinal connector element connectivity 
        meshinpfile.write('*Element, type=T3D2,elset=VisualLConns\n') 
        for i in range(0,nel_con_l):
            if i in precrack_elem:
                connector_l_precracked_index.append(count)
                count += 1
                connector_l_precracked_connectivity.append(connector_l_connectivity[i,:])
            else:
                meshinpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    meshinpfile.write(', {:d}'.format(connector_l_connectivity[i,j])) 
                meshinpfile.write('\n')
        # precracked longitudinal connector element connectivity 
        meshinpfile.write('*Element, type=T3D2,elset=VisualPrecrackLConns\n')
        meshinpfile.write('\n')
        # for i in range(0,len(connector_l_precracked_connectivity)):
        #     meshinpfile.write('{:d}'.format(connector_l_precracked_index[i]))
        #     for j in range(0,nelnode_connector):
        #         meshinpfile.write(', {:d}'.format(connector_l_precracked_connectivity[i][j]))
        #     meshinpfile.write('\n')      
    else:
        meshinpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualBotConns\n')
        for i in range(0,nel_con_tbot):
            meshinpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                meshinpfile.write(', {:d}'.format(connector_t_bot_connectivity[i,j])) 
            meshinpfile.write('\n')
        meshinpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualRegConns\n')
        for i in range(0,nel_con_treg):
            meshinpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                meshinpfile.write(', {:d}'.format(connector_t_reg_connectivity[i,j])) 
            meshinpfile.write('\n')
        meshinpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualTopConns\n')
        for i in range(0,nel_con_ttop):
            meshinpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                meshinpfile.write(', {:d}'.format(connector_t_top_connectivity[i,j])) 
            meshinpfile.write('\n')
        meshinpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualLongConns\n')
        for i in range(0,nel_con_l):
            meshinpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                meshinpfile.write(', {:d}'.format(connector_l_connectivity[i,j])) 
            meshinpfile.write('\n')
            
    meshinpfile.write('** Section: Section-1\n')
    meshinpfile.write('*Solid Section, elset=VisualBotConns, material=VisualTConns\n')
    meshinpfile.write('{:8.4E}\n'.format(props_connector_t_bot[3]*cellwallthickness_early))
    meshinpfile.write('** Section: Section-2\n')
    meshinpfile.write('*Solid Section, elset=VisualRegConns, material=VisualTConns\n')
    meshinpfile.write('{:8.4E}\n'.format(props_connector_t_bot[3]*cellwallthickness_early))
    meshinpfile.write('** Section: Section-3\n')
    meshinpfile.write('*Solid Section, elset=VisualTopConns, material=VisualTConns\n')
    meshinpfile.write('{:8.4E}\n'.format(props_connector_t_bot[3]*cellwallthickness_early))
    meshinpfile.write('** Section: Section-11\n')
    meshinpfile.write('*Solid Section, elset=AllBotConns, material=VisualTConns\n')
    meshinpfile.write('{:8.4E}\n'.format(props_connector_t_bot[3]*cellwallthickness_early))
    meshinpfile.write('** Section: Section-12\n')
    meshinpfile.write('*Solid Section, elset=AllRegConns, material=VisualTConns\n')
    meshinpfile.write('{:8.4E}\n'.format(props_connector_t_bot[3]*cellwallthickness_early))
    meshinpfile.write('** Section: Section-13\n')
    meshinpfile.write('*Solid Section, elset=AllTopConns, material=VisualTConns\n')
    meshinpfile.write('{:8.4E}\n'.format(props_connector_t_bot[3]*cellwallthickness_early))
    if precrackFlag in ['on','On','Y','y','Yes','yes']:
        meshinpfile.write('** Section: Section-4\n')
        meshinpfile.write('*Solid Section, elset=VisualPrecrackTConns, material=VisualTConns\n')
        meshinpfile.write('{:8.4E}\n'.format(props_connector_t_bot[3]*cellwallthickness_early))
        meshinpfile.write('** Section: Section-5\n')
        meshinpfile.write('*Solid Section, elset=VisualLongConns, material=VisualLConns\n')
        meshinpfile.write('{:8.4E}\n'.format(props_connector_l[3]))
        meshinpfile.write('** Section: Section-6\n')
        meshinpfile.write('*Solid Section, elset=VisualPrecrackLConns, material=VisualLConns\n')
        meshinpfile.write('{:8.4E}\n'.format(props_connector_l[3]))
        meshinpfile.write('** Section: Section-14\n')
        meshinpfile.write('*Solid Section, elset=AllPrecrackTConns, material=VisualTConns\n')
        meshinpfile.write('{:8.4E}\n'.format(props_connector_t_bot[3]*cellwallthickness_early))
        meshinpfile.write('** Section: Section-15\n')
        meshinpfile.write('*Solid Section, elset=AllLongConns, material=VisualLConns\n')
        meshinpfile.write('{:8.4E}\n'.format(props_connector_l[3]))
        meshinpfile.write('** Section: Section-16\n')
        meshinpfile.write('*Solid Section, elset=AllPrecrackLConns, material=VisualLConns\n')
        meshinpfile.write('{:8.4E}\n'.format(props_connector_l[3]))
        if NURBS_degree == 2:
            meshinpfile.write('** Section: Section-7  Profile: Profile-1\n')
            meshinpfile.write('*Beam Section, elset=VisualBeams, material=VisualBeams, temperature=GRADIENTS, section=RECT\n')
            meshinpfile.write('{:8.4E}, {:8.4E}\n'.format(props_beam[3],props_beam[4]))
            meshinpfile.write('1.,0.,0.\n')
            meshinpfile.write('** Section: Section-17  Profile: Profile-1\n')
            meshinpfile.write('*Beam Section, elset=AllBeams, material=VisualBeams, temperature=GRADIENTS, section=RECT\n')
            meshinpfile.write('{:8.4E}, {:8.4E}\n'.format(props_beam[3],props_beam[4]))
            meshinpfile.write('1.,0.,0.\n')  
    else:
        meshinpfile.write('** Section: Section-4\n')
        meshinpfile.write('*Solid Section, elset=VisualLongConns, material=VisualLConns\n')
        meshinpfile.write('{:8.4E}\n'.format(props_connector_l[3]))
        meshinpfile.write('** Section: Section-14\n')
        meshinpfile.write('*Solid Section, elset=AllLongConns, material=VisualLConns\n')
        meshinpfile.write('{:8.4E}\n'.format(props_connector_l[3]))
        if NURBS_degree == 2:
            meshinpfile.write('** Section: Section-5  Profile: Profile-1\n')
            meshinpfile.write('*Beam Section, elset=VisualBeams, material=VisualBeams, temperature=GRADIENTS, section=RECT\n')
            meshinpfile.write('{:8.4E}, {:8.4E}\n'.format(props_beam[3],props_beam[4]))
            meshinpfile.write('1.,0.,0.\n')
            meshinpfile.write('** Section: Section-15  Profile: Profile-1\n')
            meshinpfile.write('*Beam Section, elset=AllBeams, material=VisualBeams, temperature=GRADIENTS, section=RECT\n')
            meshinpfile.write('{:8.4E}, {:8.4E}\n'.format(props_beam[3],props_beam[4]))
            meshinpfile.write('1.,0.,0.\n')
            
    meshinpfile.write('*End Instance\n') 
    meshinpfile.write('** \n')
    # NODE SETS
    meshinpfile.write('*Nset, nset=AllNodes, instance=Part-1-1, generate\n') 
    meshinpfile.write('{:d}, {:d}, {:d} \n'.format(1,numnode,1))
    
    if any(x in geoName for x in ['hydrostatic_', 'Hydrostatic_']):
        # boundary nodes
        if len(boundaries) == 4:
            if any(item in boundary_conditions for item in ['left','Left','L']):
                # Nodes on the left
                LeftNodes = (np.where(woodIGAvertices[:,xaxis-1] <= x_min)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=LeftNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(LeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, LeftNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['right','Right','R']):
                # Nodes on the right
                RightNodes = (np.where(woodIGAvertices[:,xaxis-1] >= x_max)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, Nset=RightNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(RightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, RightNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['bottom','Bottom','Bot']):
                # Nodes on the bottom
                BottomNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['top','Top','T']):
                # Nodes on the top
                TopNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['back','Back']):
                # Nodes on the back
                BackNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['front','Front','F']):
                # Nodes on the front
                FrontNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')
        elif len(boundaries) == 6:      
            if any(item in boundary_conditions for item in ['Hydrostatic','hydrostatic']):
                offset = x_max*0.05
                # Nodes on the bottom-left
                BottomLeftNodes = (np.where(woodIGAvertices[:,yaxis-1] <= (-np.sqrt(3)*woodIGAvertices[:,xaxis-1] - np.sqrt(3)*x_max + offset))[0]+1).reshape(1,-1)
                # BottomLeftNodes = (np.where((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= 725.0**2)[0]+1).reshape(1,-1)
                # BottomLeftNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] < x_min/2) &\
                #                             (woodIGAvertices[:,yaxis-1] < 0))[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=BottomLeftNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomLeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, BottomLeftNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the bottom-right
                BottomRightNodes = (np.where(woodIGAvertices[:,yaxis-1] <= (np.sqrt(3)*woodIGAvertices[:,xaxis-1] - np.sqrt(3)*x_max + offset))[0]+1).reshape(1,-1)
                # BottomRightNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] > x_max/2) &\
                #                             (woodIGAvertices[:,yaxis-1] < 0))[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, Nset=BottomRightNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomRightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, BottomRightNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the bottom
                BottomNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the top-left
                TopLeftNodes = (np.where(woodIGAvertices[:,yaxis-1] >= (np.sqrt(3)*woodIGAvertices[:,xaxis-1] + np.sqrt(3)*x_max - offset))[0]+1).reshape(1,-1)
                # TopLeftNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] < x_min/2) &\
                #                             (woodIGAvertices[:,yaxis-1] > 0))[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=TopLeftNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopLeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, TopLeftNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the top-right
                TopRightNodes = (np.where(woodIGAvertices[:,yaxis-1] >= (-np.sqrt(3)*woodIGAvertices[:,xaxis-1] + np.sqrt(3)*x_max - offset))[0]+1).reshape(1,-1)
                # TopRightNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] > x_max/2) &\
                #                             (woodIGAvertices[:,yaxis-1] > 0))[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=TopRightNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopRightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, TopRightNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the top
                TopNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
                    
                # Nodes on the front
                FrontNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the back
                BackNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
            
    elif any(x in geoName for x in ['uniaxial_', 'Uniaxial_']):
        # boundary nodes
        if len(boundaries) == 4:
            if any(item in boundary_conditions for item in ['left','Left','L']):
                # Nodes on the left
                LeftNodes = (np.where(woodIGAvertices[:,xaxis-1] <= x_min)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=LeftNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(LeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, LeftNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['right','Right','R']):
                # Nodes on the right
                RightNodes = (np.where(woodIGAvertices[:,xaxis-1] >= x_max)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, Nset=RightNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(RightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, RightNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['bottom','Bottom','Bot']):
                # Nodes on the bottom
                BottomNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['top','Top','T']):
                # Nodes on the top
                TopNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['back','Back']):
                # Nodes on the back
                BackNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['front','Front','F']):
                # Nodes on the front
                FrontNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')
        elif len(boundaries) == 6:      
            if any(item in boundary_conditions for item in ['Hydrostatic','hydrostatic']):
                offset = x_max*0.05
                # Nodes on the bottom-left
                BottomLeftNodes = (np.where(woodIGAvertices[:,yaxis-1] <= (-np.sqrt(3)*woodIGAvertices[:,xaxis-1] - np.sqrt(3)*x_max + offset))[0]+1).reshape(1,-1)
                # BottomLeftNodes = (np.where((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= 725.0**2)[0]+1).reshape(1,-1)
                # BottomLeftNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] < x_min/2) &\
                #                             (woodIGAvertices[:,yaxis-1] < 0))[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=BottomLeftNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomLeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, BottomLeftNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the bottom-right
                BottomRightNodes = (np.where(woodIGAvertices[:,yaxis-1] <= (np.sqrt(3)*woodIGAvertices[:,xaxis-1] - np.sqrt(3)*x_max + offset))[0]+1).reshape(1,-1)
                # BottomRightNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] > x_max/2) &\
                #                             (woodIGAvertices[:,yaxis-1] < 0))[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, Nset=BottomRightNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomRightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, BottomRightNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the bottom
                BottomNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the top-left
                TopLeftNodes = (np.where(woodIGAvertices[:,yaxis-1] >= (np.sqrt(3)*woodIGAvertices[:,xaxis-1] + np.sqrt(3)*x_max - offset))[0]+1).reshape(1,-1)
                # TopLeftNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] < x_min/2) &\
                #                             (woodIGAvertices[:,yaxis-1] > 0))[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=TopLeftNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopLeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, TopLeftNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the top-right
                TopRightNodes = (np.where(woodIGAvertices[:,yaxis-1] >= (-np.sqrt(3)*woodIGAvertices[:,xaxis-1] + np.sqrt(3)*x_max - offset))[0]+1).reshape(1,-1)
                # TopRightNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] > x_max/2) &\
                #                             (woodIGAvertices[:,yaxis-1] > 0))[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=TopRightNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopRightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, TopRightNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the top
                TopNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
                    
                # Nodes on the front
                FrontNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the back
                BackNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min)[0]+1).reshape(1,-1)
                meshinpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    meshinpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
    else:
        if merge_operation in ['on','On','Y','y','Yes','yes']:
            # boundary nodes
            if len(boundaries) == 4:
                if any(item in boundary_conditions for item in ['left','Left','L']):
                    # Nodes on the left
                    LeftNodes = (np.where(woodIGAvertices[:,xaxis-1] <= x_min+merge_tol)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=LeftNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(LeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, LeftNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['right','Right','R']):
                    # Nodes on the right
                    RightNodes = (np.where(woodIGAvertices[:,xaxis-1] >= x_max-merge_tol)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, Nset=RightNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(RightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, RightNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['bottom','Bottom','Bot']):
                    # Nodes on the bottom
                    BottomNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min+merge_tol)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['top','Top','T']):
                    # Nodes on the top
                    TopNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max-merge_tol)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['back','Back']):
                    # Nodes on the back
                    BackNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min+merge_tol)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['front','Front','F']):
                    # Nodes on the front
                    FrontNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max-merge_tol)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')
    
            # boundary nodes
            elif len(boundaries) == 8:
                # if any(item in boundary_conditions for item in ['left','Left','L']):
                    # Nodes on the left bottom
                    LeftBottomNodes = (np.where((woodIGAvertices[:,xaxis-1] <= x_min+merge_tol) & (woodIGAvertices[:,yaxis-1] <= 0))[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=LeftBottomNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(LeftBottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, LeftBottomNodes[0][15*i:15*(i+1)])))+',\n')
                    # Nodes on the left top
                    LeftTopNodes = (np.where((woodIGAvertices[:,xaxis-1] <= x_min+merge_tol) & (woodIGAvertices[:,yaxis-1] >= 0))[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=LeftTopNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(LeftTopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, LeftTopNodes[0][15*i:15*(i+1)])))+',\n')
                # if any(item in boundary_conditions for item in ['right','Right','R']):
                    # Nodes on the right
                    RightNodes = (np.where(woodIGAvertices[:,xaxis-1] >= x_max-merge_tol)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, Nset=RightNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(RightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, RightNodes[0][15*i:15*(i+1)])))+',\n')
                # if any(item in boundary_conditions for item in ['bottom','Bottom','Bot']):
                    # Nodes on the bottom
                    BottomNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min+merge_tol)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
                # if any(item in boundary_conditions for item in ['top','Top','T']):
                    # Nodes on the top
                    TopNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max-merge_tol)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
                # if any(item in boundary_conditions for item in ['back','Back']):
                    # Nodes on the back
                    BackNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min+merge_tol)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
                # if any(item in boundary_conditions for item in ['front','Front','F']):
                    # Nodes on the front
                    FrontNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max-merge_tol)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')
        else: # not merged
            # boundary nodes
            if len(boundaries) == 4:
                if any(item in boundary_conditions for item in ['left','Left','L']):
                    # Nodes on the left
                    LeftNodes = (np.where(woodIGAvertices[:,xaxis-1] <= x_min)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=LeftNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(LeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, LeftNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['right','Right','R']):
                    # Nodes on the right
                    RightNodes = (np.where(woodIGAvertices[:,xaxis-1] >= x_max)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, Nset=RightNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(RightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, RightNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['bottom','Bottom','Bot']):
                    # Nodes on the bottom
                    BottomNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['top','Top','T']):
                    # Nodes on the top
                    TopNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['back','Back']):
                    # Nodes on the back
                    BackNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['front','Front','F']):
                    # Nodes on the front
                    FrontNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')
    
            # boundary nodes
            elif len(boundaries) == 8:
                if any(item in boundary_conditions for item in ['left','Left','L']):
                    # Nodes on the left bottom
                    LeftBottomNodes = (np.where((woodIGAvertices[:,xaxis-1] <= x_min) & (woodIGAvertices[:,yaxis-1] <= 0))[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=LeftBottomNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(LeftBottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, LeftBottomNodes[0][15*i:15*(i+1)])))+',\n')
                    # Nodes on the left top
                    LeftTopNodes = (np.where((woodIGAvertices[:,xaxis-1] <= x_min) & (woodIGAvertices[:,yaxis-1] >= 0))[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=LeftTopNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(LeftTopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, LeftTopNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['right','Right','R']):
                    # Nodes on the right
                    RightNodes = (np.where(woodIGAvertices[:,xaxis-1] >= x_max)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, Nset=RightNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(RightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, RightNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['bottom','Bottom','Bot']):
                    # Nodes on the bottom
                    BottomNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['top','Top','T']):
                    # Nodes on the top
                    TopNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['back','Back']):
                    # Nodes on the back
                    BackNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['front','Front','F']):
                    # Nodes on the front
                    FrontNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max)[0]+1).reshape(1,-1)
                    meshinpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        meshinpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')      
    # ELEMENT SETS
    meshinpfile.write('*Elset, elset=AllElles, instance=Part-1-1, generate\n') 
    meshinpfile.write('{:d}, {:d}, {:d} \n'.format(1,nel_con,1))
    meshinpfile.write('*Elset, elset=AllVisualElle, instance=Part-1-1, generate\n') 
    meshinpfile.write('{:d}, {:d}, {:d} \n'.format(count_offset+1,count-1,1))
    meshinpfile.write('*End Assembly\n')

    inpfile = open(Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName + '.inp'),'w')
    # inpfile = open (os.path.join(path,geoName)+'.inp','w')
    inpfile.write('*HEADING'+'\n')
    inpfile.write('** Job name: {0} Model name: Model-{1}\n'.format(geoName,geoName))
    inpfile.write('** Generated by Wood Mesh Generator V11.0\n')
    inpfile.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
    # PARAMETER
    inpfile.write('** \n')
    inpfile.write('*PARAMETER \n')
    inpfile.write('** \n')
    inpfile.write('********************************************************\n')
    inpfile.write('**                  Beam Properties                   **\n')
    inpfile.write('********************************************************\n')
    inpfile.write('** \n')
    inpfile.write('DensityBeam                         = {:8.4E}\n'.format(props_beam[0]))
    inpfile.write('ElasticModulusBeam                  = {:8.4E}\n'.format(props_beam[1]))
    inpfile.write('PoissonsRatioBeam                   = {:8.4E}\n'.format(props_beam[2]))
    inpfile.write('CrossSectionHeightBeam              = {:8.4E}\n'.format(props_beam[3]))
    inpfile.write('CrossSectionWidthBeam               = {:8.4E}\n'.format(props_beam[4]))
    inpfile.write('TensileStrengthBeam                 = {:8.4E}\n'.format(props_beam[5]))
    inpfile.write('TensileCharacteristicLengthBeam     = {:8.4E}\n'.format(props_beam[6]))
    inpfile.write('ShearStrengthRatioBeam              = {:8.4E}\n'.format(props_beam[7]))
    inpfile.write('SofteningExponentBeam               = {:8.4E}\n'.format(props_beam[8]))
    inpfile.write('InitialFrictionBeam                 = {:8.4E}\n'.format(props_beam[9]))
    inpfile.write('AsymptoticFrictionBeam              = {:8.4E}\n'.format(props_beam[10]))
    inpfile.write('TransitionalStressBeam              = {:8.4E}\n'.format(props_beam[11]))
    inpfile.write('TensileUnloadingBeam                = {:8.4E}\n'.format(props_beam[12]))
    inpfile.write('ShearUnloadingBeam                  = {:8.4E}\n'.format(props_beam[13]))
    inpfile.write('ShearSofteningBeam                  = {:8.4E}\n'.format(props_beam[14]))
    inpfile.write('ElasticAnalysisFlagBeam             = {:8.4E}\n'.format(props_beam[15]))
    inpfile.write('SectionTypeBeam                     = {:d}\n'.format(iprops_beam[0]))
    inpfile.write('** \n')
    inpfile.write('********************************************************\n')
    inpfile.write('**         Transverse Connector Properties            **\n')
    inpfile.write('********************************************************\n')
    inpfile.write('** \n')
    inpfile.write('DensityTConn                        = {:8.4E}\n'.format(props_connector_t_bot[0]))
    inpfile.write('ElasticModulusTConn                 = {:8.4E}\n'.format(props_connector_t_bot[1]))
    inpfile.write('ShearNormalCoeffTConn               = {:8.4E}\n'.format(props_connector_t_bot[2]))
    inpfile.write('CrossSectionHeightBot               = {:8.4E}\n'.format(props_connector_t_bot[3]))
    inpfile.write('CrossSectionHeightReg               = {:8.4E}\n'.format(props_connector_t_reg[3]))
    inpfile.write('CrossSectionHeightTop               = {:8.4E}\n'.format(props_connector_t_top[3]))
    inpfile.write('DistanceMBot                        = {:8.4E}\n'.format(props_connector_t_bot[4]))
    inpfile.write('DistanceMReg                        = {:8.4E}\n'.format(props_connector_t_reg[4]))
    inpfile.write('DistanceMTop                        = {:8.4E}\n'.format(props_connector_t_top[4]))
    inpfile.write('DistanceLBot                        = {:8.4E}\n'.format(props_connector_t_bot[5]))
    inpfile.write('DistanceLReg                        = {:8.4E}\n'.format(props_connector_t_reg[5]))
    inpfile.write('DistanceLTop                        = {:8.4E}\n'.format(props_connector_t_top[5]))
    inpfile.write('TensileStrengthTConn                = {:8.4E}\n'.format(props_connector_t_bot[6]))
    inpfile.write('TensileCharacteristicLengthTConn    = {:8.4E}\n'.format(props_connector_t_bot[7]))
    inpfile.write('ShearStrengthRatioTConn             = {:8.4E}\n'.format(props_connector_t_bot[8]))
    inpfile.write('SofteningExponentTConn              = {:8.4E}\n'.format(props_connector_t_bot[9]))
    inpfile.write('InitialFrictionTConn                = {:8.4E}\n'.format(props_connector_t_bot[10]))
    inpfile.write('AsymptoticFrictionTConn             = {:8.4E}\n'.format(props_connector_t_bot[11]))
    inpfile.write('TransitionalStressTConn             = {:8.4E}\n'.format(props_connector_t_bot[12]))
    inpfile.write('TensileUnloadingTConn               = {:8.4E}\n'.format(props_connector_t_bot[13]))
    inpfile.write('ShearUnloadingTConn                 = {:8.4E}\n'.format(props_connector_t_bot[14]))
    inpfile.write('ShearSofteningTConn                 = {:8.4E}\n'.format(props_connector_t_bot[15]))
    inpfile.write('ElasticAnalysisFlagTConn            = {:8.4E}\n'.format(props_connector_t_bot[16]))
    inpfile.write('CompressiveYieldingStrengthTConn    = {:8.4E}\n'.format(props_connector_t_bot[17]))
    inpfile.write('InitialHardeningModulusRatioTConn   = {:8.4E}\n'.format(props_connector_t_bot[18]))
    inpfile.write('TransitionalStrainRatioTConn        = {:8.4E}\n'.format(props_connector_t_bot[19]))
    inpfile.write('DeviatoricStrainThresholdRatioTConn = {:8.4E}\n'.format(props_connector_t_bot[20]))
    inpfile.write('DeviatoricDamageParameterTConn      = {:8.4E}\n'.format(props_connector_t_bot[21]))
    inpfile.write('FinalHardeningModulusRatioTConn     = {:8.4E}\n'.format(props_connector_t_bot[22]))
    inpfile.write('DensificationRatioTConn             = {:8.4E}\n'.format(props_connector_t_bot[23]))
    inpfile.write('VolumetricDeviatoricCouplingTConn   = {:8.4E}\n'.format(props_connector_t_bot[24]))
    inpfile.write('CompressiveUnloadingTConn           = {:8.4E}\n'.format(props_connector_t_bot[25]))
    inpfile.write('ConnectorTypeBot                    = {:d}\n'.format(iprops_connector_t_bot[0]))
    inpfile.write('ConnectorTypeReg                    = {:d}\n'.format(iprops_connector_t_reg[0]))
    inpfile.write('ConnectorTypeTop                    = {:d}\n'.format(iprops_connector_t_top[0]))
    inpfile.write('** \n')
    inpfile.write('********************************************************\n')
    inpfile.write('**         Longitudinal Connector Properties          **\n')
    inpfile.write('********************************************************\n')
    inpfile.write('** \n')
    inpfile.write('DensityLConn                        = {:8.4E}\n'.format(props_connector_l[0]))
    inpfile.write('ElasticModulusLConn                 = {:8.4E}\n'.format(props_connector_l[1]))
    inpfile.write('ShearNormalCoeffLConn               = {:8.4E}\n'.format(props_connector_l[2]))
    inpfile.write('CrossSectionAreaLConn               = {:8.4E}\n'.format(props_connector_l[3]))
    inpfile.write('TensileStrengthLConn                = {:8.4E}\n'.format(props_connector_l[4]))
    inpfile.write('TensileCharacteristicLengthLConn    = {:8.4E}\n'.format(props_connector_l[5]))
    inpfile.write('ShearStrengthRatioLConn             = {:8.4E}\n'.format(props_connector_l[6]))
    inpfile.write('SofteningExponentLConn              = {:8.4E}\n'.format(props_connector_l[7]))
    inpfile.write('InitialFrictionLConn                = {:8.4E}\n'.format(props_connector_l[8]))
    inpfile.write('AsymptoticFrictionLConn             = {:8.4E}\n'.format(props_connector_l[9]))
    inpfile.write('TransitionalStressLConn             = {:8.4E}\n'.format(props_connector_l[10]))
    inpfile.write('TensileUnloadingLConn               = {:8.4E}\n'.format(props_connector_l[11]))
    inpfile.write('ShearUnloadingLConn                 = {:8.4E}\n'.format(props_connector_l[12]))
    inpfile.write('ShearSofteningLConn                 = {:8.4E}\n'.format(props_connector_l[13]))
    inpfile.write('ElasticAnalysisFlagLConn            = {:8.4E}\n'.format(props_connector_l[14]))
    inpfile.write('CompressiveYieldingStrengthLConn    = {:8.4E}\n'.format(props_connector_l[15]))
    inpfile.write('InitialHardeningModulusRatioLConn   = {:8.4E}\n'.format(props_connector_l[16]))
    inpfile.write('TransitionalStrainRatioLConn        = {:8.4E}\n'.format(props_connector_l[17]))
    inpfile.write('DeviatoricStrainThresholdRatioLConn = {:8.4E}\n'.format(props_connector_l[18]))
    inpfile.write('DeviatoricDamageParameterLConn      = {:8.4E}\n'.format(props_connector_l[19]))
    inpfile.write('FinalHardeningModulusRatioLConn     = {:8.4E}\n'.format(props_connector_l[20]))
    inpfile.write('DensificationRatioLConn             = {:8.4E}\n'.format(props_connector_l[21]))
    inpfile.write('VolumetricDeviatoricCouplingLConn   = {:8.4E}\n'.format(props_connector_l[22]))
    inpfile.write('CompressiveUnloadingLConn           = {:8.4E}\n'.format(props_connector_l[23]))
    inpfile.write('ConnectorTypeLConn                  = {:d}\n'.format(iprops_connector_l[0]))
    inpfile.write('**\n')
    inpfile.write('********************************************************\n')
    inpfile.write('**            Strain Rate Effect Properties           **\n')
    inpfile.write('********************************************************\n')
    inpfile.write('**\n')
    inpfile.write('StrainRateEffectFlag                = {:8.4E}\n'.format(props_strainrate[0]))
    inpfile.write('PhysicalTimeScalingFactor           = {:8.4E}\n'.format(props_strainrate[1]))
    inpfile.write('StrainRateEffectC0                  = {:8.4E}\n'.format(props_strainrate[2]))
    inpfile.write('StrainRateEffectC1                  = {:8.4E}\n'.format(props_strainrate[3]))
    inpfile.write('**\n')
    inpfile.write('********************************************************\n')
    inpfile.write('**            Wood Lattice Model Properties           **\n')
    inpfile.write('********************************************************\n')
    inpfile.write('**\n')
    inpfile.write('NumberOfInstance = {:d}\n'.format(iprops_beam[1]))
    inpfile.write('NumberOfBeamElem = {:d}\n'.format(nelem))
    inpfile.write('NumberOfConnectorT = {:d}\n'.format(iprops_beam[2]))
    inpfile.write('NumberOfConnectorTPrecrack = {:d}\n'.format(iprops_beam[3]))
    inpfile.write('NumberOfConnectorL = {:d}\n'.format(nel_con_l))
    inpfile.write('NumberOfConnectorLPrecrack = {:d}\n'.format(iprops_beam[5]))
    inpfile.write('NumberOfSvarsBeam = {:d}\n'.format(nsvars_beam))
    inpfile.write('NumberOfSvarsTConn = {:d}\n'.format(nsvars_conn_t))
    inpfile.write('NumberOfSvarsLConn = {:d}\n'.format(nsvars_conn_l))
    inpfile.write('NumberOfSGaussPt = {:d}\n'.format(nsecgp))
    inpfile.write('NumberOfSvarsSGaussPt = {:d}\n'.format(nsvars_secgp))
    inpfile.write('NumberOfBeamPerGrain = {:d}\n'.format(nsegments))
    inpfile.write('**\n')
    inpfile.write('********************************************************\n')
    inpfile.write('**              Visualization Properties              **\n')
    inpfile.write('**     Pre-set of Visualization Svars: 1-All svars    **\n')
    inpfile.write('**     2-Stress only  3-Stress+IGAbeam geometry       **\n')
    inpfile.write('********************************************************\n')
    inpfile.write('** \n')
    inpfile.write('VisualizationSvars = 0\n')
    inpfile.write('StressFlag = 0\n')
    inpfile.write('BinaryFlag = 0\n')
    inpfile.write('TimeIncrement = {:8.4E}\n'.format(totaltime/50.0))
    inpfile.write('TotalTime = {:8.4E}\n'.format(totaltime))
    # PART
    inpfile.write('** \n')
    inpfile.write('** PARTS\n') 
    inpfile.write('** \n') 
    inpfile.write('*Part, name=Part-1\n')
    inpfile.write('*End Part\n')
    inpfile.write('** \n')
    # ASSEMBLY
    inpfile.write('** \n')
    inpfile.write('** ASSEMBLY\n') 
    inpfile.write('** \n') 
    inpfile.write('*Assembly, name=Assembly\n')
    inpfile.write('** \n') 
    # INSTANCE
    inpfile.write('*Instance, name=Part-1-1, part=Part-1\n')
    # nodes
    inpfile.write('*Node,nset=AllNodes\n')
    for i in range(0,numnode):
        inpfile.write('{:d}, {:#.9e}, {:#.9e},  {:#.9e}\n'.format(i+1,woodIGAvertices[i,0],woodIGAvertices[i,1],woodIGAvertices[i,2]))
    # beam element
    inpfile.write('*USER ELEMENT, NODES={:d}, TYPE=VU{:d}, COORDINATES=3,'.format(nelnode,eltype)) 
    inpfile.write(' VARIABLES={:d}, I PROPERTIES={:d},'.format(noGPs*nsvars_beam,len(iprops_beam)))
    inpfile.write(' PROPERTIES={:d}, INTEGRATION={:d}\n'.format(len(props_beam),noGPs))
    inpfile.write('1, 2, 3, 4, 5, 6\n') 
    # transverse connector elment 
    inpfile.write('*USER ELEMENT, NODES={:d}, TYPE=VU{:d}, COORDINATES=3,'.format(nelnode_connector,eltype_connector_t)) 
    inpfile.write(' VARIABLES={:d}, I PROPERTIES={:d},'.format(nsvars_conn_t,len(iprops_connector_t_bot)))
    inpfile.write(' PROPERTIES={:d}, INTEGRATION={:d}\n'.format(len(props_connector_t_bot)+len(props_strainrate),1))
    inpfile.write('1, 2, 3, 4, 5, 6\n')
    # longitudinal connector elment 
    inpfile.write('*USER ELEMENT, NODES={:d}, TYPE=VU{:d}, COORDINATES=3,'.format(nelnode_connector,eltype_connector_l)) 
    inpfile.write(' VARIABLES={:d}, I PROPERTIES={:d},'.format(nsvars_conn_t,len(iprops_connector_l)))
    inpfile.write(' PROPERTIES={:d}, INTEGRATION={:d}\n'.format(len(props_connector_l)+len(props_strainrate),1))
    inpfile.write('1, 2, 3, 4, 5, 6\n')
    # beam element connectivity 
    count = 1
    inpfile.write('*Element, type=VU{:d},elset=AllBeams\n'.format(eltype)) 
    for i in range(0,nelem):
        inpfile.write('{:d}'.format(count))
        count += 1
        # inpfile.write('{:d}'.format(i+1)) 
        for j in range(0,nnode):
            inpfile.write(', {:d}'.format(beam_connectivity[i,j])) 
        inpfile.write('\n')
        
    if precrackFlag in ['on','On','Y','y','Yes','yes']:
        connector_t_precracked_index = []
        connector_t_precracked_connectivity = []
        # bottom transverse connector element connectivity 
        inpfile.write('*Element, type=VU{:d},elset=AllBotConns\n'.format(eltype_connector_t)) 
        for i in range(0,nel_con_tbot):
            if i in precrack_elem:
                connector_t_precracked_index.append(count)
                count += 1
                connector_t_precracked_connectivity.append(connector_t_bot_connectivity[i,:])
            else:
                inpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    inpfile.write(', {:d}'.format(connector_t_bot_connectivity[i,j])) 
                inpfile.write('\n')
        # regular transverse connector element connectivity 
        inpfile.write('*Element, type=VU{:d},elset=AllRegConns\n'.format(eltype_connector_t)) 
        for i in range(0,nel_con_treg):
            if i in precrack_elem:
                connector_t_precracked_index.append(count)
                count += 1
                connector_t_precracked_connectivity.append(connector_t_reg_connectivity[i,:])
            else:
                inpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    inpfile.write(', {:d}'.format(connector_t_reg_connectivity[i,j])) 
                inpfile.write('\n')
        # top transverse connector element connectivity 
        inpfile.write('*Element, type=VU{:d},elset=AllTopConns\n'.format(eltype_connector_t)) 
        for i in range(0,nel_con_ttop):
            if i in precrack_elem:
                connector_t_precracked_index.append(count)
                count += 1
                connector_t_precracked_connectivity.append(connector_t_top_connectivity[i,:])
            else:
                inpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    inpfile.write(', {:d}'.format(connector_t_top_connectivity[i,j])) 
                inpfile.write('\n')
        # precracked transverse connector element connectivity 
        inpfile.write('*Element, type=VU{:d},elset=AllPrecrackTConns\n'.format(eltype_connector_t)) 
        for i in range(0,len(connector_t_precracked_connectivity)):
            inpfile.write('{:d}'.format(connector_t_precracked_index[i]))
            for j in range(0,nelnode_connector):
                inpfile.write(', {:d}'.format(connector_t_precracked_connectivity[i][j]))
            inpfile.write('\n')
    else:
        # bottom transverse connector element connectivity 
        inpfile.write('*Element, type=VU{:d},elset=AllBotConns\n'.format(eltype_connector_t)) 
        for i in range(0,nel_con_tbot):
            # inpfile.write('{:d}'.format(i+1))
            inpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                inpfile.write(', {:d}'.format(connector_t_bot_connectivity[i,j])) 
            inpfile.write('\n')
        # regular transverse connector element connectivity 
        inpfile.write('*Element, type=VU{:d},elset=AllRegConns\n'.format(eltype_connector_t)) 
        for i in range(0,nel_con_treg):
            # inpfile.write('{:d}'.format(i+1))
            inpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                inpfile.write(', {:d}'.format(connector_t_reg_connectivity[i,j])) 
            inpfile.write('\n')
        # top transverse connector element connectivity 
        inpfile.write('*Element, type=VU{:d},elset=AllTopConns\n'.format(eltype_connector_t)) 
        for i in range(0,nel_con_ttop):
            # inpfile.write('{:d}'.format(i+1))
            inpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                inpfile.write(', {:d}'.format(connector_t_top_connectivity[i,j])) 
            inpfile.write('\n')
            
    # longitudinal connector element connectivity 
    inpfile.write('*Element, type=VU{:d},elset=AllLongConns\n'.format(eltype_connector_l)) 
    for i in range(0,nel_con_l):
        # inpfile.write('{:d}'.format(i+1))
        inpfile.write('{:d}'.format(count))
        count += 1
        for j in range(0,nelnode_connector):
            inpfile.write(', {:d}'.format(connector_l_connectivity[i,j])) 
        inpfile.write('\n')
    # uel properties for beams
    inpfile.write('*UEL PROPERTY, ELSET=AllBeams\n') 
    inpfile.write('<DensityBeam>,<ElasticModulusBeam>,<PoissonsRatioBeam>,<CrossSectionHeightBeam>,<CrossSectionWidthBeam>,<TensileStrengthBeam>,<TensileCharacteristicLengthBeam>,<ShearStrengthRatioBeam>,\n')
    inpfile.write('<SofteningExponentBeam>,<InitialFrictionBeam>,<AsymptoticFrictionBeam>,<TransitionalStressBeam>,<TensileUnloadingBeam>,<ShearUnloadingBeam>,<ShearSofteningBeam>,<ElasticAnalysisFlagBeam>,\n')
    inpfile.write('<SectionTypeBeam>,<NumberOfInstance>,<NumberOfConnectorT>,<NumberOfConnectorTPrecrack>,<NumberOfConnectorL>,<NumberOfConnectorLPrecrack>\n')
    # uel properties for bottom transverse connectors
    inpfile.write('*UEL PROPERTY, ELSET=AllBotConns\n')
    inpfile.write('<DensityTConn>,<ElasticModulusTConn>,<ShearNormalCoeffTConn>,<CrossSectionHeightBot>,<DistanceMBot>,<DistanceLBot>,<TensileStrengthTConn>,<TensileCharacteristicLengthTConn>,\n')
    inpfile.write('<ShearStrengthRatioTConn>,<SofteningExponentTConn>,<InitialFrictionTConn>,<AsymptoticFrictionTConn>,<TransitionalStressTConn>,<TensileUnloadingTConn>,<ShearUnloadingTConn>,<ShearSofteningTConn>,\n')
    inpfile.write('<ElasticAnalysisFlagTConn>,<CompressiveYieldingStrengthTConn>,<InitialHardeningModulusRatioTConn>,<TransitionalStrainRatioTConn>,<DeviatoricStrainThresholdRatioTConn>,<DeviatoricDamageParameterTConn>,<FinalHardeningModulusRatioTConn>,<DensificationRatioTConn>,\n')
    inpfile.write('<VolumetricDeviatoricCouplingTConn>,<CompressiveUnloadingTConn>,<StrainRateEffectFlag>,<PhysicalTimeScalingFactor>,<StrainRateEffectC0>,<StrainRateEffectC1>,<ConnectorTypeBot>,<NumberOfInstance>,\n')
    inpfile.write('<NumberOfBeamElem>,<NumberOfConnectorT>,<NumberOfConnectorTPrecrack>,<NumberOfConnectorL>,<NumberOfConnectorLPrecrack>\n')
    # uel properties for regular transverse connectors
    inpfile.write('*UEL PROPERTY, ELSET=AllRegConns\n') 
    inpfile.write('<DensityTConn>,<ElasticModulusTConn>,<ShearNormalCoeffTConn>,<CrossSectionHeightReg>,<DistanceMReg>,<DistanceLReg>,<TensileStrengthTConn>,<TensileCharacteristicLengthTConn>,\n')
    inpfile.write('<ShearStrengthRatioTConn>,<SofteningExponentTConn>,<InitialFrictionTConn>,<AsymptoticFrictionTConn>,<TransitionalStressTConn>,<TensileUnloadingTConn>,<ShearUnloadingTConn>,<ShearSofteningTConn>,\n')
    inpfile.write('<ElasticAnalysisFlagTConn>,<CompressiveYieldingStrengthTConn>,<InitialHardeningModulusRatioTConn>,<TransitionalStrainRatioTConn>,<DeviatoricStrainThresholdRatioTConn>,<DeviatoricDamageParameterTConn>,<FinalHardeningModulusRatioTConn>,<DensificationRatioTConn>,\n')
    inpfile.write('<VolumetricDeviatoricCouplingTConn>,<CompressiveUnloadingTConn>,<StrainRateEffectFlag>,<PhysicalTimeScalingFactor>,<StrainRateEffectC0>,<StrainRateEffectC1>,<ConnectorTypeReg>,<NumberOfInstance>,\n')
    inpfile.write('<NumberOfBeamElem>,<NumberOfConnectorT>,<NumberOfConnectorTPrecrack>,<NumberOfConnectorL>,<NumberOfConnectorLPrecrack>\n')
    # uel properties for top transverse connectors
    inpfile.write('*UEL PROPERTY, ELSET=AllTopConns\n') 
    inpfile.write('<DensityTConn>,<ElasticModulusTConn>,<ShearNormalCoeffTConn>,<CrossSectionHeightTop>,<DistanceMTop>,<DistanceLTop>,<TensileStrengthTConn>,<TensileCharacteristicLengthTConn>,\n')
    inpfile.write('<ShearStrengthRatioTConn>,<SofteningExponentTConn>,<InitialFrictionTConn>,<AsymptoticFrictionTConn>,<TransitionalStressTConn>,<TensileUnloadingTConn>,<ShearUnloadingTConn>,<ShearSofteningTConn>,\n')
    inpfile.write('<ElasticAnalysisFlagTConn>,<CompressiveYieldingStrengthTConn>,<InitialHardeningModulusRatioTConn>,<TransitionalStrainRatioTConn>,<DeviatoricStrainThresholdRatioTConn>,<DeviatoricDamageParameterTConn>,<FinalHardeningModulusRatioTConn>,<DensificationRatioTConn>,\n')
    inpfile.write('<VolumetricDeviatoricCouplingTConn>,<CompressiveUnloadingTConn>,<StrainRateEffectFlag>,<PhysicalTimeScalingFactor>,<StrainRateEffectC0>,<StrainRateEffectC1>,<ConnectorTypeTop>,<NumberOfInstance>,\n')
    inpfile.write('<NumberOfBeamElem>,<NumberOfConnectorT>,<NumberOfConnectorTPrecrack>,<NumberOfConnectorL>,<NumberOfConnectorLPrecrack>\n')
    # uel properties for precracked transverse connectors
    inpfile.write('*UEL PROPERTY, ELSET=AllPrecrackTConns\n') 
    inpfile.write('<DensityTConn>,0.0E+0,<ShearNormalCoeffTConn>,<CrossSectionHeightTop>,<DistanceMTop>,<DistanceLTop>,<TensileStrengthTConn>,<TensileCharacteristicLengthTConn>,\n')
    inpfile.write('<ShearStrengthRatioTConn>,<SofteningExponentTConn>,<InitialFrictionTConn>,<AsymptoticFrictionTConn>,<TransitionalStressTConn>,<TensileUnloadingTConn>,<ShearUnloadingTConn>,<ShearSofteningTConn>,\n')
    inpfile.write('<ElasticAnalysisFlagTConn>,<CompressiveYieldingStrengthTConn>,<InitialHardeningModulusRatioTConn>,<TransitionalStrainRatioTConn>,<DeviatoricStrainThresholdRatioTConn>,<DeviatoricDamageParameterTConn>,<FinalHardeningModulusRatioTConn>,<DensificationRatioTConn>,\n')
    inpfile.write('<VolumetricDeviatoricCouplingTConn>,<CompressiveUnloadingTConn>,<StrainRateEffectFlag>,<PhysicalTimeScalingFactor>,<StrainRateEffectC0>,<StrainRateEffectC1>,<ConnectorTypeBot>,<NumberOfInstance>,\n')
    inpfile.write('<NumberOfBeamElem>,<NumberOfConnectorT>,<NumberOfConnectorTPrecrack>,<NumberOfConnectorL>,<NumberOfConnectorLPrecrack>\n')
    # uel properties for longitudinal connectors
    inpfile.write('*UEL PROPERTY, ELSET=AllLongConns\n')
    inpfile.write('<DensityLConn>,<ElasticModulusLConn>,<ShearNormalCoeffLConn>,<CrossSectionAreaLConn>,<TensileStrengthLConn>,<TensileCharacteristicLengthLConn>,<ShearStrengthRatioLConn>,<SofteningExponentLConn>,\n') 
    inpfile.write('<InitialFrictionLConn>,<AsymptoticFrictionLConn>,<TransitionalStressLConn>,<TensileUnloadingLConn>,<ShearUnloadingLConn>,<ShearSofteningLConn>,<ElasticAnalysisFlagLConn>,<CompressiveYieldingStrengthLConn>,\n') 
    inpfile.write('<InitialHardeningModulusRatioLConn>,<TransitionalStrainRatioLConn>,<DeviatoricStrainThresholdRatioLConn>,<DeviatoricDamageParameterLConn>,<FinalHardeningModulusRatioLConn>,<DensificationRatioLConn>,<VolumetricDeviatoricCouplingLConn>,<CompressiveUnloadingLConn>,\n') 
    inpfile.write('<StrainRateEffectFlag>,<PhysicalTimeScalingFactor>,<StrainRateEffectC0>,<StrainRateEffectC1>,<ConnectorTypeLConn>,<NumberOfInstance>,<NumberOfBeamElem>,<NumberOfConnectorT>,\n') 
    inpfile.write('<NumberOfConnectorTPrecrack>,<NumberOfConnectorL>,<NumberOfConnectorLPrecrack>\n')
    # uel properties for precracked longitudinal connectors
    inpfile.write('*UEL PROPERTY, ELSET=AllPrecrackLConns\n') 
    inpfile.write('<DensityLConn>,0.0E+0,<ShearNormalCoeffLConn>,<CrossSectionAreaLConn>,<TensileStrengthLConn>,<TensileCharacteristicLengthLConn>,<ShearStrengthRatioLConn>,<SofteningExponentLConn>,\n')
    inpfile.write('<InitialFrictionLConn>,<AsymptoticFrictionLConn>,<TransitionalStressLConn>,<TensileUnloadingLConn>,<ShearUnloadingLConn>,<ShearSofteningLConn>,<ElasticAnalysisFlagLConn>,<CompressiveYieldingStrengthLConn>,\n') 
    inpfile.write('<InitialHardeningModulusRatioLConn>,<TransitionalStrainRatioLConn>,<DeviatoricStrainThresholdRatioLConn>,<DeviatoricDamageParameterLConn>,<FinalHardeningModulusRatioLConn>,<DensificationRatioLConn>,<VolumetricDeviatoricCouplingLConn>,<CompressiveUnloadingLConn>,\n') 
    inpfile.write('<StrainRateEffectFlag>,<PhysicalTimeScalingFactor>,<StrainRateEffectC0>,<StrainRateEffectC1>,<ConnectorTypeLConn>,<NumberOfInstance>,<NumberOfBeamElem>,<NumberOfConnectorT>,\n') 
    inpfile.write('<NumberOfConnectorTPrecrack>,<NumberOfConnectorL>,<NumberOfConnectorLPrecrack>\n')
    # Ghost mesh for easy visualization
    # count_offset = 10**(int(math.log10(count)))
    count_offset = 10**(int(math.log10(count))+1) # set an offset with the order of magnitude of the max number + 1
    count = count_offset+1 # set an offset with the same order of magnitude
    
    if NURBS_degree == 2:
        inpfile.write('*ELEMENT, TYPE=B32, ELSET=VisualBeams\n')
        for i in range(0,nelem):
            inpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nnode):
                inpfile.write(', {:d}'.format(beam_connectivity[i,j])) 
            inpfile.write('\n')
            
    if precrackFlag in ['on','On','Y','y','Yes','yes']:
        visual_connector_precracked_index = []
        inpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualBotConns\n')
        for i in range(0,nel_con_tbot):
            if i in precrack_elem:
                visual_connector_precracked_index.append(count)
                count += 1
            else:
                inpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    inpfile.write(', {:d}'.format(connector_t_bot_connectivity[i,j])) 
                inpfile.write('\n')
        inpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualRegConns\n')
        for i in range(0,nel_con_treg):
            if i in precrack_elem:
                visual_connector_precracked_index.append(count)
                count += 1
            else:
                inpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    inpfile.write(', {:d}'.format(connector_t_reg_connectivity[i,j])) 
                inpfile.write('\n')
        inpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualTopConns\n')
        for i in range(0,nel_con_ttop):
            if i in precrack_elem:
                visual_connector_precracked_index.append(count)
                count += 1
            else:
                inpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    inpfile.write(', {:d}'.format(connector_t_top_connectivity[i,j])) 
                inpfile.write('\n')
        # precracked transverse connector element connectivity 
        inpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualPrecrackTConns\n')
        for i in range(0,len(connector_t_precracked_connectivity)):
            inpfile.write('{:d}'.format(visual_connector_precracked_index[i]))
            for j in range(0,nelnode_connector):
                inpfile.write(', {:d}'.format(connector_t_precracked_connectivity[i][j]))
            inpfile.write('\n')
        # longitudinal connector element connectivity 
        inpfile.write('*Element, type=T3D2,elset=VisualLongConns\n') 
        for i in range(0,nel_con_l):
            if i in precrack_elem:
                connector_l_precracked_index.append(count)
                count += 1
                connector_l_precracked_connectivity.append(connector_l_connectivity[i,:])
            else:
                inpfile.write('{:d}'.format(count))
                count += 1
                for j in range(0,nelnode_connector):
                    inpfile.write(', {:d}'.format(connector_l_connectivity[i,j])) 
                inpfile.write('\n')
        # precracked longitudinal connector element connectivity 
        inpfile.write('*Element, type=T3D2,elset=VisualPrecrackLConns\n') 
        inpfile.write('\n')
        # for i in range(0,len(connector_l_precracked_connectivity)):
        #     inpfile.write('{:d}'.format(connector_l_precracked_index[i]))
        #     for j in range(0,nelnode_connector):
        #         inpfile.write(', {:d}'.format(connector_l_precracked_connectivity[i][j]))
        #     inpfile.write('\n')      
    else:
        inpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualBotConns\n')
        for i in range(0,nel_con_tbot):
            inpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                inpfile.write(', {:d}'.format(connector_t_bot_connectivity[i,j])) 
            inpfile.write('\n')
        inpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualRegConns\n')
        for i in range(0,nel_con_treg):
            inpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                inpfile.write(', {:d}'.format(connector_t_reg_connectivity[i,j])) 
            inpfile.write('\n')
        inpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualTopConns\n')
        for i in range(0,nel_con_ttop):
            inpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                inpfile.write(', {:d}'.format(connector_t_top_connectivity[i,j])) 
            inpfile.write('\n')
        inpfile.write('*ELEMENT, TYPE=T3D2, ELSET=VisualLongConns\n')
        for i in range(0,nel_con_l):
            inpfile.write('{:d}'.format(count))
            count += 1
            for j in range(0,nelnode_connector):
                inpfile.write(', {:d}'.format(connector_l_connectivity[i,j])) 
            inpfile.write('\n')
            
    inpfile.write('** Section: Section-1\n')
    inpfile.write('*Solid Section, elset=VisualBotConns, material=VisualTConns\n')
    inpfile.write('{:8.4E}\n'.format(props_connector_t_bot[3]*cellwallthickness_early))
    inpfile.write('** Section: Section-2\n')
    inpfile.write('*Solid Section, elset=VisualRegConns, material=VisualTConns\n')
    inpfile.write('{:8.4E}\n'.format(props_connector_t_bot[3]*cellwallthickness_early))
    inpfile.write('** Section: Section-3\n')
    inpfile.write('*Solid Section, elset=VisualTopConns, material=VisualTConns\n')
    inpfile.write('{:8.4E}\n'.format(props_connector_t_bot[3]*cellwallthickness_early))
    if precrackFlag in ['on','On','Y','y','Yes','yes']:
        inpfile.write('** Section: Section-4\n')
        inpfile.write('*Solid Section, elset=VisualPrecrackTConns, material=VisualTConns\n')
        inpfile.write('{:8.4E}\n'.format(props_connector_t_bot[3]*cellwallthickness_early))
        inpfile.write('** Section: Section-5\n')
        inpfile.write('*Solid Section, elset=VisualLongConns, material=VisualLConns\n')
        inpfile.write('{:8.4E}\n'.format(props_connector_l[3]))
        inpfile.write('** Section: Section-6\n')
        inpfile.write('*Solid Section, elset=VisualPrecrackLConns, material=VisualLConns\n')
        inpfile.write('{:8.4E}\n'.format(props_connector_l[3]))
        if NURBS_degree == 2:
            inpfile.write('** Section: Section-7  Profile: Profile-1\n')
            inpfile.write('*Beam Section, elset=VisualBeams, material=VisualBeams, temperature=GRADIENTS, section=RECT\n')
            inpfile.write('{:8.4E}, {:8.4E}\n'.format(props_beam[3],props_beam[4]))
            inpfile.write('1.,0.,0.\n')
    else:
        inpfile.write('** Section: Section-4\n')
        inpfile.write('*Solid Section, elset=VisualLongConns, material=VisualLConns\n')
        inpfile.write('{:8.4E}\n'.format(props_connector_l[3]))
        if NURBS_degree == 2:
            inpfile.write('** Section: Section-5  Profile: Profile-1\n')
            inpfile.write('*Beam Section, elset=VisualBeams, material=VisualBeams, temperature=GRADIENTS, section=RECT\n')
            inpfile.write('{:8.4E}, {:8.4E}\n'.format(props_beam[3],props_beam[4]))
            inpfile.write('1.,0.,0.\n')
        
    inpfile.write('*End Instance\n') 
    inpfile.write('** \n')
    # NODE SETS
    inpfile.write('*Nset, nset=AllNodes, instance=Part-1-1, generate\n') 
    inpfile.write('{:d}, {:d}, {:d} \n'.format(1,numnode,1))
    
    if any(x in geoName for x in ['hydrostatic_', 'Hydrostatic_']):
        # boundary nodes
        if len(boundaries) == 4:
            if any(item in boundary_conditions for item in ['left','Left','L']):
                # Nodes on the left
                LeftNodes = (np.where(woodIGAvertices[:,xaxis-1] <= x_min)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=LeftNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(LeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, LeftNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['right','Right','R']):
                # Nodes on the right
                RightNodes = (np.where(woodIGAvertices[:,xaxis-1] >= x_max)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, Nset=RightNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(RightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, RightNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['bottom','Bottom','Bot']):
                # Nodes on the bottom
                BottomNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['top','Top','T']):
                # Nodes on the top
                TopNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['back','Back']):
                # Nodes on the back
                BackNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['front','Front','F']):
                # Nodes on the front
                FrontNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')
        elif len(boundaries) == 6:      
            if any(item in boundary_conditions for item in ['Hydrostatic','hydrostatic']):
                offset = x_max*0.05
                # Nodes on the bottom-left
                BottomLeftNodes = (np.where(woodIGAvertices[:,yaxis-1] <= (-np.sqrt(3)*woodIGAvertices[:,xaxis-1] - np.sqrt(3)*x_max + offset))[0]+1).reshape(1,-1)
                # BottomLeftNodes = (np.where((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= 725.0**2)[0]+1).reshape(1,-1)
                # BottomLeftNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] < x_min/2) &\
                #                             (woodIGAvertices[:,yaxis-1] < 0))[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=BottomLeftNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomLeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, BottomLeftNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the bottom-right
                BottomRightNodes = (np.where(woodIGAvertices[:,yaxis-1] <= (np.sqrt(3)*woodIGAvertices[:,xaxis-1] - np.sqrt(3)*x_max + offset))[0]+1).reshape(1,-1)
                # BottomRightNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] > x_max/2) &\
                #                             (woodIGAvertices[:,yaxis-1] < 0))[0]+1).reshape(1,-1)
                inpfile.write('*Nset, Nset=BottomRightNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomRightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, BottomRightNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the bottom
                BottomNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the top-left
                TopLeftNodes = (np.where(woodIGAvertices[:,yaxis-1] >= (np.sqrt(3)*woodIGAvertices[:,xaxis-1] + np.sqrt(3)*x_max - offset))[0]+1).reshape(1,-1)
                # TopLeftNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] < x_min/2) &\
                #                             (woodIGAvertices[:,yaxis-1] > 0))[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=TopLeftNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopLeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, TopLeftNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the top-right
                TopRightNodes = (np.where(woodIGAvertices[:,yaxis-1] >= (-np.sqrt(3)*woodIGAvertices[:,xaxis-1] + np.sqrt(3)*x_max - offset))[0]+1).reshape(1,-1)
                # TopRightNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] > x_max/2) &\
                #                             (woodIGAvertices[:,yaxis-1] > 0))[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=TopRightNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopRightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, TopRightNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the top
                TopNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
                    
                # Nodes on the front
                FrontNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the back
                BackNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
            
    elif any(x in geoName for x in ['uniaxial_', 'Uniaxial_']):
        # boundary nodes
        if len(boundaries) == 4:
            if any(item in boundary_conditions for item in ['left','Left','L']):
                # Nodes on the left
                LeftNodes = (np.where(woodIGAvertices[:,xaxis-1] <= x_min)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=LeftNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(LeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, LeftNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['right','Right','R']):
                # Nodes on the right
                RightNodes = (np.where(woodIGAvertices[:,xaxis-1] >= x_max)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, Nset=RightNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(RightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, RightNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['bottom','Bottom','Bot']):
                # Nodes on the bottom
                BottomNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['top','Top','T']):
                # Nodes on the top
                TopNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['back','Back']):
                # Nodes on the back
                BackNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
            if any(item in boundary_conditions for item in ['front','Front','F']):
                # Nodes on the front
                FrontNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')
        elif len(boundaries) == 6:      
            if any(item in boundary_conditions for item in ['Hydrostatic','hydrostatic']):
                offset = x_max*0.05
                # Nodes on the bottom-left
                BottomLeftNodes = (np.where(woodIGAvertices[:,yaxis-1] <= (-np.sqrt(3)*woodIGAvertices[:,xaxis-1] - np.sqrt(3)*x_max + offset))[0]+1).reshape(1,-1)
                # BottomLeftNodes = (np.where((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= 725.0**2)[0]+1).reshape(1,-1)
                # BottomLeftNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] < x_min/2) &\
                #                             (woodIGAvertices[:,yaxis-1] < 0))[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=BottomLeftNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomLeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, BottomLeftNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the bottom-right
                BottomRightNodes = (np.where(woodIGAvertices[:,yaxis-1] <= (np.sqrt(3)*woodIGAvertices[:,xaxis-1] - np.sqrt(3)*x_max + offset))[0]+1).reshape(1,-1)
                # BottomRightNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] > x_max/2) &\
                #                             (woodIGAvertices[:,yaxis-1] < 0))[0]+1).reshape(1,-1)
                inpfile.write('*Nset, Nset=BottomRightNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomRightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, BottomRightNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the bottom
                BottomNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the top-left
                TopLeftNodes = (np.where(woodIGAvertices[:,yaxis-1] >= (np.sqrt(3)*woodIGAvertices[:,xaxis-1] + np.sqrt(3)*x_max - offset))[0]+1).reshape(1,-1)
                # TopLeftNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] < x_min/2) &\
                #                             (woodIGAvertices[:,yaxis-1] > 0))[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=TopLeftNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopLeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, TopLeftNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the top-right
                TopRightNodes = (np.where(woodIGAvertices[:,yaxis-1] >= (-np.sqrt(3)*woodIGAvertices[:,xaxis-1] + np.sqrt(3)*x_max - offset))[0]+1).reshape(1,-1)
                # TopRightNodes = (np.where(((woodIGAvertices[:,xaxis-1]**2 + woodIGAvertices[:,yaxis-1]**2) >= y_max**2) &\
                #                             (woodIGAvertices[:,xaxis-1] > x_max/2) &\
                #                             (woodIGAvertices[:,yaxis-1] > 0))[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=TopRightNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopRightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, TopRightNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the top
                TopNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
                    
                # Nodes on the front
                FrontNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')
                # Nodes on the back
                BackNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min)[0]+1).reshape(1,-1)
                inpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                    inpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
    else:
        if merge_operation in ['on','On','Y','y','Yes','yes']:
            # boundary nodes
            if len(boundaries) == 4:
                if any(item in boundary_conditions for item in ['left','Left','L']):
                    # Nodes on the left
                    LeftNodes = (np.where(woodIGAvertices[:,xaxis-1] <= x_min+merge_tol)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=LeftNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(LeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, LeftNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['right','Right','R']):
                    # Nodes on the right
                    RightNodes = (np.where(woodIGAvertices[:,xaxis-1] >= x_max-merge_tol)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, Nset=RightNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(RightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, RightNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['bottom','Bottom','Bot']):
                    # Nodes on the bottom
                    BottomNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min+merge_tol)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['top','Top','T']):
                    # Nodes on the top
                    TopNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max-merge_tol)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['back','Back']):
                    # Nodes on the back
                    BackNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min+merge_tol)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['front','Front','F']):
                    # Nodes on the front
                    FrontNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max-merge_tol)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')
    
            # boundary nodes
            elif len(boundaries) == 8:
                # if any(item in boundary_conditions for item in ['left','Left','L']):
                    # Nodes on the left bottom
                    LeftBottomNodes = (np.where((woodIGAvertices[:,xaxis-1] <= x_min+merge_tol) & (woodIGAvertices[:,yaxis-1] <= (y_min+y_max)/2))[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=LeftBottomNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(LeftBottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, LeftBottomNodes[0][15*i:15*(i+1)])))+',\n')
                    # Nodes on the left top
                    LeftTopNodes = (np.where((woodIGAvertices[:,xaxis-1] <= x_min+merge_tol) & (woodIGAvertices[:,yaxis-1] >= (y_min+y_max)/2))[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=LeftTopNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(LeftTopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, LeftTopNodes[0][15*i:15*(i+1)])))+',\n')
                # if any(item in boundary_conditions for item in ['right','Right','R']):
                    # Nodes on the right
                    RightNodes = (np.where(woodIGAvertices[:,xaxis-1] >= x_max-merge_tol)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, Nset=RightNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(RightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, RightNodes[0][15*i:15*(i+1)])))+',\n')
                # if any(item in boundary_conditions for item in ['bottom','Bottom','Bot']):
                    # Nodes on the bottom
                    BottomNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min+merge_tol)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
                # if any(item in boundary_conditions for item in ['top','Top','T']):
                    # Nodes on the top
                    TopNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max-merge_tol)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
                # if any(item in boundary_conditions for item in ['back','Back']):
                    # Nodes on the back
                    BackNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min+merge_tol)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
                # if any(item in boundary_conditions for item in ['front','Front','F']):
                    # Nodes on the front
                    FrontNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max-merge_tol)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')
        else: # not merged
            # boundary nodes
            if len(boundaries) == 4:
                if any(item in boundary_conditions for item in ['left','Left','L']):
                    # Nodes on the left
                    LeftNodes = (np.where(woodIGAvertices[:,xaxis-1] <= x_min)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=LeftNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(LeftNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, LeftNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['right','Right','R']):
                    # Nodes on the right
                    RightNodes = (np.where(woodIGAvertices[:,xaxis-1] >= x_max)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, Nset=RightNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(RightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, RightNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['bottom','Bottom','Bot']):
                    # Nodes on the bottom
                    BottomNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['top','Top','T']):
                    # Nodes on the top
                    TopNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['back','Back']):
                    # Nodes on the back
                    BackNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['front','Front','F']):
                    # Nodes on the front
                    FrontNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')
    
            # boundary nodes
            elif len(boundaries) == 8:
                if any(item in boundary_conditions for item in ['left','Left','L']):
                    # Nodes on the left bottom
                    LeftBottomNodes = (np.where((woodIGAvertices[:,xaxis-1] <= x_min) & (woodIGAvertices[:,yaxis-1] <= (y_min+y_max)/2))[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=LeftBottomNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(LeftBottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, LeftBottomNodes[0][15*i:15*(i+1)])))+',\n')
                    # Nodes on the left top
                    LeftTopNodes = (np.where((woodIGAvertices[:,xaxis-1] <= x_min) & (woodIGAvertices[:,yaxis-1] >= (y_min+y_max)/2))[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=LeftTopNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(LeftTopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, LeftTopNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['right','Right','R']):
                    # Nodes on the right
                    RightNodes = (np.where(woodIGAvertices[:,xaxis-1] >= x_max)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, Nset=RightNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(RightNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, RightNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['bottom','Bottom','Bot']):
                    # Nodes on the bottom
                    BottomNodes = (np.where(woodIGAvertices[:,zaxis-1] <= z_min)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=BottomNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BottomNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, BottomNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['top','Top','T']):
                    # Nodes on the top
                    TopNodes = (np.where(woodIGAvertices[:,zaxis-1] >= z_max)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=TopNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(TopNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, TopNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['back','Back']):
                    # Nodes on the back
                    BackNodes = (np.where(woodIGAvertices[:,yaxis-1] <= y_min)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=BackNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(BackNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, BackNodes[0][15*i:15*(i+1)])))+',\n')
                if any(item in boundary_conditions for item in ['front','Front','F']):
                    # Nodes on the front
                    FrontNodes = (np.where(woodIGAvertices[:,yaxis-1] >= y_max)[0]+1).reshape(1,-1)
                    inpfile.write('*Nset, nset=FrontNodes, instance=Part-1-1\n')
                    for i in range(0,math.ceil(len(FrontNodes[0])/15)): # Abaqus only accepts maximum 15 items per row
                        inpfile.write(''.join(','.join(map(str, FrontNodes[0][15*i:15*(i+1)])))+',\n')      
    # ELEMENT SETS
    inpfile.write('*Elset, elset=AllElles, instance=Part-1-1, generate\n') 
    inpfile.write('{:d}, {:d}, {:d} \n'.format(1,nel_con,1))
    inpfile.write('*Elset, elset=AllVisualElle, instance=Part-1-1, generate\n') 
    inpfile.write('{:d}, {:d}, {:d} \n'.format(count_offset+1,count-1,1))
    inpfile.write('*End Assembly\n')
    # AMPLITUDES
    inpfile.write('** \n') 
    inpfile.write('*Amplitude, name=Amp-1, time=TOTAL TIME, definition=SMOOTH STEP \n') 
    inpfile.write('{:8.2E}, {:8.2E}, {:8.2E}, {:8.2E}, {:8.2E}, {:8.2E} \n'.format(0., 0., 0.2*totaltime, 1.0, totaltime, 1.0))
    # MATERIALS
    inpfile.write('**\n')
    inpfile.write('** MATERIALS\n')
    inpfile.write('**\n')
    inpfile.write('*Material, name=VisualTConns\n')
    inpfile.write('*Density\n')
    inpfile.write('<DensityTConn>,\n')
    inpfile.write('*Depvar\n')
    inpfile.write('<NumberOfSvarsTConn>\n')
    inpfile.write('1, SigN, SigN\n')
    inpfile.write('2, SigM, SigM\n')
    inpfile.write('3, SigL, SigL\n')
    inpfile.write('4, EpsN, EpsN\n')
    inpfile.write('5, EpsM, EpsM\n')
    inpfile.write('6, EpsL, EpsL\n')
    inpfile.write('7, ENx, ENx\n')
    inpfile.write('8, ENy, ENy\n')
    inpfile.write('9, ENz, ENz\n')
    inpfile.write('10, EMx, EMx\n')
    inpfile.write('11, EMy, EMy\n')
    inpfile.write('12, EMz, EMz\n')
    inpfile.write('13, ELx, ELx\n')
    inpfile.write('14, ELy, ELy\n')
    inpfile.write('15, ELz, ELz\n')
    inpfile.write('16, Width, Width\n')
    inpfile.write('17, Height, Height\n')
    inpfile.write('18, Length, Length\n')
    inpfile.write('19, EpsNmax, EpsNmax\n')
    inpfile.write('20, EpsTmax, EpsTmax\n')
    inpfile.write('21, ft, ft\n')
    inpfile.write('22, kpost, kpost\n')
    inpfile.write('23, wN, wN\n')
    inpfile.write('24, wM, wM\n')
    inpfile.write('25, wL, wL\n')
    inpfile.write('26, wtotal, wtotal\n')
    inpfile.write('27, EpsNmin, EpsNmin\n')
    inpfile.write('28, EpsV, EpsV\n')
    inpfile.write('29, Ed, Ed\n')
    inpfile.write('30, Edrate, Edrate\n')
    inpfile.write('31, Etd, Etd\n')
    inpfile.write('32, Etdrate, Etdrate\n')
    inpfile.write('*User Material, constants=1\n')
    inpfile.write('0.,\n')
    #
    inpfile.write('*Material, name=VisualLConns\n')
    inpfile.write('*Density\n')
    inpfile.write('<DensityLConn>,\n')
    inpfile.write('*Depvar\n')
    inpfile.write('<NumberOfSvarsLConn>\n')
    inpfile.write('1, SigN, SigN\n')
    inpfile.write('2, SigM, SigM\n')
    inpfile.write('3, SigL, SigL\n')
    inpfile.write('4, EpsN, EpsN\n')
    inpfile.write('5, EpsM, EpsM\n')
    inpfile.write('6, EpsL, EpsL\n')
    inpfile.write('7, ENx, ENx\n')
    inpfile.write('8, ENy, ENy\n')
    inpfile.write('9, ENz, ENz\n')
    inpfile.write('10, EMx, EMx\n')
    inpfile.write('11, EMy, EMy\n')
    inpfile.write('12, EMz, EMz\n')
    inpfile.write('13, ELx, ELx\n')
    inpfile.write('14, ELy, ELy\n')
    inpfile.write('15, ELz, ELz\n')
    inpfile.write('16, Width, Width\n')
    inpfile.write('17, Height, Height\n')
    inpfile.write('18, Length, Length\n')
    inpfile.write('19, EpsNmax, EpsNmax\n')
    inpfile.write('20, EpsTmax, EpsTmax\n')
    inpfile.write('21, ft, ft\n')
    inpfile.write('22, kpost, kpost\n')
    inpfile.write('23, wN, wN\n')
    inpfile.write('24, wM, wM\n')
    inpfile.write('25, wL, wL\n')
    inpfile.write('26, wtotal, wtotal\n')
    inpfile.write('27, EpsNmin, EpsNmin\n')
    inpfile.write('28, EpsV, EpsV\n')
    inpfile.write('29, Ed, Ed\n')
    inpfile.write('30, Edrate, Edrate\n')
    inpfile.write('31, Etd, Etd\n')
    inpfile.write('32, Etdrate, Etdrate\n')
    inpfile.write('*User Material, constants=1\n')
    inpfile.write('0.,\n')
    if NURBS_degree == 2:
        inpfile.write('*Material, name=VisualBeams\n')
        inpfile.write('*Density\n')
        inpfile.write('<DensityBeam>,\n')
        inpfile.write('*Depvar\n')
        inpfile.write('<NumberOfSvarsBeam>\n')
        inpfile.write('1, Nt, Nt\n')
        inpfile.write('2, Qn, Qn\n')
        inpfile.write('3, Qb, Qb\n')
        inpfile.write('4, Mt, Mt\n')
        inpfile.write('5, Mn, Mn\n')
        inpfile.write('6, Mb, Mb\n')
        inpfile.write('7, eps0tt, eps0tt\n')
        inpfile.write('8, gma0tn, gma0tn\n')
        inpfile.write('9, gma0tb, gma0tb\n')
        inpfile.write('10, chit, chit\n')
        inpfile.write('11, chin, chin\n')
        inpfile.write('12, chib, chib\n')
        inpfile.write('13, gpx, gpx\n')
        inpfile.write('14, gpy, gpy\n')
        inpfile.write('15, gpz, gpz\n')
        inpfile.write('16, tx, tx\n')
        inpfile.write('17, ty, ty\n')
        inpfile.write('18, tz, tz\n')
        inpfile.write('19, nx, nx\n')
        inpfile.write('20, ny, ny\n')
        inpfile.write('21, nz, nz\n')
        inpfile.write('22, bx, bx\n')
        inpfile.write('23, by, by\n')
        inpfile.write('24, bz, bz\n')
        inpfile.write('25, kappa, kappa\n')
        inpfile.write('26, tau, tau\n')
        inpfile.write('27, Blength, Blength\n')
        inpfile.write('*User Material, constants=1\n')
        inpfile.write('0.,\n')
        
    # BOUNDARY CONDITIONS
    inpfile.write('** \n')  
    inpfile.write('** BOUNDARY CONDITIONS\n') 
    inpfile.write('** \n')  
    inpfile.write('** Name: BC-Fixed Type: Displacement/Rotation\n') 
    inpfile.write('*Boundary\n')
    if 'hydrostatic_' in geoName:
        if 'free' in geoName:
            inpfile.write('LeftNodes, 1, 1\n')
            inpfile.write('LeftNodes, 4, 4\n')
            inpfile.write('LeftNodes, 5, 5\n')
            inpfile.write('LeftNodes, 6, 6\n')
            
            inpfile.write('BottomNodes, 4, 4\n')
            inpfile.write('BottomNodes, 5, 5\n')
            inpfile.write('BottomNodes, 6, 6\n')
            inpfile.write('TopNodes, 4, 4\n')
            inpfile.write('TopNodes, 5, 5\n')
            inpfile.write('TopNodes, 6, 6\n')
            inpfile.write('BottomLeftNodes, 4, 4\n')
            inpfile.write('BottomLeftNodes, 5, 5\n')
            inpfile.write('BottomLeftNodes, 6, 6\n')
            inpfile.write('BottomRightNodes, 4, 4\n')
            inpfile.write('BottomRightNodes, 5, 5\n')
            inpfile.write('BottomRightNodes, 6, 6\n')
            inpfile.write('TopLeftNodes, 4, 4\n')
            inpfile.write('TopLeftNodes, 5, 5\n')
            inpfile.write('TopLeftNodes, 6, 6\n')
            inpfile.write('TopRightNodes, 4, 4\n')
            inpfile.write('TopRightNodes, 5, 5\n')
            inpfile.write('TopRightNodes, 6, 6\n')
        elif 'free' in geoName:
            inpfile.write('AllNodes, 4, 6\n')
            
    elif 'uniaxial_' in geoName:
        if 'free' in geoName:
            inpfile.write('LeftNodes, 1, 1\n')
            inpfile.write('LeftNodes, 4, 4\n')
            inpfile.write('LeftNodes, 5, 5\n')
            inpfile.write('LeftNodes, 6, 6\n')
            
            inpfile.write('BottomNodes, 4, 4\n')
            inpfile.write('BottomNodes, 5, 5\n')
            inpfile.write('BottomNodes, 6, 6\n')
            inpfile.write('TopNodes, 4, 4\n')
            inpfile.write('TopNodes, 5, 5\n')
            inpfile.write('TopNodes, 6, 6\n')
            inpfile.write('BottomLeftNodes, 4, 4\n')
            inpfile.write('BottomLeftNodes, 5, 5\n')
            inpfile.write('BottomLeftNodes, 6, 6\n')
            inpfile.write('BottomRightNodes, 4, 4\n')
            inpfile.write('BottomRightNodes, 5, 5\n')
            inpfile.write('BottomRightNodes, 6, 6\n')
            inpfile.write('TopLeftNodes, 4, 4\n')
            inpfile.write('TopLeftNodes, 5, 5\n')
            inpfile.write('TopLeftNodes, 6, 6\n')
            inpfile.write('TopRightNodes, 4, 4\n')
            inpfile.write('TopRightNodes, 5, 5\n')
            inpfile.write('TopRightNodes, 6, 6\n')
        elif 'free' in geoName:
            inpfile.write('AllNodes, 4, 6\n')
    else:   
        inpfile.write('LeftNodes, 4, 6\n')
        inpfile.write('RightNodes, 4, 6\n')
        inpfile.write('BottomNodes, 4, 6\n')
        inpfile.write('TopNodes, 4, 6\n')
        inpfile.write('FrontNodes, 4, 6\n')
        inpfile.write('BackNodes, 4, 6\n')

    inpfile.write('** ------------------------------------------------------- \n') 
    inpfile.write('** \n')  
    inpfile.write('** STEP: Load\n') 
    inpfile.write('** \n')  
    inpfile.write('*Step, name=Load, nlgeom=NO\n') 
    inpfile.write('*Dynamic, Explicit, DIRECT USER CONTROL\n') 
    inpfile.write('{:8.4e}, {:8.4e} \n'.format(timestep, totaltime))
    inpfile.write('*Bulk Viscosity\n') 
    inpfile.write('0.06, 1.2\n')
    inpfile.write('** Mass Scaling: Semi-Automatic\n')
    inpfile.write('**               Whole Model\n')
    inpfile.write('*Fixed Mass Scaling, dt={:8.4e}, type=below min\n'.format(timestep))
    inpfile.write('** \n')  
    inpfile.write('** BOUNDARY CONDITIONS\n') 
    inpfile.write('** \n')  
    inpfile.write('** Name: BC-velo Type: Velocity/Angular velocity\n') 
    
    # inpfile.write('*Boundary, type=VELOCITY\n') 
    # inpfile.write('TopNodes, {:d}, {:d}, {:e}\n'.format(BC_velo_dof,BC_velo_dof,BC_velo_value))
    # inpfile.write('*Boundary, type=VELOCITY\n') 
    # inpfile.write('RightNodes, {:d}, {:d}, {:e}\n'.format(BC_velo_dof,BC_velo_dof,BC_velo_value))
    if 'hydrostatic_' in geoName:
        inpfile.write('*Boundary, type=VELOCITY, amp=Amp-1\n')
        inpfile.write('TopNodes, 2, 2, {:e}\n'.format(BC_velo_value))
        inpfile.write('BottomNodes, 2, 2, {:e}\n'.format(-BC_velo_value))
        inpfile.write('TopRightNodes, 1, 1, {:e}\n'.format(BC_velo_value*np.cos(np.pi/6)))
        inpfile.write('TopRightNodes, 2, 2, {:e}\n'.format(BC_velo_value*np.sin(np.pi/6)))
        inpfile.write('BottomRightNodes, 1, 1, {:e}\n'.format(BC_velo_value*np.cos(-np.pi/6)))
        inpfile.write('BottomRightNodes, 2, 2, {:e}\n'.format(BC_velo_value*np.sin(-np.pi/6)))
        inpfile.write('TopLeftNodes, 1, 1, {:e}\n'.format(BC_velo_value*np.cos(np.pi - np.pi/6)))
        inpfile.write('TopLeftNodes, 2, 2, {:e}\n'.format(BC_velo_value*np.sin(np.pi - np.pi/6)))
        inpfile.write('BottomLeftNodes, 1, 1, {:e}\n'.format(BC_velo_value*np.cos(np.pi + np.pi/6)))
        inpfile.write('BottomLeftNodes, 2, 2, {:e}\n'.format(BC_velo_value*np.sin(np.pi + np.pi/6)))   
    
    elif 'uniaxial_' in geoName:
        inpfile.write('*Boundary, type=VELOCITY, amp=Amp-1\n')
        inpfile.write('TopNodes, 2, 2, {:e}\n'.format(BC_velo_value))
        inpfile.write('BottomNodes, 2, 2, {:e}\n'.format(-BC_velo_value))
        inpfile.write('TopRightNodes, 1, 1, {:e}\n'.format(BC_velo_value*np.cos(np.pi/6)))
        inpfile.write('TopRightNodes, 2, 2, {:e}\n'.format(BC_velo_value*np.sin(np.pi/6)))
        inpfile.write('BottomRightNodes, 1, 1, {:e}\n'.format(BC_velo_value*np.cos(-np.pi/6)))
        inpfile.write('BottomRightNodes, 2, 2, {:e}\n'.format(BC_velo_value*np.sin(-np.pi/6)))
        inpfile.write('TopLeftNodes, 1, 1, {:e}\n'.format(BC_velo_value*np.cos(np.pi - np.pi/6)))
        inpfile.write('TopLeftNodes, 2, 2, {:e}\n'.format(BC_velo_value*np.sin(np.pi - np.pi/6)))
        inpfile.write('BottomLeftNodes, 1, 1, {:e}\n'.format(BC_velo_value*np.cos(np.pi + np.pi/6)))
        inpfile.write('BottomLeftNodes, 2, 2, {:e}\n'.format(BC_velo_value*np.sin(np.pi + np.pi/6)))   
        
    # OUTPUT
    inpfile.write('** \n') 
    inpfile.write('** OUTPUT REQUESTS\n') 
    inpfile.write('** \n') 
    inpfile.write('*Output, Field, Number interval=100\n') 
    inpfile.write('*Node Output, Nset=AllNodes\n') 
    inpfile.write('U, V, A, RF \n')
    inpfile.write('*Element Output, elset=AllVisualElle, directions=NO\n')
    inpfile.write('SDV\n')
    inpfile.write('**\n') 
    inpfile.write('** HISTORY OUTPUT: H-Output-1\n') 
    inpfile.write('**\n') 
    inpfile.write('*Output, history, time interval={:8.4e}\n'.format(totaltime/100.0)) 
    inpfile.write('*Energy Output\n') 
    inpfile.write('ETOTAL, ALLWK, ALLKE\n') 
    inpfile.write('**\n') 
    inpfile.write('** HISTORY OUTPUT: H-Output-2\n') 
    inpfile.write('**\n') 
    inpfile.write('*Output, history, time interval={:8.4e}\n'.format(totaltime/100.0)) 
    inpfile.write('*Energy Output, elset=AllElles\n') 
    inpfile.write('ALLIE\n') 
    inpfile.write('*End Step') 
    inpfile.close()


def ChronoMesh(geoName,woodIGAvertices,beam_connectivity,NURBS_degree,nctrlpt_per_beam,nconnector_t_per_beam,npatch,knotVec):

    # Beam
   
    numnode = woodIGAvertices.shape[0]
    nelem = beam_connectivity.shape[0]
    nnode = beam_connectivity.shape[1]


    # Generate a .inp file which can be directly imported and played in Abaqus
    nodefile = open(Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName + '-chronoNodes.dat'),'w')
    elementfile = open(Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName + '-chronoElements.dat'),'w')

    # nodes
    for i in range(0,numnode):
        nodefile.write('{:#.9e}, {:#.9e},  {:#.9e}\n'.format(woodIGAvertices[i,0],woodIGAvertices[i,1],woodIGAvertices[i,2]))

    # beam element connectivity 
    for i in range(0,nelem):
        elementfile.write('{:d}, {:d},  {:d}\n'.format(beam_connectivity[i,0],beam_connectivity[i,1],beam_connectivity[i,2]))

    elementfile.close()
    nodefile.close()

    # IGA file
    igafile = open (Path(App.ConfigGet('UserHomePath') + '/woodWorkbench' + '/' + geoName + '/' + geoName + '-chronoIGA.dat'),'w')
    igafile.write('# Dimension of beam elements \n')
    igafile.write('1 \n')
    igafile.write('# Order of basis function \n')
    igafile.write('{:d} \n'.format(NURBS_degree))
    igafile.write('# Number of control points per patch \n')
    igafile.write('{:d} \n'.format(nctrlpt_per_beam))
    igafile.write('# Number of elements per patch \n') 
    igafile.write('{:d} \n'.format(nconnector_t_per_beam-1))
    igafile.write('# Number of Patches \n') 
    igafile.write('{:d} \n'.format(npatch))
    # Loop over patches
    for i in range(0,npatch):
        igafile.write('{:s} \n'.format('PATCH-'+str(i+1)))
        igafile.write('Size of knot vectors \n') 
        igafile.write('{:d} \n'.format(knotVec.shape[1])) 
        igafile.write('knot vectors \n')
        for j in range(0,knotVec.shape[1]):
            igafile.write('{:f} '.format(knotVec[i,j])) 
        igafile.write('\n')
        igafile.write('{:d}, {:d}, {:d}, {:d}, {:d}\n'.format(beam_connectivity[2*i,0],beam_connectivity[2*i,1],beam_connectivity[2*i,2],beam_connectivity[2*i+1,1],beam_connectivity[2*i+1,2]))
    
    igafile.close()


def ModelInfo(height,net_area,sec_area):

    #Calculate the material properties of generated geometry
    
    wall_density = 1.5e-9 # unit: tonne/mm3
    # Kellogg, Robert M., and Frederick F. Wangaard. "Variation in the cell-wall density of wood." Wood and Fiber Science (1969): 180-204.
    # Plötze, Michael, and Peter Niemz. "Porosity and pore size distribution of different wood types as determined by mercury intrusion porosimetry." \
    # European journal of wood and wood products 69.4 (2011): 649-657.
    # https://link.springer.com/article/10.1007/s00107-010-0504-0
    # norway spruce = 73% according to above

    net_volume = net_area*height
    mass = wall_density*net_volume
    
    bulk_volume = height*sec_area
    bulk_density = mass/bulk_volume
    
    mass_if_all_solid = wall_density*bulk_volume # if all skeleton phase
    porosity = 1 - mass/mass_if_all_solid

    return mass,bulk_volume,bulk_density,porosity


def LogFile(geoName,outDir,mass,bulk_volume,bulk_density,porosity,net_area,gross_area):
    
    #Generate the log file (geoName.log file) for the generation procedure.
    logfile = open(Path(outDir + '/' + geoName + '/' + geoName + '-input.cwPar'),'a')
    
    # get current local time
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Add to log file                                     
    logfile.write('END' + '\n')
    logfile.write('[assuming mm kg tonnes]\n')
    logfile.write('net_area= ' + str('{:0.3e}'.format(net_area)) + ' [mm^2] \n')
    logfile.write('gross_area= ' + str('{:0.3e}'.format(gross_area)) + ' [mm^2] \n')
    logfile.write('mass= ' + str('{:0.3e}'.format(mass)) + ' [tonne] \n')
    logfile.write('bulk_volume= ' + str('{:0.3e}'.format(bulk_volume)) + ' [mm^3] \n')
    logfile.write('bulk_density= ' + str('{:0.3e}'.format(bulk_density)) + ' [tonne/mm^3] \n')
    logfile.write('porosity= ' + str('{:0.1f}'.format(porosity*100)) + ' [%] \n')
    logfile.write('current_time= ' + str(current_time) + '\n')
    logfile.write('\n')

    logfile.close()


