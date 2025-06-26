## ================================================================================
## CHRONO WOOD WORKBENCH
##
## Copyright (c) 2025 
## All rights reserved. 
##
## Use of this source code is governed by a BSD-style license that can be found
## in the LICENSE file at the top level of the distribution
##
## ================================================================================
## Developed by Cusatis Computational Serivces, Inc.
## Primary Authors: Susan Alexis Brown, Hao Yin
## This is a test
## ===========================================================================

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.tri as mtri
from pylab import *
import os
import time
# import cProfile
# import pstats
# from line_profiler import LineProfiler

from pathlib import Path
import triangle as tr
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.path as mpltPath
import FreeCAD as App # type: ignore
import FreeCADGui as Gui # type: ignore
import feminout.importVTKResults as importVTK # type: ignore
import ObjectsFem # type: ignore
import FemGui # type: ignore
from PySide import QtCore, QtGui # type: ignore

from freecad.woodWorkbench.src.inputParams import inputParams
from freecad.woodWorkbench.src.readLog import readLog

import freecad.woodWorkbench.tools.WoodMeshGenTools_v11 as WoodMeshGen
from freecad.woodWorkbench.tools.rf_generator import RandomField

# cProfile.run("",sort='ncalls')

def main(self):

    # performance check only
    # pr = cProfile.Profile()
    # pr.enable()
    
    # profiler = LineProfiler()

    # ==================================================================
    self.form[3].progressBar.setValue(10) 
    self.form[3].statusWindow.setText("Status: Reading Parameters.") 
    # ==================================================================
    # Input parameters 
    [geoName, radial_growth_rule, iter_max, \
        nrings, width_heart, width_early, width_late, generation_center, \
        cellsize_early, cellsize_late, cellwallthickness_early, cellwallthickness_late, \
        boundaryFlag, box_shape, box_center, box_size, box_height, \
        nsegments, theta_max, theta_min, z_max, z_min, long_connector_ratio, \
        x_notch_size, y_notch_size, precrack_start,precrack_end, \
        mergeFlag, merge_tol, precrackFlag, \
        inpType, randomFlag, randomParams, NURBS_degree, box_width, box_depth, visFlag, \
        knotFlag, knotParams, outDir,flowFlag,rayFlag]\
            = inputParams(self.form)
    
    
    # ==================================================================
    self.form[3].progressBar.setValue(15) 
    self.form[3].statusWindow.setText("Status: Initializing Files.") 
    # ==================================================================

    
    # Prep freecad naming variables
    analysisName = geoName + "_analysis"
    materialName = geoName + "_material"
    visualFilesName = geoName + '_visualFiles'
        

    # Make new freecad document and set view if does not exisit
    try:
        docGui.activeView().viewAxonometric()
    except:
        App.newDocument(geoName)
        docGui = Gui.activeDocument()
        docGui.activeView().viewAxonometric()
    Gui.runCommand('Std_PerspectiveCamera',1)

    # Generate new analysis object with associated material
    try:
        test = (App.getDocument(App.ActiveDocument.Name).getObjectsByLabel(analysisName)[0] != None)
    except:
        test = (App.getDocument(App.ActiveDocument.Name).getObjectsByLabel(analysisName) != [])
    if test == False:
        # Analysis
        analysis_object = ObjectsFem.makeAnalysis(App.ActiveDocument,analysisName)
        # Store Material
        material_object = ObjectsFem.makeMaterialSolid(App.ActiveDocument, materialName)
        mat = material_object.Material
        mat['Name'] = materialName
        material_object.Material = mat
        analysis_object.addObject(material_object)

    # # Start figure
    plt.close()
    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    print('\n')
    print(geoName,':')
    # ==================================================================
    self.form[3].progressBar.setValue(20) 
    self.form[3].statusWindow.setText("Status: Placing Cells.") 
    # ==================================================================
    # Place cells with a specific radial growth pattern

    x_min,x_max,y_min,y_max,boundaries,boundary_points_original, sec_area = \
        WoodMeshGen.Clipping_Box(box_shape,box_center,box_size,box_width,box_depth,x_notch_size,y_notch_size,self.form)
    
    if radial_growth_rule == 'binary_lloyd':
        # ---------------------------------------------
        # binary with Lloyd's algorithm (e.g. wood microstructure with earlywood-latewood alternations, but more regular cell shapes)
        sites_vor, radii = WoodMeshGen.CellPlacement_Binary_Lloyd(nrings,width_heart,width_early,width_late,\
                                                    cellsize_early,cellsize_late,iter_max,\
                                                    mergeFlag,boundary_points_original,omega=1)

    elif radial_growth_rule == 'debug':
        # ---------------------------------------------
        # debug run
        sites_vor,radii = WoodMeshGen.CellPlacement_Debug(nrings,width_heart,width_early,width_late)  

    elif radial_growth_rule == 'readsites':
        # ---------------------------------------------
        # read in from file
        oldgeoName = self.form[0].geoName.text()
        sites_vor = np.load(Path(outDir + '/' + oldgeoName + '/' + oldgeoName + '_sites.npy'))
        radii = np.load(Path(outDir + '/' + oldgeoName + '/' + oldgeoName + '_radii.npy'))

    nParticles = len(sites_vor)
    print(int(nParticles), 'cells placed')

    # ==================================================================
    self.form[3].progressBar.setValue(30) 
    self.form[3].statusWindow.setText("Status: Generating Flow Mesh.") 
    # ==================================================================

    num_bound = np.shape(boundary_points_original)[0]  # create boundary segements to enforce boundaries 
    boundary_segments = np.array([np.linspace(0,num_bound-1,num_bound),np.concatenate((np.linspace(1,num_bound-1,num_bound-1),np.array([0])))]).transpose()
    boundary_region = np.array([[box_center[0],box_center[1],1,0]])

    boundary_coords = boundary_points_original[np.array(boundary_segments,dtype='int')]
    for seg in boundary_coords:
        ax.plot(seg[:,0],seg[:,1],'ko--',markersize=1.,linewidth=0.1)

    # ax.plot(sites_vor[:,0],sites_vor[:,1],'ks',markersize=3.)

    
    # ---------------------------------------------
    # # Delaunay triangulation             
    # based on old sites to get vor
    delaunay_vertices = np.concatenate((boundary_points_original,np.array(sites_vor))) 
    # important the boundary points are listed first for region index reasons, and thus not included in tessellation
    tri_inp = {'vertices': delaunay_vertices,'segments':boundary_segments,'regions':boundary_region}
    conforming_delaunay = tr.triangulate(tri_inp, 'peq3D') 

    # vertsD = np.array(conforming_delaunay['vertices'])
    # ax.triplot(vertsD[:, 0], vertsD[:, 1], conforming_delaunay['triangles'], 'r^-',markersize=2.,linewidth=0.15)
    # vorsci = Voronoi(conforming_delaunay.get('vertices')) # location of vor sites
    # ax.plot(vorsci.vertices[:,0],vorsci.vertices[:,1],'bo',markersize=3.)
    # vortri_vertices = tr.voronoi(conforming_delaunay.get('vertices'))[0]
    # ax.plot(vortri_vertices[:,0],vortri_vertices[:,1],'g^',markersize=3.)
    # plt.show()

    print('pre-flow')
    # ---------------------------------------------
    # # Build flow mesh
    if flowFlag in ['on','On','Y','y','Yes','yes']:
        flow_nodes, flow_elems = WoodMeshGen.BuildFlowMesh(outDir,geoName,nsegments,long_connector_ratio,z_min,z_max, \
                                                        boundaries,conforming_delaunay)
        # flow_nodes are conforming_delaunay vertices layered with z-value added, but volume is calculated based on old sites to match 
        # mechanical cell shapes
    else:
        flow_nodes = np.empty([0,3])
        flow_elems = np.empty([0,2])

    # ---------------------------------------------
    # # Build mechanical mesh 
    vor_vertices, vor_edges, ray_origins, ray_directions = tr.voronoi(conforming_delaunay.get('vertices'))


    # ==================================================================
    self.form[3].progressBar.setValue(40) 
    self.form[3].statusWindow.setText("Status: Building 2D Mesh.") 
    # ==================================================================
    # Rebuild the Voronoi mesh

    [voronoi_vertices,finite_ridges_new,\
        boundary_ridges_new,nvertex,nvertices_in,nfinite_ridge,nboundary_ridge,\
        nboundary_pts,voronoi_ridges,nridge] = \
        WoodMeshGen.RebuildVoronoi_ConformingDelaunay_New(vor_vertices,vor_edges,ray_origins,ray_directions,\
                                                        boundaries,boundaryFlag,boundary_points_original)

    # ---------------------------------------------
    # # Visualize the meshes

    # # Main cells
    for beg, end in voronoi_ridges.astype(int):
        x0, y0 = voronoi_vertices[beg, :]
        x1, y1 = voronoi_vertices[end, :]
        ax.plot(
            [x0, x1],
            [y0, y1],'k-',linewidth=0.25,markersize=0.15)    
    # for x,y in zip(voronoi_vertices[:,0], voronoi_vertices[:,1]):   
    #     ax.annotate('({:.2e}, \n{:.2e})'.format(x,y),(x,y),size=3,color='k')

    # Flow
    # vertsD = np.array(conforming_delaunay['vertices'])
    # ax.triplot(vertsD[:, 0], vertsD[:, 1], conforming_delaunay['triangles'], 'b^-',markersize=2.,linewidth=0.15)
    for el in flow_elems:
        beg = int(el[0])
        end = int(el[1])
        x0,y0 = flow_nodes[beg,1:3]
        x1,y1 = flow_nodes[end,1:3]
        ax.plot(
            [x0, x1],
            [y0, y1],'r-',linewidth=0.15)
    # ax.plot(flow_nodes[:,1],flow_nodes[:,2],'r^',markersize=0.1)
    # for x,y in zip(flow_nodes[:,1], flow_nodes[:,2]):   
    #     ax.annotate('({:.2e}, \n{:.2e})'.format(x,y),(x,y),size=5,color='r')


    # plt.show()
    plt.savefig(Path(outDir + '/' + geoName + '/' + geoName + '.png'), format='png', dpi=1000) 


    if randomFlag in ['on','On','Y','y','Yes','yes']:
        # ==================================================================
        self.form[3].progressBar.setValue(45) 
        self.form[3].statusWindow.setText("Status: Generating Random Field.") 
        # ==================================================================
        # Calculate random field
        RF_dist_types = randomParams.get('RF_dist_types')
        RF_dist_params = randomParams.get('RF_dist_params')
        RF_corr_l = randomParams.get('RF_corr_l')
        RF_sampling_type = randomParams.get('RF_sampling_type')

        random_field = RandomField(dimension = 1, readFromFolder = "none", dist_types=RF_dist_types, CC = np.array([[1]]), \
                         dist_params=RF_dist_params, name=geoName,out=outDir,corr_l =RF_corr_l, corr_f='square_exponential',\
                         x_range=[z_min,z_max], sampling_type = RF_sampling_type, periodic = True, filesavetype="binary",sparse=True)
    else:
        random_field = []
    # ==================================================================
    self.form[3].progressBar.setValue(50) 
    self.form[3].statusWindow.setText("Status: Building 3D Mesh.") 
    # ================================================================== 
    [voronoi_vertices_3D,nvertices_3D,nlayers,segment_length,nctrlpt_per_elem,nctrlpt_per_beam,nconnector_t_per_beam,\
           nconnector_t_per_grain,theta,z_coord,npt_per_layer,finite_ridges_3D,boundary_ridges_3D,voronoi_vertices_2D] = \
    WoodMeshGen.LayerOperation(NURBS_degree,nsegments,theta_min,theta_max,finite_ridges_new,boundary_ridges_new,nfinite_ridge,nboundary_ridge,\
                   z_min,z_max,long_connector_ratio,voronoi_vertices,nvertex,generation_center,knotFlag, knotParams, box_center,box_depth)
    

    # Insert mid and quarter points on the Voronoi ridges (can be used as potential failure positions on cell walls)
    [all_pts_3D,npt_per_layer_vtk,all_pts_2D,all_ridges] = \
    WoodMeshGen.RidgeMidQuarterPts(voronoi_vertices_3D,nvertex,nvertices_in,voronoi_ridges,\
                    finite_ridges_new,boundary_ridges_new,finite_ridges_3D,boundary_ridges_3D,nfinite_ridge,\
                    nboundary_ridge,nboundary_pts,nlayers,voronoi_vertices)


    # ---------------------------------------------
    # Generate a file for vertices and ridges info
    [all_vertices_2D, max_wings, flattened_all_vertices_2D, all_ridges,vert_area] = \
        WoodMeshGen.VertexandRidgeinfo(all_pts_2D,all_ridges,npt_per_layer,\
                        geoName,radii,generation_center,\
                        cellwallthickness_early,cellwallthickness_late)
    

    # ==================================================================
    self.form[3].progressBar.setValue(60) 
    self.form[3].statusWindow.setText("Status: Generating Beams.") 
    # ==================================================================
    # Extrude in the parallel-to-grain (longitudinal) direction
    
    [IGAvertices,beam_connectivity_original,nbeam_total,\
    beam_connectivity,nbeamElem,\
    connector_t_bot_connectivity,connector_t_top_connectivity,\
    connector_t_reg_connectivity,connector_l_connectivity,\
    nconnector_t,nconnector_l,connector_l_vertex_dict] = \
    WoodMeshGen.GenerateBeamElement(voronoi_vertices_3D,nvertices_3D,NURBS_degree,nctrlpt_per_beam,nctrlpt_per_elem,nsegments,\
                        npt_per_layer,\
                        nvertex,voronoi_ridges,nridge,all_vertices_2D,\
                        nconnector_t_per_beam,nconnector_t_per_grain)

    BeamTime = time.time() 
    print(nbeamElem,'beam elements generated')


    # ---------------------------------------------
    # Insert precracks
    if precrackFlag in ['on','On','Y','y','Yes','yes']:

        x_start, y_start = precrack_start
        x_end, y_end = precrack_end
        precrack_nodes = np.array([[x_start, y_start, x_end, y_end]])

        [precrack_elem,nconnector_t_precrack,nconnector_l_precrack] = \
            WoodMeshGen.InsertPrecrack(all_pts_2D,all_ridges,nridge,precrack_nodes,\
                                    cellsize_early,nsegments)
        plt.savefig(Path(outDir + '/' + geoName + '/' + geoName + '.png'), format='png', dpi=1000) 
        # plt.close()
    else:
        precrack_nodes = []
        precrack_elem = []
        nconnector_t_precrack = 0
        nconnector_l_precrack = 0
        # plt.close()

    # ==================================================================
    self.form[3].progressBar.setValue(70) 
    self.form[3].statusWindow.setText("Status: Calculating Mesh Info.") 
    # ==================================================================
    # Calculate mesh info

    # lp = LineProfiler()
    # lp_wrapper = lp(WoodMeshGen.ConnectorMeshFile)
    # [ConnMeshData,conn_l_tangents,height_connector_t] = lp_wrapper(geoName,IGAvertices,connector_t_bot_connectivity,\
    #                 connector_t_reg_connectivity,connector_t_top_connectivity,\
    #                 connector_l_connectivity,all_vertices_2D,\
    #                 max_wings,flattened_all_vertices_2D,nsegments,segment_length,\
    #                 nctrlpt_per_beam,theta,nridge,\
    #                 randomFlag,random_field,knotParams,knotFlag,box_center,voronoi_vertices_2D,precrack_elem)
    # lp.print_stats()
    [ConnMeshData,conn_l_tangents,height_connector_t] = \
        WoodMeshGen.ConnectorMeshFile(geoName,IGAvertices,connector_t_bot_connectivity,\
                    connector_t_reg_connectivity,connector_t_top_connectivity,\
                    connector_l_connectivity,all_vertices_2D,\
                    max_wings,flattened_all_vertices_2D,nsegments,segment_length,long_connector_ratio,\
                    nctrlpt_per_beam,theta,nridge,\
                    randomFlag,random_field,knotParams,knotFlag,box_center,voronoi_vertices_2D,precrack_elem,cellwallthickness_early,radii,z_max,rayFlag)

    # ---------------------------------------------

    # Calculate model properties
    [mass,bulk_volume,bulk_density,porosity] = WoodMeshGen.ModelInfo(box_height,vert_area,sec_area)

    # ---------------------------------------------
    # Bezier extraction 
    knotVec = WoodMeshGen.BezierExtraction(NURBS_degree,nbeam_total)
    npatch = beam_connectivity_original.shape[0]

    WoodMeshGen.BezierBeamFile(geoName,NURBS_degree,nctrlpt_per_beam,\
                    nconnector_t_per_beam,npatch,knotVec)

    # ==================================================================
    self.form[3].progressBar.setValue(80) 
    self.form[3].statusWindow.setText("Status: Generating Model Files.") 
    # ==================================================================
    # Generate input files for numerical simulations

    [nsvars_beam, nsecgp, nsvars_secgp, iprops_beam, props_beam, \
        nsvars_conn_t, iprops_connector_t_bot, props_connector_t_bot, iprops_connector_t_reg, props_connector_t_reg, \
        iprops_connector_t_top, props_connector_t_top, \
        nsvars_conn_l, iprops_connector_l, props_connector_l, props_strainrate] = \
            WoodMeshGen.ABQParams(nconnector_l,nconnector_l_precrack,nconnector_t,nconnector_t_precrack,\
                 cellwallthickness_early,cellwallthickness_late,height_connector_t,nbeamElem)

    timestep     = 5.0E-9
    totaltime    = 5.0E-3
    z_min = np.min(z_coord)
    z_max = np.max(z_coord)
    boundary_conditions = ['Bottom','Top','Left','Right','Front','Back']
    BC_velo_dof = 1 # 1-x, 2-y, 3-z
    BC_velo_value = box_size*0.03 # mm/s

    if inpType == 'abaqus':
        WoodMeshGen.AbaqusFile(geoName,NURBS_degree,npatch,nsegments,IGAvertices,beam_connectivity,\
                    connector_t_bot_connectivity,connector_t_reg_connectivity,\
                    connector_t_top_connectivity,connector_l_connectivity,\
                    segment_length,props_beam,iprops_beam,props_connector_t_bot,iprops_connector_t_bot,\
                    props_connector_t_reg,iprops_connector_t_reg,props_connector_t_top,\
                    iprops_connector_t_top,props_connector_l,iprops_connector_l,props_strainrate,\
                    timestep,totaltime,boundary_conditions,BC_velo_dof,BC_velo_value,\
                    x_max,x_min,y_max,y_min,z_max,z_min,boundaries,nsvars_beam,nsvars_conn_t,\
                    nsvars_conn_l,nsecgp,nsvars_secgp,cellwallthickness_early,mergeFlag,merge_tol,\
                    precrackFlag,precrack_elem)
    elif inpType == 'project chrono':
        # chronoInput(self.form)
        WoodMeshGen.ChronoMesh(geoName,IGAvertices,beam_connectivity,NURBS_degree,nctrlpt_per_beam,nconnector_t_per_beam,npatch,knotVec)
    elif inpType == 'all':
        WoodMeshGen.AbaqusFile(geoName,NURBS_degree,npatch,nsegments,IGAvertices,beam_connectivity,\
                    connector_t_bot_connectivity,connector_t_reg_connectivity,\
                    connector_t_top_connectivity,connector_l_connectivity,\
                    segment_length,props_beam,iprops_beam,props_connector_t_bot,iprops_connector_t_bot,\
                    props_connector_t_reg,iprops_connector_t_reg,props_connector_t_top,\
                    iprops_connector_t_top,props_connector_l,iprops_connector_l,props_strainrate,\
                    timestep,totaltime,boundary_conditions,BC_velo_dof,BC_velo_value,\
                    x_max,x_min,y_max,y_min,z_max,z_min,boundaries,nsvars_beam,nsvars_conn_t,\
                    nsvars_conn_l,nsecgp,nsvars_secgp,cellwallthickness_early,mergeFlag,merge_tol,\
                    precrackFlag,precrack_elem)
        WoodMeshGen.ChronoMesh(geoName,IGAvertices,beam_connectivity,NURBS_degree,nctrlpt_per_beam,nconnector_t_per_beam,npatch,knotVec)
        np.save(Path(outDir + '/' + geoName + '/' + geoName + '_sites.npy'),sites_vor)
        np.save(Path(outDir + '/' + geoName + '/' + geoName + '_radii.npy'),radii)
    else:
        np.save(Path(outDir + '/' + geoName + '/' + geoName + '_sites.npy'),sites_vor)
        np.save(Path(outDir + '/' + geoName + '/' + geoName + '_radii.npy'),radii)
        print('Generated cells and rings info has been saved.')


    # ---------------------------------------------
    # Add to log file for the generation
    WoodMeshGen.LogFile(geoName,outDir,mass,bulk_volume,bulk_density,porosity,vert_area,sec_area)
    
    # ==================================================================
    self.form[3].progressBar.setValue(90) 
    self.form[3].statusWindow.setText("Status: Generating Vizualiation Files.") 
    # ==================================================================
    # Generate Paraview visulization files
    if visFlag in ['on','On','Y','y','Yes','yes']:
        
        # # Create visualization files
        # lp = LineProfiler()
        # lp_wrapper = lp(WoodMeshGen.VisualizationFiles)
        # lp_wrapper(geoName,NURBS_degree,nlayers,npt_per_layer_vtk,all_pts_3D,\
        #                nsegments,nridge,voronoi_ridges,all_ridges,nvertex,\
        #                nconnector_t,nconnector_l,nctrlpt_per_beam,ConnMeshData,\
        #                conn_l_tangents,all_vertices_2D, flow_nodes, flow_elems)
        # lp.print_stats()
        WoodMeshGen.VisualizationFiles(geoName,NURBS_degree,nlayers,npt_per_layer_vtk,all_pts_3D,\
                       nsegments,nridge,voronoi_ridges,all_ridges,nvertex,\
                       nconnector_t,nconnector_l,nctrlpt_per_beam,ConnMeshData,\
                       conn_l_tangents,all_vertices_2D, flow_nodes, flow_elems)  

        App.activeDocument().addObject('App::DocumentObjectGroup',visualFilesName)
        App.activeDocument().getObject(visualFilesName).Label = 'Visualization Files'

        importVTK.insert(str(outDir + '/' + geoName + '/' + geoName + '_beams.vtu'),App.ActiveDocument.Name)
        CBLbeamsVTU = App.getDocument(App.ActiveDocument.Name).getObject(geoName + '_beams')
        CBLbeamsVTU.Label = geoName + '_beams'
        App.getDocument(App.ActiveDocument.Name).getObject(visualFilesName).addObject(CBLbeamsVTU)
        CBLbeamsVTU.addProperty("App::PropertyFile",'Location','Paraview VTK File','Location of Paraview VTK file').Location=str(Path(outDir + '/' + geoName + '/' + geoName + '_beams.vtu'))
        CBLbeamsVTU.Visibility = True

        # ==================================================================    
        self.form[3].progressBar.setValue(100) 
        self.form[3].statusWindow.setText("Status: Complete.") 
        # ==================================================================

        App.ActiveDocument.recompute()
        Gui.Control.closeDialog()
        # Gui.activateWorkbench("FemWorkbench")
        FemGui.setActiveAnalysis(App.activeDocument().getObject(analysisName))

        # Set view
        Gui.SendMsgToActiveView("ViewFit")
        docGui.activeView().viewIsometric()
        Gui.runCommand('Std_ViewGroup',0)
    
    else:

        # ==================================================================    
        self.form[3].progressBar.setValue(100) 
        self.form[3].statusWindow.setText("Status: Complete.") 
        # ==================================================================
        
        App.ActiveDocument.recompute()
        Gui.Control.closeDialog()

    # pr.disable()
    # loc = os.path.join(outDir, geoName, 'fullProfile.cProf')
    # pr.dump_stats(loc)
    # p = pstats.Stats(loc)
    # p.strip_dirs().sort_stats('cumulative').print_stats(10)


if __name__ == '__main__':
    main()
    
