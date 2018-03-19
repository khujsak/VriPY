# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import scipy.spatial as ss
from numpy.linalg import inv
import tess as ts
from Utils import unit_normal, poly_area, calculate_local_properties,
from Utils import get_xyz_data
from Utils import calculate_ionic_character, calculate_path_weights_for_atom
     

#Read in a list of geometry files in ase format
#stored in a series of index directories
#with a master index in a training file

if __name__ == "__main__":
    
    for zz in range(2100):
        
        structure_index=zz
        train=pd.read_csv('./train.csv')
        
        idx = train.id.values[structure_index]
        fn = "./train/{}/geometry.xyz".format(idx)
        train_xyz, train_lat = get_xyz_data(fn)
        
        d = {'structure': [train.id.values[structure_index]]}
        
        df = pd.DataFrame(data=d)
        
        A=np.transpose(train_lat)
        B = inv(A)
        xyz=[]
        
        #Conver to reduced coordinates
        for i in enumerate(train_xyz):
            r = np.matmul(B, i[1][0])
            xyz.append(r)
        
        #Perform Tesselation    
        cntr=ts.Container(xyz, limits=((0,0,0),(1,1,1)), periodic=True)
        
        
        
        
        #Now we need to convert vertices back to normal coordinates
        #and recalculate face areas
        vertices=np.array([v.vertices() for v in cntr])
        face_vertices=[v.face_vertices() for v in cntr]
        
        face_areas_r=[]
        
        for j in range(len(face_vertices)):    
            areas=[]    
            tmp=np.array(vertices[j])
            
            for i in range(len(face_vertices[j])):
                corrected=[]        
                dummy=tmp[face_vertices[j][i], :]
                
                for k in range(len(dummy)):    
                    corrected.append(np.matmul(A, dummy[k]))
                    
                areas.append(poly_area(corrected))
                
            face_areas_r.append(areas)
            
        #Calculate coordination from face_sharing
    
        
        coord=[]
        
        for v in face_areas_r:
            
            num=np.sum(v)**2
            denom=np.sum(np.power(v, 2))
            coord.append(num/denom)
            
        
        df['max_coordination']=np.max(coord)
        df['min_coordination']=np.min(coord)
        df['mean_coordination']=np.mean(coord)
        df['mad_coordination']=np.sum(np.abs(coord-np.mean(coord)))/len(coord)
        
        features_coord=[np.max(coord), np.min(coord), np.mean(coord), np.sum(np.abs(coord-np.mean(coord)))/len(coord)]
        
        
        electronegativities={"Al":1.61, "In":1.78, "Ga": 1.81, "O":3.44}
        electron_affinities = {"Al" : 0.432, "In": 0.3, "Ga": 0.43,  "O": 1.461 }
        covalent_radius = {"Al" : 125, "In": 155, "Ga": 130,  "O": 60 }
        valence = {"Al" : 3, "In": 3, "Ga": 3,  "O": 2 }
        melting_T = {"Al" :660.32, "In": 156.6, "Ga": 29.76,  "O": -218.3 }
        mendeleev_num={"Al" :73, "In": 75, "Ga": 74,  "O": 87 }
        atomic_weight={"Al" :26.98, "In": 114.8, "Ga": 69.72,  "O": 15.99 }
        effective_local_charge={"Al" :4.066, "In": 8.470, "Ga": 6.222,  "O": 4.453 }
        heat_capacity={'Al': 24.2, 'Ga': 25.86, 'In': 26.74, 'O': 29.378}   
        heat_of_fusion={'Al': 10700.0, 'Ga': 5590.0, 'In': 3260.0, 'O': 222.0}
        heat_of_vaporization={'Al': 294, 'Ga': 254, 'In': 231.8, 'O':  6.820}  
        first_ionization = {'Al': 577.5, 'Ga': 578.8, 'In': 558.3, 'O': 1313.9}
        second_ionization = {'Al': 1816.7, 'Ga': 1979.3, 'In': 1820.7, 'O': 3388.3}    
        third_ionization = {'Al': 2744.8, 'Ga': 2963, 'In': 2704, 'O': 5300.5}      
        thermal_conductivity = {'Al': 235.0, 'Ga': 29.0, 'In': 82.0, 'O': 0.02658}
        molar_volume={'Al': 10.00, 'Ga': 11.803, 'In': 15.76, 'O': 22.4134}
        chemical_hardness={'Al': 2.77, 'Ga': 2.9, 'In': 2.8, 'O': 6.08}  
        polarizability = {'Al': 57.74, 'Ga': 51.4, 'In': 68.7, 'O': 6.1}  
            
        local_properties=[electronegativities,
                          electron_affinities,
                          covalent_radius,
                          valence,
                          melting_T,
                          mendeleev_num,
                          atomic_weight,
                          effective_local_charge,
                          heat_capacity,
                          heat_of_fusion,
                          heat_of_vaporization,
                          first_inoization,
                          second_ionization,
                          third_ionization,
                          thermal_conductivity,
                          molar_volume,
                          chemical_hardness,
                          polarizability]
        
        
        
        
        neighbors=[v.neighbors() for v in cntr]
        train_array=np.array(train_xyz)
        core_atom=train_array[:,1]
        
        
        
        for dictionary in local_properties:
            
            property_name=[ k for k,v in locals().items() if v is dictionary][0]    
            features=calculate_local_properties(train_array, dictionary, neighbors, core_atom, face_areas_r)
            
            df[property_name +'_max']=features[0]
            df[property_name +'_min']=features[1]    
            df[property_name +'_mean']=features[2]      
            df[property_name +'_mad']=features[3]  
            
        
        
        features=calculate_ionic_character(train_array, electronegativities, neighbors, core_atom, face_areas_r)
        
        df['ionic_character_max']=features[0]
        df['ionic_character_min']=features[1]    
        df['ionich_character_mean']=features[2]      
        df['ionic_character_mad']=features[3]  
            
        
        #Now we'll calculate structural homogeneity features
        #Calculate distance between core and neighbor
        #Weight by area of that face
        #Calculate mean, min, max, mad
            
        bond_dist=[]
        
        for i,atom in enumerate(train_xyz):
            
            neighbor_atoms=train_array[neighbors[i], 0]
            dist=[]
        
            for k in range(len(neighbor_atoms)):
                dist.append( np.linalg.norm(atom[1][0]-neighbor_atoms[k]))
                
            bond_dist.append(np.dot(dist, face_areas_r[i])/np.sum(face_areas_r[i]))   
            bond_dist_var.append(np.abs(np.dot(dist, face_areas_r[i])-bond_dist[i])/(np.sum(face_areas_r[i]))*bond_dist[i])
            
        features=[np.max(bond_dist), np.min(bond_dist), np.mean(bond_dist), np.sum(np.abs(bond_dist-np.mean(bond_dist)))/len(bond_dist)]   
    
        
                
        df['mean_bond_length']=np.mean(bond_dist)
        df['max_bond_length']=np.max(bond_dist)
        df['min_bond_length']=np.min(bond_dist)
        df['mad_bond_length']=np.sum(np.abs(bond_dist-np.mean(bond_dist)))/len(bond_dist)
        
                
        df['mean_var_bond_length']=np.mean(bond_dist_var)
        df['max_var_bond_length']=np.max(bond_dist_var)
        df['min_var_bond_length']=np.min(bond_dist_var)
        df['mad_var_bond_length']=np.sum(np.abs(bond_dist_var-np.mean(bond_dist_var)))/len(bond_dist_var)
        
        
        #The mean absolute deviation of the volume of the Voronoi cell about each atom 
        #over the mean cell volume
        
        cell_vols_r=[]
        
        for j in range(len(vertices)):
                 
            corrected=[]     
            
            for k in range(len(vertices[j])):
                   
                corrected.append(np.matmul(A, vertices[j][k]))
                
            hull=ss.ConvexHull(corrected)     
            cell_vols_r.append(hull.volume)
    
                
        df['mad_cell_volume']=np.sum(np.abs(cell_vols_r-np.mean(cell_vols_r)))/(len(cell_vols_r)*np.mean(cell_vols_r))          
        
        
        #Maximum Packing Efficiency
        #Compute Center of each face
        #Then calculate distance from center atom to that center
        #Take minimum as packing efficiency
        
        face_centers_r=[]
            
        for j in range(len(face_vertices)):
             
            centers=[]
            tmp=np.array(vertices[j])
            
            for i in range(len(face_vertices[j])):
                corrected=[]        
                dummy=tmp[face_vertices[j][i], :]
                
                for k in range(len(dummy)):
                    
                    corrected.append(np.matmul(A, dummy[k]))
                
                centers.append(np.mean(corrected, axis=0))
                
            face_centers_r.append(centers)
        
        maximum_packing=0
        
        for i,atom in enumerate(train_xyz):
            
            neighbor_atoms=face_centers_r[i]
            dist=[]
        
            for k in range(len(neighbor_atoms)):
                
                dist.append(np.linalg.norm(atom[1][0]-neighbor_atoms[k]))
          
            maximum_packing+=4/3*(np.pi*np.power(np.min(dist), 3))
        
        df['maximum_packing_efficiency']=maximum_packing/np.sum(cell_vols_r)
        
        
        #now we'll calculate first shell ordering parameters
        #Go through all the core atoms
        #For each face shared with a particular element
        #Calculate the parameter and sum for the order
        #Parameter for that element
        
        #Get Atomic Fractions
        unique_elements, atomic_fraction = np.unique(train_array[:,1], return_counts= True)
        
        atomic_fraction=atomic_fraction/len(train_array)
        oxygen_fraction_index=np.argwhere(unique_elements=='O')
        aluminum_fraction_index=np.argwhere(unique_elements=='Al')
        gallium_fraction_index=np.argwhere(unique_elements=='Ga')
        indium_fraction_index=np.argwhere(unique_elements=='In')
        
        first_order_param_O=[]
        first_order_param_In=[]
        first_order_param_Ga=[]
        first_order_param_Al=[]
        
        
        
        for i in range(len(train_array[:,1])):
            
            #For each cell
            element=train_array[:,1][i]
            element_list_neigh=train_array[neighbors[i], 1]
            areas=face_areas_r[i]
            
            numerator_area_indices=np.argwhere(element_list_neigh==element)
            
            if element=='O':       
                first_order_param_O.append(1-(np.sum([areas[int(v)] for v in numerator_area_indices]))/(np.sum(areas)*(atomic_fraction[oxygen_fraction_index])))
            
            if element=='Ga':
                first_order_param_Ga.append(1-(np.sum([areas[int(v)] for v in numerator_area_indices]))/(np.sum(areas)*(atomic_fraction[gallium_fraction_index])))
    
            if element=='In':
                first_order_param_In.append(1-(np.sum([areas[int(v)] for v in numerator_area_indices]))/(np.sum(areas)*(atomic_fraction[indium_fraction_index])))
                  
            if element=='Al':
                first_order_param_Al.append(1-(np.sum([areas[int(v)] for v in numerator_area_indices]))/(np.sum(areas)*(atomic_fraction[aluminum_fraction_index])))
    
            
            
        if bool(first_order_param_O)==True:
            first_order_param_O=np.mean(np.abs(first_order_param_O))
            df['first_ordering_param_O']=first_order_param_O
            
        if bool(first_order_param_Ga)==True:
            first_order_param_Ga=np.mean(np.abs(first_order_param_Ga)) 
            df['first_ordering_param_Ga']=first_order_param_Ga
                
        if bool(first_order_param_In)==True:
            first_order_param_In=np.mean(np.abs(first_order_param_In)) 
            df['first_ordering_param_In']=first_order_param_In 
                 
        if bool(first_order_param_Al)==True:
            first_order_param_Al=np.mean(np.abs(first_order_param_Al))  
            df['first_ordering_param_Al']=first_order_param_Al
        
    
        
        #Second and Third Order Parameters
            
        #Construct Graph
        
        G=nx.Graph()
        
        
        G.add_nodes_from(range(len(train_xyz)))
        
        for i in range(len(neighbors)):
            l=[]
            l=neighbors[i]
            l=[(i,x) for x in l]
        
            G.add_edges_from(l)
        
        cutoff=2
        w_tot=np.zeros([len(face_areas_r)])
        
        for i in range(len(face_areas_r)):    
            w_tot[i]=calculate_path_weights_for_atom(i, cutoff, G, neighbors, face_areas_r)
    
        
        indicator_O=np.argwhere(train_array[:,1]=='O')
        indicator_In=np.argwhere(train_array[:,1]=='In')
        indicator_Ga=np.argwhere(train_array[:,1]=='Ga')
        indicator_Al=np.argwhere(train_array[:,1]=='Al')
         
        if bool(first_order_param_O)==True:
            second_order_param_O=np.mean(np.abs(1-(w_tot[indicator_O]/atomic_fraction[oxygen_fraction_index])))    
            df['second_ordering_param_O']=second_order_param_O
            
        if bool(first_order_param_Ga)==True:
            second_order_param_Ga=np.mean(np.abs(1-(w_tot[indicator_Ga]/atomic_fraction[gallium_fraction_index])))
            df['second_ordering_param_Ga']=second_order_param_Ga
                
        if bool(first_order_param_In)==True:  
            second_order_param_In=np.mean(np.abs(1-(w_tot[indicator_In]/atomic_fraction[indium_fraction_index])))
            df['second_ordering_param_In']=second_order_param_In 
            
            
        if bool(first_order_param_Al)==True:
            second_order_param_Al=np.mean(np.abs(1-(w_tot[indicator_Al]/atomic_fraction[aluminum_fraction_index])))        
            df['second_ordering_param_Al']=second_order_param_Al 
        
        
        #Third Order Param
        
        cutoff=3
        w_tot=np.zeros([len(face_areas_r)])
        
        for i in range(len(face_areas_r)):    
            w_tot[i]=calculate_path_weights_for_atom(i, cutoff, G, neighbors, face_areas_r)
        
        if bool(first_order_param_O)==True:
            third_order_param_O=np.mean(np.abs(1-(w_tot[indicator_O]/atomic_fraction[oxygen_fraction_index])))    
            df['third_ordering_param_O']=third_order_param_O
            
        if bool(first_order_param_Ga)==True:
            third_order_param_Ga=np.mean(np.abs(1-(w_tot[indicator_Ga]/atomic_fraction[gallium_fraction_index])))
            df['third_ordering_param_Ga']=third_order_param_Ga
                
        if bool(first_order_param_In)==True:  
            third_order_param_In=np.mean(np.abs(1-(w_tot[indicator_In]/atomic_fraction[indium_fraction_index])))
            df['third_ordering_param_In']=third_order_param_In 
            
            
        if bool(first_order_param_Al)==True:
            third_order_param_Al=np.mean(np.abs(1-(w_tot[indicator_Al]/atomic_fraction[aluminum_fraction_index])))        
            df['third_ordering_param_Al']=third_order_param_Al 
        
        print(zz)
        
        if zz != 0:
            
            result = pd.concat([result, df])
        else:
            result=df
        