# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import networkx as nx

def unit_normal(a, b, c):
    x = np.linalg.det([[1,a[1],a[2]],
         [1,b[1],b[2]],
         [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
         [b[0],1,b[2]],
         [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
         [b[0],b[1],1],
         [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

def poly_area(poly):
    if len(poly) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)


def calculate_local_properties(train_array, dictionary, neighbors, core_atom, face_areas_r):
    
    dict_core=[dictionary[x] for x in core_atom]
    dict_tot=[]
    
    
    for i in range(len(face_areas_r)):
        
        element_list_neigh=train_array[neighbors[i], 1]
        areas=face_areas_r[i]
        
    
        values_n=np.array([dictionary[x] for x in element_list_neigh])
        
        dict_tot.append(np.sum(areas*np.abs(values_n-dict_core[i]))/np.sum(areas))
        
    features=[np.max(dict_tot), np.min(dict_tot), np.mean(dict_tot), np.sum(np.abs(dict_tot-np.mean(dict_tot)))/len(dict_tot)]

    return features

def calculate_ionic_character(train_array, dictionary, neighbors, core_atom, face_areas_r):
    
    dict_core=[dictionary[x] for x in core_atom]
    dict_tot=[]
    
    
    for i in range(len(face_areas_r)):
        
        element_list_neigh=train_array[neighbors[i], 1]
        areas=face_areas_r[i]
        
    
        values_n=np.array([dictionary[x] for x in element_list_neigh])
        
        dict_tot.append(np.sum(areas*np.abs(1-np.exp(-0.25*np.power((values_n-dict_core[i]), 2))))/np.sum(areas))
        
    features=[np.max(dict_tot), np.min(dict_tot), np.mean(dict_tot), np.sum(np.abs(dict_tot-np.mean(dict_tot)))/len(dict_tot)]

    return features


def calculate_path_weights_for_atom(target, cutoff, G, neighbors ,face_areas_r):

    w_tot=0
    
    for l in range(len(face_areas_r)):
              
        paths = nx.all_simple_paths(G, source=l, target=target, cutoff=cutoff)
    
    
        for path in map(nx.utils.pairwise, paths):
        
            dummy=[]
             #paths2.append((list(path)))
            dummy.append((list(path)))
            w=1
            
            if len(dummy[0])==cutoff:
        
                 #add the total weight for each path
                 #multiple each step
        #        print(len(dummy[0]))      
                for i in range(len(dummy[0])):
                    
                    tmp=dummy[0][i]
            
                    areas=face_areas_r[tmp[0]]
                    nn=neighbors[tmp[0]]
                    face_index=np.argwhere(np.array(nn)==tmp[1])
                     
                    if i==0:             
                        denom=np.sum(areas)
                        num=areas[face_index[0][0]]
                        w=w*(num/denom)
                    else:
                        last=dummy[0][i-1][0]
                        last_index=np.argwhere(np.array(nn)==last)
                        denom=np.sum(areas)-areas[last_index[0][0]]
                        num=areas[face_index[0][0]]
                        w=w*num/denom
                w_tot+=w
    return w_tot



def get_xyz_data(filename):
    pos_data = []
    lat_data = []
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                pos_data.append([np.array(x[1:4], dtype=np.float),x[4]])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))
    return pos_data, np.array(lat_data)


