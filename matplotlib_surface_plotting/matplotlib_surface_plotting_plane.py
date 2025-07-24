import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm

# (Helper functions unchanged)

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2)
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr


def normal_vectors(vertices,faces):
    tris = vertices[faces]
    n = np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0])
    return normalize_v3(n)


def vertex_normals(vertices,faces):
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    tris = vertices[faces]
    n = np.cross(tris[:,1] - tris[:,0], tris[:,2] - tris[:,0])
    n = normalize_v3(n)
    norm[faces[:,0]] += n
    norm[faces[:,1]] += n
    norm[faces[:,2]] += n
    return normalize_v3(norm)


def frustum(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M


def perspective(fovy, aspect, znear, zfar):
    h = np.tan(0.5*np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)


def translate(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)


def xrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0,  0, 0], [0, c, -s, 0], [0, s,  c, 0], [0, 0,  0, 1]], dtype=float)


def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([[ c, 0, s, 0], [ 0, 1, 0, 0], [-s, 0, c, 0], [ 0, 0, 0, 1]], dtype=float)


def zrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([[ c, -s, 0, 0], [ s, c, 0, 0], [0, 0, 1, 0], [ 0, 0, 0, 1]], dtype=float)


def shading_intensity(vertices,faces, light = np.array([0,0,1]),shading=0.7):
    face_normals = normal_vectors(vertices,faces)
    intensity = np.dot(face_normals, light)
    intensity[np.isnan(intensity)] = 1
    # top 20% all become fully coloured
    intensity = (1-shading) + shading*(intensity - np.min(intensity))/(
                 (np.percentile(intensity,80) - np.min(intensity)))
    intensity[intensity>1] = 1
    return intensity


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def get_ring_of_neighbours(island, neighbours, vertex_indices=None, ordered=False):
    if not vertex_indices:
        vertex_indices = np.arange(len(island))
    if not ordered:
        neighbours_island = neighbours[island]
        unfiltered_neighbours = []
        for n in neighbours_island:
            unfiltered_neighbours.extend(n)
        unique_neighbours = np.setdiff1d(np.unique(unfiltered_neighbours), vertex_indices[island])
        return unique_neighbours


def get_neighbours_from_tris(tris, label=None):
    n_vert = np.max(tris+1)
    neighbours = [[] for _ in range(n_vert)]
    for tri in tris:
        neighbours[tri[0]].extend([tri[1],tri[2]])
        neighbours[tri[1]].extend([tri[2],tri[0]])
        neighbours[tri[2]].extend([tri[0],tri[1]])
    for k in range(len(neighbours)):
        if label is not None:
            neighbours[k] = set(neighbours[k]).intersection(label)
        else:
            neighbours[k] = f7(neighbours[k])
    return np.array(neighbours, dtype=object)


def mask_colours(colours, triangles, mask, mask_colour=None):
    if mask is not None:
        if mask_colour is None:
            mask_colour = np.array([0.86,0.86,0.86,1])
        verts_masked = mask[triangles].any(axis=1)
        colours[verts_masked,:] = mask_colour
    return colours


def adjust_colours_pvals(colours, pvals, triangles, mask=None, mask_colour=None,
                          border_colour = np.array([1.0,0,0,1])):
    colours = mask_colours(colours,triangles,mask,mask_colour)
    neighbours = get_neighbours_from_tris(triangles)
    ring = get_ring_of_neighbours(pvals<0.05, neighbours)
    if len(ring)>0:
        ring_label = np.zeros(len(neighbours)).astype(bool)
        ring_label[ring] = 1
        ring = get_ring_of_neighbours(ring_label, neighbours)
        ring_label[ring] = 1
        colours[ring_label[triangles].any(axis=1),:] = border_colour
    grey_out = pvals<0.05
    verts_grey_out = grey_out[triangles].any(axis=1)
    colours[verts_grey_out,:] = (1.5*colours[verts_grey_out] + np.array([0.86,0.86,0.86,1]))/2.5
    return colours


def add_parcellation_colours(colours, parcel, triangles, labels=None,
                              mask=None, filled=False, mask_colour=None, neighbours=None):
    colours = mask_colours(colours, triangles, mask, mask_colour=mask_colour)
    rois = list(set(parcel))
    if 0 in rois:
        rois.remove(0)
    if labels is None:
        labels = dict(zip(rois, np.random.rand(len(rois),4)))
    if filled:
        colours = np.zeros_like(colours)
        for label in rois:
            colours[np.median(parcel[triangles], axis=1) == label] = labels[label]
        return colours
    if neighbours is None:
        neighbours = get_neighbours_from_tris(triangles)
    matrix_colored = np.zeros([len(triangles), len(rois)])
    for l, label in enumerate(rois):
        ring = get_ring_of_neighbours(parcel != label, neighbours)
        if len(ring)>0:
            ring_label = np.zeros(len(neighbours)).astype(bool)
            ring_label[ring]=1
            matrix_colored[:,l] = np.median(ring_label[triangles],axis=1)
    maxis = [max(matrix_colored[i,:]) for i in range(len(colours))]
    colours = np.array([
        labels[rois[np.random.choice(np.where(matrix_colored[i,:] == maxi)[0])]]
        if maxi != 0 else colours[i] for i, maxi in enumerate(maxis)
    ])
    return colours


def adjust_colours_alpha(colours, alpha):
    alpha_rescaled = 0.1 + 0.9*(alpha - np.min(alpha))/(np.max(alpha) - np.min(alpha))
    colours = (alpha_rescaled*colours.T).T + ((1-alpha_rescaled)*np.array([0.86,0.86,0.86,1]).reshape(-1,1)).T
    return np.clip(colours, 0,1)


def frontback(T):
    Z = (T[:,1,0] - T[:,0,0])*(T[:,1,1] + T[:,0,1]) + \
        (T[:,2,0] - T[:,1,0])*(T[:,2,1] + T[:,1,1]) + \
        (T[:,0,0] - T[:,2,0])*(T[:,0,1] + T[:,2,1])
    return Z < 0, Z >= 0


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def ras2coords(mri_img, ras=(0, 0, 0)):
    return tuple(mri_img.affine[:3, :3].dot(ras) + mri_img.affine[:3, 3])


def compute_plane_from_mri(mri_img, slice_i, slice_axis):
    shape = mri_img.shape
    if slice_axis == 0:
        coord = slice_i
        v0 = ras2coords(mri_img, (coord, 0, 0))
        v1 = ras2coords(mri_img, (coord, shape[1], 0))
        v2 = ras2coords(mri_img, (coord, 0, shape[2]))
        v3 = ras2coords(mri_img, (coord, shape[1], shape[2]))
    elif slice_axis == 1:
        coord = slice_i
        v0 = ras2coords(mri_img, (0, coord, 0))
        v1 = ras2coords(mri_img, (shape[0], coord, 0))
        v2 = ras2coords(mri_img, (0, coord, shape[2]))
        v3 = ras2coords(mri_img, (shape[0], coord, shape[2]))
    else:
        coord = slice_i
        v0 = ras2coords(mri_img, (0, 0, coord))
        v1 = ras2coords(mri_img, (shape[0], 0, coord))
        v2 = ras2coords(mri_img, (0, shape[1], coord))
        v3 = ras2coords(mri_img, (shape[0], shape[1], coord))
    return np.array([v0, v1, v2, v3]), np.array([[0,1,2],[1,3,2]])


def get_plane_intersections(mesh_vertices, mesh_faces, plane_coords, plane_faces):
    """
    Categorize mesh faces as behind, intersecting, or in front of a plane defined by its vertices and faces.

    Parameters:
        mesh_vertices (np.ndarray): Array of shape (N, 3).
        mesh_faces (np.ndarray): Array of shape (M, 3), indices into mesh_vertices.
        plane_coords (np.ndarray): Array of shape (P, 3) with plane vertices.
        plane_faces (np.ndarray): Array of shape (Q, 3) with indices into plane_coords defining plane triangles.

    Returns:
        behind_faces (List[int]): Indices of faces completely behind the plane.
        intersected_faces (List[int]): Indices of faces intersecting the plane.
        in_front_faces (List[int]): Indices of faces completely in front of the plane.
    """
    # Compute plane normal using the first triangle of the plane
    idx0, idx1, idx2 = plane_faces[0]
    v0 = plane_coords[idx0]
    v1 = plane_coords[idx1]
    v2 = plane_coords[idx2]
    # Edge vectors
    e1 = v1 - v0
    e2 = v2 - v0
    plane_normal = np.cross(e1, e2)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    plane_point = v0  # any point on the plane

    behind_faces = np.zeros(len(mesh_faces), dtype=bool)
    intersected_faces = np.zeros(len(mesh_faces), dtype=bool)
    in_front_faces = np.zeros(len(mesh_faces), dtype=bool)

    for i, face in enumerate(mesh_faces):
        verts = mesh_vertices[face]
        
        signed_dists = np.dot(verts - plane_point, plane_normal)

        if np.all(signed_dists < 0):
            behind_faces[i] = True
        elif np.all(signed_dists > 0):
            in_front_faces[i] = True
        else:
            intersected_faces[i] = True

    return behind_faces, intersected_faces, in_front_faces

def bounds_from_mesh(axis, coord, mesh_vertices):
    min_vals = mesh_vertices.min(axis=0)
    max_vals = mesh_vertices.max(axis=0)
    #border is 10% of the mesh size
    border = 0.1 * (max_vals - min_vals)
    min_vals -= border
    max_vals += border
    if axis == 0:
        return np.array([
            [coord, min_vals[1], min_vals[2]],
            [coord, max_vals[1], min_vals[2]],
            [coord, min_vals[1], max_vals[2]],
            [coord, max_vals[1], max_vals[2]]
        ]), np.array([[0,1,2],[1,3,2]])
    elif axis == 1:
        return np.array([
            [min_vals[0], coord, min_vals[2]],
            [max_vals[0], coord, min_vals[2]],
            [min_vals[0], coord, max_vals[2]],
            [max_vals[0], coord, max_vals[2]]
        ]),np.array([[0,1,2],[1,3,2]])
    else:  # axis == 'z'
        return np.array([
            [min_vals[0], min_vals[1], coord],
            [max_vals[0], min_vals[1], coord],
            [min_vals[0], max_vals[1], coord],
            [max_vals[0], max_vals[1], coord]
        ]),np.array([[0,1,2],[1,3,2]])

def plot_surf(vertices, faces,overlay, rotate=[90,270], cmap='viridis',
             filename='plot.png', label=False,
             vmax=None, vmin=None, x_rotate=270, pvals=None, colorbar=True, cmap_label='value',
             title=None, mask=None, base_size=6, arrows=None,arrow_subset=None,arrow_size=0.5,
             arrow_colours = None,arrow_head=0.05,arrow_width=0.001,
             mask_colour=None,transparency=1,show_back=False,border_colour = np.array([1,0,0,1]),
             alpha_colour = None,flat_map=False, z_rotate=0,neighbours=None,
             parcel=None, parcel_cmap=None,filled_parcels=False,return_ax=False,
             plane=None):
    """ This function plot mesh surface with a given overlay. 
        Features available : display in flat surface, display parcellation on top, display gradients arrows on top

    
    Parameters:
    ----------
        vertices     : numpy array  
                       vertex locations
        faces        : numpy array
                       triangles of vertex indices definings faces
        overlay      : numpy array
                       array to be plotted
        rotate       : tuple, optional
                       rotation angle for lateral on lh,  and medial 
        cmap         : string, optional
                       matplotlib colormap
        filename     : string, optional
                       name of the figure to save
        label        : bool, optional
                       colours smoothed (mean) or median if label         
        vmin, vmax   : float, optional
                       min and max value for display intensity
        x_rotate     : int, optional
        
        pvals        : bool, optional
        
        colorbar     : bool, optional
                       display or not colorbar
        cmap_label   : string, optional
                       label of the colorbar 
        title        : string, optional
                       title of the figure
        mask         : numpy array, optional
                       vector to mask part of the surface
        base_size    : int, optional
        
        arrows       : numpy array, optional
                       dipsplay arrows in the directions of gradients on top of the surface
        arrow_subset : numpy array, optional
                       vector containing at which vertices display an arrow
        arrow_size   : float, optional
                       size of the arrow
        arrow_colours: 
        alpha_colour : float, optional
                       value to play with transparency of the overlay
        flat_map     : bool, optional
                       display on flat map 
        z_rotate     : int, optional
        transparency : float, optional
                       value between 0-1 to play with mesh transparency
        show_back    : bool, optional
                       display or hide the faces in the back of the mesh (z<0) 
        parcel       : numpy array, optional
                       delineate rois on top of the surface
        parcel_cmap  : dictionary, optional
                       dic containing labels and colors associated for the parcellation
        filled_parcels: fill the parcel colours
        neighbours    : provided neighbours in case faces is only a subset of all vertices
        plane        : dictionary, optional 
                      {'mri_img': img, 'slice_i': 30, 'slice_axis': 1,
                      'plane_colour': np.array([1,1,1,1]),
                      'plane_alpha': 0.5}
            
    """
    default_plane = {'mri_img': None, 'slice_i': 100, 'slice_axis': 1,
                 'plane_colour':np.array([.5,.5,.5,1]),
                 'plane_alpha':0.8,
                 'intersected_colour':np.array([0,0,0,1])}
    if plane is not None:
        if not isinstance(plane,dict):
            raise ValueError('Plane should be a dictionary with keys: mri_img, slice_i, slice_axis, plane_colour, plane_alpha')
        for key in default_plane.keys():
            if key not in plane:
                plane[key] = default_plane[key]
    vertices=vertices.astype(np.float32)
    F=faces.astype(int)
    
    if not isinstance(rotate,list):
        rotate=[rotate]
    if not isinstance(overlay,list):
        overlays=[overlay]
    else:
        overlays=overlay
    if parcel is not None:
        if parcel.sum() == 0:
            parcel = None
    if flat_map:
        z_rotate=90
        rotate=[90]
        intensity = np.ones(len(F))
        if plane is not None:
            print('Plane is not supported for flat maps, ignoring plane.')
    else:
        #change light source if z is rotate
        light = np.array([0,0,1,1]) @ yrotate(z_rotate)
        intensity=shading_intensity(vertices, F, light=light[:3],shading=0.7)
        if plane is not None:
            if plane['mri_img'] is None:
                plane_coords, plane_faces = bounds_from_mesh(plane['slice_axis'], plane['slice_i'], vertices)
            else:
                plane_coords, plane_faces = compute_plane_from_mri(plane['mri_img'], plane['slice_i'], plane['slice_axis'])
            behind_faces, intersected_faces, in_front_faces = get_plane_intersections(vertices, F, plane_coords, plane_faces)
   
            plane_intensity = shading_intensity(plane_coords, plane_faces, light=light[:3],shading=0.7)
            plane_colours = np.ones((len(plane_faces), 4))
            plane_colours[:,:3] = plane['plane_colour'][:3]
            #intensity
            #plane_colours[:,0] *= plane_intensity
            #plane_colours[:,1] *= plane_intensity
            #plane_colours[:,2] *= plane_intensity
            #alpha
            plane_colours[:,3] = plane['plane_alpha']
    
    if plane is not None:
        vertices = (vertices-(plane_coords.max(0)+plane_coords.min(0))/2)/max(plane_coords.max(0)-plane_coords.min(0))
        plane_coords = (plane_coords-(plane_coords.max(0)+plane_coords.min(0))/2)/max(plane_coords.max(0)-plane_coords.min(0))
    else:
        vertices = (vertices-(vertices.max(0)+vertices.min(0))/2)/max(vertices.max(0)-vertices.min(0))
     #make figure dependent on rotations
    
    fig = plt.figure(figsize=(base_size*len(rotate)+colorbar*(base_size-2),
                              (base_size-1)*len(overlays)))
    if title is not None:
        plt.title(title, fontsize=25)
    plt.axis('off')
    for k,overlay in enumerate(overlays):
        #colours smoothed (mean) or median if label
        if label:
            colours = np.median(overlay[F],axis=1)
        else:
            colours = np.mean(overlay[F],axis=1)
        if vmax is not None:
            colours = (colours - vmin)/(vmax-vmin)
            colours = np.clip(colours,0,1)
        else: 
            vmax = colours.max()
            vmin = colours.min()
            colours = (colours - colours.min())/(colours.max()-colours.min())
        C = plt.get_cmap(cmap)(colours)
        if alpha_colour is not None:
            C = adjust_colours_alpha(C,np.mean(alpha_colour[F],axis=1))
        if pvals is not None:
            C = adjust_colours_pvals(C,pvals,F,mask,mask_colour=mask_colour, 
                                     border_colour=border_colour)
        elif mask is not None:
            C = mask_colours(C,F,mask,mask_colour=mask_colour)
        if parcel is not None :
            C = add_parcellation_colours(C,parcel,F,parcel_cmap,
                     mask,mask_colour=mask_colour,
                  filled=filled_parcels, neighbours=neighbours)
        if plane is not None:
            C[intersected_faces,:] = plane['intersected_colour']
            
        #adjust intensity based on light source here
        C[:,0] *= intensity
        C[:,1] *= intensity
        C[:,2] *= intensity
        
        collection = PolyCollection([], closed=True, linewidth=0,antialiased=False, facecolor=C, cmap=cmap)
            
        for i,view in enumerate(rotate):
            MVP = perspective(25,1,1,100)  @ translate(0,0,-3) @ yrotate(view) @ zrotate(z_rotate)  @ xrotate(x_rotate) @ zrotate(270*flat_map)
            #translate coordinates based on viewing position
            V = np.c_[vertices, np.ones(len(vertices))]  @ MVP.T
            V /= V[:,3].reshape(-1,1)
            if plane is not None:
                #transform plane coordinates
                P = np.c_[plane_coords, np.ones(len(plane_coords))] @ MVP.T
                P /= P[:,3].reshape(-1,1)
                

            center = np.array([0, 0, 0, 1]) @ MVP.T;
            center /= center[3];
            # add vertex positions to A_dir before transforming them
            if arrows is not None: 
                #calculate arrow position + small shift in surface normal direction
                vertex_normal_orig = vertex_normals(vertices,faces)
                A_base = np.c_[vertices+vertex_normal_orig*0.01, np.ones(len(vertices))]  @ MVP.T
                A_base /= A_base[:,3].reshape(-1,1)
                
                #calculate arrow direction
                A_dir = np.copy(arrows) 
                #normalise arrow size
                max_arrow = np.max(np.linalg.norm(arrows,axis=1))
                A_dir = arrow_size*A_dir/max_arrow
                A_dir = np.c_[A_dir, np.ones(len(A_dir))] @ MVP.T
                A_dir /= A_dir[:,3].reshape(-1,1)
               # A_dir *= 0.1;

            V = V[F]
            if plane is not None:
                #transform plane coordinates
                plane_faces=np.array([[0,1,3,2]])
                P = P[plane_faces]
                PT = P[:,:,:2]
               
        #triangle coordinates
            T =  V[:,:,:2]
        #get Z values for ordering triangle plotting
            Z = -V[:,:,2].mean(axis=1)
        #sort the triangles based on their z coordinate. If front/back views then need to sort a different axis
            front, back = frontback(T)
            if show_back == False:
                T=T[front]
                s_C = C[front]
                Z = Z[front]
                if plane is not None:
                    behind_faces_f = behind_faces[front]
                    intersected_faces_f = intersected_faces[front]
                    in_front_faces_f = in_front_faces[front]
            else:
                s_C = C
            I = np.argsort(Z)
            #sort triangles and colours
            if plane is not None:
                if sum(intersected_faces_f) > 0:
                    #we may need some logic to figure out behind in front depending on view.
                    #reorder I. Behind first, then intersected, then plane then in front, but preserve order internally
                    if np.median(Z[behind_faces_f]) > np.median(Z[in_front_faces_f]):
                        #if behind faces are in front of in front faces, then we need to swap them
                        behind_faces_f, intersected_faces_f, in_front_faces_f = \
                            in_front_faces_f, intersected_faces_f, behind_faces_f
                    behind_I = np.argsort(Z[behind_faces_f])
                    intersected_I = np.argsort(Z[intersected_faces_f])
                    in_front_I = np.argsort(Z[in_front_faces_f])
                    ax = fig.add_subplot(len(overlays),len(rotate)+1,2*k+i+1, xlim=[-.98,+.98], ylim=[-.98,+.98],aspect=1, frameon=False,
                xticks=[], yticks=[])
                    collection = PolyCollection(T[behind_faces_f][behind_I,:], closed=True, linewidth=0,
                                                antialiased=False, 
                                                facecolor=s_C[behind_faces_f][behind_I,:], cmap=cmap)
                    collection.set_alpha(transparency)
                    ax.add_collection(collection)
                    
                    collection = PolyCollection(T[intersected_faces_f][intersected_I,:], closed=True, linewidth=0,
                                                antialiased=False, 
                                                facecolor=s_C[intersected_faces_f][intersected_I,:], cmap=cmap)
                    collection.set_alpha(transparency)
                    ax.add_collection(collection)
                    
                    collection = PolyCollection(PT, closed=True, linewidth=0,
                                                antialiased=True, 
                                                facecolor=plane_colours, cmap=cmap)
                    collection.set_alpha(plane_colours[:,3].mean())
                    ax.add_collection(collection)
                    collection = PolyCollection(T[in_front_faces_f][in_front_I,:], closed=True, linewidth=0,
                                                antialiased=False, 
                                                facecolor=s_C[in_front_faces_f][in_front_I,:], cmap=cmap)
                    collection.set_alpha(transparency)
                    ax.add_collection(collection)

            else:        
                T, s_C = T[I,:], s_C[I,:]
                ax = fig.add_subplot(len(overlays),len(rotate)+1,2*k+i+1, xlim=[-.98,+.98], ylim=[-.98,+.98],aspect=1, frameon=False,
                xticks=[], yticks=[])
                collection = PolyCollection(T, closed=True, linewidth=0,antialiased=False, facecolor=s_C, cmap=cmap)
                collection.set_alpha(transparency)
                ax.add_collection(collection)
            #add arrows to image
            if arrows is not None:
                front_arrows = F[front].ravel()
                for arrow_index,i in enumerate(arrow_subset):
                    if i in front_arrows and A_base[i,2] < center[2] + 0.01:
                        arrow_colour = 'k'
                        if arrow_colours is not None:
                            arrow_colour = arrow_colours[arrow_index]
                        #if length of arrows corresponds perfectly with coordinates
                        # assume 1:1 matching
                        if len(A_dir) == len(A_base):
                            direction = A_dir[i]
                        #otherwise, assume it is a custom list matching the 
                        elif len(A_dir) == len(arrow_subset):
                            direction  = A_dir[arrow_index]
                        half = direction * 0.5
                        
                        ax.arrow(A_base[i,0] - half[0],
                                 A_base[i,1] - half[1], 
                                 direction[0], direction[1], 
                                 head_width=arrow_head,width =arrow_width,
                                color = arrow_colour)
                    # ax.arrow(A_base[idx,0], A_base[idx,1], A_dir[i,0], A_dir[i,1], head_width=0.01)
            plt.subplots_adjust(left =0 , right =1, top=1, bottom=0,wspace=0, hspace=0)
        
    if colorbar:
        l=0.7
        if len(rotate)==1:
            l=0.5
        cbar_size= [l, 0.3, 0.03, 0.38]
        cbar = fig.colorbar(collection,
                            ticks=[0,0.5, 1],
                            cax = fig.add_axes(cbar_size),
                           )
        cbar.ax.set_yticklabels([np.round(vmin,decimals=2), np.round(np.mean([vmin,vmax]),decimals=2),
                         np.round(vmax,decimals=2)])
        cbar.ax.tick_params(labelsize=25)
        cbar.ax.set_title(cmap_label, fontsize=25, pad = 30)
    if filename is not None:
        fig.savefig(filename,bbox_inches = 'tight',pad_inches=0,transparent=True)
    if return_ax:
        return fig,ax,MVP
    return 


