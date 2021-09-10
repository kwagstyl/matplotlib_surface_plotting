import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def normal_vectors(vertices,faces):
    norm = np.zeros( vertices.shape, dtype=vertices.dtype )
    tris = vertices[faces]
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    n=normalize_v3(n)
    return n

def vertex_normals(vertices,faces):
    norm = np.zeros( vertices.shape, dtype=vertices.dtype )
    tris = vertices[faces]
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    n=normalize_v3(n)
    norm[ faces[:,0] ] += n
    norm[ faces[:,1] ] += n
    norm[ faces[:,2] ] += n
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
    return np.array([[1, 0, 0, x], [0, 1, 0, y],
                     [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)

def xrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0,  0, 0], [0, c, -s, 0],
                     [0, s,  c, 0], [0, 0,  0, 1]], dtype=float)

def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c, 0, s, 0], [ 0, 1, 0, 0],
                      [-s, 0, c, 0], [ 0, 0, 0, 1]], dtype=float)

def zrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c, -s, 0, 0], 
                      [ s, c, 0, 0],
                      [0, 0, 1, 0], 
                      [ 0, 0, 0, 1]], dtype=float)

def shading_intensity(vertices,faces, light = np.array([0,0,1]),shading=0.7):
    """shade calculation based on light source
       default is vertical light.
       shading controls amount of shading.
       Also saturates so top 20 % of vertices all have max intensity."""
    face_normals=normal_vectors(vertices,faces)
    intensity = np.dot(face_normals, light)
    intensity[np.isnan(intensity)]=1
    shading = 0.7    
    #top 20% all become fully coloured
    intensity = (1-shading)+shading*(intensity-np.min(intensity))/((np.percentile(intensity,80)-np.min(intensity)))
    #saturate
    intensity[intensity>1]=1
    
    return intensity


def f7(seq):
    #returns uniques but in order to retain neighbour triangle relationship
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))];


def get_ring_of_neighbours(island, neighbours, vertex_indices=None, ordered=False):
    """Calculate ring of neighbouring vertices for an island of cortex
    If ordered, then vertices will be returned in connected order"""
    if not vertex_indices:
        vertex_indices=np.arange(len(island))
    if not ordered:

        neighbours_island = neighbours[island]
        unfiltered_neighbours = []
        for n in neighbours_island:
            unfiltered_neighbours.extend(n)
        unique_neighbours = np.setdiff1d(np.unique(unfiltered_neighbours), vertex_indices[island])
        return unique_neighbours

def get_neighbours_from_tris(tris, label=None):
    """Get surface neighbours from tris
        Input: tris
         Returns Nested list. Each list corresponds 
        to the ordered neighbours for the given vertex"""
    n_vert=np.max(tris+1)
    neighbours=[[] for i in range(n_vert)]
    for tri in tris:
        neighbours[tri[0]].extend([tri[1],tri[2]])
        neighbours[tri[2]].extend([tri[0],tri[1]])
        neighbours[tri[1]].extend([tri[2],tri[0]])
    #Get unique neighbours
    for k in range(len(neighbours)):      
        if label is not None:
            neighbours[k] = set(neighbours[k]).intersection(label)
        else :
            neighbours[k]=f7(neighbours[k])
    return np.array(neighbours,dtype=object)

def mask_colours(colours,triangles,mask):
    """grey out mask"""
    if mask is not None:
        verts_masked = mask[triangles].any(axis=1)
        colours[verts_masked,:] = np.array([0.86,0.86,0.86,1])
    return colours

def adjust_colours_pvals(colours, pvals,triangles,mask=None):
    """red ring around clusters and greying out non-significant vertices"""
    if mask is not None:
        verts_masked = mask[triangles].any(axis=1)
        colours[verts_masked,:] = np.array([0.86,0.86,0.86,1])
    neighbours=get_neighbours_from_tris(triangles)
    ring=get_ring_of_neighbours(pvals<0.05,neighbours)
    if len(ring)>0:
        ring_label = np.zeros(len(neighbours)).astype(bool)
        ring_label[ring]=1
        ring=get_ring_of_neighbours(ring_label,neighbours)
        ring_label[ring]=1
        colours[ring_label[triangles].any(axis=1),:] = np.array([1.0,0,0,1])
    grey_out=pvals<0.05
    verts_grey_out= grey_out[triangles].any(axis=1)
    colours[verts_grey_out,:] = (1.5*colours[verts_grey_out] + np.array([0.86,0.86,0.86,1]))/2.5
    return colours

def add_parcelation_colours(colours,parcel,triangles,labels=None,mask=None):
    """delineate regions"""
    if mask is not None:
        verts_masked = mask[triangles].any(axis=1)
        colours[verts_masked,:] = np.array([0.86,0.86,0.86,1])    
    #normalise rois and colors
    rois=list(set(parcel))
    if 0 in rois:
        rois.remove(0)
    if labels is  None : 
        labels = dict(zip(rois, np.random.rand(len(rois),4)))
    
    #remove transparent rois
    #find vertices that delineate rois
    neighbours=get_neighbours_from_tris(triangles)
    matrix_colored = np.zeros([len(triangles), len(rois)])
    for l,label in enumerate(rois):
        ring=get_ring_of_neighbours(parcel!=label,neighbours)
        if len(ring)>0:
            ring_label = np.zeros(len(neighbours)).astype(bool)
            ring_label[ring]=1
#            ring=get_ring_of_neighbours(ring_label,neighbours)
#            ring_label[ring]=1
            matrix_colored[:,l] = np.median(ring_label[triangles],axis=1) #ring_label[triangles].sum(axis=1)
    #update colours with delineation
    maxis = [max(matrix_colored[i,:]) for i in range(0,len(colours))]
    colours = np.array([labels[rois[np.random.choice(np.where(matrix_colored[i,:] == maxi)[0])]] 
                        if maxi!=0 else colours[i] for i,maxi in enumerate(maxis)])
    return colours


def adjust_colours_alpha(colours,alpha):
    """grey out vertices according to scalar"""
    #rescale alpha to 0.2-1.0
    alpha_rescaled = 0.1+0.9*(alpha-np.min(alpha))/(np.max(alpha)-np.min(alpha))
    colours = (alpha_rescaled*colours.T).T + ((1-alpha_rescaled)*np.array([0.86,0.86,0.86,1]).reshape(-1,1)).T
    colours = np.clip(colours, 0,1)
    return colours

def frontback(T):
    """
    Sort front and back facing triangles
    Parameters:
    -----------
    T : (n,3) array
       Triangles to sort
    Returns:
    --------
    front and back facing triangles as (n1,3) and (n2,3) arrays (n1+n2=n)
    """
    Z = (T[:,1,0]-T[:,0,0])*(T[:,1,1]+T[:,0,1]) + \
        (T[:,2,0]-T[:,1,0])*(T[:,2,1]+T[:,1,1]) + \
        (T[:,0,0]-T[:,2,0])*(T[:,0,1]+T[:,2,1])
    return Z < 0, Z >= 0

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def plot_surf(vertices, faces,overlay, rotate=[90,270], cmap='viridis', filename='plot.png', label=False,
             vmax=None, vmin=None, x_rotate=270, pvals=None, colorbar=True, cmap_label='value',
             title=None, mask=None, base_size=6, arrows=None,arrow_subset=None,arrow_size=0.5,
              arrow_colours = None,arrow_head=0.05,arrow_width=0.001,
            alpha_colour = None,flat_map=False, z_rotate=0,parcel=None, parcel_cmap=None,):
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
        
        parcel       : numpy array, optional
                       delineate rois on top of the surface
        parcel_cmap  : dictionary, optional
                       dic containing labels and colors associated for the parcellation
                         
    """
    vertices=vertices.astype(np.float)
    F=faces.astype(int)
    vertices = (vertices-(vertices.max(0)+vertices.min(0))/2)/max(vertices.max(0)-vertices.min(0))
    if not isinstance(rotate,list):
        rotate=[rotate]
    if not isinstance(overlay,list):
        overlays=[overlay]
    else:
        overlays=overlay
    if flat_map:
        z_rotate=90
        rotate=[90]
        intensity = np.ones(len(F))
    else:
        #change light source if z is rotate
        light = np.array([0,0,1,1]) @ yrotate(z_rotate)
        intensity=shading_intensity(vertices, F, light=light[:3],shading=0.7)
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
            C = adjust_colours_pvals(C,pvals,F,mask)
        elif mask is not None:
            C = mask_colours(C,F,mask)
        if parcel is not None :
            C = add_parcelation_colours(C,parcel,F,parcel_cmap,mask)
            
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
            
        #triangle coordinates
            T =  V[:,:,:2]
        #get Z values for ordering triangle plotting
            Z = -V[:,:,2].mean(axis=1)
        #sort the triangles based on their z coordinate. If front/back views then need to sort a different axis
            front, back = frontback(T)
            T=T[front]
            s_C = C[front]
            Z = Z[front]
            I = np.argsort(Z)
            T, s_C = T[I,:], s_C[I,:]
            ax = fig.add_subplot(len(overlays),len(rotate)+1,2*k+i+1, xlim=[-.98,+.98], ylim=[-.98,+.98],aspect=1, frameon=False,
             xticks=[], yticks=[])
            collection = PolyCollection(T, closed=True, linewidth=0,antialiased=False, facecolor=s_C, cmap=cmap)
            collection.set_alpha(1)
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
    return 

