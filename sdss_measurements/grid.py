"""
structures to dilenaeate a regions and grid of regions
"""

from __future__ import print_function, division
import numpy as np


class Region(object):
    """
    a rectangular region
    """
    
    def __init__(self, left, right, bottom, top):
        """
        Parameters
        ==========
        left : float
            minimum x-coordinate of region
             
        right : float
            maximum x-coordinate of region
            
        bottom : float
            minimum y-coordinate of region
            
        top : float
            maximum y-coordinate of region
        """
        
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
    
    def inside(self, x, y):
        """
        Parameters
        ==========
        x : array-like
            array of x-coordinates
            
        y : array-like
            array of y-coordinates
            
        Returns
        =======
        mask : numpy.array
            array of booleans indicating whether the points are within the region
        """
        
        mask_1 = (x>self.left) & (x<=self.right)
        mask_2 = (y>self.bottom) & (y<=self.top)
        
        mask = mask_1 & mask_2
        
        return mask

class Grid(object):
    """
    grid of rectangular regions
    """
    
    def __init__(self, left, right, bottom, top, Nlr, Nud):
        """
        Parameters
        ==========
        left : float
            minimum x-coordinate of region
             
        right : float
            maximum x-coordinate of region
            
        bottom : float
            minimum y-coordinate of region
            
        top : float
            maximum y-coordinate of region
        
        Nlr : int
            number of region columns
        
        Nud : int
            number of region rows
        """
        
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.Nlr = Nlr
        self.Nud = Nud
        self.N_params = (self.Nud-1) + (self.Nlr-1)*self.Nud
        
        #initialize grid
        init_row_params, init_col_params = self.initial_params()
        self.define_regions(init_row_params, init_col_params)
    
    def initial_params(self):
        """
        calculate intial parameters for grid
        """
        
        row_params = np.zeros(self.Nud-1)
        col_params = np.zeros((self.Nud,self.Nlr-1))
        
        dlr = (self.right - self.left)/self.Nlr
        dud = (self.top - self.bottom)/self.Nud
        
        row_params[:] = self.bottom + dud*np.arange(1,self.Nud)
        for i in range(0,self.Nud):
            col_params[i,:] = self.left + dlr*np.arange(1,self.Nlr)
        
        return row_params, col_params
    
    def define_regions(self, row_params, col_params):
        """
        set regions given parameters
        """
        
        self.regions = []
        
        #bottom left corner
        self.regions.append(Region(self.left, col_params[0,0],
                                   self.bottom, row_params[0]))
        #the rest of the row
        for i in range(1,self.Nlr-1):
            self.regions.append(Region(col_params[0,i-1], col_params[0,i],
                                       self.bottom, row_params[0]))
        #bottom right corner
        self.regions.append(Region(col_params[0,-1], self.right,
                                   self.bottom, row_params[0]))
                                   
        #the rest of the row
        for i in range(1,self.Nud-1): #loop over rows
            bottom = row_params[i-1]
            top = row_params[i]
            #first cell
            left = self.left
            right = col_params[i,0]
            self.regions.append(Region(left, right, bottom, top))
            #in-between cells
            for j in range(1,self.Nlr-1): #loop of columns
                left = col_params[i,j-1]
                right = col_params[i,j]
                self.regions.append(Region(left, right, bottom, top))
            #last cell
            left = col_params[i,-1]
            right = self.right
            self.regions.append(Region(left, right, bottom, top))
        
        #top left corner
        self.regions.append(Region(self.left, col_params[-1,0],
                                   row_params[-1], self.top))
        #the rest of the row
        for i in range(1,self.Nlr-1):
            self.regions.append(Region(col_params[-1,i-1], col_params[-1,i],
                                       row_params[-1], self.top))
        #top right corner
        self.regions.append(Region(col_params[-1,-1], self.right,
                                   row_params[-1], self.top))
        
        self.row_params = row_params
        self.col_params = col_params
     
    def region_id(self, x, y):
        """
        which region is each point within
        """
        result = np.zeros(len(x))
        for i,r in enumerate(self.regions):
            mask = r.inside(x,y)
            result[mask] = i+1
        return result
    
    def number_per_region(self, x, y):
        """
        number of points per region
        """
        result = self.region_id(x, y)
        unq_labels, counts = np.unique(result, return_counts=True)
        
        possible_labels = np.arange(1, self.Nlr*self.Nud+1)
        counts_per_label = np.zeros(len(possible_labels))
        mask = np.in1d(possible_labels,unq_labels)
        
        counts_per_label[mask] = counts
        
        return counts_per_label
    
    def number_per_row(self, x, y):
        """
        number of points pwer row
        """
        counts = self.number_per_region(x, y)
        
        counts = np.reshape(counts,(self.Nud,self.Nlr))
        return np.sum(counts,axis=1)


