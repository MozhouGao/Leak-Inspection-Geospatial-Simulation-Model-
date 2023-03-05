import pandas as pd 
import numpy as np 
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt 
import random
import datetime
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox 

from CVT_funs import * 
from LDAR_crews import LDAR_agent

class LDAR_travel: 
    def __init__(self,dutyholder,ID,epsg,mode):
        '''
        dutyholder: define the name of a LDAR duty holder/an operator
        ID: define the ID of a LDAR duty holder/an operator
        epsg: define the epsg id/number of projection system used in the simulation
        num_of_crews: number of crews
        '''
        self.duty_holder = dutyholder 
        self.ID = ID
        self.epsg = epsg
        self.mode = mode 
        self.crews = []
        return
    
    def read_service_cnt(self,hbd,Lon,Lat,prj=None): 
        '''
        read a service center csv file 
        hbd: the path of homebases csv file
        Lon: the fieldname for longitudes columns 
        Lat: the fieldname for latitudes columns 
        prj: the coordiates of homebases used, if needed  
        '''
        ori = pd.read_csv(hbd,sep=',')
        self.sc = gpd.GeoDataFrame(ori,geometry=gpd.points_from_xy(ori[Lon],ori[Lat]))
        self.cy = self.sc.geometry.y[0]
        self.cx = self.sc.geometry.x[0]
        if prj: 
            self.sc = self.sc.set_crs("EPSG:{}".format(prj))
        else: 
            self.sc = self.sc.set_crs("EPSG:{}".format(4326))
        self.sc.geometry = self.sc.geometry.to_crs(epsg=self.epsg)
        
        return
    
    def read_site_list(self,site_csv,lon,lat,prj=None):
        '''
        To roed the csv file that include LDAR sites and convert it to geodataframe
        site_csv: path of homebases csv file 
        lon: the fieldname for longitudes columns 
        lat: the fieldname for latitudes columns 
        prj: the projection system used, if needed 
        ''' 
        sites = pd.read_csv(site_csv,sep=',')
        self.sites = gpd.GeoDataFrame(sites,geometry=gpd.points_from_xy(sites[lon],sites[lat]))
        if prj: 
            self.sites = self.sites.set_crs("EPSG:{}".format(prj))
        else: 
            self.sites = self.sites.set_crs("EPSG:{}".format(4326))
        self.sites.geometry = self.sites.geometry.to_crs(epsg=self.epsg)
        self.sites_geo = list(self.sites.geometry)
        return
    
    def download_graph_from_OSM(self,dist,network_type):
        '''
        To download graph from OpenStreetMap
        cx:x of service center
        cy:y of service center
        network_type : string, {"all_private", "all", "bike", "drive", "drive_service", "walk"} 
        '''

        self.G = ox.graph_from_point([self.cy,self.cx],dist = dist,dist_type='bbox', network_type=network_type)
        G_proj = ox.project_graph(self.G)
        self.G = ox.project_graph(G_proj, to_crs='epsg:{}'.format(self.epsg))
        # add driving speed and travel time to graph
        self.G = ox.speed.add_edge_speeds(self.G)
        self.G = ox.speed.add_edge_travel_times(self.G)
        self.nodes, self.edges = ox.graph_to_gdfs(self.G, nodes=True, edges=True)
        
        self.cy = self.sc.geometry.y[0]
        self.cx = self.sc.geometry.x[0]
        return 
    
    def get_home_nodes(self):
    
        self.home_node = [] 
        for index, row in self.sc.iterrows():
            rx = row.geometry.x 
            ry = row.geometry.y
            hn = ox.distance.nearest_nodes(self.G,rx,ry)
            self.home_node.append(hn)
        return 
    
    def get_site_nodes(self): 
        site_nodes = [] 
        for sg in self.sites.geometry:
            sgx = sg.x
            sgy = sg.y 
            snode = ox.distance.nearest_nodes(self.G,sgx,sgy)
            site_nodes.append(snode)
    
        return 
    
    def sample_inspection_time(self,timetable):
        
        tdf = pd.read_csv(timetable,sep=',')
        time_on_site = np.array(tdf['time'])
        
        TOS = [] 
        while len(TOS) < len(self.sites):
            itime = random.choice(time_on_site)
            itime = np.round(itime/60,2)
            TOS.append(itime)
        
        
        self.sites['TOS'] = TOS
        
        return

    def add_Google_correction(self,correct_table):
        
        
        cdf = pd.read_csv(correct_table,sep=',')
        dist_diff = list(cdf.distance_corr)
        time_diff = list(cdf.time_corr)
        site_id = np.arange(1,len(self.sites)+1,1)
        self.sites['dist_diff'] = dist_diff
        self.sites['time_diff'] = time_diff
        self.sites['siteID'] = site_id
        
        return
    
    def create_empty_graph_with_sc(self,X,Y): 
        '''
        To create an empty graph with service center located in the 
        center of the graph with node id 0, start from 0 for multi
        service center case scenario.
        X: x of the service center 
        Y: y of the service center
        '''
        self.G = nx.Graph()
        # add service center
        if len(X) != len(Y):
            print ('Error! X and Y should have the same length.')
        if len(X) > 1: 
            sid = 0 
            self.home_node= [] 
            for e in zip(X,Y): 
                x = e[0]
                y = e[1]
                self.G.add_node(sid)
                self.G.nodes[sid]['x'] = x
                self.G.nodes[sid]['y'] = y
                self.home_node.append(sid)
                sid += 1 
        else:
            x = X[0]
            y = Y[0]
            self.G.add_node(0)
            self.G.nodes[0]['x'] = x
            self.G.nodes[0]['y'] = y
            self.home_node  = [0] 
            
        
        # service centers
        PTs = [] 
        for hn in self.home_node: 
            px = self.G.nodes[hn]['x']
            py = self.G.nodes[hn]['y']
            PTs.append(Point(px,py))
        self.sc= gpd.GeoDataFrame(PTs) 
        self.sc.rename(columns ={0:"geometry"},inplace=True)
        self.sc = self.sc.set_crs("EPSG:{}".format(self.epsg))
        self.sc.geometry = self.sc.geometry.to_crs(epsg=self.epsg)
        
        self.cy = self.sc.geometry.y[0]
        self.cx = self.sc.geometry.x[0]

        return
    
    def create_highways(self,eid,nid,ran,xs,highway_speed,setting = "Agriculture"):
        '''
        To create a gridded road network
        nid: initial id for nodes 
        eid: initial id for edges
        ran: length of horizontal highways  
        xs: space of grid
        highway_speed: the maximum speed limits of highways 
        '''
        if setting == "Agriculture":
            # create highways
            spaces = [(xs,xs),(xs,-xs),(-xs,-xs),(-xs,xs)]
            ranges = [(ran,ran),(ran,-ran),(-ran,-ran),(-ran,ran)]

            self.edge_manager = []
            self.node_manager = [] 
            self.highwayedges = [] 
            for i in range(4): 
                xr = ranges[i][0]
                yr = ranges[i][1]
                xspace = spaces[i][0]
                yspace = spaces[i][1]
                self.G,nid,eid,node_manager1,edge_manager1,edges1 = create_grid_roads(self.G,i+1,xr,yr,xspace,yspace,
                                                           eid,nid,highway_speed)
                self.edge_manager.append(edge_manager1)
                self.node_manager.append(node_manager1)
                self.highwayedges.append(edges1)
            
            Highways_roads =[] 
            for highway in self.highwayedges:
                Highways_roads += highway
                
            EM = []
            for em in self.edge_manager:
                EM += em
                
            NM = []
            for nm in self.node_manager:
                NM += nm
                
            if len(self.home_node) > 1:
                self.G,self.home_edges,nid,eid = connect_highway_homenode(self.G,self.home_node,self.sc.geometry,
                                                                     EM,Highways_roads,nid,eid,highway_speed)
        
        elif setting == "Forested": 
            self.G, self.highwayedges, eid, nid  = build_forest_highway(self.G,nid,eid,ran,highway_speed)
           
        else: 
            print ('This package only support Agriculture and Forested setting')

            
        
        return nid,eid
    
    
    def create_random_points_as_LDAR_sites (self,num,ran,stratified = False):
        # lx ux ly uy
        if stratified == True: 
            n = num/4
            site1 = sample_sites(n,0,ran,0,ran)
            site2 = sample_sites(n,0,ran,ran,0)
            site3 = sample_sites(n,ran,0,ran,0)
            site4 = sample_sites(n,ran,0,0,ran)
            sites = site1 + site2 + site3 + site4
        else: 
            sites = sample_sites(num,-ran,ran,-ran,ran)
        
        self.sites_geo = [Point(s) for s in sites]
        
        self.sites = gpd.GeoDataFrame(self.sites_geo) 
        self.sites.rename(columns ={0:"geometry"},inplace=True)
        self.sites = self.sites.set_crs("EPSG:{}".format(self.epsg))
        self.sites.geometry = self.sites.geometry.to_crs(epsg=self.epsg)
        
        return 
    
    
    def create_gravel_roads_forest (self,ran,margin,num_segment,nid,eid,highway_speed,gravelroad_speed):
        
        self.Loopnodes = []
        self.loops = [] 
        self.Loopends = [] 
        rans = [(ran,ran),(ran,-ran),(-ran,-ran),(-ran,ran)]
        for i in range(4): 
            ran1 = rans[i][0]
            ran2 = rans[i][1]
            loopnode,loop,loop_end_nodes,self.G,eid,nid = generate_loop_roads(self.G,ran1,ran2,margin,i+1,num_segment,
                                                                              eid,nid,highway_speed,gravelroad_speed)
            self.Loopnodes.append(loopnode)
            self.loops.append(loop)
            self.Loopends.append(loop_end_nodes)
            
        
        Loops_roads = [] 
        for loop in self.loops: 
            Loops_roads += loop
        Loop_ends  = [] 
        for end in self.Loopends:
            Loop_ends += end
        
        if len(self.home_node) > 1:
            self.G,self.home_edges,nid,eid = connect_highway_homenode(self.G,self.home_node,self.sc.geometry,
                                                                  Loop_ends,Loops_roads,nid,eid,highway_speed)
    
        return nid,eid
    
    def create_backroads_forest (self,nid,eid,backroad_speed,gravelroad_speed):
        self.SiteQ = [] 
        self.Site_nodes = [] 
        self.Loop_pts = [] 
        self.Backroads = [] 
        for i in range (4): 
            self.G,site_q,site_nodes,loop_pts,backroads,nid,eid = generate_back_roads(self.G,self.sites,self.loops[i],
                                                                                     i+1,nid,eid,backroad_speed,
                                                                                      gravelroad_speed,self.Loopends[i])
            
            self.SiteQ += site_q
            self.Site_nodes += site_nodes
            self.Loop_pts += loop_pts
            self.Backroads.append(backroads) 
        
        return nid,eid
    
    def create_backroads_grid (self,nid,eid,highway_speed,backroad_speed): 
        
        wells = [[],[],[],[]]
        for sg in self.sites_geo: 
            if sg.x > 0 and sg.y > 0:
                wells[0].append(sg)
            elif sg.x > 0 and sg.y<0: 
                wells[1].append(sg)
            elif sg.x <0 and sg.y <0:
                wells[2].append(sg)
            else: 
                wells[3].append(sg)
        
        self.Grid_pts = []
        self.Backroads = [] 
        self.Site_nodes = [] 
        for i in range(4):
            grid_pts,backroads, site_nodes,self.G,nid,eid = generate_backroads_grid(wells[i],self.highwayedges[i],self.G,self.edge_manager[i],
                                                                                nid,eid,highway_speed,backroad_speed)
            self.Site_nodes += site_nodes
            self.Grid_pts += grid_pts
            self.Backroads.append(backroads) 
        
        return nid,eid
            

    def create_GDB(self,setting,epsg):
        # Highways
        if setting == "Agriculture": 
            Highways_roads =[] 
            for highway in self.highwayedges:
                Highways_roads += highway
            
        elif setting == "Forested": 
            Highways_roads = self.highwayedges
            Loops_roads = [] 
            for loop in self.loops: 
                Loops_roads += loop

            self.gravelroads = gpd.GeoDataFrame(Loops_roads) 
            self.gravelroads.rename(columns ={0:"geometry"},inplace=True)
            self.gravelroads = self.gravelroads.set_crs("EPSG:{}".format(self.epsg))
            self.gravelroads.geometry = self.gravelroads.geometry.to_crs(epsg=self.epsg)
            
        else: 
            print ('This package only support Agriculture and Forested setting')
        
        # Create backroads geodataframe
        Back_roads = [] 
        for back in self.Backroads: 
            Back_roads += back
            
        if len(self.home_node) >  1:
            Highways_roads = Highways_roads + self.home_edges
        
        self.BR = gpd.GeoDataFrame(Back_roads) 
        self.BR.rename(columns ={0:"geometry"},inplace=True)
        self.BR = self.BR.set_crs("EPSG:{}".format(self.epsg))
        self.BR.geometry = self.BR.geometry.to_crs(epsg=self.epsg)
        # create highway geo dataframe 
        self.highways = gpd.GeoDataFrame(Highways_roads) 
        self.highways.rename(columns ={0:"geometry"},inplace=True)
        self.highways = self.highways.set_crs("EPSG:{}".format(self.epsg))
        self.highways.geometry = self.highways.geometry.to_crs(epsg=self.epsg)
        
        
        return 
    
    def create_plot(self,setting):
        # plot everything 
        fig, ax = plt.subplots(figsize=(5,5))
        
        if setting == "empirical": 
            self.sc.plot(ax=ax, color='red', markersize=50)
            # Plot road 
            self.edges.plot(ax=ax, color='black',alpha = 0.1)
        else:
            self.sc.plot(ax=ax,color='red',markersize=70,marker='o',zorder=4)
            self.highways.plot(ax=ax, color='black',linewidth=0.5)
            if setting == 'Forested':
                self.gravelroads.plot(ax=ax, color='orange')
            self.BR.plot(ax=ax, color='green')
        self.sites.plot(ax=ax,color='blue',markersize=35,marker='*',zorder=4)
        plt.tight_layout()
        ax.set_axis_off()
        
        return
    
    def create_agent(self,n_crews,start_time,end_time,Shared = True, Dutyholder = None ): 
        '''
        Initialize the individual aircraft crews (the agents)
        
        '''
        
        self.crews = [] 
        self.sites_copy = self.sites.copy()
        if Shared: 
            for i in range(n_crews): 
                self.crews.append(LDAR_agent(self.G,self.sc,self.cx,self.cy,self.sites_copy,self.home_node,self.duty_holder,
                                             start_time,end_time,self.mode))
        else: 
            self.subsites = []
            self.crews = [] 
            for comp in self.sites_copy[Dutyholder].unique():
                subsite = self.sites_copy[self.sites_copy[Dutyholder] == comp]
                self.subsite_copy = subsite.copy()
                self.opt_crew  = []            
                for i in range(n_crews):

                    self.opt_crew.append(LDAR_agent(self.G,self.sc,self.cx,self.cy,self.subsite_copy,self.home_node,self.duty_holder,
                                         start_time,end_time,self.mode))
                
                self.subsites.append(self.subsite_copy)
                self.crews.append(self.opt_crew)
            
            
        return
    
    def deploy_agents(self,Shared = True): 
        
        if Shared:
            while len(self.sites_copy) > 0:
                for i in self.crews:
                    i.survey_a_day()
                    
        
        else:
            for so in zip(self.subsites,self.crews):
                
                while len(so[0]) > 0: 
                    for i in so[1]: 
                            i.survey_a_day()
                    
        return 
                    
        
    def generate_inspection_report(self, Shared = True): 
        self.Report = []
        
        if Shared:         
            for i in self.crews: 
                travel_report =  pd.DataFrame(data = {"transit_between_sites":i.tbs_list,
                                   "dist_between_sites": i.dbs_list,
                                   "transit_home_site":i.tth_list,
                                   "dist_home_site": i.dth_list,
                                   "daily_num_sites":i.nod_list,
                                   "Nodes":i.node_list})
                self.Report.append(travel_report)
        else:    
            for opt_crew in self.crews:
                for i in opt_crew:
                    travel_report =  pd.DataFrame(data = {"transit_between_sites":i.tbs_list,
                                                   "dist_between_sites": i.dbs_list,
                                                   "transit_home_site":i.tth_list,
                                                   "dist_home_site": i.dth_list,
                                                   "daily_num_sites":i.nod_list,
                                                   "Nodes":i.node_list})
                
                    
                    self.Report.append(travel_report)
        return 