import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
import random 
import math

import networkx as nx

def dot(v,w):

    x,y, = v
    X,Y = w
    return x*X + y*Y

def length(v):
    x,y = v
    return math.sqrt(x*x + y*y )

def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)

def unit(v):
    x,y = v
    mag = length(v)
    return (x/mag, y/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y = v
    return (x * sc, y * sc,)

def add(v,w):
    x,y= v
    X,Y = w
    return (x+X, y+Y)


def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)

def build_single_road(x1,y1,x2,y2): 
    road = LineString([(x1,y1), (x2, y2)])
    return road 

def find_node(coor,N): 
    i = 0 
    node = False 
    for n in N: 
        if coor == n[0]: 
            node = N[i][1]
            break 
        i += 1
    if not node: 
        return False 
    else:
        return node
    
def build_forest_highway(G,nid,eid,max_dist,highway_speed): 
    
    highway_speed = highway_speed * 0.278
    
    
    # highway1 
    r1 = build_single_road(-max_dist,0,max_dist,0)
    # highway2 
    r2 = build_single_road(0,max_dist,0,-max_dist)
    Roads = [r1,r2]
    for geo in Roads:
        # first node 
        G.add_node(nid)
        pt1x = geo.xy[0][0]
        pt1y = geo.xy[1][0]
        G.nodes[nid]['x'] = pt1x
        G.nodes[nid]['y'] = pt1y
        # connect service center to the boundary of highway 
        G.add_edges_from([(0,nid)], ID=eid,
                         speed=highway_speed,
                         time = (max_dist)/highway_speed, 
                         distance=max_dist,nodes=(0,nid))
        eid += 1 
        # connect service center to the boundary of highway 
        G.add_edges_from([(nid,0)], ID=eid,
                         speed=highway_speed,
                         time = (max_dist)/highway_speed,
                         distance=max_dist,nodes=(nid,0))
        nid += 1
        eid += 1 
        # second node 
        G.add_node(nid)
        pt2x = geo.xy[0][1]
        pt2y = geo.xy[1][1]
        G.nodes[nid]['x'] = pt2x
        G.nodes[nid]['y'] = pt2y
        # connect service center to the boundary of highway 
        G.add_edges_from([(0,nid)], ID=eid,
                         speed=highway_speed,
                         time = (max_dist)/highway_speed,
                         distance=max_dist,nodes=(0,nid))
        eid += 1 
        G.add_edges_from([(nid,0)], ID=eid,
                         speed=highway_speed,
                         time = (max_dist)/highway_speed,
                         distance=max_dist,nodes=(nid,0))
        eid += 1 
        # connect two end points of highways 
        G.add_edges_from([(nid-1,nid)], ID=eid,
                         speed=highway_speed,
                         time = (max_dist*2)/highway_speed,
                         distance=max_dist*2,
                         nodes=(nid-1,nid))
        # connect two end points of highways 
        G.add_edges_from([(nid,nid-1)], ID=eid,
                         speed=highway_speed,distance=max_dist*2,
                         time = (max_dist*2)/highway_speed,
                         nodes=(nid,nid-1))
        nid += 1
        eid += 1 
        
    
    return G, Roads, eid, nid 

def generate_loop_roads(G, edgex,edgey,margin,quadrant,n,eid,nid,highway_speed,loop_speed):
    
    highway_speed = highway_speed * 0.278 
    loop_speed = loop_speed * 0.278
    
    # random generate a point on the highway
    if quadrant%2 == 0: 
        num = random.uniform(0,edgey)
        origin = (0,num)
    else:
        num = random.uniform(0,edgex)
        origin = (num,0)
    
    if quadrant == 1: 
        q_node = 2
        e_node = 3
    elif quadrant == 2: 
        q_node = 4
        e_node = 2
    elif quadrant == 3: 
        q_node = 1
        e_node = 4
    else: 
        q_node = 3
        e_node = 1
    
    max_dist = np.abs(origin[0] + origin[1])
    # create node for origin 
    G.add_node(nid)
    G.nodes[nid]['x'] = origin[0]
    G.nodes[nid]['y'] = origin[1]
    # connect service center to the the origin 
    G.add_edges_from([(0,nid)], ID=eid,
                     speed=highway_speed,
                     time = (max_dist)/highway_speed,
                     distance=max_dist,nodes=(0,nid))
    eid += 1 
    G.add_edges_from([(nid,0)], ID=eid,
                 speed=highway_speed,
                 time = (max_dist)/highway_speed,
                 distance=max_dist,nodes=(nid,0))
    eid += 1
    
    # connet loop entrance to the edge of highway
    G.add_edges_from([(nid,q_node)], ID=eid,
                 speed=highway_speed,
                 time = ((np.abs(edgex) - max_dist))/highway_speed,
                 distance=np.abs(edgex) - max_dist,nodes=(nid,q_node))            
    eid += 1 
    
    G.add_edges_from([(q_node,nid)], ID=eid,
             speed=highway_speed,
             time = ((np.abs(edgex) - max_dist))/highway_speed,
             distance=np.abs(edgex) - max_dist,nodes=(q_node,nid))                
    eid += 1 
    nid += 1 
     
    if edgex >0: 
        boundx = edgex - margin
    else: 
        boundx = edgex + margin
    
    if edgey >0: 
        boundy = edgey - margin
    else: 
        boundy = edgey + margin
    
    xy = [] 
    for i in range(n):
        ex = random.uniform(0,boundx)
        ey = random.uniform(0,boundy)
        pt = (ex,ey)
        xy.append(pt)
    
    # sort the loop road nodes
    dist = [] 
    for e in xy:
        d = ((origin[0] - e[0])**2 + (origin[1] - e[1])**2)**0.5
        dist.append(d)
    sort_xy = [x for _, x in sorted(zip(dist,xy))]
    
    # create loop road linestring and edges of graph 
    start = origin
    Loops = [] 
    # a disctionary to store nodes 
    loop_end_nodes = [] 
    for pt in sort_xy: 
        road_seg = LineString([start, pt])
        
        # add nodes and edges 
        G.add_node(nid)
        G.nodes[nid]['x'] = pt[0]
        G.nodes[nid]['y'] = pt[1]
        
        loop_end_nodes.append([nid-1,nid])
        
        # add edges 
        dist = road_seg.length
        G.add_edges_from([(nid-1,nid)], ID=eid,
                     speed=loop_speed,
                     time = (dist)/loop_speed,
                     distance=dist,nodes=(nid-1,nid))
        eid += 1
        G.add_edges_from([(nid,nid-1)], ID=eid,
             speed=loop_speed,
             time = (dist)/loop_speed,
             distance=dist,nodes=(nid,nid-1))
        eid += 1 
        nid += 1 
        
        Loops.append(road_seg)
        start = pt 
    dx = sort_xy[-1][0]
    dy = sort_xy[-1][1]
    if quadrant%2 == 0: 
        dest = (dx,0)
    else: 
        dest = (0,dy)
    last_road_seg = LineString([sort_xy[-1], dest])
    Loops.append(last_road_seg)
    # add thhe last loop road to graph 
    G.add_node(nid)
    G.nodes[nid]['x'] = dest[0]
    G.nodes[nid]['y'] = dest[1]
    loop_end_nodes.append([nid-1,nid])
    # add edges 
    dist = last_road_seg.length
    G.add_edges_from([(nid-1,nid)], ID=eid,
                 speed=loop_speed,
                 time = (dist)/loop_speed,
                 distance=dist,nodes=(nid-1,nid))
    eid += 1
    G.add_edges_from([(nid,nid-1)], ID=eid,
         speed=loop_speed,
         time = (dist)/loop_speed,
         distance=dist,nodes=(nid,nid-1))
    eid += 1  
    loopnodes = [origin] + sort_xy + [dest]
    
    # connect loop exit to the service center  
    max_dist = np.abs(dest[0] + dest[1])
    G.add_edges_from([(0,nid)], ID=eid,
                     speed=highway_speed,
                     time = (max_dist)/loop_speed,
                     distance=max_dist,nodes=(0,nid))
    eid += 1 
    G.add_edges_from([(nid,0)], ID=eid,
                 speed=highway_speed,
                 time = (max_dist)/loop_speed,
                 distance=max_dist,nodes=(nid,0))
    eid += 1 

    # connet loop exit to the edge of highway
    G.add_edges_from([(nid,e_node)], ID=eid,
                 speed=highway_speed,
               time = ((np.abs(edgex) - max_dist))/highway_speed,
               distance=np.abs(edgex) - max_dist,nodes=(nid,e_node))            
    eid += 1 
    
    G.add_edges_from([(e_node,nid)], ID=eid,
             speed=highway_speed,
             time = ((np.abs(edgex) - max_dist))/highway_speed,
             distance=np.abs(edgex) - max_dist,nodes=(e_node,nid))                
    eid += 1
    nid += 1
    
    return loopnodes, Loops,loop_end_nodes,G,eid,nid


def generate_back_roads(G,sites,Loops,quadrant,nid,eid,backroad_speed,looproad_speed,loop_end_nodes):
    backroad_speed = backroad_speed * 0.278
    looproad_speed = looproad_speed * 0.278 
    # filter sites 
    Site_Q1 = [] 
    for s in sites.geometry: 
        x = s.x
        y = s.y 
        if quadrant  == 1: 
            if x > 0 and y > 0: 
                Site_Q1.append((x,y))
        elif quadrant  == 2: 
            if x > 0 and y < 0: 
                Site_Q1.append((x,y))
        elif quadrant  == 3: 
            if x < 0 and y < 0: 
                Site_Q1.append((x,y))
        else: 
            if x < 0 and y > 0: 
                Site_Q1.append((x,y))
                
    Loop_pts = []
    backroads = []
    Sites_node = [] 
    for s in Site_Q1: 
        pt1 = s
        # add site to the nodes of graph 
        G.add_node(nid)
        Sites_node.append(nid)
        G.nodes[nid]['x'] = pt1[0]
        G.nodes[nid]['y'] = pt1[1]
        nid += 1 
        dist = []
        pt4s = [] 
        for line in Loops: 
            pt2 = (line.xy[0][0],line.xy[1][0])
            pt3 = (line.xy[0][1],line.xy[1][1])
            d, pt4 = pnt2line(pt1, pt2, pt3)
            dist.append(d) 
            pt4s.append(pt4) 
        min_dist = min(dist)
        ind = dist.index(min_dist)
        lpt = pt4s[ind]
        Loop_pts.append(lpt)
        # roads 
        br = LineString([pt1,lpt])
        backroads.append(br)  
        # build road 
        line_num = loop_end_nodes[ind]
        line_coor = Loops[ind]
        G.add_node(nid)
        G.nodes[nid]['x'] = lpt[0]
        G.nodes[nid]['y'] = lpt[1]
        # build edge from site to loop road connection
        d1 = br.length
        G.add_edges_from([(nid-1,nid)], ID=eid,
                     speed=backroad_speed,
                     time = (d1)/backroad_speed,
                     distance=d1,nodes=(nid-1,nid))
        eid += 1 
        G.add_edges_from([(nid,nid-1)], ID=eid,
                     speed=backroad_speed,
                     time = (d1)/backroad_speed,
                     distance=d1,nodes=(nid,nid-1))
        eid += 1 
        # connect connection node to both ends of loop edge 
        node1 = line_num[0]
        node2 = line_num[1]
        nc1 = (line_coor.xy[0][0],line_coor.xy[1][0])
        nc2 = (line_coor.xy[0][1],line_coor.xy[1][1])
        d2 = ((nc1[0] - lpt[0])**2 + (nc1[1] - lpt[1])**2)**0.5
        d3 = ((nc2[0] - lpt[0])**2 + (nc2[1] - lpt[1])**2)**0.5
        
        # edge to one end 
        G.add_edges_from([(nid,node1)], ID=eid,
                     speed=looproad_speed,
                     time = (d2)/looproad_speed,
                     distance=d2,nodes=(nid,node1))
        eid += 1 
        G.add_edges_from([(node1,nid)], ID=eid,
                     speed=looproad_speed,
                     time = (d2)/looproad_speed,
                     distance=d2,nodes=(node1,nid))
        eid += 1
        # edge to second end 
        G.add_edges_from([(nid,node2)], ID=eid,
                     speed=looproad_speed,
                     time = (d3)/looproad_speed,
                     distance=d3,nodes=(nid,node2))
        eid += 1 
        G.add_edges_from([(node2,nid)], ID=eid,
                     speed=looproad_speed,
                     time = (d3)/looproad_speed,
                     distance=d3,nodes=(node2,nid))
        eid += 1
        nid += 1 
        
    return G,Site_Q1,Sites_node,Loop_pts,backroads,nid,eid

def create_grid_roads(G,quadrant,xran,yran,xs,ys,nid,eid,highway_speed): 
    
    highway_speed = highway_speed*0.278
    
    X = np.arange(0,xran + xs,xs)
    Y = np.arange(0,yran + ys,ys)
    space = np.abs(xs)
    # generate nid 
    nid = nid 
    node_manager = []
    for x in X:
        for y in Y:
            if x!= 0 or y!= 0: 
                G.add_node(nid)
                G.nodes[nid]['x'] = x
                G.nodes[nid]['y'] = y
                nid += 1 

                node_manager.append([[x,y],nid])
    
    edge_manager = [] 
    edges = [] 
    for x in X: 
        for y in Y: 
            x1 = x 
            y1 = y 
            init_nid = find_node ([x1,y1],node_manager)
            forward = True 
            up = True 
            backward = True
            down = True 
            if x1 == 0:  
                foward = True 
                up = True 
                backward = False  
                down = True  
                
                if y1 == 0: 
                    init_nid = 0
                    down = False
                if y1 == yran: 
                    up = False
                    
            if x1 == xran: 
                forward = False 
                up = True 
                backward = True
                down = True 
                
                if y1  == 0: 
                    down = False 
                if y1 == yran: 
                    up = False 
                    
            if y1 == 0: 
                forward = True 
                up = True 
                backward = True
                down = False
                
                if x1 == 0: 
                    init_nid = 0
                    backward = False
                if x1 == xran: 
                    forward = False
                    
            
            if y1 == yran: 
                forward = True 
                up = False  
                backward = True
                down = True
                if x1 == 0: 
                    backward = False
                if x1 == xran: 
                    forward = False 

            # forward 
            if forward: 
                x2 = x1 + xs
                y2 = y1 
                nid2 = find_node ([x2,y2],node_manager)
                if [init_nid,nid2] not in edge_manager:
                    # edge between pt1 and pt2 
                    G.add_edges_from([(init_nid,nid2)], ID=eid,
                                     speed=highway_speed,distance=space,
                                     time = (space)/highway_speed,
                                     nodes=(init_nid,nid2))
                    eid += 1 

                    G.add_edges_from([(nid2,init_nid)], ID=eid,
                                     speed=highway_speed,distance=space,
                                     time = (space)/highway_speed,
                                     nodes=(nid2,init_nid))
                    eid += 1
                    edge_manager.append([init_nid,nid2])
                    L = LineString([Point(x1,y1), Point(x2,y2)])
                    edges.append(L)
            # up 
            if up: 
                x3 = x1 
                y3 = y1 + ys 
                nid3 = find_node ([x3,y3],node_manager)
                
                if [init_nid,nid3] not in edge_manager:
                    # edge between pt1 and pt3
                    G.add_edges_from([(init_nid,nid3)], ID=eid,
                                     speed=highway_speed,distance=space,
                                     time = (space)/highway_speed,
                                     nodes=(init_nid,nid3))
                    eid += 1 

                    G.add_edges_from([(nid3,init_nid)], ID=eid,
                                     speed=highway_speed,distance=space,
                                     time = (space)/highway_speed,
                                     nodes=(nid3,init_nid))
                    eid += 1 
                    edge_manager.append([init_nid,nid3])
                    L = LineString([Point(x1,y1), Point(x3,y3)])
                    edges.append(L)
                    
            # backward 
            
            if backward: 
                x4 = x1 - xs 
                y4 = y1 
                nid4 = find_node ([x4,y4],node_manager)
                if [init_nid,nid4] not in edge_manager:
                    
                    G.add_edges_from([(init_nid,nid4)], ID=eid,
                                 speed=highway_speed,distance=space,
                                 time = (space)/highway_speed,
                                 nodes=(init_nid,nid4))
                    eid += 1 

                    G.add_edges_from([(nid4,init_nid)], ID=eid,
                                     speed=highway_speed,distance=space,
                                     time = (space)/highway_speed,
                                     nodes=(nid4,init_nid))
                    eid += 1 
                    edge_manager.append([init_nid,nid4])
                    L = LineString([Point(x1,y1), Point(x4,y4)])
                    edges.append(L)
            
            # down
            if down: 
                x5 = x1 
                y5 = y1 - ys 
                nid5 = find_node ([x5,y5],node_manager)
                if [init_nid,nid5] not in edge_manager:
                    G.add_edges_from([(init_nid,nid5)], ID=eid,
                                 speed=highway_speed,distance=space,
                                 time = (space)/highway_speed,
                                 nodes=(init_nid,nid5))
                    eid += 1 

                    G.add_edges_from([(nid5,init_nid)], ID=eid,
                                     speed=highway_speed,distance=space,
                                     time = (space)/highway_speed,
                                     nodes=(nid5,init_nid))
                    eid += 1 
        
                    edge_manager.append([init_nid,nid5])
                    L = LineString([Point(x1,y1), Point(x5,y5)])
                    edges.append(L)
            
    return G,nid,eid,node_manager,edge_manager,edges

def generate_backroads_grid(sites,edges,G,edge_manager,nid,eid,looproad_speed,backroad_speed): 
    
    looproad_speed = looproad_speed * 0.278
    backroad_speed = backroad_speed * 0.278
    grid_pts = [] 
    backroads = []
    sites_node = [] 
    
    for s in sites: 
        sx = s.x
        sy = s.y 
        G.add_node(nid)
        sites_node.append(nid)
        G.nodes[nid]['x'] = sx
        G.nodes[nid]['y'] = sy
        nid += 1 
        pt1 = (sx,sy)
        
        dist = []
        pt4s = [] 
        for line in edges: 
            pt2 = (line.xy[0][0],line.xy[1][0])
            pt3 = (line.xy[0][1],line.xy[1][1])
            d, pt4 = pnt2line(pt1, pt2, pt3)
            dist.append(d) 
            pt4s.append(pt4) 
            
        min_dist = min(dist)
        ind = dist.index(min_dist)
        lpt = pt4s[ind]
        grid_pts.append(lpt)
        line_coor = edges[ind]
        # roads 
        br = LineString([pt1,lpt])
        backroads.append(br)  
        line_num = edge_manager[ind]
        
        # build back roads in graph 
        G.add_node(nid)
        G.nodes[nid]['x'] = lpt[0]
        G.nodes[nid]['y'] = lpt[1]
        # build edge from site to loop road connection
        d1 = br.length
        G.add_edges_from([(nid-1,nid)], ID=eid,
                     speed=backroad_speed,
                     time = (d1)/backroad_speed,
                     distance=d1,nodes=(nid-1,nid))
        eid += 1 
        G.add_edges_from([(nid,nid-1)], ID=eid,
                     speed=backroad_speed,
                     time = (d1)/backroad_speed,
                     distance=d1,nodes=(nid,nid-1))
        eid += 1 
        # connect connection node to both ends of loop edge 
        node1 = line_num[0]
        node2 = line_num[1]
        nc1 = (line_coor.xy[0][0],line_coor.xy[1][0])
        nc2 = (line_coor.xy[0][1],line_coor.xy[1][1])
        d2 = ((nc1[0] - lpt[0])**2 + (nc1[1] - lpt[1])**2)**0.5
        d3 = ((nc2[0] - lpt[0])**2 + (nc2[1] - lpt[1])**2)**0.5
        
        # edge to one end 
        G.add_edges_from([(nid,node1)], ID=eid,
                     speed=looproad_speed,
                     time = (d2)/looproad_speed,
                     distance=d2,nodes=(nid,node1))
        eid += 1 
        G.add_edges_from([(node1,nid)], ID=eid,
                     speed=looproad_speed,
                     time = (d2)/looproad_speed,
                     distance=d2,nodes=(node1,nid))
        eid += 1
        # edge to second end 
        G.add_edges_from([(nid,node2)], ID=eid,
                     speed=looproad_speed,
                     time = (d3)/looproad_speed,
                     distance=d3,nodes=(nid,node2))
        eid += 1 
        G.add_edges_from([(node2,nid)], ID=eid,
                     speed=looproad_speed,
                     time = (d3)/looproad_speed,
                     distance=d3,nodes=(node2,nid))
        eid += 1
        nid += 1 
        
    return grid_pts,backroads, sites_node,G,nid,eid

def sample_sites(num,lx,ux,ly,uy):
    sites = [] 
    while len(sites)<=num:
        x = random.uniform(lx,ux)
        x = np.round(x,2)
        y = random.uniform(ly,uy)
        y = np.round(y,2)
        
        sites.append((x,y))
    return sites

def create_empty_graph_with_sc(Xs,Ys,epsg): 
    '''
    To create an empty graph with service center located in the 
    center of the graph with node id 0
    x: x of the service center 
    y: y of the service center
    '''
    G = nx.Graph()
    # add service center
    if len(Xs) == len(Ys) == 1:
        x = Xs[0]
        y = Ys[0]
        G.add_node(0)
        G.nodes[0]['x'] = x
        G.nodes[0]['y'] = y
        # convert a service center to a geodataframe 
        sc= gpd.GeoDataFrame([Point(x,y)]) 
        sc.rename(columns ={0:"geometry"},inplace=True)
        sc = sc.set_crs("EPSG:{}".format(epsg))
        sc.geometry = sc.geometry.to_crs(epsg=epsg)

        home_node  = [0]
    else:
        sid = 0 
        home_node= [] 
        PTs = [] 
        for e in zip(Xs,Ys):
            x = e[0]
            y = e[1]
            G.add_node(sid)
            G.nodes[sid]['x'] = x
            G.nodes[sid]['y'] = y
            PTs.append(Point(x,y))
            home_node.append(sid)
            sid += 1 
        
        sc= gpd.GeoDataFrame(PTs) 
        sc.rename(columns ={0:"geometry"},inplace=True)
        sc = sc.set_crs("EPSG:{}".format(epsg))
        sc.geometry = sc.geometry.to_crs(epsg=epsg)
        
    return home_node, sc, G

def connect_highway_homenode(G,home_node,home_geo,highway_edge,highway_geo,nid,eid,highway_speed):
    highway_speed = highway_speed * 0.278
    ind = 1
    Roads = [] 
    while ind < len(home_node):
        hn = home_node[ind]
        h_geo = home_geo[ind]       
        
        pt1x = h_geo.x
        pt1y = h_geo.y
        pt1 = (pt1x,pt1y)
        
        Ds = [] 
        Connects = []
        ends = []
        for hw in highway_geo: 
            pt2x = hw.xy[0][0]
            pt2y = hw.xy[1][0]
            pt2 = (pt2x,pt2y)
            pt3x = hw.xy[0][1]
            pt3y = hw.xy[1][1]
            pt3 = (pt3x,pt3y)
            d, pt4 = pnt2line(pt1, pt2, pt3)
            Ds.append(d)
            Connects.append(pt4)
            ends.append([pt2,pt3])
            
        min_ds = min(Ds)
        j = Ds.index(min_ds)
        connect = Connects[j]
        edge = highway_edge[j]
        end = ends[j]
        
        pt2 = end[0]
        pt3 = end[1]
        
        n1 = edge[0]
        n2 = edge[1]
        # build edge between homenode and connector 
        G.add_node(nid)
        G.nodes[nid]['x'] = connect[0]
        G.nodes[nid]['y'] = connect[1]
        
        G.add_edges_from([(hn,nid)], ID=eid,
                     speed=highway_speed,
                     time = (min_ds)/highway_speed,
                     distance=min_ds,nodes=(hn,nid))
        eid += 1 
        G.add_edges_from([(nid,hn)], ID=eid,
                     speed=highway_speed,
                     time = (min_ds)/highway_speed,
                     distance=min_ds,nodes=(nid,hn))
        eid += 1 
        
        Roads.append(LineString([pt1,connect]))
        
        # build edge between highway ends and home base connect
        d1 = ((connect[0] - pt2[0])**2 + (connect[1] - pt2[1])**2)**0.5
        G.add_edges_from([(n1,nid)], ID=eid,
                     speed=highway_speed,
                     time = (d1)/highway_speed,
                     distance=d1,nodes=(n1,nid))
        eid += 1 
        G.add_edges_from([(nid,n1)], ID=eid,
                     speed=highway_speed,
                     time = (d1)/highway_speed,
                     distance=d1,nodes=(nid,n1))
        
        Roads.append(LineString([connect,pt2]))
        
        d2 = ((connect[0] - pt3[0])**2 + (connect[1] - pt3[1])**2)**0.5
        G.add_edges_from([(n2,nid)], ID=eid,
                     speed=highway_speed,
                     time = (d2)/highway_speed,
                     distance=d2,nodes=(n2,nid))
        eid += 1 
        G.add_edges_from([(nid,n2)], ID=eid,
                     speed=highway_speed,
                     time = (d2)/highway_speed,
                     distance=d2,nodes=(nid,n2))
        eid += 1 
        
        Roads.append(LineString([connect,pt3]))
        
        nid += 1 
        ind += 1     
        
    return G,Roads,nid,eid

