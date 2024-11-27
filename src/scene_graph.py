from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
import os, pickle
from typing import Optional

from src import ObjectNode, DrawerNode, LightSwitchNode
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import datetime, time
import random

class SceneGraph:
    """
    Represents a scene graph to manage and organize connections between nodes in a 3D scene.

    The SceneGraph class is designed to structure relationships between various nodes
    in a 3D scene, maintaining connectivity, labels, and hierarchical relationships, also
    during transformations applied to the scene/nodes. 
    It supports efficient queries and manages node attributes.

    Attributes:
        index (int): Counter to keep track of unique indices for nodes in the scene graph.
        nodes (dict): Dictionary to store nodes in the scene graph, keyed by unique node IDs.
        labels (dict): Mapping of node IDs to semantic labels.
        outgoing (dict): Tracks outgoing connections for each node, representing directed edges.
        ingoing (dict): Tracks ingoing connections for each node, representing directed edges, mutiple ingoing connections are possible.
        tree (Optional[spatial.KDTree]): KDTree for efficient spatial queries, if applicable.
        ids (list): List of node IDs present in the scene graph, is updated after addition or deletion of nodes.
        k (int): Number of nearest neighbors to consider for spatial relations. Defaults to 2.
        label_mapping (dict): Dictionary for mapping semantic labels.
        min_confidence (float): Minimum confidence threshold for including nodes in the scene.
        immovable (list): List of IDs or labels representing nodes that cannot be moved.
        pose (Optional[np.ndarray]): Pose of the entire scene graph, if applicable.
        mesh (Optional[o3d.geometry.TriangleMesh]): Central mesh representation of the scene, if applicable.
        pcd (Optional[o3d.geometry.PointCloud]): Central point cloud representation of the scene, if applicable.
    """

    def __init__(
        self,
        label_mapping: dict = dict(),
        min_confidence: float = 0.0,
        k: int = 2,
        immovable: list = [],
        pose: Optional[np.ndarray] = None
    ):
        """
        Initializes a SceneGraph with default configurations and empty connectivity structures.

        This constructor sets up the initial data structures for managing nodes and connections in the 
        scene graph, including labels, ingoing and outgoing connections, and spatial properties. 
        Additional parameters allow configuration of label mappings, confidence thresholds, and immovable nodes.

        :param label_mapping: Dictionary for mapping semantic labels to node IDs or categories.
        :param min_confidence: Minimum confidence threshold for including nodes in the scene. Defaults to 0.0.
        :param k: Number of nearest neighbors to consider for spatial relations. Defaults to 2.
        :param immovable: List of IDs or labels representing nodes that cannot be moved. Defaults to an empty list.
        :param pose: Optional 4x4 numpy array representing the pose of the entire scene graph. Defaults to None.
        """
        self.index = 0
        self.nodes = dict()
        self.labels = dict()
        self.outgoing = dict()
        self.ingoing = dict()
        self.tree = None
        self.ids = []
        self.k = k
        self.label_mapping = label_mapping
        self.min_confidence = min_confidence
        self.immovable = immovable
        self.pose = pose
        self.mesh = None
        self.pcd = None
    
    def change_coordinate_system(self, transformation: np.ndarray) -> None:
        """
        Applies a transformation to change the coordinate system of the entire scene graph.

        This method updates the coordinate system of all nodes and associated spatial data in the 
        scene graph by applying a given transformation matrix. The transformation affects the pose 
        of each node and any central scene representations such as point clouds or meshes.

        :param transformation: A 4x4 transformation matrix to apply to the scene graph’s coordinate system.
        :return: None. The coordinate system of nodes and spatial data is modified in place.
        """
        if self.pose is not None:
            trans_inv = np.linalg.inv(self.pose)
            transformation = np.dot(transformation, trans_inv)
        for node in self.nodes.values():
            node.transform(transformation, force=True)
        if self.mesh is not None:
            self.mesh.transform(transformation)
        if self.pcd is not None:
            self.pcd.transform(transformation)
        self.tree = KDTree(np.array([self.nodes[index].centroid for index in self.ids]))

    def add_node(
        self,
        color: tuple,
        sem_label: str,
        points: np.ndarray,
        tracking_points: list,
        mesh_mask: np.ndarray,
        confidence: float
    ) -> int:
        """
        Adds a new node to the scene graph with specified attributes.

        This method creates a new node with the given properties, such as color, semantic label, 
        and 3D point data, and adds it to the scene graph. The node is assigned a unique identifier, 
        and its spatial properties and metadata are stored within the graph.
        Special node types, such as drawers or light switches, are handeled as well.

        :param color: RGB color tuple representing the node's color.
        :param sem_label: Semantic label categorizing the node (e.g., "drawer", "light switch").
        :param points: Array of 3D points defining the node's geometry.
        :param tracking_points: List of key points used for tracking the node’s position.
        :param mesh_mask: Binary mask representing the node's mesh structure.
        :param confidence: Confidence score associated with the node's detection or classification.
        """
        if self.label_mapping.get(sem_label, "ID not found") in self.immovable:
            # mark objects as immovable if a list was given
            # TODO: could in the future either be a complete list or LLM api request for open vocabulary
            self.nodes[self.index] = ObjectNode(self.index, np.array([0.5, 0.5, 0.5]), sem_label, points, tracking_points, mesh_mask, confidence, movable=False)
        elif sem_label == 25:
            self.nodes[self.index] = DrawerNode(self.index, color, sem_label, points, tracking_points, mesh_mask, confidence)
        elif sem_label == 232:
            self.nodes[self.index] = LightSwitchNode(self.index, color, sem_label, points, tracking_points, mesh_mask, confidence)
        else:
            self.nodes[self.index] = ObjectNode(self.index, color, sem_label, points, tracking_points, mesh_mask, confidence)
        self.labels.setdefault(sem_label, []).append(self.index)
        self.ids.append(self.index)
        self.index += 1
    
    def update_connection(self, node: ObjectNode) -> None:
        """
        Updates the connection of the specified node to its nearest neighboring node in the scene graph.

        This method identifies the closest neighboring node to the provided `node` object and updates 
        its connections accordingly. Any existing connections to other nodes are removed before 
        establishing the new connection.

        :param node: The node (of type ObjectNode or one of its subclasses) whose connections need to be updated.
        :return: None. The method updates the connections in the scene graph in place.
        """
        min_index, min_dist = None, None
        # add the regular node based on the closest other node
        if isinstance(node, ObjectNode):
            # TODO: if I am really just doing this for the closest node, I could just use the tree query
            for other in self.nodes.values():
                if other.object_id != node.object_id:
                    dist = np.linalg.norm(node.centroid - other.centroid)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        min_index = other.object_id
        else:
            raise TypeError("Invalid node type. Expected ObjectNode or DrawerNode.")
        
        tmp = self.outgoing.get(node.object_id, None)
        if min_index is not None and tmp != min_index:                               
            if tmp is not None:
                self.ingoing[tmp].remove(node.object_id)
            self.outgoing[node.object_id] = min_index
            self.ingoing.setdefault(min_index, []).append(node.object_id)
        if min_index is not None:
            if isinstance(self.nodes[min_index], DrawerNode) and self.nodes[min_index].box is not None:
                inside = self.nodes[min_index].box.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector([node.centroid]))
                if len(inside) and node not in self.nodes[min_index].contains:
                    self.nodes[min_index].contains.append(node)
                    node.visible = False

    def get_node_info(self):
        # TODO: json representation of the graph?
        for node in self.nodes.values():
            print(f"Object ID: {node.object_id}")
            print(f"Centroid: {node.centroid}")
            print("Semantic Label: " + self.label_mapping.get(node.sem_label, "ID not found"))
            print(f"Confidence: {node.confidence}")
    
    def save(self, file_path: str) -> None:
        """
        Saves the entire scene graph to a specified file using pickle serialization.

        This method serializes the scene graph object and writes it to a file in pickle format, 
        allowing for easy storage and retrieval. The file can later be loaded to reconstruct the 
        scene graph in its current state.

        :param file_path: Path to the file where the scene graph will be saved.
        :return: None. The scene graph is saved to the specified file.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    def init_graph(self) -> None:
        """
        Initializes the scene graph by establishing initial connections between nodes.

        This method sets up the initial structure of the scene graph, assuming no connections have 
        been established beforehand. It must be called once after the graph is built to create 
        the foundational connections and relationships between nodes.

        :return: None. The scene graph's initial connections are established in place.
        """
        for node in self.nodes.values():
            self.update_connection(node)
            if self.outgoing.get(node.object_id, None) is None:
                continue
            if isinstance(node, DrawerNode):
                # iterate through all added shelfs in the scene
                min_index, min_dist = None, None
                labels = [item for sublist in [self.labels.get(l, []) for l in [7, 8, 18, 24, 44, 51]] for item in sublist]
                for idx in labels:
                    other = self.nodes[idx]
                    if other.object_id != node.object_id:
                        dist = np.linalg.norm(node.centroid - other.centroid)
                        if min_dist is None or dist < min_dist:
                            min_dist = dist
                            min_index = other.object_id
                node.belongs_to = min_index
                node.add_box(self.nodes[node.belongs_to].centroid)
    
    def build(
        self,
        scan_dir: str,
        aria_dir: Optional[str] = None,
        drawers: bool = False,
        light_switches: bool = False
    ) -> None:
        """
        Constructs the scene graph by loading data from the specified directories and initializing nodes 
        based on available object types.

        This method populates the scene graph with nodes derived from data in the specified scan directory 
        and, if provided, an optional Aria directory. Aria directory must be specified if one would like to
        perform tracking of egocentric Aria recordings within the scene graph. Flags allow inclusion of 
        specific object types, such as drawers and light switches, during the scene graph construction.
        The respective preprocessing on the scan data must be performed before including these object types.

        :param scan_dir: Path to the directory containing scan data for building the graph.
        :param aria_dir: Optional path to an additional data directory for further context. Defaults to None.
        :param drawers: Boolean flag to include drawer nodes in the scene graph. Defaults to False.
        :param light_switches: Boolean flag to include light switch nodes in the scene graph. Defaults to False.
        :return: None. The scene graph is built and populated in place.
        """
        lines = []
        
        with open(os.path.join(scan_dir, 'predictions.txt'), 'r') as file:
            lines += file.readlines()
        
        if drawers and os.path.exists(os.path.join(scan_dir, 'predictions_drawers.txt')):
            with open(os.path.join(scan_dir, 'predictions_drawers.txt'), 'r') as file:
                lines += file.readlines()
        if light_switches and os.path.exists(os.path.join(scan_dir, 'predictions_light_switches.txt')):
            with open(os.path.join(scan_dir, 'predictions_light_switches.txt'), 'r') as file:
                lines += file.readlines()

        
        file_paths = []
        values = []
        confidences = []

        for line in lines:
            parts = line.split()
            file_paths.append(parts[0])
            values.append(int(parts[1]))
            confidences.append(float(parts[2]))
        
        base_dir = os.path.dirname(os.path.abspath(os.path.join(scan_dir, 'predictions.txt')))

        self.mesh = o3d.io.read_triangle_mesh(scan_dir + "/textured_output.obj", enable_post_processing=True)

        self.pcd = o3d.io.read_point_cloud(scan_dir + "/mesh_labeled.ply")

        if aria_dir is not None:
            T = np.load(aria_dir + "/icp_aligned_pose.npy")
            self.pcd.transform(T)
            self.mesh.transform(T)
            self.aria_pcd = o3d.io.read_point_cloud(aria_dir + "/aria_pointcloud.ply")
            
            aria_points, ipad_points = np.array(self.aria_pcd.points), np.array(self.pcd.points)
            aria_tree, ipad_tree = KDTree(aria_points), KDTree(ipad_points)

            _, nn_aria_to_ipad = aria_tree.query(ipad_points)
            _, nn_ipad_to_aria = ipad_tree.query(aria_points)

            mutual_indeces = -np.ones_like(nn_aria_to_ipad, dtype=np.int64)
            for i, j in enumerate(nn_aria_to_ipad):
                if nn_ipad_to_aria[j] == i:
                    mutual_indeces[i] = j

        np_points = np.array(self.pcd.points)
        np_colors = np.array(self.pcd.colors)


        mask3d_labels = np.ones((np_points.shape[0], 2), dtype=np.int64) * -1
        
        for i, relative_path in enumerate(file_paths):
            if confidences[i] < self.min_confidence:
                continue
            file_path = os.path.join(base_dir, relative_path)
            labels = np.loadtxt(file_path, dtype=np.int64)
            index, counts = np.unique(mask3d_labels[labels == 1, 1], return_counts=True)
            if index.shape[0] == 1:
                if index[0] == -1 or np.sum(np.loadtxt(os.path.join(base_dir, file_paths[index[0]]), dtype=np.int64)) > np.sum(labels):
                    mask3d_labels[labels == 1, 0] = values[i]
                    mask3d_labels[labels == 1, 1] = i
            else:
                if index[np.argmax(counts)] == -1:
                    mask3d_labels[np.logical_and(labels == 1, mask3d_labels[:, 0] == -1), 0] = values[i]
                    mask3d_labels[np.logical_and(labels == 1, mask3d_labels[:, 1] == -1), 1] = i
                elif np.max(counts) < 10000 and np.max(counts) / np.sum(counts) > 0.75:
                    mask3d_labels[labels == 1, 0] = values[i]
                    mask3d_labels[labels == 1, 1] = i
        
        for i, relative_path in enumerate(file_paths):
            file_path = os.path.join(base_dir, relative_path)
            labels = np.loadtxt(file_path, dtype=np.int64)
            
            mask = np.logical_and.reduce((labels == 1, mask3d_labels[:, 0] == values[i], mask3d_labels[:, 1] == i))
            node_points = np_points[mask]
            tracking_points = np_points[mask]
            colors = np_colors[mask]

            mesh_mask = np.copy(mask)

            if aria_dir is not None:
                mask = mutual_indeces[(mask==1) & (mutual_indeces>=0)]
                tracking_points = aria_points[mask]
            
            if confidences[i] > self.min_confidence and node_points.shape[0] > 10:
                self.add_node(colors[0], values[i], node_points, tracking_points, mesh_mask, confidences[i])
        
        self.init_graph()
        self.tree = KDTree(np.array([self.nodes[index].centroid for index in self.ids]))
        self.color_with_ibm_palette()

    def get_centroid_distance(self, point: np.ndarray) -> float:
        """
        Calculates the distance from a given point to the nearest node's centroid.

        :param point: A 3D numpy array representing the point of interest.
        :return: Euclidean distance as a float between the point and the nearest node's centroid.
        """
        _, idx = self.tree.query(point)
        return np.linalg.norm(point - self.nodes[self.ids[idx]].centroid)

    def query(self, point: np.ndarray) -> int:
        """
        Finds the ID of the node closest to the given point.

        :param point: A 3D numpy array representing the point of interest.
        :return: The unique ID of the node closest to the specified point.
        """
        _, idx = self.tree.query(point)
        return self.ids[idx]

    def nearest_node(self, point: Optional[np.ndarray]) -> tuple[float, Optional[int]]:
        """
        Finds the nearest movable and visible node to the specified point.

        :param point: A 3D numpy array representing the point of interest. If None, returns infinity and None.
        :return: A tuple containing:
            - minimum_distance: Minimum distance as a float to the nearest valid node.
            - nearest_neighbor_id: ID of the nearest valid node, or None if no valid node is found.
        """
        if point is None:
            return np.inf, None
        _, neighbor_indices = self.tree.query(point, k=4)
        neighbor_indices = [
            self.ids[n_idx]
            for n_idx in neighbor_indices
            if (self.nodes[self.ids[n_idx]].movable and self.nodes[self.ids[n_idx]].visible)
        ]

        if len(neighbor_indices) == 0:
            return None, None
        else:
            nearest_neighbor = np.array([
                self.nodes[neighbor_idx].hull_tree.query(point, k=1)[0]
                for neighbor_idx in neighbor_indices
            ])
            return np.min(nearest_neighbor), neighbor_indices[np.argmin(nearest_neighbor)]

        
    def remove_node(self, remove_index: int) -> None:
        """
        This method deletes the specified node from the scene graph, removes its connections, 
        and updates the connections of other nodes that were linked to it. Finally, the KD-tree 
        is rebuilt to reflect the removal.

        :param remove_index: The unique ID of the node to be removed from the scene graph.
        :return: None. The node and its connections are removed in place.
        """
        self.nodes.pop(remove_index, None)
        self.ids.remove(remove_index)
        deleted = self.outgoing.pop(remove_index, None)  
        # update the connections of the other nodes that were connected to the removed node
        for id in self.ingoing.get(remove_index, []):
            del self.outgoing[id]
            self.update_connection(self.nodes[id])
        self.ingoing.pop(remove_index, None)
        self.ingoing.get(deleted, []).remove(remove_index)
        self.tree = KDTree(np.array([self.nodes[index].centroid for index in self.ids]))

    def remove_category(self, category: str) -> None:
        """
        Removes all nodes belonging to a specified category from the scene graph.

        This method identifies nodes in the specified category based on the label mapping 
        and removes them from the scene graph. Connections are updated accordingly.

        :param category: The category of nodes to be removed from the scene graph.
        :return: None. All nodes of the specified category are removed in place.
        """
        label_to_remove = next((label for label, cat in self.label_mapping.items() if cat == category), None)
        for index in self.labels.get(label_to_remove, []):
            self.remove_node(index)
        self.labels.pop(label_to_remove, None)

    def query_object(self) -> tuple[np.ndarray, int]:
        """
        Visualizes the scene graph and allows the user to select a point interactively.

        This method creates a fused point cloud of all nodes in the scene graph and visualizes it.
        The user can select a point, which the function then identifies within the scene. The nearest 
        node's ID is returned alongside the selected point coordinates.

        :return: Tuple containing:
            - picked_point: The selected point as a numpy array.
            - nearest_node_id: The unique ID of the node closest to the selected point.
        """
        fused_pcd = o3d.geometry.PointCloud()
        for node_id in sorted(self.nodes):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.nodes[node_id].points)
            pcd.paint_uniform_color(self.nodes[node_id].color)
            fused_pcd += pcd
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Pick a point using [shift + left click] (undo: [shift + right click]), when finished press [Q]")
        vis.add_geometry(fused_pcd)
        vis.run()
        vis.destroy_window()
        picked_point = vis.get_picked_points()[0]
        picked_point = np.array(fused_pcd.points)[picked_point]

        return picked_point, self.ids[self.tree.query(picked_point)[1]]
    
    def get_tracker_offset(self, object_coordinates=False):
        tracker_point, obj_id = self.query_object()
        tracker_offset = tracker_point - self.nodes[obj_id].centroid
        if object_coordinates:
            tracker_offset = np.dot(self.nodes[obj_id].pose[:3, :3].T, tracker_offset)
        return tracker_offset

    
    def transform(self, idx: int, *args) -> None:
        """
        Applies a transformation to the node with the specified index and updates its connections.

        This method transforms the node identified by `idx` based on the provided transformation arguments.
        After transforming the node, it ensures that the connections between the node and its neighbors 
        are updated to reflect the change in position or orientation.

        :param idx: The unique ID of the node to be transformed.
        :param args: Variable arguments representing the transformation parameters (e.g., transformation, rotation, translation).
        :return: None. The node is transformed, and connections are updated in place.
        """
        # node is transformed
        self.nodes[idx].transform(*args)
        # all the nodes that this node is connected to might change their connection to a closer other node, hence the updating
        try:
            for neighbor in self.ingoing.get(idx, []):
                self.update_connection(self.nodes[neighbor])
        except KeyError:
            print(idx)
            print(self.nodes.keys())
            print(self.ingoing)
            raise KeyError("Key not found.")
        # update the own connection
        self.update_connection(self.nodes[idx])
        # the newly connected node might change their connections as well
        self.update_connection(self.nodes[self.outgoing[idx]])
        # tree needs to be built again (TODO: optimize this)
        self.tree = KDTree(np.array([self.nodes[index].centroid for index in self.ids]))
    
    def color_with_ibm_palette(self):
        """ manual definition of the IBM palette including 10 colors """
        colors = np.array([[0.39215686, 0.56078431, 1.], [0.47058824, 0.36862745, 0.94117647], [0.8627451 , 0.14901961, 0.49803922],
                [0.99607843, 0.38039216, 0], [1., 0.69019608, 0.], [0.29803922, 0.68627451, 0.31372549], [0., 0.6, 0.8],
                [0.70196078, 0.53333333, 1.], [0.89803922, 0.22352941, 0.20784314], [1., 0.25098039, 0.50588235]])

        random.seed(10)
        for node in self.nodes.values():
            if node.movable:
                node.color = colors[random.randint(0, len(colors)-1)]
    
    def scene_geometries(self, centroids: bool = True, connections: bool = True) -> list[tuple]:
        """
        Retrieves all geometries in the scene graph, optionally including centroids and connections.

        This method compiles a list of all geometries associated with nodes in the scene graph. 
        Based on the provided arguments, it can include visual representations of the object centroids 
        and the connections between them. The geometries are returned as a list of tuples, where each 
        tuple contains the geometry data and metadata for rendering.

        :param centroids: Boolean flag indicating whether to include centroids of objects in the output. Defaults to True.
        :param connections: Boolean flag indicating whether to include connections between nodes in the output. Defaults to True.
        :return: List of tuples, each representing a geometry in the scene graph along with associated visualization data.
        """
        geometries = []
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"

        line_mat = rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 5
        
        for node in self.nodes.values():
            if node.visible:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(node.points)
                pcd_color = np.array(node.color, dtype=np.float64)
                pcd.paint_uniform_color(pcd_color)
                geometries.append((pcd, "node_" + str(node.object_id), material))
                if isinstance(node, DrawerNode) and node.box is not None:
                    geometries.append((node.box, "box_" + str(node.object_id), line_mat))
        
        if centroids:
            centroid_pcd = o3d.geometry.PointCloud()
            centroids_xyz = np.array([node.centroid for node in self.nodes.values()])
            centroids_colors = np.array([node.color for node in self.nodes.values()], dtype=np.float64) / 255.0
            centroid_pcd.points = o3d.utility.Vector3dVector(centroids_xyz)
            centroid_pcd.colors = o3d.utility.Vector3dVector(centroids_colors)
            geometries.append((centroid_pcd, "centroids", material))

        if connections:
            line_points = []
            line_indices = []
            idx = 0
            for start, end in self.outgoing.items():
                line_points.append(self.nodes[start].centroid)
                line_points.append(self.nodes[end].centroid)
                line_indices.append([idx, idx + 1])
                idx += 2
            if line_points:
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(line_points),
                    lines=o3d.utility.Vector2iVector(line_indices)
                )
                line_set.paint_uniform_color([0, 0, 0])
                geometries.append((line_set, "connections", line_mat))
            drawer_points = []
            drawer_indices = []
            idx = 0
            for node in self.nodes.values():
                if isinstance(node, DrawerNode) and node.belongs_to is not None:
                    drawer_points.append(node.centroid)
                    drawer_points.append(self.nodes[node.belongs_to].centroid)
                    drawer_indices.append([idx, idx + 1])
                    idx += 2
            if drawer_points:
                drawer_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(drawer_points),
                    lines=o3d.utility.Vector2iVector(drawer_indices)
                )
                drawer_set.paint_uniform_color([0.5, 0.5, 0.5])
                geometries.append((drawer_set, "drawer_connections", line_mat))
        
        return geometries
    
    
    def set_camera(self, size, intrinsics) -> tuple[o3d.camera.PinholeCameraIntrinsic, np.ndarray] | None:
        """
        Opens an interactive Open3D window to set the desired camera view for the scene graph.

        This method generates the scene graph geometries and opens an Open3D visualization window, allowing 
        the user to navigate and set the preferred camera view. By pressing 's', the function captures 
        the current view's extrinsic and intrinsic camera parameters and returns them as a tuple. 
        If 's' is not pressed, the function exits without returning parameters.

        :return: 
            - Tuple containing:
                - extrinsics: A numpy array representing the extrinsic parameters of the camera view.
                - intrinsics: A numpy array representing the intrinsic parameters of the camera view.
            - None
        """
        geometries = self.scene_geometries()

        gui.Application.instance.initialize()

        window = gui.Application.instance.create_window("Press <S> to save the pose", 600, int(600*size[1]/size[0]))
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        scene.scene.set_view_size(size[0], size[1])
        scene.scene.set_background(np.array([255.0, 255.0, 255.0, 1.0], dtype=np.float32))
        scene.scene.camera.set_projection(intrinsics, 1, 5, size[0], size[1])
        scene.scene.camera.look_at(np.array([0, 0, 0]), [0, 3, 0], [0, 1, 0])
        window.add_child(scene)
        
        for geometry, name, material in geometries:
            scene.scene.add_geometry(name, geometry, material)

        camera_params = [None, None]
        
        # Set a key event callback to capture the screen
        def on_key_event(event):
            if event.type == gui.KeyEvent.Type.DOWN:
                if event.key == gui.KeyName.S:  # Capture screen when 'S' key is pressed
                    model_matrix = np.array(scene.scene.camera.get_model_matrix())
                    intrinsics_cam = o3d.camera.PinholeCameraIntrinsic(size[0], size[1], intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2])
                    extrinsics = np.linalg.inv(model_matrix)
                    extrinsics[1] = -extrinsics[1]
                    extrinsics[2] = -extrinsics[2]
                    camera_params[0] = intrinsics_cam
                    camera_params[1] = extrinsics
                    gui.Application.instance.quit()
                    return True
            return False

        window.set_on_key(on_key_event)
        
        # Run the application
        gui.Application.instance.run()
        
        if camera_params[0] is None:
            print("camera parameters not set correctly")
            return None
        else:
            return camera_params

    def headpose_track(
            self,
            scan_dir: str,
            video_path: Optional[str] = None,
            force_object_detection: bool = False,
            save_pose: bool = False,
            tracker_offset: Optional[np.ndarray] = None,
            use_hand: bool = True
        ) -> None:
        """ Employs same strategy as track method but with headpose heuristic for the object rotation instead of point tracking.
        Served as a baseline in the experiments. Less accurate but more robust to non-rigid objects. Consider this if your applications
        involve non-rigid objects.

        :param scene_graph: The scene graph containing object and spatial information.
        :param scan_dir: Directory containing scan data and where results will be saved.
        :param video_path: Path where tracking visualization video will be saved.
        :param force_object_detection: Whether to force object detection in each frame.
        :param save_pose: Whether to save tracked poses to disk.
        :param tracker_offset: Optional offset to apply to tracked poses.
        :param use_hand: Whether to use hand positions for pose estimation.
        :return: None   
        """  
        from .tracking import headpose_track        
        headpose_track(self, scan_dir, video_path, force_object_detection, save_pose, tracker_offset, use_hand)


    def track(
        self,
        scan_dir: str,
        video_path: Optional[str] = None,
        force_object_detection: bool = False,
        save_pose: bool = False,
        tracker_offset: Optional[np.ndarray] = None,
        use_hand: bool = True
    ) -> None:
        """
        Executes the tracking process for objects within the scene.

        This method iterates over incoming observations from the `aria_data` iterator,
        processes each observation to update object poses based on hand detections and
        object detections, and manages the rendering and saving of tracking data.

        The tracking process involves:
            - Updating the context window with new observations.
            - Checking for hand-object detections and updating object associations.
            - Processing left and right hands to manage tracked objects.
            - Saving poses and timestamps if enabled.
            - Rendering tracking states to images if video rendering is enabled.

        :param scene_graph: The scene graph containing object and spatial information.
        :param scan_dir: Directory containing scan data and where results will be saved.
        :param video_path: Path where tracking visualization video will be saved.
        :param force_object_detection: Whether to force object detection in each frame.
        :param save_pose: Whether to save tracked poses to disk.
        :param tracker_offset: Optional offset to apply to tracked poses.
        :param use_hand: Whether to use hand positions for pose estimation.
        :return: None
        """
        from .tracking import track
        track(self, scan_dir, video_path, force_object_detection, save_pose, tracker_offset, use_hand)
    
    def visualize(self, centroids: bool = True, connections: bool = True, labels: bool = False) -> None:
        """
        Visualizes the scene graph in its current state, with customizable visualization options.

        :param centroids: Boolean flag to display centroids of objects in the scene. Defaults to True.
        :param connections: Boolean flag to display connections between nodes in the scene graph. Defaults to True.
        :param labels: Boolean flag to display labels for each node in the scene. Defaults to False.
        :return: None. The scene graph is visualized in an Open3D window.
        """
        
        # add the geometries to the scene
        geometries = self.scene_geometries(centroids, connections)

        gui.Application.instance.initialize()
        window = gui.Application.instance.create_window("Press <S> to capture a screenshot or <ESC> to quit the application.", 1024, 1024)
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        scene.scene.set_background(np.array([255.0, 255.0, 255.0, 1.0], dtype=np.float32))
        window.add_child(scene)

        for geometry, name, mat in geometries:
            scene.scene.add_geometry(name, geometry, mat)

        if geometries:
            bounds = geometries[0][0].get_axis_aligned_bounding_box()
            for geometry, _, _ in geometries[1:]:
                bounds += geometry.get_axis_aligned_bounding_box()
            scene.setup_camera(60, bounds, bounds.get_center())

        if labels:
            offset = np.array([0, 0, 0.01])
            for node in self.nodes.values():
                label = self.label_mapping.get(node.sem_label, "ID not found")
                point = node.centroid
                scene.add_3d_label(point + offset, label)
        
        # Set a key event callback to capture the screen
        def on_key_event(event):
            if event.type == gui.KeyEvent.Type.DOWN:
                if event.key == gui.KeyName.S:  # Capture screen when 'S' key is pressed
                    image = gui.Application.instance.render_to_image(scene.scene, 1024, 1024)
                    current_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
                    filename = f"screenshot_{current_time}.png"
                    image = gui.Application.instance.render_to_image(scene.scene, 1024, 1024)
                    o3d.io.write_image(filename, image)
                    time.sleep(0.5)
                    return True
                if event.key == gui.KeyName.ESCAPE:  # Quit application when 'ESC' key is pressed
                    gui.Application.instance.quit()
                    return True
            return False

        window.set_on_key(on_key_event)
        
        # Run the application
        gui.Application.instance.run()



