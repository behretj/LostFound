import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from typing import Optional
from src import compute_pose

class ObjectNode:
    """
    Represents a generic object within a 3D scene, storing properties such as geometry, 
    semantic information, and state attributes.

    The ObjectNode class serves as a base for other scene objects, providing properties 
    like centroid calculation, color, confidence, and a convex hull tree for spatial queries.
    This class can be extended to add specific functionalities for different types of objects.

    Attributes:
        object_id (int): Unique identifier for the object.
        centroid (np.ndarray): The centroid of the object, computed from the provided 3D points.
        points (np.ndarray): Array of 3D points defining the object's geometry.
        sem_label (str): Semantic label categorizing the object (e.g., "drawer," "table").
        color (tuple): RGB color representation of the object.
        movable (bool): Indicates if the object is movable. Defaults to True.
        confidence (float, optional): Confidence score associated with the detection.
        visible (bool): Indicates if the object is currently visible in the scene. Defaults to True.
        tracking_points (list): Key points used for tracking the object's position.
        mesh_mask (np.ndarray): Binary mask representing the object's mesh.
        hull_tree (spatial.KDTree): Spatial KD-tree structure for the object's convex hull.
        pose (np.ndarray): Estimated pose matrix for the object based on its points and centroid.
    """

    def __init__(
        self,
        object_id: int,
        color: tuple,
        sem_label: str,
        points: np.ndarray,
        tracking_points: list,
        mesh_mask: np.ndarray,
        confidence: Optional[float] = None,
        movable: bool = True
    ):
        """
        Initializes an ObjectNode with specified attributes, computes its centroid. Generic building block for scene graph.

        :param object_id: Unique identifier for the object.
        :param color: RGB color tuple representing the object.
        :param sem_label: Semantic label for the object.
        :param points: Array of 3D points representing the object's geometry.
        :param tracking_points: List of key points used for tracking the object's position.
        :param mesh_mask: Binary mask representing the object's mesh.
        :param confidence: Confidence score of the detection. Defaults to None.
        :param movable: Flag indicating if the object is movable. Defaults to True.
        """
        self.object_id = object_id
        self.centroid = np.mean(points, axis=0)
        self.points = points
        self.sem_label = sem_label
        self.color = color
        self.movable = movable
        self.confidence = confidence
        self.visible = True
        self.tracking_points = tracking_points
        self.mesh_mask = mesh_mask
        self.update_hull_tree()
        self.pose = compute_pose(self.points, self.centroid)
    
    def update_hull_tree(self):
        """
        Updates the convex hull tree for the object using a KD-tree.

        This method constructs or updates a KD-tree based on the object's 3D points, allowing for 
        efficient spatial queries within the object's convex hull.

        :return: None. Updates the `hull_tree` attribute in place.
        """
        self.hull_tree = KDTree(self.points)
    
    def transform(self, transformation: np.ndarray, force: bool = False) -> None:
        """
        Applies a transformation to the node.

        If the `force` flag is set to True, the transformation is 
        applied even if certain conditions might normally prevent it (e.g., immovable objects).

        :param transformation: A translation, a 3x3 rotation or a 4x4 homogeneous transformation matrix to apply to the node's points.
        :param force: Flag to force the transformation, regardless of the node's movable status. Defaults to False.
        :return: None. The node's points are modified in place.
        """
        if isinstance(transformation, np.ndarray):
            if transformation.shape == (3,):
                self.centroid += transformation
                self.points += transformation
                self.tracking_points += transformation
                self.pose[:3, 3] += transformation
                self.update_hull_tree()
            elif transformation.shape == (3, 3):
                self.points = np.dot(transformation, self.points.T).T
                self.centroid = np.dot(transformation, self.centroid)
                self.tracking_points = np.dot(transformation, self.tracking_points.T).T
                self.pose = np.dot(transformation, self.pose[:3, :3])
                self.update_hull_tree()
            elif transformation.shape == (4, 4):
                self.points = np.dot(transformation, np.vstack((self.points.T, np.ones(self.points.shape[0])))).T[:, :3]
                self.centroid = np.dot(transformation, np.append(self.centroid, 1))[:3]
                self.tracking_points = np.dot(transformation, np.vstack((self.tracking_points.T, np.ones(self.tracking_points.shape[0])))).T[:, :3]
                self.pose = np.dot(transformation, self.pose)
                self.update_hull_tree()
            else:
                raise ValueError("Invalid argument shape. Expected (3,) for translation, (3,3) for rotation, or (4,4) for homogeneous transformation.")
        else:
            raise TypeError("Invalid argument type. Expected numpy.ndarray.")


class DrawerNode(ObjectNode):
    """
    Represents a drawer in the 3D scene, inheriting properties and methods from ObjectNode.

    The DrawerNode class includes additional properties and methods specific to drawers, such as 
    plane segmentation, containment relationships, and sign checking. The plane segmentation helps 
    define the drawer's orientation and position in the scene, and `contains` tracks items related 
    to the drawer.

    Attributes:
        equation (tuple): Plane equation of the drawer derived via RANSAC segmentation.
        box (optional): 3D bounding box of the drawer, if applicable.
        belongs_to (optional): Parent object or relationship attribute.
        contains (list): List of objects contained within the drawer.
    """

    def __init__(
        self,
        object_id: int,
        color: tuple,
        sem_label: str,
        points: np.ndarray,
        tracking_points: list,
        mesh_mask: np.ndarray,
        confidence: float = 1.0,
        movable: bool = True
    ):
        """
        Initializes a DrawerNode with specified attributes and performs plane segmentation.

        This constructor initializes DrawerNode properties inherited from ObjectNode and performs 
        plane segmentation to determine the drawer's orientation in the 3D scene. Additional attributes 
        are set to manage the drawer's bounding box, relationships, and contained objects.

        :param object_id: Unique identifier for the drawer.
        :param color: RGB color tuple representing the drawer's color.
        :param sem_label: Semantic label for the drawer.
        :param points: Array of 3D points representing the drawer.
        :param tracking_points: List of key points used for tracking the drawer's position.
        :param mesh_mask: Binary mask representing the drawer's mesh.
        :param confidence: Confidence score of the detection. Defaults to 1.0.
        :param movable: Flag indicating if the drawer is movable. Defaults to True.
        """
        super().__init__(object_id, color, sem_label, points, tracking_points, mesh_mask, confidence, movable)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        self.equation, _ = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        self.box = None
        self.belongs_to = None
        self.contains = []
    
    def sign_check(self, point: np.ndarray) -> bool:
        """
        Determines whether a given point lies on the positive side of the object's plane.

        This method checks if the provided 3D point is located on the positive side of the plane 
        represented by the object's plane equation. It uses the dot product of the plane normal 
        with the point coordinates and an offset to make this determination.

        :param point: A 3D point as a numpy array to be checked against the plane equation.
        :return: True if the point lies on the positive side of the plane, False otherwise.
        """
        return np.dot(self.equation[:3], point) + self.equation[3] > 0
    
    def add_box(self, shelf_centroid: np.ndarray) -> None:
        """
        Adds a bounding box to indicate the drawer's spatial appearance.

        Adds the box attribute of the DrawerNode based on a heuristic by computing the intersection with a parallel
        plane (to the initially estimated Drawer plane) anchored at the shelf centroid.

        :param shelf_centroid: A 3D numpy array representing the centroid of the shelf, used as a reference.
        :return: None. The bounding box is added to the object's attributes.
        """
        intersection = self.compute_intersection(shelf_centroid)
        
        bbox_points = []
        for point in self.points:
            bbox_points.append(point)
            bbox_points.append(point + 2* (shelf_centroid - intersection))

        points = np.array(bbox_points)

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(points)
          
        self.box = tmp_pcd.get_minimal_oriented_bounding_box()
    
    def compute_intersection(self, ray_start: np.ndarray) -> np.ndarray | None:
        """
        This method calculates the intersection point of a ray, starting from `ray_start`, with the 
        plane defined by the object's equation.

        :param ray_start: A 3D numpy array representing the starting point of the ray.
        :return: A 3D numpy array representing the intersection point with the plane, or None if 
                the ray is parallel to the plane.
        """
        signed_distance = (np.dot(self.equation[:3], ray_start) + self.equation[3]) / np.linalg.norm(self.equation[:3])
        
        if signed_distance > 0:
            direction = -self.equation[:3]
        else:
            direction = self.equation[:3]

        numerator = - (np.dot(self.equation[:3], ray_start) + self.equation[3])
        denominator = np.dot(self.equation[:3], direction)

        if denominator == 0:
            print("The ray is parallel to the plane and does not intersect it.")
            return
        
        t = numerator / denominator
        intersection_point = ray_start + t * direction

        return intersection_point
    
    def transform(self, transformation: np.ndarray, force: bool = False) -> None:
        """
        Applies a transformation to the drawer node. The DrawerNode is restricted in a way that it can only be moved along the plane normal.
        Compared to the ObjectNode, the DrawerNode's box is also updated based on the transformation.
        If the `force` flag is set to True, arbitrary transformations are allowed.

        :param transformation: A translation, a 3x3 rotation or a 4x4 homogeneous transformation matrix to apply to the node's points.
        :param force: Flag to force the transformation, regardless of the node's movable status. Defaults to False.
        :return: None. The node's points are modified in place.
        """
        if force:
            super().transform(transformation)
            if isinstance(transformation, np.ndarray):
                if transformation.shape == (3,):
                    self.box.translate(transformation)
                elif transformation.shape == (4, 4):
                    translation = transformation[:3, 3]
                    rotation = transformation[:3, :3]
                    self.box = self.box.rotate(rotation, center=np.array([0, 0, 0]))
                    self.box.translate(translation)
        else:
            if isinstance(transformation, np.ndarray) and (transformation.shape == (3,) or transformation.shape == (4, 4)):
                normal = self.equation[:3]
                normal /= np.linalg.norm(normal)
                new_location = np.dot(transformation, np.append(self.centroid, 1))[:3] - self.centroid
                translation = np.dot(new_location, normal) * normal
                self.centroid += translation
                self.points += translation
                self.tracking_points += translation
                self.pose[:3, 3] += translation
                self.box.translate(translation)
                self.update_hull_tree()
                for node in self.contains:
                    node.transform(translation)
            else:
                raise TypeError("Invalid argument type. Expected numpy.ndarray of shape (3,) or (4,4).")

class LightSwitchNode(ObjectNode):
    """
    Represents a light switch within a 3D scene, inheriting from ObjectNode.

    The LightSwitchNode class extends the functionality of ObjectNode to include attributes 
    specific to light switches, such as associated lamps, button count, interaction state, 
    and normal vector. This class enables modeling of light switches and their interactions 
    within the scene.

    Attributes:
        lamps (list): List of lamps associated with the light switch.
        button_count (int or None): Number of buttons on the light switch, if known.
        interaction (any): Represents interaction data or state, if applicable.
        normal (np.ndarray or None): Normal vector for the light switch's surface, if calculated.
    """

    def __init__(
        self,
        object_id: int,
        color: tuple,
        sem_label: str,
        points: np.ndarray,
        tracking_points: list,
        mesh_mask: np.ndarray,
        confidence: float = 1.0,
        movable: bool = True
    ):
        """
        Initializes a LightSwitchNode with specified attributes and additional light switch properties.

        This constructor sets up the base ObjectNode properties and initializes additional attributes 
        relevant to light switches, such as an empty list for associated lamps, button count, 
        interaction state, and the surface normal vector.

        :param object_id: Unique identifier for the light switch.
        :param color: RGB color tuple representing the light switch.
        :param sem_label: Semantic label for the light switch.
        :param points: Array of 3D points representing the light switch's geometry.
        :param tracking_points: List of key points used for tracking the light switch's position.
        :param mesh_mask: Binary mask representing the light switch's mesh.
        :param confidence: Confidence score of the detection. Defaults to 1.0.
        :param movable: Flag indicating if the light switch is movable. Defaults to True.
        """
        super().__init__(object_id, color, sem_label, points, tracking_points, mesh_mask, confidence, movable)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        self.equation, _ = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        self.lamps = []
        self.button_count = None
        self.interaction = None
    
    def transform(self, transformation, force=False):
        # light switch can't be moved, only if the whole scene is moved
        if force:
            super().transform(transformation)

    def add_lamp(self, lamp_id: int) -> None:
        """
        Associates a lamp with this light switch.

        :param lamp_id: Unique identifier of the lamp associated with this light switch.
        :return: None.
        """
        self.lamps.append(lamp_id)

    def set_button_count(self, count: int) -> None:
        """
        Sets the number of buttons on the light switch.

        :param count: Integer specifying the number of buttons.
        :return: None.
        """
        self.button_count = count

    def set_interaction_type(self, interaction: str) -> None:
        """
        Defines the interaction type for the light switch.

        :param interaction: String representing the type of interaction.
        :return: None.
        """
        self.interaction = interaction

    def set_normal(self, equation: np.ndarray) -> None:
        """
        Sets the surface normal for the light switch using a plane equation.

        :param equation: Array representing the plane equation of the light switch's surface.
        :return: None.
        """
        self.equation = equation