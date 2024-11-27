import numpy as np
import open3d as o3d
import copy
import cv2
from projectaria_tools.core import calibration
import open3d.visualization.rendering as rendering


def project_pcd_to_image(
    tracking_points_3d: np.ndarray,
    img: np.ndarray,
    extrinsics: np.ndarray,
    scan_dir: str,
    vis: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D tracking points onto a 2D image plane using camera extrinsics.

    This function takes a set of 3D tracking points and projects them onto a 2D image based on the 
    provided extrinsic parameters. Optionally, visualization of the projected points on the image 
    can be enabled. The projected 2D points are returned, along with the 3D points that belong to these points.

    :param tracking_points_3d: Array of 3D points to be projected, typically of shape (N, 3).
    :param img: The 2D image as a numpy array onto which points will be projected.
    :param extrinsics: 4x4 matrix representing the extrinsic parameters for the projection.
    :param scan_dir: Path to the scan directory containing related data for the projection.
    :param vis: Flag indicating whether to visualize the projection on the image. Defaults to False.
    :return: Tuple containing:
        - Numpy array of projected 2D points on the image.
        - Numpy array of 3D points that correspond to the projected 2D points.
    """
    if vis:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    w, h = img.shape[:2]

    original_indices = np.arange(tracking_points_3d.shape[0])
    points_cam = np.dot(np.linalg.inv(extrinsics), np.hstack((tracking_points_3d, np.ones((tracking_points_3d.shape[0], 1)))).T)
    
    points_cam = points_cam.T[:, :3]

    pinhole = calibration.get_linear_camera_calibration(w, h, 611.43)

    points_list = []
    indices_list = []
    
    for idx, point in enumerate(points_cam):
        image_point = pinhole.project(point)
        if image_point is None:
            continue
        image_point_int = np.round(image_point).astype(int)
        x, y = image_point_int
        
        points_list.append([0, x, y])
        indices_list.append(original_indices[idx])
        if vis:
            img = cv2.circle(img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)  # Red color
    
    points_array = np.array(points_list, dtype=np.float32)
    indices_array = np.array(indices_list)

    tracking_points_2d, unique_indices = np.unique(points_array, axis=0, return_index=True)
    
    indices = indices_array[unique_indices]
    
    if vis:
        cv2.imwrite(scan_dir + "/tmp_mask.jpg", img)

    return tracking_points_2d, tracking_points_3d[indices].astype(np.float32)

def project_mesh_to_image(
    mask: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    pose: np.ndarray,
    camera_pose: np.ndarray
) -> tuple[np.ndarray, o3d.geometry.TriangleMesh]:
    """
    Projects a 3D mesh onto a 2D image plane, returning the modified mask and transformed mesh.

    This function uses the provided pose and camera pose to project a 3D mesh onto a 2D image plane, 
    updating the mask to reflect the projected mesh area. The function returns a tuple containing the 
    modified 2D mask and the transformed 3D mesh.

    :param mask: A 2D numpy array representing the initial mask, which will be updated with the projection.
    :param mesh: An Open3D TriangleMesh object representing the 3D mesh to project.
    :param pose: 4x4 numpy array representing the pose transformation matrix for the mesh.
    :param camera_pose: 4x4 numpy array representing the camera's pose in the world frame.
    :return: Tuple containing:
        - mask: 2D numpy array with the updated mask reflecting the projected mesh.
        - transformed_mesh: Transformed Open3D TriangleMesh object after projection.
    """
    object_mesh = copy.deepcopy(mesh)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(1296, 1296, 611.428, 611.428, 647.5, 647.5)

    inverted_mask = np.ones(mask.shape) - mask
    object_mesh.remove_vertices_by_mask(inverted_mask.astype(bool))
    return_mesh = copy.deepcopy(object_mesh)
    return_mesh.transform(np.linalg.inv(pose))


    render = rendering.OffscreenRenderer(1296, 1296)

    render.scene.set_background([0.0, 0.0, 0.0, 1.0])

    object_mesh.paint_uniform_color([1.0, 1.0, 1.0])

    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]
    mtl.shader = "defaultUnlit"

    render.scene.add_geometry("mesh", object_mesh, mtl)
    
    extrinsics = np.linalg.inv(camera_pose)
    render.setup_camera(intrinsics, extrinsics)
    
    render.scene.set_view_size(1296, 1296)

    projected_mask = np.asarray(render.render_to_image())
    projected_mask = projected_mask.astype(np.float32) / 255.0

    projected_mask = (projected_mask.sum(axis=-1)>0.5).astype(np.uint8)

    render.scene.clear_geometry()

    return projected_mask, return_mesh