# -*- coding: utf-8 -*-
"""
boundary_refinement.py

Created on Thu Apr 24 12:32:25 2025
Updated on Fri Apr 03 16:20:26 2026

@author: Chaim

Boundary refinement for RO-NOUS clustering.
Enhanced with image alignment-based structural element detection.
"""

import numpy as np
import logging
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

# Check if OpenCV is available
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("OpenCV is available for boundary refinement")
except ImportError:
    logger.warning("OpenCV not installed. Some boundary refinement features will be limited.")
    logger.warning("Install with: pip install opencv-python")

# Check if scikit-image is available for advanced boundary processing
SKIMAGE_AVAILABLE = False
try:
    from skimage import measure, morphology, filters
    SKIMAGE_AVAILABLE = True
    logger.info("scikit-image is available for advanced boundary processing")
except ImportError:
    logger.warning("scikit-image not installed. Advanced boundary processing will be limited.")
    logger.warning("Install with: pip install scikit-image")


def compute_alpha_shape(points, alpha=0.5):
    """
    Compute the alpha shape (concave hull) of a set of points.
    
    Args:
        points: Nx2 array of points
        alpha: Alpha value to control the detail level (smaller values -> more detailed)
        
    Returns:
        List of point indices representing the hull boundary
    """
    if len(points) < 4:
        # Not enough points for alpha shape, return convex hull
        try:
            hull = ConvexHull(points)
            return hull.vertices
        except:
            # If convex hull fails, return simple point indices
            return np.arange(len(points))
    
    # Compute the Delaunay triangulation
    try:
        tri = Delaunay(points)
        
        # Get circumradius for each triangle
        circumcenters = np.zeros((len(tri.simplices), 2))
        radius = np.zeros(len(tri.simplices))
        
        for i, simplex in enumerate(tri.simplices):
            # Get the points of the triangle
            pts = points[simplex]
            
            # Calculate circumcenter and circumradius
            a = np.linalg.norm(pts[0] - pts[1])
            b = np.linalg.norm(pts[1] - pts[2])
            c = np.linalg.norm(pts[2] - pts[0])
            s = (a + b + c) / 2.0
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            
            # Avoid division by zero
            if area < 1e-10:
                radius[i] = np.inf
            else:
                radius[i] = a * b * c / (4.0 * area)
        
        # Find triangles with radius less than alpha
        triangles = tri.simplices[radius < 1/alpha]
        
        # Build edge list
        edges = set()
        for tri in triangles:
            edges.add((tri[0], tri[1]))
            edges.add((tri[1], tri[2]))
            edges.add((tri[2], tri[0]))
        
        # Find boundary edges (those that appear only once)
        boundary_edges = set()
        for i, j in edges:
            if (j, i) not in edges:
                boundary_edges.add((i, j))
        
        # Order boundary edges to form a continuous path
        if not boundary_edges:
            hull = ConvexHull(points)
            return hull.vertices
        
        boundary_points = []
        edge_start, edge_end = next(iter(boundary_edges))
        boundary_points.append(edge_start)
        boundary_points.append(edge_end)
        boundary_edges.remove((edge_start, edge_end))
        
        while boundary_edges:
            current = boundary_points[-1]
            found = False
            
            # Find edge starting with current point
            to_remove = None
            for edge in boundary_edges:
                if edge[0] == current:
                    boundary_points.append(edge[1])
                    to_remove = edge
                    found = True
                    break
            
            if found:
                boundary_edges.remove(to_remove)
            else:
                # If no edge found, start new path
                if boundary_edges:
                    edge_start, edge_end = next(iter(boundary_edges))
                    boundary_points.append(edge_start)
                    boundary_points.append(edge_end)
                    boundary_edges.remove((edge_start, edge_end))
        
        return np.array(boundary_points)
        
    except Exception as e:
        logger.error(f"Error computing alpha shape: {str(e)}")
        try:
            # Fall back to convex hull
            hull = ConvexHull(points)
            return hull.vertices
        except:
            # If all else fails, return simple point indices
            return np.arange(len(points))


def create_boundary_mask(boundary, image_shape):
    """
    Create a binary mask from boundary points.
    
    Args:
        boundary: Array of (x,y) boundary points
        image_shape: Shape of the image (height, width)
        
    Returns:
        Binary mask of the boundary region
    """
    if not CV2_AVAILABLE:
        return None
    
    try:
        # Create empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Convert points to integer coordinates
        points = np.round(boundary).astype(np.int32)
        
        # Create a polygon
        polygon = points.reshape((-1, 1, 2))
        
        # Fill the polygon
        cv2.fillPoly(mask, [polygon], 255)
        
        return mask
    
    except Exception as e:
        logger.error(f"Error creating boundary mask: {str(e)}")
        return None


def compute_cluster_boundaries(points, labels, projected_points):
    """
    Compute 2D boundaries for each cluster using projected points.
    
    Args:
        points: Nx3 array of 3D points
        labels: Cluster labels
        projected_points: Nx2 array of projected 2D points
        
    Returns:
        Dictionary mapping cluster labels to boundary points
    """
    # Get unique labels (excluding noise)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]
    
    # Dictionary to store boundaries
    boundaries = {}
    
    for label in unique_labels:
        try:
            # Get mask for this cluster
            mask = (labels == label)
            
            # Skip if no points with this label
            if np.sum(mask) < 3:
                logger.warning(f"Cluster {label} has too few points for boundary calculation")
                continue
            
            # Get projected points for this cluster
            cluster_projected = projected_points[mask]
            
            # Compute alpha shape boundary
            boundary_indices = compute_alpha_shape(cluster_projected)
            boundary = cluster_projected[boundary_indices]
            
            # Store boundary
            boundaries[label] = boundary
            
            logger.info(f"Computed boundary for cluster {label} with {len(boundary)} points")
            
        except Exception as e:
            logger.error(f"Error computing boundary for cluster {label}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    return boundaries


def identify_edge_regions(edge_map, dilate_radius=5):
    """
    Identify regions near edges in the image.
    
    Args:
        edge_map: Binary edge map
        dilate_radius: Radius for edge dilation
        
    Returns:
        Dilated edge mask
    """
    if not CV2_AVAILABLE:
        return None
    
    try:
        # Dilate edges to create regions of interest
        kernel = np.ones((dilate_radius, dilate_radius), np.uint8)
        dilated_edges = cv2.dilate(edge_map, kernel, iterations=1)
        
        return dilated_edges
    
    except Exception as e:
        logger.error(f"Error identifying edge regions: {str(e)}")
        return edge_map


def find_potential_cluster_points(all_points, filtered_points, projected_points, 
                                 filtered_projected, edge_regions, labels, max_distance=10):
    """
    Find potential cluster points that were filtered but are near edges.
    
    Args:
        all_points: All original 3D points
        filtered_points: Filtered out 3D points
        projected_points: Projected 2D points for clustered points
        filtered_projected: Projected 2D points for filtered points
        edge_regions: Binary mask of edge regions
        max_distance: Maximum distance for point recovery
        
    Returns:
        Dictionary of {label: [indices to recover]}
    """
    if not CV2_AVAILABLE or edge_regions is None:
        return {}
    
    # Dictionary to store potential points for each cluster
    potential_points = {}
    
    try:
        # Find filtered points that are in edge regions
        edge_point_mask = np.zeros(len(filtered_projected), dtype=bool)
        
        for i, (x, y) in enumerate(filtered_projected):
            px, py = int(round(x)), int(round(y))
            
            # Skip if outside image bounds
            if px < 0 or px >= edge_regions.shape[1] or py < 0 or py >= edge_regions.shape[0]:
                continue
            
            # Check if point is in edge region
            if edge_regions[py, px] > 0:
                edge_point_mask[i] = True
        
        # Get filtered points in edge regions
        edge_points = filtered_projected[edge_point_mask]
        edge_point_indices = np.where(edge_point_mask)[0]
        
        # For each edge point, find nearest clustered point
        if len(edge_points) > 0 and len(projected_points) > 0:
            # Using NearestNeighbors for efficient nearest neighbor search
            nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(projected_points)
            distances, indices = nn.kneighbors(edge_points)
            
            # For each filtered point in edge regions, find nearest clustered point
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if dist[0] <= max_distance:
                    original_idx = edge_point_indices[i]
                    nearest_point_idx = idx[0]
                    
                    # Get label of nearest point
                    label = labels[nearest_point_idx]
                    
                    # Add to potential points for this label
                    if label not in potential_points:
                        potential_points[label] = []
                    
                    potential_points[label].append(original_idx)
        
        # Log results
        total_recovered = sum(len(indices) for indices in potential_points.values())
        logger.info(f"Found {total_recovered} potential points to recover near edges")
        
        return potential_points
    
    except Exception as e:
        logger.error(f"Error finding potential cluster points: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def recover_filtered_points(all_points, labels, filtered_indices, potential_points):
    """
    Recover filtered points and add them to appropriate clusters.
    
    Args:
        all_points: All original 3D points
        labels: Current cluster labels
        filtered_indices: Indices of filtered points
        potential_points: Dictionary of {label: [indices of filtered points to recover]}
        
    Returns:
        Updated labels for all points
    """
    # Create updated labels array initialized with noise label (-1)
    updated_labels = np.full(len(all_points), -1)
    
    # Copy existing labels
    valid_indices = np.where(labels != -1)[0]
    updated_labels[valid_indices] = labels[valid_indices]
    
    # Add recovered points
    for label, indices in potential_points.items():
        for idx in indices:
            # Convert filtered index to original index
            original_idx = filtered_indices[idx]
            updated_labels[original_idx] = label
    
    # Count recovered points
    recovered_count = sum(len(indices) for indices in potential_points.values())
    logger.info(f"Recovered {recovered_count} points ({recovered_count/len(all_points)*100:.2f}%)")
    
    return updated_labels


def refine_clusters(all_points, labels, filtered_indices, projected_points, 
                  edge_map, max_distance=10, dilate_radius=5):
    """
    Refine clusters by recovering filtered points near edges.
    
    Args:
        all_points: All original 3D points
        labels: Current cluster labels for non-filtered points
        filtered_indices: Indices of filtered points
        projected_points: Projected 2D points for non-filtered points
        edge_map: Edge detection result from OpenCV
        max_distance: Maximum distance for point recovery
        dilate_radius: Radius for edge region dilation
        
    Returns:
        Updated labels for all points
    """
    if not CV2_AVAILABLE or edge_map is None:
        return labels
    
    # Get filtered points
    filtered_points = all_points[filtered_indices]
    
    # Project filtered points to 2D
    try:
        from . import opencv_integration
    except ImportError:
        try:
            import opencv_integration
        except ImportError:
            logger.error("Could not import opencv_integration module")
            return labels
    
    # Use the same projection parameters as for the original points
    filtered_projected = opencv_integration.project_3d_to_2d(
        filtered_points,
        camera_position=(-47,28,-20.0),  # Default values, should be passed from caller
        camera_target=(-25.1,123.8,-28.3),
        focal_length=27.5,
        field_of_view=66,
        image_size=(480, 864)
    )
    
    # Identify regions near edges
    edge_regions = identify_edge_regions(edge_map, dilate_radius)
    
    # Find potential points to recover
    potential_points = find_potential_cluster_points(
        all_points, filtered_points, projected_points, 
        filtered_projected, edge_regions, labels, max_distance
    )
    
    # Recover filtered points
    updated_labels = recover_filtered_points(
        all_points, labels, filtered_indices, potential_points
    )
    
    return updated_labels


def refine_clusters_with_image_features(all_points, labels, filtered_indices, 
                                      original_frame, edge_features=None, color_regions=None):
    """
    Refine clusters using image features to recover structural elements.
    Enhanced method that uses image alignment.
    
    Args:
        all_points: All original 3D points
        labels: Current cluster labels
        filtered_indices: Indices of filtered points
        original_frame: Original video frame
        edge_features: Edge detection results (optional)
        color_regions: Color region segmentation (optional)
        
    Returns:
        Updated labels with reclaimed structural elements
    """
    # Try to import image_alignment module
    try:
        import image_alignment
    except ImportError:
        try:
            from . import image_alignment
        except ImportError:
            logger.warning("Could not import image_alignment module. Using standard refinement.")
            # Fall back to standard refinement
            try:
                from . import opencv_integration
            except ImportError:
                try:
                    import opencv_integration
                except ImportError:
                    logger.error("Could not import opencv_integration module")
                    return labels
                    
            # Project points to 2D
            projected_points = opencv_integration.project_3d_to_2d(
                all_points,
                camera_position=(-47,28,-20.0),
                camera_target=(25.1,123.8,-28.3),
                focal_length=27.5,
                field_of_view=66,
                image_size=(original_frame.shape[1], original_frame.shape[0])
            )
            
            # Extract edge features if not provided
            if edge_features is None and CV2_AVAILABLE:
                gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edge_features = cv2.Canny(blurred, 50, 150)
            
            # Use standard refinement
            return refine_clusters(
                all_points, labels, filtered_indices, projected_points, 
                edge_features, max_distance=10, dilate_radius=5
            )
    
    try:
        # Try to import opencv_integration
        try:
            from . import opencv_integration
        except ImportError:
            try:
                import opencv_integration
            except ImportError:
                logger.error("Could not import opencv_integration module")
                return labels
        
        if not CV2_AVAILABLE or original_frame is None:
            logger.error("OpenCV not available or original_frame is None")
            return labels
        
        # Project points to 2D
        projected_points = opencv_integration.project_3d_to_2d(
            all_points,
            camera_position=(-47,28,-20.0),  # Default values, should be passed from caller
            camera_target=(25.1,123.8,-28.3),
            focal_length=27.5,
            field_of_view=66,
            image_size=(original_frame.shape[1], original_frame.shape[0])
        )
        
        # Create point cloud visualization
        point_cloud_vis = image_alignment.create_point_cloud_visualization(
            all_points, labels, projected_points,
            image_size=(original_frame.shape[1], original_frame.shape[0])
        )
        
        # Apply alignment with default parameters
        # Note: In a real implementation, the parameters should be passed from the caller
        alignment_params = {'dx': 52, 'dy': -74, 'scale_x': 1.1, 'scale_y': 1.1}
        
        aligned_point_cloud, overlay, color_mask = image_alignment.align_point_cloud_to_frame(
            original_frame, 
            point_cloud_vis,
            alignment_params['dx'], 
            alignment_params['dy'],
            alignment_params['scale_x'], 
            alignment_params['scale_y']
        )
        
        # If edge features not provided, extract them
        if edge_features is None:
            edge_features = opencv_integration.extract_lines_directly_RECOMMENDED(original_frame)
        
        # Detect structural elements
        structural_mask = image_alignment.detect_structural_elements(
            edge_features, color_mask if color_mask is not None else np.zeros_like(edge_features)
        )
        
        # Match structural elements to filtered points
        reclaimed_indices = image_alignment.match_structural_elements_to_3d(
            structural_mask, 
            aligned_point_cloud, 
            all_points, 
            projected_points, 
            filtered_indices
        )
        
        # Update clustering to include reclaimed points
        updated_labels = image_alignment.recluster_with_reclaimed_points(
            all_points, labels, reclaimed_indices
        )
        
        return updated_labels
    
    except Exception as e:
        logger.error(f"Error refining clusters with image features: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return labels


def enhance_cluster_boundaries_with_alignment(all_points, labels, original_frame, 
                                           camera_parameters, alignment_params=None):
    """
    Enhance cluster boundaries using alignment and edge detection.
    
    Args:
        all_points: All original 3D points
        labels: Current cluster labels
        original_frame: Original video frame
        camera_parameters: Dictionary with camera parameters
        alignment_params: Optional alignment parameters
        
    Returns:
        Updated labels with enhanced boundaries
    """
    # Import required modules
    try:
        import image_alignment
        import opencv_integration
    except ImportError:
        try:
            from . import image_alignment
            from . import opencv_integration
        except ImportError:
            logger.error("Could not import required modules")
            return labels
    
    if not CV2_AVAILABLE or original_frame is None:
        logger.error("OpenCV not available or original_frame is None")
        return labels
    
    try:
        # Project points to 2D
        projected_points = opencv_integration.project_3d_to_2d(
            all_points,
            camera_position=camera_parameters.get('position', (-47,28,-20.0)),
            camera_target=camera_parameters.get('target', (-25.1,123.8,-28.3)),
            focal_length=camera_parameters.get('focal_length', 27.5),
            field_of_view=camera_parameters.get('field_of_view', 66),
            image_size=(original_frame.shape[1], original_frame.shape[0])
        )
        
        # Create point cloud visualization
        point_cloud_vis = image_alignment.create_point_cloud_visualization(
            all_points, labels, projected_points,
            image_size=(original_frame.shape[1], original_frame.shape[0])
        )
        
        # Find optimal alignment parameters if not provided
        if alignment_params is None:
            alignment_params = image_alignment.find_optimal_alignment_parameters(
                original_frame, point_cloud_vis
            )
        
        # Apply alignment
        aligned_point_cloud, overlay, color_mask = image_alignment.align_point_cloud_to_frame(
            original_frame, 
            point_cloud_vis,
            alignment_params['dx'], 
            alignment_params['dy'],
            alignment_params['scale_x'], 
            alignment_params['scale_y']
        )
        
        # Extract edge features
        edge_map = image_alignment.extract_edge_features(original_frame, method='combined')
        
        # Identify edge regions
        edge_regions = identify_edge_regions(edge_map, dilate_radius=5)
        
        # Compute boundaries for existing clusters
        boundaries = compute_cluster_boundaries(all_points, labels, projected_points)
        
        # Enhance boundaries based on edge regions
        enhanced_labels = labels.copy()
        
        # For each cluster, check if points that weren't assigned to it should be
        for cluster_label, boundary in boundaries.items():
            # Create a mask of the boundary
            boundary_mask = create_boundary_mask(boundary, (original_frame.shape[0], original_frame.shape[1]))
            
            if boundary_mask is None:
                continue
            
            # Find intersection of boundary and edge regions
            boundary_edges = cv2.bitwise_and(boundary_mask, edge_regions)
            
            # Find points that project to these regions but aren't in the cluster
            for i, (x, y) in enumerate(projected_points):
                px, py = int(round(x)), int(round(y))
                
                # Skip if outside image bounds
                if px < 0 or px >= boundary_edges.shape[1] or py < 0 or py >= boundary_edges.shape[0]:
                    continue
                
                # Check if point is in boundary edge region but not in cluster
                if boundary_edges[py, px] > 0 and labels[i] != cluster_label:
                    # If point is unlabeled or in a different cluster with lower confidence
                    if labels[i] == -1:
                        # Add to this cluster
                        enhanced_labels[i] = cluster_label
        
        # Count enhanced points
        enhanced_count = np.sum(enhanced_labels != labels)
        logger.info(f"Enhanced {enhanced_count} boundary points using alignment and edge detection")
        
        return enhanced_labels
    
    except Exception as e:
        logger.error(f"Error enhancing cluster boundaries: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return labels