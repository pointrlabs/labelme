
import numpy as np
import cv2
from PIL import Image
import math
from skimage.filters import gaussian
from scipy import ndimage as ndi
import visvalingamwyatt as vw
from tqdm import tqdm 

def executeSnake(imagePath, initialPolygon, obstaclePolygons, ignorePolygons, down_sampling_ratio, simplification_threshold=100):

    print(imagePath)

    img = np.array(Image.open(imagePath).convert("L"))

    # Add ignore regions
    for ignore in ignorePolygons:
        contours = np.asarray(ignore, dtype=np.int32)
        cv2.fillPoly(img, pts = [contours], color =(255,255,255))

    # Add obstacles 
    for obstacle in obstaclePolygons:
        contours = np.asarray(obstacle, dtype=np.int32)
        cv2.fillPoly(img, pts = [contours], color =(0,0,0))

    # Draw initial contour
    imgCopy = img.copy()
    contour_initial = []
    for vertex in initialPolygon:
        contour_initial.append([vertex])
    contour_initial = np.asarray(contour_initial, dtype=np.int32)
    cv2.drawContours(imgCopy, contour_initial, -1, (0,0,0), 20)

    img = Image.fromarray(img).convert('L')

    img_init_level = np.zeros((img.size[1],img.size[0]), dtype=np.uint8)

    contours = np.asarray(initialPolygon, dtype=np.int32)
    cv2.fillPoly(img_init_level, pts = [contours], color =(255,255,255))    
    
    img_init_level = Image.fromarray(img_init_level).convert("L")
    

    if not down_sampling_ratio:
        down_sampling_ratio = math.floor(max(img.size) / 1024)
    
    img = np.asarray(img.resize((img.size[0] // down_sampling_ratio, img.size[1] // down_sampling_ratio))) / 255.0
    img_init_level = np.asarray(img_init_level.resize((img_init_level.size[0] // down_sampling_ratio, img_init_level.size[1] // down_sampling_ratio)), dtype=np.bool)

    eps = 1e-5
    img[img < img.max()-eps] = 0

    binary_img = morphological_geodesic_active_contour(img, iterations=1000,
                                            init_level_set=img_init_level,
                                            smoothing=0, threshold=0.2,
                                            balloon=-1, early_stop_mode=True)
    binary_img = np.uint8(binary_img)
    binary_img *= 255

    binary_img = Image.fromarray(binary_img).convert("L")

    binary_img = np.asarray(binary_img.resize((binary_img.size[0] * down_sampling_ratio, binary_img.size[1] * down_sampling_ratio)))

    contour_raw, _ = find_contours_hull(binary_img)
    
    if simplification_threshold == 0:
        contour = contour_raw
    else:
        contour = simplify_polygon(contour_raw, [{'method_name': 'visvalingam', 'threshold': simplification_threshold}])

    img_tmp = np.asarray(binary_img, dtype=np.uint8)
    print('Snake converged!')
    return contour

def morphological_geodesic_active_contour(gimage, iterations,
                                          init_level_set='circle', smoothing=1,
                                          threshold='auto', balloon=0, early_stop_mode = False,
                                          iter_callback=lambda x: None):
    """Morphological Geodesic Active Contours (MorphGAC).

    Geodesic active contours implemented with morphological operators. It can
    be used to segment objects with visible but noisy, cluttered, broken
    borders.

    Parameters
    ----------
    gimage : (M, N) or (L, M, N) array
        Preprocessed image or volume to be segmented. This is very rarely the
        original image. Instead, this is usually a preprocessed version of the
        original image that enhances and highlights the borders (or other
        structures) of the object to segment.
        `morphological_geodesic_active_contour` will try to stop the contour
        evolution in areas where `gimage` is small. See
        `morphsnakes.inverse_gaussian_gradient` as an example function to
        perform this preprocessing. Note that the quality of
        `morphological_geodesic_active_contour` might greatly depend on this
        preprocessing.
    iterations : uint
        Number of iterations to run.
    init_level_set : str, (M, N) array, or (L, M, N) array
        Initial level set. If an array is given, it will be binarized and used
        as the initial level set. If a string is given, it defines the method
        to generate a reasonable initial level set with the shape of the
        `image`. Accepted values are 'checkerboard' and 'circle'. See the
        documentation of `checkerboard_level_set` and `circle_level_set`
        respectively for details about how these level sets are created.
    smoothing : uint, optional
        Number of times the smoothing operator is applied per iteration.
        Reasonable values are around 1-4. Larger values lead to smoother
        segmentations.
    threshold : float, optional
        Areas of the image with a value smaller than this threshold will be
        considered borders. The evolution of the contour will stop in this
        areas.
    balloon : float, optional
        Balloon force to guide the contour in non-informative areas of the
        image, i.e., areas where the gradient of the image is too small to push
        the contour towards a border. A negative value will shrink the contour,
        while a positive value will expand the contour in these areas. Setting
        this to zero will disable the balloon force.
    iter_callback : function, optional
        If given, this function is called once per iteration with the current
        level set as the only argument. This is useful for debugging or for
        plotting intermediate results during the evolution.

    Returns
    -------
    out : (M, N) or (L, M, N) array
        Final segmentation (i.e., the final level set)

    See also
    --------
    inverse_gaussian_gradient, circle_level_set, checkerboard_level_set

    Notes
    -----

    This is a version of the Geodesic Active Contours (GAC) algorithm that uses
    morphological operators instead of solving partial differential equations
    (PDEs) for the evolution of the contour. The set of morphological operators
    used in this algorithm are proved to be infinitesimally equivalent to the
    GAC PDEs (see [1]_). However, morphological operators are do not suffer
    from the numerical stability issues typically found in PDEs (e.g., it is
    not necessary to find the right time step for the evolution), and are
    computationally faster.

    The algorithm and its theoretical derivation are described in [1]_.

    References
    ----------
    .. [1] A Morphological Approach to Curvature-based Evolution of Curves and
           Surfaces, Pablo Márquez-Neila, Luis Baumela, Luis Álvarez. In IEEE
           Transactions on Pattern Analysis and Machine Intelligence (PAMI),
           2014, DOI 10.1109/TPAMI.2013.106
    """

    image = gimage
    init_level_set = _init_level_set(init_level_set, image.shape)

    _check_input(image, init_level_set)

    if threshold == 'auto':
        threshold = np.percentile(image, 40)

    structure = np.ones((3,) * len(image.shape), dtype=np.int8)
    dimage = np.gradient(image)
    # threshold_mask = image > threshold
    if balloon != 0:
        threshold_mask_balloon = image > threshold / np.abs(balloon)

    u = np.int8(init_level_set > 0)

    iter_callback(u)

    cur_u_length = 0
    cur_iter = 0
    for _ in tqdm(range(iterations)):

        # Balloon
        if balloon > 0:
            aux = ndi.binary_dilation(u, structure)
        elif balloon < 0:
            aux = ndi.binary_erosion(u, structure)
        if balloon != 0:
            u[threshold_mask_balloon] = aux[threshold_mask_balloon]

        # Image attachment
        aux = np.zeros_like(image)
        du = np.gradient(u)
        for el1, el2 in zip(dimage, du):
            aux += el1 * el2
        u[aux > 0] = 1
        u[aux < 0] = 0

        # Smoothing
        for _ in range(smoothing):
            u = _curvop(u)

        iter_callback(u)

        if early_stop_mode:
            delta_u = (cur_u_length / np.sum(u)) - 1
            if np.abs(delta_u) < 1e-5:
                break
            cur_iter += 1
            cur_u_length = np.sum(u)
    return u

def _init_level_set(init_level_set, image_shape):
    """Auxiliary function for initializing level sets with a string.

    If `init_level_set` is not a string, it is returned as is.
    """
    if isinstance(init_level_set, str):
        if init_level_set == 'checkerboard':
            res = checkerboard_level_set(image_shape)
        elif init_level_set == 'circle':
            res = circle_level_set(image_shape)
        elif init_level_set == 'ellipsoid':
            res = ellipsoid_level_set(image_shape)
        else:
            raise ValueError("`init_level_set` not in "
                             "['checkerboard', 'circle', 'ellipsoid']")
    else:
        res = init_level_set
    return res

def _check_input(image, init_level_set):
    """Check that shapes of `image` and `init_level_set` match."""
    if image.ndim not in [2, 3]:
        raise ValueError("`image` must be a 2 or 3-dimensional array.")

    if len(image.shape) != len(init_level_set.shape):
        raise ValueError("The dimensions of the initial level set do not "
                         "match the dimensions of the image.")


def find_contours_hull(binary_img, auto_num_detection=False, scale_coords = None, img=None, area_threshold=None, ML=False, ML_preprocessing=True):
    """ Finds and returns contours and hull points of the biggest blop in the binary image

    Parameters
    ----------
    binary_img : (M, N) binary image array
                 dtype = uint8
                 values = 0 or 255

    Returns
    -------
    out : (M, N) and (M, N) array
    """
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        contours, hierarchy = cv2.findContours(255 * np.ones(binary_img.shape, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    hull = []
    contour_areas = np.zeros(len(contours))
    max_area_contour = 0
    ind_max_area_contour = 0
    cur_area_contour = 0
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))
        cur_area_contour = cv2.contourArea(contours[i])
        contour_areas[i] = cur_area_contour
        if scale_coords:
            contours[i][:,:,0] = contours[i][:,:,0] * scale_coords
            contours[i][:,:,1] = contours[i][:,:,1] * scale_coords
            hull[i][:,:,0] = hull[i][:,:,0] * scale_coords
            hull[i][:,:,1] = hull[i][:,:,1] * scale_coords

        if cur_area_contour > max_area_contour:
            max_area_contour = cur_area_contour
            ind_max_area_contour = i

    if auto_num_detection:
        if True:
            contours_to_keep = []
            contours_to_filt_out = []
            for i in range(len(contours)):
                binary_mask = np.zeros([binary_img.shape[0], binary_img.shape[1]], dtype=np.int8)

                contours[i][:, :, 0] = contours[i][:, :, 0] / scale_coords
                contours[i][:, :, 1] = contours[i][:, :, 1] / scale_coords
                binary_mask = cv2.fillConvexPoly(binary_mask, contours[i], 1)
                contours[i][:, :, 0] = contours[i][:, :, 0] * scale_coords
                contours[i][:, :, 1] = contours[i][:, :, 1] * scale_coords

                #binary_mask = cv2.drawContours(binary_mask, contours, i, color=(255,255,255), thickness=cv2.FILLED)
                #plt.imshow(binary_mask)
                intersection_mask = binary_img/255 * binary_mask
                #print(np.sum(intersection_mask))
                w, h = binary_img.shape
                if np.sum(intersection_mask)>w*h*0.0025:
                    #plt.imshow(intersection_mask)
                    contours_to_keep.append(i)
                else:
                    contours_to_filt_out.append(i)
            contours_tmp = [contours[x] for x in contours_to_keep]
            contours_filt_out = [contours[x] for x in contours_to_filt_out]
            contours = contours_tmp
            hull = [hull[x] for x in contours_to_keep]

        if ML:
            if img is not None:
                if area_threshold:
                    X, ind_return = feature_extractor_contours(contours, hull, img=img, area_threshold=area_threshold)
                    contours = [contours[x] for x in ind_return]
                    hull = [hull[x] for x in ind_return]
                else:
                    X = feature_extractor_contours(contours, hull, img=img)
            else:
                X = feature_extractor_contours(contours, hull)

            pca = decomposition.PCA(n_components=4)
            pca.fit(X)
            X = pca.transform(X)

            kmeans = KMeans(n_clusters=2).fit(X)
            prediction = kmeans.labels_

            contours_to_return = list(np.argwhere(np.logical_or(prediction == 1, prediction == 1)).flatten())
            contours_to_filt_out = list(np.argwhere(prediction == 0).flatten())
            return [contours[x] for x in contours_to_return], [hull[x] for x in contours_to_return], [contours[x] for x in contours_to_filt_out]
        else:
            return contours, hull, contours_filt_out
    else:
        return contours[ind_max_area_contour], hull[ind_max_area_contour]



def simplify_polygon(polygon, simplification_methods=[{'method_name': 'visvalingam', 'threshold': 25}]):
    for method in simplification_methods:
        if method["method_name"] == 'douglas_peucker':
            # OpenCV version > 4.5.3 doesn't accept tuple anymore. Need to extract nparray inside tuple obj
            if type(polygon) == tuple and len(polygon) > 0:
                polygon = polygon[0]
            polygon = cv2.approxPolyDP(polygon, epsilon=method['threshold'], closed=True)
        elif method["method_name"] == 'visvalingam':
            simplifier = vw.Simplifier(np.squeeze(polygon))
            polygon = simplifier.simplify(threshold=method['threshold'])
            polygon = np.expand_dims(polygon, axis=1)
            # polygon = del_duplicate_vertices(polygon)
    return polygon