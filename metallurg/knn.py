def create_knn(filename):
    import cv2

    fs_read = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    node_read = fs_read.getNode('opencv_ml_knn')

    knn = cv2.ml.KNearest_create()
    knn.read(node_read)

    fs_read.release()

    return knn
