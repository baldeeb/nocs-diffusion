import trimesh


def load_clouds_from_obj_files(obj_files, points_per_cloud):
    objs = []
    for file in obj_files:
        mesh = trimesh.load(file, 
                            force='mesh',
                            skip_materials=True)
        points = mesh.sample(points_per_cloud)
        objs.append(points)
    return objs