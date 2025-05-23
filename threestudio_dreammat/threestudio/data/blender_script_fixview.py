import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import mathutils
from mathutils import Vector, Matrix
import numpy as np
import bpy
import pickle
import time

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pkl_path):
    # os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

parser = argparse.ArgumentParser()
parser.add_argument("--param_dir", type=str, required=True)
parser.add_argument("--env_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--seg_model_path", type=str)
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--camera_type", type=str, default='fixed')
parser.add_argument("--num_images", type=int, required=True)
parser.add_argument("--device", type=str, default='CUDA')
parser.add_argument("--test_seg", type=bool, default=False, help='Only render segmentation images (bool)')


argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)
print('===================', args.engine, '===================')

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"
render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"

render.resolution_percentage = 100
render.use_persistent_data = True
scene.cycles.device = "GPU"
scene.cycles.samples = 64
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.get_devices()

# Set the device_type
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = args.device # or "OPENCL"
bpy.context.scene.cycles.tile_size = 8192

def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x,y,z],-1) #

def set_camera_location(cam_pt):
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = cam_pt
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    return camera

def get_calibration_matrix_K_from_blender(camera):
    f_in_mm = camera.data.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camera.data.sensor_width
    sensor_height_in_mm = camera.data.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if camera.data.sensor_fit == 'VERTICAL':
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_u
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = np.asarray(((alpha_u, skew, u_0),
                    (0, alpha_v, v_0),
                    (0, 0, 1)),np.float32)
    return K


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    bpy.context.view_layer.update()
    location, rotation = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rotation.to_matrix())
    t = np.asarray(location)

    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R = R.T
    t = -R @ t
    R_world2cv = cam_rec @ R
    t_world2cv = cam_rec @ t
    t_world2cv  = -R_world2cv.T @ t_world2cv
    RT = np.concatenate([R_world2cv.T,t_world2cv[:,None]],1)
    RT = np.vstack([RT, np.array([0, 0, 0, 1])])
    return RT

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

def calc_view_matrix(cam):
    loc,rot = cam.matrix_world.decompose()[0:2]
    rot = rot.to_matrix()
    rot.transpose()
    return rot

def create_material():
    mat = bpy.data.materials.new('RM')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.0
    bsdf.inputs['Metallic'].default_value = 1.0
    bsdf.inputs['Specular'].default_value = 1.0
    bsdf.inputs['Transmission'].default_value = 0.0
    bsdf.inputs['Sheen Tint'].default_value = 0.0
    bsdf.inputs['Sheen'].default_value = 0.0
    bsdf.inputs['Clearcoat Roughness'].default_value = 0.0
    bsdf.inputs['IOR'].default_value = 1.0
    bsdf.inputs['Specular Tint'].default_value = 0.0
    bsdf.inputs['Anisotropic'].default_value = 0.0
    return mat
def create_group_view_normal():
    node_group = bpy.data.node_groups.new('View_Normal', 'CompositorNodeTree')

    input_node = node_group.nodes.new('NodeGroupInput')
    output_node = node_group.nodes.new('NodeGroupOutput')

    normal_input = node_group.inputs.new('NodeSocketVector', 'Normal')
    depth_input = node_group.inputs.new('NodeSocketFloat', 'Depth')

    vn0 = node_group.nodes.new("CompositorNodeNormal")
    vn1 = node_group.nodes.new("CompositorNodeNormal")
    vn2 = node_group.nodes.new("CompositorNodeNormal")
    vn0.label = "0"
    vn1.label = "1"
    vn2.label = "2"

    map0= node_group.nodes.new("CompositorNodeMath")
    map1= node_group.nodes.new("CompositorNodeMath")
    map2= node_group.nodes.new("CompositorNodeMath")
    map0.operation = "MULTIPLY_ADD"
    map1.operation = "MULTIPLY_ADD"
    map2.operation = "MULTIPLY_ADD"
    # only yz need to inverse
    map0.inputs[1].default_value = 0.5
    map0.inputs[2].default_value = 0.5
    map1.inputs[1].default_value = -0.5
    map1.inputs[2].default_value = 0.5
    map2.inputs[1].default_value = 0.5
    map2.inputs[2].default_value = 0.5

    # mask
    mask_lessthan = node_group.nodes.new("CompositorNodeMath")
    mask_inverse = node_group.nodes.new("CompositorNodeMath")
    mask_ma = node_group.nodes.new("CompositorNodeMath")

    mask_lessthan.operation = "LESS_THAN"
    mask_inverse.operation = "MULTIPLY_ADD"
    mask_ma.operation = "MULTIPLY_ADD"
     
    mask_lessthan.inputs[1].default_value = 100.0
    mask_inverse.inputs[1].default_value = -1.0
    mask_inverse.inputs[2].default_value = 1.0 
    mask_ma.inputs[1].default_value = -1.0
     
    node_group.links.new(input_node.outputs['Normal'],vn0.inputs[0])
    node_group.links.new(input_node.outputs['Normal'],vn1.inputs[0])
    node_group.links.new(input_node.outputs['Normal'],vn2.inputs[0])

    node_group.links.new(vn0.outputs[1], map0.inputs[0])
    node_group.links.new(vn1.outputs[1], map1.inputs[0])
    #node_group.links.new(vn2.outputs[1], map0.inputs[0])

    node_group.links.new(vn2.outputs[1], mask_ma.inputs[0])
    node_group.links.new(input_node.outputs['Depth'], mask_lessthan.inputs[0])
    node_group.links.new(mask_lessthan.outputs[0], mask_inverse.inputs[0])
    node_group.links.new(mask_inverse.outputs[0], mask_ma.inputs[2])
    node_group.links.new(mask_ma.outputs[0], map2.inputs[0])

    #output
    combine = node_group.nodes.new("CompositorNodeCombineXYZ")
    node_group.links.new(map0.outputs[0],combine.inputs[0])
    node_group.links.new(map1.outputs[0],combine.inputs[1])
    node_group.links.new(map2.outputs[0],combine.inputs[2])
    node_group.outputs.new('NodeSocketVector', 'viewNormal')
    node_group.links.new(combine.outputs[0],output_node.inputs[0])

    group_node = bpy.context.scene.node_tree.nodes.new('CompositorNodeGroup')
    bpy.context.scene.node_tree.nodes[-1].node_tree = node_group
    return (group_node,vn0,vn1,vn2)

def is_smooth(polys): 
    smoothed=False 
    for poly in polys: 
        if not smoothed: 
            smoothed = poly.use_smooth 
    return smoothed 


def save_images(object_file: str) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    lightdir = os.path.join(args.output_dir,"light")
    promptdir=os.path.join(args.output_dir)
    depthdir=os.path.join(args.output_dir,"depth")
    normaldir=os.path.join(args.output_dir,"normal")
    segdir = os.path.join(args.output_dir, "seg")
    if not args.test_seg:
        if not os.path.exists(lightdir): Path(lightdir).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(promptdir): Path(promptdir).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(depthdir): Path(depthdir).mkdir(exist_ok=True, parents=True)
        if not os.path.exists(normaldir): Path(normaldir).mkdir(exist_ok=True, parents=True)
    
    if not os.path.exists(segdir): Path(segdir).mkdir(exist_ok=True, parents=True)

    param = read_pickle(object_file)
    reset_scene()
    # load the object
    # create new 1st mesh 
    n_mesh_1 = bpy.data.meshes.new("new_mesh_1")
    # create new object and link 1st mesh to object
    obj_1 = bpy.data.objects.new("new_object_1",n_mesh_1)
    # link object to scene
    bpy.context.collection.objects.link(obj_1)
    # update mesh data from given points
    n_mesh_1.from_pydata(param['v_pos'],[],param['t_pos_idx'])
    n_mesh_1.update()

    # 세그멘테이션 처리 - 방법 2로 통일 (원본 메시의 정점 색상 사용)
    seg_obj = None
    if 'v_color' in param and obj_1 is not None:
        try:
            print("Applying vertex colors from pickle data to original mesh for segmentation")
            # 정점 색상 레이어 생성
            if not obj_1.data.vertex_colors:
                obj_1.data.vertex_colors.new(name="Col")
                
            color_layer = obj_1.data.vertex_colors["Col"]
            
            # 메시에 정점 색상 적용
            v_colors = param['v_color']
            
            # 적절한 형식인지 확인
            if v_colors.shape[1] >= 3:  # RGB 값 최소 필요
                mesh = obj_1.data
                # 각 폴리곤(face)를 순회
                for poly_idx, poly in enumerate(mesh.polygons):
                    # 해당 폴리곤을 구성하는 인덱스 순회
                    for loop_idx in range(poly.loop_start, poly.loop_start + poly.loop_total):
                        vertex_idx = mesh.loops[loop_idx].vertex_index
                        # v_colors에서 색상을 가져와 색상 레이어에 할당
                        if vertex_idx < len(v_colors):
                            color_layer.data[loop_idx].color = (
                                v_colors[vertex_idx][0],
                                v_colors[vertex_idx][1],
                                v_colors[vertex_idx][2],
                                1.0 if v_colors.shape[1] < 4 else v_colors[vertex_idx][3]
                            )
                
                # 원본 메시 복제하여 세그멘테이션 메시 생성
                bpy.ops.object.select_all(action='DESELECT')
                obj_1.select_set(True)
                bpy.context.view_layer.objects.active = obj_1
                bpy.ops.object.duplicate()
                
                # 복제된 객체 가져오기
                seg_obj = bpy.context.active_object
                seg_obj.name = "SegmentationMesh"
                
                # 세그멘테이션 마테리얼 생성
                seg_mat = bpy.data.materials.new(name="SegmentationMaterial")
                seg_mat.use_nodes = True
                seg_tree = seg_mat.node_tree
                seg_nodes = seg_tree.nodes
                
                # 기본 노드 제거
                for node in seg_nodes:
                    seg_nodes.remove(node)
                
                # Emission 노드와 Material Output 노드 추가
                emission_node = seg_nodes.new(type='ShaderNodeEmission')
                output_node = seg_nodes.new(type='ShaderNodeOutputMaterial')
                
                # 정점 색상 노드 추가
                vc_node = seg_nodes.new(type='ShaderNodeVertexColor')
                vc_node.layer_name = "Col"  # 생성한 정점 색상 레이어 사용
                
                # 노드 연결
                seg_tree.links.new(vc_node.outputs['Color'], emission_node.inputs['Color'])
                seg_tree.links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])
                
                # 복제된 객체에 세그멘테이션 마테리얼 할당
                if seg_obj.data.materials:
                    seg_obj.data.materials[0] = seg_mat
                else:
                    seg_obj.data.materials.append(seg_mat)
                
                # 세그멘테이션 객체 초기 렌더링 숨김
                seg_obj.hide_render = True
                
                print("Successfully set up segmentation using vertex colors from pickle data")
            else:
                print(f"Invalid vertex color data shape: {v_colors.shape}")
        except Exception as e:
            print(f"Error applying vertex colors from pickle: {e}")
            # 세그멘테이션 없이 계속 진행
    else:
        print("No vertex color data found in pickle, segmentation will not be available")

    # set smooth mesh
    for polygon in n_mesh_1.polygons:
       polygon.use_smooth = True
    obj_1.data.update()
    bpy.context.view_layer.objects.active = obj_1
    # soft edge, removing overlapping vertices
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.remove_doubles(use_sharp_edge_from_normals=True)
    bpy.ops.object.mode_set(mode='OBJECT')

    mat = create_material()
    for mesh in scene_meshes():
        mesh.active_material = mat

    # Create segmentation material if seg_obj exists
    seg_mat = None
    if seg_obj:
        seg_mat = bpy.data.materials.new(name="SegmentationMaterial")
        seg_mat.use_nodes = True
        seg_tree = seg_mat.node_tree
        seg_nodes = seg_tree.nodes
        # Clear default nodes
        for node in seg_nodes:
            seg_nodes.remove(node)
        # Add Emission node connected to Material Output
        emission_node = seg_nodes.new(type='ShaderNodeEmission')
        output_node = seg_nodes.new(type='ShaderNodeOutputMaterial')
        # Add Vertex Color node to get vertex colors
        # This assumes the GLB loader correctly creates a vertex color layer
        vc_node = seg_nodes.new(type='ShaderNodeVertexColor')
        # Try to use the first available vertex color layer name if the object has one
        if seg_obj and seg_obj.data.vertex_colors:
            vc_node.layer_name = seg_obj.data.vertex_colors[0].name
        else:
            # Fallback or default name if needed, though ideally the layer exists
            print("Warning: Segmentation object has no vertex color layer. Using default or first found.")
            # vc_node.layer_name = "Col" # Keep a fallback? Or rely on default?

        seg_tree.links.new(vc_node.outputs['Color'], emission_node.inputs['Color'])
        seg_tree.links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

        # Assign material to seg_obj
        if seg_obj.data.materials:
            seg_obj.data.materials[0] = seg_mat
        else:
            seg_obj.data.materials.append(seg_mat)

    # load env_map
    bpy.context.scene.world.use_nodes = True

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    
    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes['Background']

    map_path = args.env_dir
    env_map_prompts = []

    mat_list = [(0.0,0.0),(0.0,0.5),(0.0,1.0),(1.0,0.0),(1.0,0.5),(1.0,1.0)]
    env_texture_node = world_tree.nodes.new(type="ShaderNodeTexEnvironment")
    world_tree.links.new(env_texture_node.outputs[0], back_node.inputs[0])

    default_color_management = bpy.context.scene.view_settings.view_transform# = 'Filmic'
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    bpy.context.scene.view_layers["ViewLayer"].use_pass_mist = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
    render_layer = tree.nodes.new(type="CompositorNodeRLayers")
    composite_node = tree.nodes.new(type="CompositorNodeComposite")
    tree.nodes.active = composite_node

    # depth control Nodes
    subtract_node = tree.nodes.new(type="CompositorNodeMath")
    subtract_node.operation = "SUBTRACT"
    MAX_DIST = 100.0
    subtract_node.inputs[1].default_value = MAX_DIST
    lessthan_node = tree.nodes.new(type="CompositorNodeMath")
    lessthan_node.operation = "LESS_THAN"
    lessthan_node.inputs[1].default_value = 0.0
    tree.links.new(subtract_node.outputs[0], lessthan_node.inputs[0])

    multip_node_0 = tree.nodes.new(type="CompositorNodeMath")
    multip_node_1 = tree.nodes.new(type="CompositorNodeMath")
    divide_node = tree.nodes.new(type="CompositorNodeMath")
    multip_node_0.operation = "MULTIPLY"
    multip_node_1.operation = "MULTIPLY"
    divide_node.operation = "DIVIDE"
    NORMALIZING_FACTOR = 1000.0
    NORMALIZING_BIT_DEPTH = 65535.0
    multip_node_1.inputs[1].default_value = NORMALIZING_FACTOR
    divide_node.inputs[1].default_value = NORMALIZING_BIT_DEPTH
    tree.links.new(multip_node_0.outputs[0], multip_node_1.inputs[0])
    tree.links.new(multip_node_1.outputs[0], divide_node.inputs[0])

    tree.links.new(render_layer.outputs["Depth"], subtract_node.inputs[0])
    tree.links.new(render_layer.outputs["Depth"], multip_node_0.inputs[0])
    tree.links.new(lessthan_node.outputs[0],multip_node_0.inputs[1])
    output_depth_slot = divide_node.outputs[0]
    # output - divide_node.outputs[0]
    (node_group,vn0,vn1,vn2) = create_group_view_normal()
    tree.links.new(render_layer.outputs["Depth"], node_group.inputs['Depth'])
    tree.links.new(render_layer.outputs["Normal"], node_group.inputs['Normal'])
    
    cam_poses = param['c2w']

    # output render image
    camera = bpy.data.objects["Camera"]
    render.resolution_x = param['width']
    render.resolution_y = param['height']

    # Store original render settings to restore later
    original_color_mode = render.image_settings.color_mode
    original_color_depth = render.image_settings.color_depth
    original_view_transform = bpy.context.scene.view_settings.view_transform
    original_film_transparent = scene.render.film_transparent
    original_cycles_samples = scene.cycles.samples
    original_max_bounces = scene.cycles.max_bounces # Store original bounces

    # Store original world background
    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes['Background']
    original_bg_color = back_node.inputs['Color'].default_value[:]
    original_bg_strength = back_node.inputs['Strength'].default_value

    for i in range(args.num_images):
        lens_scale = render.resolution_x / cam.data.sensor_width
        cam.data.lens = float(param['focal_length'].flatten()[i])/lens_scale# focus_length transorm
        camera.matrix_world = Matrix(cam_poses[i])

        # Ensure correct object visibility for non-segmentation renders
        obj_1.hide_render = False
        if seg_obj:
            seg_obj.hide_render = True

        if not args.test_seg:
            # output depth image
            bpy.context.scene.view_settings.view_transform = "Raw"
            render.image_settings.color_mode = "BW"
            render.image_settings.color_depth = "16"
            tree.links.new(output_depth_slot, composite_node.inputs[0])
            render_path = os.path.join(depthdir,f"{i:03d}.png")
            scene.render.filepath = os.path.abspath(render_path)
            bpy.ops.render.render(write_still=True)

            # output normal image
            view_matrix= calc_view_matrix(camera)
            vn0.outputs[0].default_value = view_matrix[0]
            vn1.outputs[0].default_value = view_matrix[1]
            vn2.outputs[0].default_value = view_matrix[2]

            render.image_settings.color_mode = "RGB"
            render.image_settings.color_depth = "8"
            tree.links.new(node_group.outputs["viewNormal"], composite_node.inputs[0])
            render_path = os.path.join(normaldir,f"{i:03d}.png")
            scene.render.filepath = os.path.abspath(render_path)
            bpy.ops.render.render(write_still=True)

        # --- Render Segmentation Map if seg_obj exists ---
        if seg_obj:
            obj_1.hide_render = True
            seg_obj.hide_render = False

            bpy.context.scene.view_settings.view_transform = 'Raw'
            render.image_settings.color_mode = 'RGB'
            render.image_settings.color_depth = '8'
            scene.render.film_transparent = True
            scene.cycles.samples = 1
            scene.cycles.max_bounces = 0

            back_node.inputs['Color'].default_value = (0, 0, 0, 1)
            back_node.inputs['Strength'].default_value = 0.0

            if composite_node.inputs['Image'].is_linked:
                 tree.links.remove(composite_node.inputs['Image'].links[0])
            tree.links.new(render_layer.outputs["Image"], composite_node.inputs[0])

            render_path = os.path.join(segdir, f"{i:03d}.png")
            scene.render.filepath = os.path.abspath(render_path)
            bpy.ops.render.render(write_still=True)

            obj_1.hide_render = False
            seg_obj.hide_render = True

            scene.render.film_transparent = original_film_transparent
            scene.cycles.samples = original_cycles_samples
            scene.cycles.max_bounces = original_max_bounces
            back_node.inputs['Color'].default_value = original_bg_color
            back_node.inputs['Strength'].default_value = original_bg_strength

    if not args.test_seg:
        # Restore settings before light rendering loop
        bpy.context.scene.view_settings.view_transform = original_view_transform
        render.image_settings.color_mode = original_color_mode
        render.image_settings.color_depth = original_color_depth
        if composite_node.inputs['Image'].is_linked:
            tree.links.remove(composite_node.inputs['Image'].links[0])
        tree.links.new(render_layer.outputs["Image"], composite_node.inputs[0])

        obj_1.hide_render = False
        if seg_obj:
            seg_obj.hide_render = True

        back_node.inputs['Color'].default_value = original_bg_color
        back_node.inputs['Strength'].default_value = original_bg_strength

        for env in range(0,5):
            env_map = bpy.data.images.load(os.path.join(map_path,'map'+str(env+1), 'map'+str(env+1)+".exr"))
            env_texture_node.image = env_map
            back_node.inputs['Strength'].default_value = 1.0

            for i in range(args.num_images):
                lens_scale = render.resolution_x / cam.data.sensor_width
                cam.data.lens = float(param['focal_length'].flatten()[i])/lens_scale
                camera.matrix_world = Matrix(cam_poses[i])

                for j in range(len(mat_list)):
                    render_path = os.path.join(lightdir,f"{i:03d}_m"+str(mat_list[j][0])+"r"+str(mat_list[j][1])+"_env"+str(env+1)+".png")
                    mat.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = mat_list[j][0]
                    mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = mat_list[j][1]
                    scene.render.filepath = os.path.abspath(render_path)
                    bpy.ops.render.render(write_still=True)


        
if __name__ == "__main__":
    save_images(args.param_dir)