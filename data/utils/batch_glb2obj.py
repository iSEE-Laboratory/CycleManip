import trimesh
import os
import json
import numpy as np
from PIL import Image
import argparse

def load_scale_from_json(json_path):
    """从JSON文件读取scale因子"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            if 'scale' in data:
                return data['scale']
    except Exception as e:
        print(f"读取scale失败 {json_path}: {e}")
    return [1.0, 1.0, 1.0]  # 默认缩放

def apply_scale(mesh, scale_factor):
    """对网格应用缩放"""
    scaled_mesh = mesh.copy()
    
    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = scale_factor[0]  # X轴缩放
    scale_matrix[1, 1] = scale_factor[1]  # Y轴缩放
    scale_matrix[2, 2] = scale_factor[2]  # Z轴缩放
    
    scaled_mesh.apply_transform(scale_matrix)
    
    return scaled_mesh

def export_textures(mesh, output_dir, prefix):
    """导出网格的纹理贴图"""
    if hasattr(mesh.visual, 'material'):
        material = mesh.visual.material
        
        # 基础颜色贴图
        if hasattr(material, 'baseColorTexture') and material.baseColorTexture:
            save_texture(material.baseColorTexture, output_dir, f"{prefix}_base_color.png")
        
        # 法线贴图
        if hasattr(material, 'normalTexture') and material.normalTexture:
            save_texture(material.normalTexture, output_dir, f"{prefix}_normal.png")
        
        # 金属粗糙度贴图
        if hasattr(material, 'metallicRoughnessTexture') and material.metallicRoughnessTexture:
            save_texture(material.metallicRoughnessTexture, output_dir, f"{prefix}_metallic_roughness.png")

def save_texture(texture, output_dir, filename):
    """保存原始纹理到文件"""
    try:
        img_path = None
        
        if hasattr(texture, 'source') and hasattr(texture.source, 'filename') and texture.source.filename:
            img_path = texture.source.filename
        elif hasattr(texture, 'pil') and texture.pil:
            texture.pil.save(os.path.join(output_dir, filename))
            print(f"    导出纹理: {filename}")
            return
        
        if img_path:
            img = Image.open(img_path)
            img.save(os.path.join(output_dir, filename))
            print(f"    导出纹理: {filename}")

    except Exception as e:
        print(f"    导出纹理失败 {filename}: {e}")

def export_glb(glb_path, output_dir, scale_factor):
    """导出单个GLB文件"""
    scene = trimesh.load(glb_path)
    
    if isinstance(scene, trimesh.Scene):
        # 处理场景中的每个网格
        for i, (name, mesh) in enumerate(scene.geometry.items()):
            scaled_mesh = apply_scale(mesh, scale_factor)
            obj_path = os.path.join(output_dir, f"{name}_{i}.obj")
            scaled_mesh.export(obj_path)
            export_textures(mesh, output_dir, name)
    else:
        # 单个网格
        scaled_mesh = apply_scale(scene, scale_factor)
        obj_path = os.path.join(output_dir, "mesh.obj")
        scaled_mesh.export(obj_path)
        export_textures(scene, output_dir, "mesh")

def process_object_folder(object_folder):
    """处理单个对象文件夹"""
    print(f"\n处理对象文件夹: {os.path.basename(object_folder)}")
    
    # 查找visual文件夹
    visual_folder = os.path.join(object_folder, "visual")
    if not os.path.exists(visual_folder):
        print(f"  未找到visual文件夹，跳过")
        return
    
    # 创建output文件夹
    output_folder = os.path.join(object_folder, "output")
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有GLB文件
    glb_files = sorted([f for f in os.listdir(visual_folder) if f.endswith('.glb')])
    
    if not glb_files:
        print(f"  未找到GLB文件，跳过")
        return
    
    print(f"  找到 {len(glb_files)} 个GLB文件")
    
    # 处理每个GLB文件
    for idx, glb_file in enumerate(glb_files):
        # 提取编号（例如 base0.glb -> 0）
        base_name = os.path.splitext(glb_file)[0]  # 例如 "base0"
        number = base_name.replace('base', '')  # 例如 "0"
        
        # 查找对应的JSON文件
        json_file = f"model_data{number}.json"
        json_path = os.path.join(object_folder, json_file)
        
        # 读取scale因子
        if os.path.exists(json_path):
            scale_factor = load_scale_from_json(json_path)
            print(f"  [{idx+1}/{len(glb_files)}] 处理 {glb_file} (scale: {scale_factor})")
        else:
            scale_factor = [1.0, 1.0, 1.0]
            print(f"  [{idx+1}/{len(glb_files)}] 处理 {glb_file} (未找到JSON，使用默认scale)")
        
        # 创建mesh子文件夹
        mesh_output_dir = os.path.join(output_folder, f"mesh{number}")
        os.makedirs(mesh_output_dir, exist_ok=True)
        
        # 导出GLB
        glb_path = os.path.join(visual_folder, glb_file)
        try:
            export_glb(glb_path, mesh_output_dir, scale_factor)
            print(f"    ✓ 完成 mesh{number}")
        except Exception as e:
            print(f"    ✗ 失败 mesh{number}: {e}")

def batch_process(root_folder, recursive=True, only_numeric=False):
    """批量处理根文件夹下的所有对象文件夹"""
    print(f"开始批量处理: {root_folder}")
    print(f"递归模式: {'是' if recursive else '否'}")
    print(f"仅处理数字开头的文件夹: {'是' if only_numeric else '否'}\n")
    
    # 如果root_folder本身就包含visual文件夹，则直接处理
    if os.path.exists(os.path.join(root_folder, "visual")):
        process_object_folder(root_folder)
    elif recursive:
        # 递归遍历所有子文件夹
        processed_count = 0
        skipped_count = 0
        for item in sorted(os.listdir(root_folder)):
            item_path = os.path.join(root_folder, item)
            if os.path.isdir(item_path):
                # 如果启用了only_numeric，检查文件夹名是否以数字开头
                if only_numeric and not item[0].isdigit():
                    skipped_count += 1
                    continue
                
                # 检查是否包含visual文件夹
                if os.path.exists(os.path.join(item_path, "visual")):
                    process_object_folder(item_path)
                    processed_count += 1
        
        print(f"\n\n处理完成! 共处理 {processed_count} 个对象文件夹")
        if only_numeric and skipped_count > 0:
            print(f"跳过 {skipped_count} 个非数字开头的文件夹")
    else:
        print("未找到visual文件夹，且未启用递归模式")
    
    print("\n所有处理完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量将GLB文件转换为OBJ文件')
    parser.add_argument('input_folder', type=str, nargs='?', 
                        default="/home/dex/haoran/LoopBreaker/assets/objects",
                        help='输入文件夹路径')
    parser.add_argument('--no-recursive', action='store_true',
                        help='不递归处理子文件夹')
    parser.add_argument('--only-numeric', action='store_true',
                        help='仅处理以数字开头的文件夹')
    
    args = parser.parse_args()
    
    # 开始批量处理
    batch_process(args.input_folder, recursive=not args.no_recursive, only_numeric=args.only_numeric)
