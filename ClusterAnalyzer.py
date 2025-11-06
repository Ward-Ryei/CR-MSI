import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from collections import defaultdict
from scipy.spatial import distance
import openpyxl
import warnings
warnings.filterwarnings('ignore')

class ClusterAnalyzer:
    def __init__(self):
        self.colors = plt.cm.tab10
        self.matrix = None
    
    def load_excel_matrix(self, file_path, sheet_name=0, header=0):
        """
        从Excel文件加载聚类像素矩阵
        """
        try:
            print(f"正在加载Excel文件: {file_path}")
            
            # 读取Excel文件
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=header,index_col=0)
            
            # 转换为numpy数组
            self.matrix = df.values.astype(np.int32)
            
            print(f"成功加载矩阵，形状: {self.matrix.shape}")
            print(f"矩阵中的类别: {np.unique(self.matrix)}")
            
            return self.matrix
            
        except Exception as e:
            print(f"加载Excel文件时出错: {e}")
            return None
    
    def visualize_matrix(self, matrix=None, title="聚类像素矩阵"):
        """可视化矩阵"""
        if matrix is None:
            matrix = self.matrix
        
        plt.figure(figsize=(12, 6))
        plt.imshow(matrix, cmap=self.colors, vmin=0, vmax=10)
        plt.colorbar(label='类别ID')
        plt.title(title)
        plt.xlabel('列')
        plt.ylabel('行')
        plt.tight_layout()
        plt.show()
    
    def find_centers_by_area_rank(self, matrix=None, min_area=3, rank_threshold=1, 
                                 target_classes=None, include_all_ranks=False):
        """
        找到面积排名大于等于rank_threshold的色块中心点
        
        参数:
        matrix: 输入矩阵
        min_area: 最小色块面积
        rank_threshold: 面积排名阈值 (1=最大, 2=第二大, 以此类推)
        target_classes: 指定要处理的类别列表，如果为None则处理所有类别
        include_all_ranks: 是否包含所有排名>=threshold的色块，False则只包含指定排名的色块
        
        返回:
        中心点字典
        """
        if matrix is None:
            matrix = self.matrix
        
        centers = defaultdict(list)
        
        # 如果没有指定目标类别，则处理所有非零类别
        if target_classes is None:
            target_classes = [class_id for class_id in np.unique(matrix) if class_id != 0]
        
        for class_id in target_classes:
            if class_id == 0:  # 跳过背景
                continue
            
            # 创建该类别的二值掩码
            class_mask = (matrix == class_id).astype(np.uint8)
            
            # 找到连通组件
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                class_mask, connectivity=8
            )
            
            # 如果没有找到色块，跳过
            if num_labels <= 1:
                print(f"类别 {class_id}: 未找到色块")
                continue
            
            # 获取所有符合条件的色块信息
            blocks_info = []
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area >= min_area:
                    block_mask = (labels == label).astype(np.uint8)
                    component_coords = np.column_stack(np.where(block_mask == 1))
                    
                    # 找到中心点
                    dist_transform = cv2.distanceTransform(block_mask, cv2.DIST_L2, 5)
                    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
                    center_row, center_col = max_loc[1], max_loc[0]
                    
                    blocks_info.append({
                        'label': label,
                        'area': area,
                        'center': (center_row, center_col),
                        'coords': component_coords,
                        'rank': 0  # 稍后计算排名
                    })
            
            # 如果没有符合条件的色块
            if not blocks_info:
                continue
            
            # 按面积排序并计算排名
            blocks_info.sort(key=lambda x: x['area'], reverse=True)
            for i, block in enumerate(blocks_info):
                block['rank'] = i + 1  # 排名从1开始
            
            # 选择符合条件的色块
            selected_blocks = []
            for block in blocks_info:
                if include_all_ranks:
                    # 包含所有排名>=threshold的色块
                    if block['rank'] <= rank_threshold:
                        selected_blocks.append(block)
                else:
                    # 只包含指定排名的色块
                    if block['rank'] == rank_threshold:
                        selected_blocks.append(block)
            
            # 处理选中的色块
            for block in selected_blocks:
                center_row, center_col = block['center']
                centers[class_id].append({
                    'center': (int(center_row), int(center_col)),
                    'area': int(block['area']),
                    'rank': int(block['rank']),
                    'method': f'area_rank_{block["rank"]}',
                    'is_selected': True
                })
            
            # 打印选择信息
            if selected_blocks:
                ranks = [block['rank'] for block in selected_blocks]
                print(f"类别 {class_id}: 选择了排名 {ranks} 的色块，面积 {[block['area'] for block in selected_blocks]}")
            else:
                print(f"类别 {class_id}: 没有找到排名 {rank_threshold} 的色块")
        
        return centers
    
    def get_blocks_by_area_rank(self, matrix=None, min_area=3, target_classes=None):
        """
        获取每个类别所有色块的面积排名信息
        
        参数:
        matrix: 输入矩阵
        min_area: 最小色块面积
        target_classes: 指定要处理的类别列表
        
        返回:
        每个类别的色块排名信息字典
        """
        if matrix is None:
            matrix = self.matrix
        
        rank_info = defaultdict(list)
        
        # 如果没有指定目标类别，则处理所有非零类别
        if target_classes is None:
            target_classes = [class_id for class_id in np.unique(matrix) if class_id != 0]
        
        for class_id in target_classes:
            if class_id == 0:  # 跳过背景
                continue
            
            # 创建该类别的二值掩码
            class_mask = (matrix == class_id).astype(np.uint8)
            
            # 找到连通组件
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                class_mask, connectivity=8
            )
            
            # 如果没有找到色块，跳过
            if num_labels <= 1:
                rank_info[class_id] = []
                continue
            
            # 获取所有符合条件的色块信息
            blocks_info = []
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area >= min_area:
                    block_mask = (labels == label).astype(np.uint8)
                    component_coords = np.column_stack(np.where(block_mask == 1))
                    
                    # 找到中心点
                    dist_transform = cv2.distanceTransform(block_mask, cv2.DIST_L2, 5)
                    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
                    center_row, center_col = max_loc[1], max_loc[0]
                    
                    blocks_info.append({
                        'label': label,
                        'area': area,
                        'center': (center_row, center_col),
                        'coords_count': len(component_coords),
                        'max_distance': float(max_val)
                    })
            
            # 按面积排序并计算排名
            blocks_info.sort(key=lambda x: x['area'], reverse=True)
            for i, block in enumerate(blocks_info):
                block['rank'] = i + 1
            
            rank_info[class_id] = blocks_info
        
        return rank_info
    
    def find_top_n_blocks(self, matrix=None, min_area=3, top_n=1, target_classes=None):
        """
        找到每个类别面积前top_n大的色块
        
        参数:
        matrix: 输入矩阵
        min_area: 最小色块面积
        top_n: 选择前n大的色块
        target_classes: 指定要处理的类别列表
        
        返回:
        中心点字典
        """
        return self.find_centers_by_area_rank(
            matrix=matrix,
            min_area=min_area,
            rank_threshold=1,
            target_classes=target_classes,
            include_all_ranks=True  # 包含所有排名>=1的色块，即前top_n大的色块
        )
    
    def _find_nearest_point_to_center(self, coords):
        """计算几何中心并找到最近的色块点"""
        if len(coords) == 0:
            return (0, 0)
        
        # 计算几何中心
        geometric_center = np.mean(coords, axis=0)
        
        # 找到距离几何中心最近的点
        distances = distance.cdist([geometric_center], coords, 'euclidean')
        nearest_idx = np.argmin(distances)
        nearest_point = coords[nearest_idx]
        
        return (int(nearest_point[0]), int(nearest_point[1]))
    
    def verify_centers_on_blocks(self, matrix, centers):
        """验证所有中心点确实在对应的色块上"""
        valid_centers = defaultdict(list)
        invalid_count = 0
        
        for class_id, blocks in centers.items():
            for block in blocks:
                center_row, center_col = block['center']
                
                # 检查中心点是否在正确的类别上
                if (0 <= center_row < matrix.shape[0] and 
                    0 <= center_col < matrix.shape[1] and 
                    matrix[center_row, center_col] == class_id):
                    valid_centers[class_id].append(block)
                else:
                    invalid_count += 1
                    print(f"警告: 中心点 ({center_row}, {center_col}) 不在类别 {class_id} 的色块上")
        
        if invalid_count > 0:
            print(f"总共发现 {invalid_count} 个无效中心点")
        
        return valid_centers
    
    def visualize_with_centers(self, matrix, centers, title="色块中心检测", highlight_rank=None):
        """可视化矩阵和中心点"""
        plt.figure(figsize=(15, 8))
        
        # 绘制带中心点的矩阵
        plt.imshow(matrix, cmap=self.colors, vmin=0, vmax=10, alpha=0.8)
        
        # 定义不同排名的颜色和标记
        rank_colors = {
            1: 'gold',      # 最大 - 金色
            2: 'red',       # 第二大 - 红色
            3: 'blue',      # 第三大 - 蓝色
            4: 'green',     # 第四大 - 绿色
            5: 'purple'     # 第五大 - 紫色
        }
        
        rank_markers = {
            1: '*',         # 最大 - 星号
            2: 'o',         # 第二大 - 圆圈
            3: 's',         # 第三大 - 正方形
            4: '^',         # 第四大 - 三角形
            5: 'D'          # 第五大 - 菱形
        }
        
        # 绘制中心点
        for class_id, blocks in centers.items():
            for block in blocks:
                center_row, center_col = block['center']
                rank = block.get('rank', 1)
                
                # 选择颜色和标记
                color = rank_colors.get(rank, 'gray')
                marker = rank_markers.get(rank, 'x')
                size = 150 - (rank - 1) * 20  # 排名越高，标记越大
                
                # 绘制标记
                plt.scatter(center_col, center_row, 
                           color=color, s=size, marker=marker, linewidth=2,
                           edgecolors='black' if rank <= 5 else 'white')
                
                # 标记类别ID和排名
                marker_text = f'C{class_id}-R{rank}'
                plt.text(center_col + 2, center_row, 
                        marker_text, color='white', 
                        fontweight='bold', fontsize=8,
                        bbox=dict(facecolor='black', alpha=0.7))
        
        # 添加图例
        legend_elements = []
        for rank in sorted(rank_colors.keys()):
            if rank <= 5:
                legend_elements.append(plt.Line2D([0], [0], 
                                                 marker=rank_markers[rank], 
                                                 color='w', 
                                                 markerfacecolor=rank_colors[rank],
                                                 markersize=10, 
                                                 label=f'排名 {rank}'))
        
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper right')
        
        plt.colorbar(label='类别ID')
        plt.title(title)
        plt.xlabel('列')
        plt.ylabel('行')
        plt.tight_layout()
        plt.show()
    
    def print_centers_info(self, centers, method_name):
        """打印中心点信息"""
        print(f"\n=== {method_name} 方法结果 ===")
        print("=" * 60)
        
        total_blocks = 0
        for class_id, blocks in centers.items():
            print(f"类别 {class_id}: {len(blocks)} 个色块")
            for i, block in enumerate(blocks):
                center_row, center_col = block['center']
                area = block.get('area', 'N/A')
                rank = block.get('rank', 'N/A')
                method = block.get('method', 'unknown')
                print(f"  排名 {rank}: 位置 ({center_row}, {center_col}), 面积: {area}, 方法: {method}")
            total_blocks += len(blocks)
            print()
        
        print(f"总共找到 {total_blocks} 个色块")
    
    def export_centers_to_excel(self, centers, output_file):
        """将中心点信息导出到Excel文件"""
        try:
            data = []
            
            for class_id, blocks in centers.items():
                for block in blocks:
                    center_row, center_col = block['center']
                    data.append({
                        '类别ID': class_id,
                        '行坐标': center_row,
                        '列坐标': center_col,
                        '面积': block.get('area', 0),
                        '排名': block.get('rank', 0),
                        '方法': block.get('method', 'unknown'),
                        '是否选中': '是' if block.get('is_selected', False) else '否'
                    })
            
            # 创建DataFrame并导出
            df = pd.DataFrame(data)
            df.to_excel(output_file, index=False)
            print(f"中心点信息已导出到: {output_file}")
            
            return df
            
        except Exception as e:
            print(f"导出Excel文件时出错: {e}")
            return None
    
    def print_rank_statistics(self, rank_info):
        """打印排名统计信息"""
        print("\n=== 色块面积排名统计 ===")
        print("=" * 50)
        
        for class_id, blocks in rank_info.items():
            if blocks:
                print(f"\n类别 {class_id}:")
                print(f"色块数量: {len(blocks)}")
                print("排名\t面积\t中心点")
                print("-" * 30)
                for block in blocks:
                    center_row, center_col = block['center']
                    print(f"{block['rank']}\t{block['area']}\t({center_row}, {center_col})")
            else:
                print(f"\n类别 {class_id}: 没有符合条件的色块")