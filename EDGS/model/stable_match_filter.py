#!/usr/bin/env python3
"""
稳定特征匹配滤波器
解决EfficientLoFTR匹配结果不稳定的问题
"""

import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from collections import deque
import time
import colorsys


class StableMatchManager:
    """稳定匹配点管理器"""
    
    def __init__(self, history_frames=6, spatial_threshold=8.0, stability_threshold=3, min_matches=10):
        self.history_frames = history_frames
        self.spatial_threshold = spatial_threshold  # 像素距离阈值
        self.stability_threshold = stability_threshold  # 最小稳定帧数
        self.min_matches = min_matches
        
        # 历史数据存储
        self.match_history = deque(maxlen=history_frames)  # 存储 (mkpts0, mkpts1, mconf, timestamp)
        self.stable_clusters = []  # 当前稳定的匹配点集群
        self.cluster_histories = {}  # 集群的历史记录 {cluster_id: [frame_indices]}
        
    def update_matches(self, mkpts0, mkpts1, mconf):
        """
        更新匹配点并返回稳定的匹配
        
        Args:
            mkpts0, mkpts1: 匹配点对
            mconf: 匹配置信度
            
        Returns:
            stable_mkpts0, stable_mkpts1, stable_mconf: 稳定的匹配点
        """
        current_time = time.time()
        
        # 添加当前帧到历史
        current_matches = {
            'mkpts0': mkpts0.copy(),
            'mkpts1': mkpts1.copy(), 
            'mconf': mconf.copy(),
            'timestamp': current_time,
            'frame_id': len(self.match_history)
        }
        self.match_history.append(current_matches)
        
        # 如果历史帧不够，返回原始匹配
        if len(self.match_history) < self.stability_threshold:
            return mkpts0, mkpts1, mconf
        
        # 进行稳定性分析
        stable_matches = self._extract_stable_matches()
        
        if len(stable_matches['mkpts0']) < self.min_matches:
            # 如果稳定匹配太少，返回高置信度的原始匹配
            high_conf_mask = mconf > np.percentile(mconf, 70)  # 保留前30%置信度的点
            return mkpts0[high_conf_mask], mkpts1[high_conf_mask], mconf[high_conf_mask]
        
        return stable_matches['mkpts0'], stable_matches['mkpts1'], stable_matches['mconf']
    
    def _extract_stable_matches(self):
        """提取稳定的匹配点"""
        if len(self.match_history) < self.stability_threshold:
            return {'mkpts0': np.array([]), 'mkpts1': np.array([]), 'mconf': np.array([])}
        
        # 获取最近几帧用于稳定性分析
        recent_frames = list(self.match_history)[-self.stability_threshold:]
        
        # 对最新帧的匹配点进行聚类
        current_frame = self.match_history[-1]
        current_clusters = self._cluster_matches(current_frame['mkpts0'], current_frame['mkpts1'], current_frame['mconf'])
        
        # 验证每个聚类的稳定性
        stable_matches = {'mkpts0': [], 'mkpts1': [], 'mconf': []}
        
        for cluster in current_clusters:
            if self._is_cluster_stable(cluster, recent_frames):
                # 添加稳定的聚类中心点
                stable_matches['mkpts0'].append(cluster['center0'])
                stable_matches['mkpts1'].append(cluster['center1'])
                stable_matches['mconf'].append(cluster['avg_conf'])
        
        # 转换为numpy数组
        for key in stable_matches:
            stable_matches[key] = np.array(stable_matches[key])
        
        return stable_matches
    
    def _cluster_matches(self, mkpts0, mkpts1, mconf):
        """对匹配点进行空间聚类"""
        if len(mkpts0) == 0:
            return []
        
        # 使用第一个图像的坐标进行聚类
        try:
            if len(mkpts0) < 3:
                # 点太少，每个点自成一类
                clusters = []
                for i in range(len(mkpts0)):
                    clusters.append({
                        'center0': mkpts0[i],
                        'center1': mkpts1[i],
                        'avg_conf': mconf[i],
                        'points': [i],
                        'stability_count': 1
                    })
                return clusters
            
            # 使用DBSCAN进行聚类
            clustering = DBSCAN(eps=self.spatial_threshold, min_samples=1).fit(mkpts0)
            labels = clustering.labels_
            
            clusters = []
            for label in np.unique(labels):
                if label == -1:  # 噪声点
                    continue
                
                mask = labels == label
                cluster_mkpts0 = mkpts0[mask]
                cluster_mkpts1 = mkpts1[mask]
                cluster_mconf = mconf[mask]
                
                # 计算聚类中心（加权平均）
                weights = cluster_mconf / np.sum(cluster_mconf)
                center0 = np.average(cluster_mkpts0, axis=0, weights=weights)
                center1 = np.average(cluster_mkpts1, axis=0, weights=weights)
                avg_conf = np.mean(cluster_mconf)
                
                clusters.append({
                    'center0': center0,
                    'center1': center1,
                    'avg_conf': avg_conf,
                    'points': np.where(mask)[0].tolist(),
                    'stability_count': 1
                })
            
            return clusters
            
        except Exception as e:
            print(f"Clustering error: {e}")
            # 备选方案：简单的距离聚类
            return self._simple_clustering(mkpts0, mkpts1, mconf)
    
    def _simple_clustering(self, mkpts0, mkpts1, mconf):
        """简单的距离聚类备选方案"""
        clusters = []
        used = np.zeros(len(mkpts0), dtype=bool)
        
        for i in range(len(mkpts0)):
            if used[i]:
                continue
                
            # 找到所有在阈值距离内的点
            distances = np.linalg.norm(mkpts0 - mkpts0[i], axis=1)
            cluster_mask = distances < self.spatial_threshold
            
            if np.sum(cluster_mask) == 0:
                continue
                
            used[cluster_mask] = True
            
            cluster_mkpts0 = mkpts0[cluster_mask]
            cluster_mkpts1 = mkpts1[cluster_mask]
            cluster_mconf = mconf[cluster_mask]
            
            # 加权中心
            weights = cluster_mconf / np.sum(cluster_mconf)
            center0 = np.average(cluster_mkpts0, axis=0, weights=weights)
            center1 = np.average(cluster_mkpts1, axis=0, weights=weights)
            avg_conf = np.mean(cluster_mconf)
            
            clusters.append({
                'center0': center0,
                'center1': center1,
                'avg_conf': avg_conf,
                'points': np.where(cluster_mask)[0].tolist(),
                'stability_count': 1
            })
        
        return clusters
    
    def _is_cluster_stable(self, cluster, recent_frames):
        """检查聚类是否稳定"""
        cluster_center0 = cluster['center0']
        stable_count = 0
        
        # 检查这个聚类在历史帧中的出现次数
        for frame in recent_frames[:-1]:  # 排除当前帧
            frame_mkpts0 = frame['mkpts0']
            if len(frame_mkpts0) == 0:
                continue
                
            # 计算与历史帧中所有点的距离
            distances = np.linalg.norm(frame_mkpts0 - cluster_center0, axis=1)
            
            # 如果有点在阈值距离内，认为这个聚类在该帧中存在
            if np.min(distances) < self.spatial_threshold * 1.5:  # 稍微放宽阈值
                stable_count += 1
        
        # 需要在至少 stability_threshold-1 帧中出现才算稳定
        return stable_count >= (self.stability_threshold - 1)
    
    def get_statistics(self):
        """获取滤波统计信息"""
        if len(self.match_history) == 0:
            return {
                'total_frames': 0,
                'avg_matches_per_frame': 0,
                'current_stable_matches': 0
            }
        
        total_matches = sum(len(frame['mkpts0']) for frame in self.match_history)
        avg_matches = total_matches / len(self.match_history)
        
        current_stable = len(self.stable_clusters)
        
        return {
            'total_frames': len(self.match_history),
            'avg_matches_per_frame': avg_matches,
            'current_stable_matches': current_stable,
            'stability_ratio': current_stable / avg_matches if avg_matches > 0 else 0
        }


class GeometryValidator:
    """几何一致性验证器"""
    
    def __init__(self, camera_matrix, max_epipolar_error=2.0, min_inliers=8):
        self.camera_matrix = camera_matrix
        self.max_epipolar_error = max_epipolar_error
        self.min_inliers = min_inliers
        self.reference_essential_matrix = None
        self.reference_update_counter = 0
        
    def validate_matches(self, mkpts0, mkpts1, mconf):
        """
        验证匹配点的几何一致性
        
        Args:
            mkpts0, mkpts1: 匹配点对
            mconf: 匹配置信度
            
        Returns:
            valid_mkpts0, valid_mkpts1, valid_mconf: 几何一致的匹配点
        """
        if len(mkpts0) < self.min_inliers:
            return mkpts0, mkpts1, mconf
        
        try:
            # 计算本质矩阵
            E, inlier_mask = cv2.findEssentialMat(
                mkpts0, mkpts1, self.camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=self.max_epipolar_error
            )
            
            if E is None or inlier_mask is None or inlier_mask.sum() < self.min_inliers:
                # 如果几何验证失败，返回高置信度的原始匹配
                high_conf_mask = mconf > np.percentile(mconf, 80)
                return mkpts0[high_conf_mask], mkpts1[high_conf_mask], mconf[high_conf_mask]
            
            # 更新参考几何约束
            if self.reference_update_counter % 10 == 0:  # 每10帧更新一次参考
                self.reference_essential_matrix = E
            self.reference_update_counter += 1
            
            # 返回几何一致的匹配点
            valid_mask = inlier_mask.ravel().astype(bool)
            return mkpts0[valid_mask], mkpts1[valid_mask], mconf[valid_mask]
            
        except Exception as e:
            print(f"Geometry validation error: {e}")
            # 出错时返回原始匹配
            return mkpts0, mkpts1, mconf
    
    def check_epipolar_constraint(self, mkpts0, mkpts1):
        """检查极线约束"""
        if self.reference_essential_matrix is None or len(mkpts0) < 4:
            return np.ones(len(mkpts0), dtype=bool)
        
        try:
            # 将点转换为齐次坐标
            pts0_normalized = cv2.undistortPoints(
                mkpts0.reshape(-1, 1, 2), 
                self.camera_matrix, 
                None
            ).reshape(-1, 2)
            pts1_normalized = cv2.undistortPoints(
                mkpts1.reshape(-1, 1, 2), 
                self.camera_matrix, 
                None
            ).reshape(-1, 2)
            
            # 转换为齐次坐标
            pts0_homo = np.column_stack([pts0_normalized, np.ones(len(pts0_normalized))])
            pts1_homo = np.column_stack([pts1_normalized, np.ones(len(pts1_normalized))])
            
            # 计算极线距离
            epipolar_errors = []
            for i in range(len(pts0_homo)):
                # x1^T * E * x0
                error = np.abs(pts1_homo[i] @ self.reference_essential_matrix @ pts0_homo[i].T)
                epipolar_errors.append(error)
            
            epipolar_errors = np.array(epipolar_errors)
            valid_mask = epipolar_errors < self.max_epipolar_error
            
            return valid_mask
            
        except Exception as e:
            print(f"Epipolar constraint check error: {e}")
            return np.ones(len(mkpts0), dtype=bool)


class AdaptiveThresholdController:
    """自适应阈值控制器"""
    
    def __init__(self, base_threshold=0.2, adaptation_rate=0.1, stability_history_size=10):
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold
        self.adaptation_rate = adaptation_rate
        self.stability_history = deque(maxlen=stability_history_size)
        self.match_count_history = deque(maxlen=stability_history_size)
        
        # 阈值范围
        self.min_threshold = 0.1
        self.max_threshold = 0.5
        
    def update_threshold(self, current_matches_count, stable_matches_count):
        """
        根据匹配稳定性更新阈值
        
        Args:
            current_matches_count: 当前帧总匹配数
            stable_matches_count: 稳定匹配数
        """
        self.match_count_history.append(current_matches_count)
        
        # 计算稳定性指标
        stability_ratio = stable_matches_count / max(current_matches_count, 1)
        self.stability_history.append(stability_ratio)
        
        if len(self.stability_history) < 3:
            return self.current_threshold
        
        # 计算平均稳定性
        avg_stability = np.mean(list(self.stability_history))
        
        # 计算匹配数量变化
        if len(self.match_count_history) >= 2:
            recent_counts = list(self.match_count_history)[-5:]  # 最近5帧
            count_variation = np.std(recent_counts) / max(np.mean(recent_counts), 1)
        else:
            count_variation = 0
        
        # 自适应调整逻辑
        target_threshold = self.base_threshold
        
        if avg_stability > 0.7 and count_variation < 0.2:
            # 场景很稳定，可以降低阈值获取更多匹配
            target_threshold = self.base_threshold * 0.8
        elif avg_stability < 0.3 or count_variation > 0.5:
            # 场景不稳定，提高阈值保证质量
            target_threshold = self.base_threshold * 1.3
        
        # 平滑调整
        self.current_threshold += self.adaptation_rate * (target_threshold - self.current_threshold)
        
        # 限制阈值范围
        self.current_threshold = np.clip(self.current_threshold, self.min_threshold, self.max_threshold)
        
        return self.current_threshold
    
    def get_adaptive_threshold(self):
        """获取当前自适应阈值"""
        return self.current_threshold
    
    def get_statistics(self):
        """获取控制器统计信息"""
        if len(self.stability_history) == 0:
            return {
                'current_threshold': self.current_threshold,
                'avg_stability': 0,
                'match_count_variation': 0
            }
        
        avg_stability = np.mean(list(self.stability_history))
        
        if len(self.match_count_history) >= 2:
            count_variation = np.std(list(self.match_count_history)) / max(np.mean(list(self.match_count_history)), 1)
        else:
            count_variation = 0
            
        return {
            'current_threshold': self.current_threshold,
            'avg_stability': avg_stability,
            'match_count_variation': count_variation,
            'base_threshold': self.base_threshold
        }


class OutlierRemover:
    """离群点移除器"""
    
    def __init__(self, method='statistical', neighborhood_size=20, std_multiplier=2.0, 
                 isolation_threshold=0.1, min_cluster_size=5):
        self.method = method  # 'statistical', 'isolation', 'cluster'
        self.neighborhood_size = neighborhood_size
        self.std_multiplier = std_multiplier
        self.isolation_threshold = isolation_threshold
        self.min_cluster_size = min_cluster_size
        
    def remove_outliers(self, mkpts0, mkpts1, mconf):
        """
        移除离群点
        
        Args:
            mkpts0, mkpts1: 匹配点对
            mconf: 匹配置信度
            
        Returns:
            filtered_mkpts0, filtered_mkpts1, filtered_mconf: 过滤后的匹配点
        """
        if len(mkpts0) < self.min_cluster_size:
            return mkpts0, mkpts1, mconf
        
        if self.method == 'statistical':
            valid_mask = self._statistical_outlier_removal(mkpts0, mkpts1)
        elif self.method == 'isolation':
            valid_mask = self._isolation_forest_removal(mkpts0, mkpts1)
        elif self.method == 'cluster':
            valid_mask = self._cluster_based_removal(mkpts0, mkpts1)
        else:
            valid_mask = np.ones(len(mkpts0), dtype=bool)
        
        return mkpts0[valid_mask], mkpts1[valid_mask], mconf[valid_mask]
    
    def _statistical_outlier_removal(self, mkpts0, mkpts1):
        """统计学离群点移除"""
        try:
            # 计算每个点到其邻近点的平均距离
            distances = []
            
            for i in range(len(mkpts0)):
                # 计算到其他所有点的距离
                dists_to_others = np.linalg.norm(mkpts0 - mkpts0[i], axis=1)
                # 排除自己，取最近的k个邻居
                k = min(self.neighborhood_size, len(mkpts0) - 1)
                if k > 0:
                    nearest_distances = np.partition(dists_to_others, k)[:k+1]
                    avg_dist = np.mean(nearest_distances[1:])  # 排除距离为0的自己
                    distances.append(avg_dist)
                else:
                    distances.append(0)
            
            distances = np.array(distances)
            
            # 计算距离的统计特征
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # 使用标准差倍数来识别离群点
            threshold = mean_dist + self.std_multiplier * std_dist
            valid_mask = distances <= threshold
            
            return valid_mask
            
        except Exception as e:
            print(f"Statistical outlier removal error: {e}")
            return np.ones(len(mkpts0), dtype=bool)
    
    def _isolation_forest_removal(self, mkpts0, mkpts1):
        """孤立森林离群点移除"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # 组合特征：位置 + 位移向量
            features = np.hstack([
                mkpts0,
                mkpts1,
                mkpts1 - mkpts0  # 位移向量
            ])
            
            # 训练孤立森林
            iso_forest = IsolationForest(
                contamination=self.isolation_threshold,
                random_state=42,
                n_estimators=100
            )
            
            outlier_labels = iso_forest.fit_predict(features)
            valid_mask = outlier_labels == 1  # 1表示正常点，-1表示离群点
            
            return valid_mask
            
        except ImportError:
            print("Warning: sklearn not available, using statistical method")
            return self._statistical_outlier_removal(mkpts0, mkpts1)
        except Exception as e:
            print(f"Isolation forest outlier removal error: {e}")
            return np.ones(len(mkpts0), dtype=bool)
    
    def _cluster_based_removal(self, mkpts0, mkpts1):
        """基于聚类的离群点移除"""
        try:
            from sklearn.cluster import DBSCAN
            
            # 使用DBSCAN聚类
            clustering = DBSCAN(
                eps=15.0,  # 稍大的eps值
                min_samples=self.min_cluster_size
            ).fit(mkpts0)
            
            labels = clustering.labels_
            
            # 移除噪声点（标签为-1）
            valid_mask = labels != -1
            
            return valid_mask
            
        except ImportError:
            print("Warning: sklearn not available, using statistical method")
            return self._statistical_outlier_removal(mkpts0, mkpts1)
        except Exception as e:
            print(f"Cluster-based outlier removal error: {e}")
            return np.ones(len(mkpts0), dtype=bool)


class DensificationEnhancer:
    """密集点稠密化增强器"""
    
    def __init__(self, min_density_threshold=3, search_radius=15.0, 
                 max_interpolated_per_region=5, confidence_decay=0.8):
        self.min_density_threshold = min_density_threshold  # 最小密度阈值
        self.search_radius = search_radius  # 搜索半径
        self.max_interpolated_per_region = max_interpolated_per_region
        self.confidence_decay = confidence_decay  # 插值点置信度衰减因子
        
    def densify_matches(self, mkpts0, mkpts1, mconf):
        """
        对匹配点进行稠密化
        
        Args:
            mkpts0, mkpts1: 匹配点对
            mconf: 匹配置信度
            
        Returns:
            enhanced_mkpts0, enhanced_mkpts1, enhanced_mconf: 稠密化后的匹配点
        """
        if len(mkpts0) < 10:  # 点太少时不进行稠密化
            return mkpts0, mkpts1, mconf
        
        try:
            # 1. 识别密集区域
            dense_regions = self._identify_dense_regions(mkpts0, mkpts1, mconf)
            
            # 2. 在密集区域之间进行插值
            interpolated_points = self._interpolate_sparse_regions(mkpts0, mkpts1, mconf, dense_regions)
            
            # 3. 合并原始点和插值点
            if len(interpolated_points['mkpts0']) > 0:
                enhanced_mkpts0 = np.vstack([mkpts0, interpolated_points['mkpts0']])
                enhanced_mkpts1 = np.vstack([mkpts1, interpolated_points['mkpts1']])
                enhanced_mconf = np.hstack([mconf, interpolated_points['mconf']])
            else:
                enhanced_mkpts0, enhanced_mkpts1, enhanced_mconf = mkpts0, mkpts1, mconf
            
            return enhanced_mkpts0, enhanced_mkpts1, enhanced_mconf
            
        except Exception as e:
            print(f"Densification error: {e}")
            return mkpts0, mkpts1, mconf
    
    def _identify_dense_regions(self, mkpts0, mkpts1, mconf):
        """识别密集区域"""
        try:
            from sklearn.cluster import DBSCAN
            
            # 使用DBSCAN识别密集区域
            clustering = DBSCAN(
                eps=self.search_radius,
                min_samples=self.min_density_threshold
            ).fit(mkpts0)
            
            labels = clustering.labels_
            
            # 统计每个聚类的信息
            dense_regions = []
            for label in np.unique(labels):
                if label == -1:  # 跳过噪声点
                    continue
                
                cluster_mask = labels == label
                cluster_points0 = mkpts0[cluster_mask]
                cluster_points1 = mkpts1[cluster_mask]
                cluster_conf = mconf[cluster_mask]
                
                if len(cluster_points0) >= self.min_density_threshold:
                    center0 = np.mean(cluster_points0, axis=0)
                    center1 = np.mean(cluster_points1, axis=0)
                    avg_conf = np.mean(cluster_conf)
                    
                    dense_regions.append({
                        'center0': center0,
                        'center1': center1,
                        'points0': cluster_points0,
                        'points1': cluster_points1,
                        'confidence': avg_conf,
                        'size': len(cluster_points0)
                    })
            
            return dense_regions
            
        except ImportError:
            # 备选方案：简单的距离-密度分析
            return self._simple_dense_region_detection(mkpts0, mkpts1, mconf)
        except Exception as e:
            print(f"Dense region identification error: {e}")
            return []
    
    def _simple_dense_region_detection(self, mkpts0, mkpts1, mconf):
        """简单的密集区域检测"""
        dense_regions = []
        used = np.zeros(len(mkpts0), dtype=bool)
        
        for i in range(len(mkpts0)):
            if used[i]:
                continue
            
            # 找到在半径内的所有点
            distances = np.linalg.norm(mkpts0 - mkpts0[i], axis=1)
            nearby_mask = distances <= self.search_radius
            nearby_count = np.sum(nearby_mask)
            
            if nearby_count >= self.min_density_threshold:
                used[nearby_mask] = True
                
                nearby_points0 = mkpts0[nearby_mask]
                nearby_points1 = mkpts1[nearby_mask]
                nearby_conf = mconf[nearby_mask]
                
                center0 = np.mean(nearby_points0, axis=0)
                center1 = np.mean(nearby_points1, axis=0)
                avg_conf = np.mean(nearby_conf)
                
                dense_regions.append({
                    'center0': center0,
                    'center1': center1,
                    'points0': nearby_points0,
                    'points1': nearby_points1,
                    'confidence': avg_conf,
                    'size': nearby_count
                })
        
        return dense_regions
    
    def _interpolate_sparse_regions(self, mkpts0, mkpts1, mconf, dense_regions):
        """在稀疏区域之间进行插值"""
        interpolated_points = {'mkpts0': [], 'mkpts1': [], 'mconf': []}
        
        if len(dense_regions) < 2:
            return interpolated_points
        
        # 在密集区域之间进行插值
        for i in range(len(dense_regions)):
            for j in range(i + 1, len(dense_regions)):
                region1 = dense_regions[i]
                region2 = dense_regions[j]
                
                # 计算两个区域之间的距离
                dist = np.linalg.norm(region1['center0'] - region2['center0'])
                
                # 如果距离适中，进行插值
                if self.search_radius * 2 < dist < self.search_radius * 6:
                    interp_points = self._create_interpolated_points(region1, region2)
                    
                    for point in interp_points:
                        interpolated_points['mkpts0'].append(point['pt0'])
                        interpolated_points['mkpts1'].append(point['pt1'])
                        interpolated_points['mconf'].append(point['conf'])
        
        # 转换为numpy数组
        for key in interpolated_points:
            if len(interpolated_points[key]) > 0:
                interpolated_points[key] = np.array(interpolated_points[key])
            else:
                interpolated_points[key] = np.array([]).reshape(0, 2 if 'mkpts' in key else 0)
        
        return interpolated_points
    
    def _create_interpolated_points(self, region1, region2):
        """在两个密集区域之间创建插值点"""
        interpolated_points = []
        
        # 限制插值点数量
        num_points = min(self.max_interpolated_per_region, 
                        int((region1['size'] + region2['size']) / 4))
        
        if num_points <= 0:
            return interpolated_points
        
        for i in range(num_points):
            # 线性插值因子
            t = (i + 1) / (num_points + 1)
            
            # 在两个区域中心之间插值
            interp_pt0 = region1['center0'] * (1 - t) + region2['center0'] * t
            interp_pt1 = region1['center1'] * (1 - t) + region2['center1'] * t
            
            # 添加一些随机扰动，使插值更自然
            noise_scale = 3.0
            noise0 = np.random.normal(0, noise_scale, 2)
            noise1 = np.random.normal(0, noise_scale, 2)
            
            interp_pt0 += noise0
            interp_pt1 += noise1
            
            # 置信度衰减
            interp_conf = (region1['confidence'] + region2['confidence']) / 2 * self.confidence_decay
            
            interpolated_points.append({
                'pt0': interp_pt0,
                'pt1': interp_pt1,
                'conf': interp_conf
            })
        
        return interpolated_points


class ColorSelectiveProcessor:
    """基于颜色的选择性处理器"""
    
    def __init__(self):
        # 预定义颜色配置
        self.color_configs = {
            'cyan_tweezers': {
                'color_center': [180, 255, 255],  # HSV format: 天蓝色
                'color_tolerance': [20, 100, 100],  # HSV容忍度
                'densify_only': True,
                'enhanced_densification': True,
                'name': 'Cyan Tweezers'
            }
        }
        
        # 当前激活的颜色配置
        self.active_configs = ['cyan_tweezers']
        
        # 缓存上一帧的图像用于颜色分析
        self.last_frame0 = None
        self.last_frame1 = None
    
    def update_frames(self, frame0, frame1):
        """更新当前帧图像用于颜色分析"""
        self.last_frame0 = frame0.copy() if frame0 is not None else None
        self.last_frame1 = frame1.copy() if frame1 is not None else None
    
    def analyze_match_colors(self, mkpts0, mkpts1):
        """分析匹配点的颜色信息"""
        if self.last_frame0 is None or self.last_frame1 is None or len(mkpts0) == 0:
            return np.array([]), {}
        
        color_classifications = []
        color_details = {}
        
        for i, (pt0, pt1) in enumerate(zip(mkpts0, mkpts1)):
            # 确保坐标在图像范围内
            x0, y0 = int(np.clip(pt0[0], 0, self.last_frame0.shape[1]-1)), int(np.clip(pt0[1], 0, self.last_frame0.shape[0]-1))
            x1, y1 = int(np.clip(pt1[0], 0, self.last_frame1.shape[1]-1)), int(np.clip(pt1[1], 0, self.last_frame1.shape[0]-1))
            
            # 从两个视图提取颜色（BGR格式）
            if len(self.last_frame0.shape) == 3:
                color0_bgr = self.last_frame0[y0, x0]
                color1_bgr = self.last_frame1[y1, x1]
            else:
                # 灰度图像，转换为伪彩色
                gray0 = self.last_frame0[y0, x0] 
                gray1 = self.last_frame1[y1, x1]
                color0_bgr = np.array([gray0, gray0, gray0])
                color1_bgr = np.array([gray1, gray1, gray1])
            
            # 平均两个视图的颜色
            avg_color_bgr = (color0_bgr.astype(float) + color1_bgr.astype(float)) / 2
            
            # 转换为HSV进行颜色匹配
            avg_color_bgr_uint8 = avg_color_bgr.astype(np.uint8).reshape(1, 1, 3)
            avg_color_hsv = cv2.cvtColor(avg_color_bgr_uint8, cv2.COLOR_BGR2HSV)[0, 0]
            
            # 检查每个活跃的颜色配置
            matched_config = None
            min_distance = float('inf')
            
            for config_name in self.active_configs:
                if config_name in self.color_configs:
                    config = self.color_configs[config_name]
                    distance = self._calculate_color_distance(avg_color_hsv, config)
                    
                    if distance < min_distance:
                        min_distance = distance
                        if distance < 1.0:  # 归一化距离阈值
                            matched_config = config_name
            
            color_classifications.append(matched_config)
            color_details[i] = {
                'avg_color_bgr': avg_color_bgr,
                'avg_color_hsv': avg_color_hsv,
                'matched_config': matched_config,
                'color_distance': min_distance
            }
        
        return np.array(color_classifications), color_details
    
    def _calculate_color_distance(self, color_hsv, config):
        """计算颜色距离（HSV空间）"""
        center_hsv = np.array(config['color_center'])
        tolerance = np.array(config['color_tolerance'])
        
        # 处理色调的循环性（0-180度）
        hue_diff = abs(color_hsv[0] - center_hsv[0])
        if hue_diff > 90:  # HSV中H的范围是0-179
            hue_diff = 180 - hue_diff
        
        # 计算HSV各分量的归一化距离
        h_dist = hue_diff / tolerance[0]
        s_dist = abs(color_hsv[1] - center_hsv[1]) / tolerance[1]
        v_dist = abs(color_hsv[2] - center_hsv[2]) / tolerance[2]
        
        # 计算总距离
        total_distance = np.sqrt(h_dist**2 + s_dist**2 + v_dist**2) / np.sqrt(3)
        
        return total_distance
    
    def should_preserve_from_removal(self, point_idx, color_classifications):
        """判断某个点是否应该被保护不被移除"""
        if point_idx >= len(color_classifications):
            return False
        
        config_name = color_classifications[point_idx]
        if config_name and config_name in self.color_configs:
            return self.color_configs[config_name]['densify_only']
        
        return False
    
    def should_enhance_densification(self, point_idx, color_classifications):
        """判断某个点是否应该增强稠密化"""
        if point_idx >= len(color_classifications):
            return False
        
        config_name = color_classifications[point_idx]
        if config_name and config_name in self.color_configs:
            return self.color_configs[config_name]['enhanced_densification']
        
        return False
    
    def get_special_color_points(self, mkpts0, mkpts1, mconf, color_classifications):
        """获取特殊颜色的匹配点"""
        special_indices = []
        for i, config_name in enumerate(color_classifications):
            if config_name is not None:
                special_indices.append(i)
        
        if len(special_indices) == 0:
            return np.array([]), np.array([]), np.array([])
        
        special_indices = np.array(special_indices)
        return mkpts0[special_indices], mkpts1[special_indices], mconf[special_indices]
    
    def add_color_config(self, name, color_center_hsv, tolerance_hsv, densify_only=True, enhanced_densification=True):
        """添加新的颜色配置"""
        self.color_configs[name] = {
            'color_center': color_center_hsv,
            'color_tolerance': tolerance_hsv,
            'densify_only': densify_only,
            'enhanced_densification': enhanced_densification,
            'name': name
        }
    
    def set_active_configs(self, config_names):
        """设置激活的颜色配置"""
        self.active_configs = [name for name in config_names if name in self.color_configs]
    
    def get_color_statistics(self, color_classifications, color_details):
        """获取颜色分析统计信息"""
        stats = {
            'total_points': len(color_classifications),
            'special_color_points': 0,
            'color_breakdown': {}
        }
        
        for config_name in color_classifications:
            if config_name is not None:
                stats['special_color_points'] += 1
                if config_name in stats['color_breakdown']:
                    stats['color_breakdown'][config_name] += 1
                else:
                    stats['color_breakdown'][config_name] = 1
        
        return stats


class StableMatchFilter:
    """稳定匹配滤波器 - 整合所有组件"""
    
    def __init__(self, camera_matrix, 
                 history_frames=6, spatial_threshold=8.0, stability_threshold=3,
                 max_epipolar_error=2.0, base_confidence_threshold=0.2,
                 enable_outlier_removal=True, enable_densification=True,
                 outlier_method='statistical', densify_threshold=3,
                 enable_color_selective=True):
        
        self.stable_manager = StableMatchManager(
            history_frames=history_frames,
            spatial_threshold=spatial_threshold, 
            stability_threshold=stability_threshold
        )
        
        self.geometry_validator = GeometryValidator(
            camera_matrix=camera_matrix,
            max_epipolar_error=max_epipolar_error
        )
        
        self.threshold_controller = AdaptiveThresholdController(
            base_threshold=base_confidence_threshold
        )
        
        # 新增：离群点移除器
        self.outlier_remover = None
        self.enable_outlier_removal = enable_outlier_removal
        if enable_outlier_removal:
            self.outlier_remover = OutlierRemover(
                method=outlier_method,
                neighborhood_size=20,
                std_multiplier=2.0,
                isolation_threshold=0.1,
                min_cluster_size=5
            )
        
        # 新增：密集点稠密化器
        self.densification_enhancer = None
        self.enable_densification = enable_densification
        if enable_densification:
            self.densification_enhancer = DensificationEnhancer(
                min_density_threshold=densify_threshold,
                search_radius=15.0,
                max_interpolated_per_region=5,
                confidence_decay=0.8
            )
        
        # 新增：颜色选择性处理器
        self.color_processor = None
        self.enable_color_selective = enable_color_selective
        if enable_color_selective:
            self.color_processor = ColorSelectiveProcessor()
        
        self.enabled = True
        
        # 统计信息
        self.stats = {
            'original_matches': 0,
            'after_confidence': 0,
            'after_outlier_removal': 0,
            'after_stability': 0,
            'after_geometry': 0,
            'after_densification': 0,
            'densification_added': 0,
            'special_color_points': 0,
            'protected_from_removal': 0
        }
    
    def process_matches(self, mkpts0, mkpts1, mconf, frame0=None, frame1=None):
        """
        处理匹配点，返回稳定且几何一致的匹配
        
        Args:
            mkpts0, mkpts1: 原始匹配点对
            mconf: 原始匹配置信度
            frame0, frame1: 当前帧图像（用于颜色分析）
            
        Returns:
            filtered_mkpts0, filtered_mkpts1, filtered_mconf: 过滤后的匹配点
        """
        if not self.enabled or len(mkpts0) == 0:
            return mkpts0, mkpts1, mconf
        
        # 更新图像帧用于颜色分析
        if self.color_processor is not None:
            self.color_processor.update_frames(frame0, frame1)
        
        # 更新统计信息
        self.stats['original_matches'] = len(mkpts0)
        
        # 1. 自适应置信度过滤
        adaptive_threshold = self.threshold_controller.get_adaptive_threshold()
        conf_mask = mconf > adaptive_threshold
        
        if conf_mask.sum() == 0:
            # 如果过滤后无点，使用较低阈值
            conf_mask = mconf > (adaptive_threshold * 0.7)
            
        mkpts0_conf = mkpts0[conf_mask]
        mkpts1_conf = mkpts1[conf_mask] 
        mconf_conf = mconf[conf_mask]
        self.stats['after_confidence'] = len(mkpts0_conf)
        
        # 2. 颜色分析和选择性离群点移除
        color_classifications = np.array([])
        protected_indices = set()
        
        if self.enable_color_selective and self.color_processor is not None and len(mkpts0_conf) > 0:
            # 分析颜色
            color_classifications, color_details = self.color_processor.analyze_match_colors(mkpts0_conf, mkpts1_conf)
            
            # 识别需要保护的点（特殊颜色且设置为densify_only）
            for i in range(len(color_classifications)):
                if self.color_processor.should_preserve_from_removal(i, color_classifications):
                    protected_indices.add(i)
            
            self.stats['special_color_points'] = np.sum(color_classifications != None)
            self.stats['protected_from_removal'] = len(protected_indices)
        
        # 3. 选择性离群点移除（保护特殊颜色点）
        if self.enable_outlier_removal and self.outlier_remover is not None and len(mkpts0_conf) > 5:
            if len(protected_indices) > 0:
                # 分离保护的点和普通点
                all_indices = set(range(len(mkpts0_conf)))
                removal_indices = list(all_indices - protected_indices)
                protected_indices_list = list(protected_indices)
                
                if len(removal_indices) > 5:  # 只对足够多的普通点进行离群点移除
                    # 对非保护点进行离群点移除
                    mkpts0_removal = mkpts0_conf[removal_indices]
                    mkpts1_removal = mkpts1_conf[removal_indices]
                    mconf_removal = mconf_conf[removal_indices]
                    
                    mkpts0_clean, mkpts1_clean, mconf_clean = self.outlier_remover.remove_outliers(
                        mkpts0_removal, mkpts1_removal, mconf_removal
                    )
                    
                    # 合并保护的点和清理后的点
                    if len(protected_indices_list) > 0:
                        mkpts0_protected = mkpts0_conf[protected_indices_list]
                        mkpts1_protected = mkpts1_conf[protected_indices_list]
                        mconf_protected = mconf_conf[protected_indices_list]
                        
                        mkpts0_outlier = np.vstack([mkpts0_clean, mkpts0_protected])
                        mkpts1_outlier = np.vstack([mkpts1_clean, mkpts1_protected])
                        mconf_outlier = np.hstack([mconf_clean, mconf_protected])
                    else:
                        mkpts0_outlier, mkpts1_outlier, mconf_outlier = mkpts0_clean, mkpts1_clean, mconf_clean
                else:
                    # 保护点太多，跳过离群点移除
                    mkpts0_outlier, mkpts1_outlier, mconf_outlier = mkpts0_conf, mkpts1_conf, mconf_conf
            else:
                # 没有保护点，正常进行离群点移除
                mkpts0_outlier, mkpts1_outlier, mconf_outlier = self.outlier_remover.remove_outliers(
                    mkpts0_conf, mkpts1_conf, mconf_conf
                )
        else:
            mkpts0_outlier, mkpts1_outlier, mconf_outlier = mkpts0_conf, mkpts1_conf, mconf_conf
        self.stats['after_outlier_removal'] = len(mkpts0_outlier)
        
        # 4. 时域稳定性滤波
        mkpts0_stable, mkpts1_stable, mconf_stable = self.stable_manager.update_matches(
            mkpts0_outlier, mkpts1_outlier, mconf_outlier
        )
        self.stats['after_stability'] = len(mkpts0_stable)
        
        # 5. 几何一致性验证
        mkpts0_geom, mkpts1_geom, mconf_geom = self.geometry_validator.validate_matches(
            mkpts0_stable, mkpts1_stable, mconf_stable
        )
        self.stats['after_geometry'] = len(mkpts0_geom)
        
        # 6. 增强稠密化（对特殊颜色点增强处理）
        if self.enable_densification and self.densification_enhancer is not None and len(mkpts0_geom) >= 10:
            # 检查是否有特殊颜色需要增强稠密化
            enhanced_densification = False
            if len(color_classifications) > 0:
                # 重新分析最终点的颜色（因为经过了多轮过滤）
                final_color_classifications, _ = self.color_processor.analyze_match_colors(mkpts0_geom, mkpts1_geom)
                for i in range(len(final_color_classifications)):
                    if self.color_processor.should_enhance_densification(i, final_color_classifications):
                        enhanced_densification = True
                        break
            
            # 如果有特殊颜色，使用增强参数
            if enhanced_densification:
                # 临时调整稠密化参数
                original_max_per_region = self.densification_enhancer.max_interpolated_per_region
                original_search_radius = self.densification_enhancer.search_radius
                
                self.densification_enhancer.max_interpolated_per_region = 8  # 增加插值点
                self.densification_enhancer.search_radius = 20.0  # 扩大搜索半径
                
                mkpts0_final, mkpts1_final, mconf_final = self.densification_enhancer.densify_matches(
                    mkpts0_geom, mkpts1_geom, mconf_geom
                )
                
                # 恢复原始参数
                self.densification_enhancer.max_interpolated_per_region = original_max_per_region
                self.densification_enhancer.search_radius = original_search_radius
            else:
                # 正常稠密化
                mkpts0_final, mkpts1_final, mconf_final = self.densification_enhancer.densify_matches(
                    mkpts0_geom, mkpts1_geom, mconf_geom
                )
            
            self.stats['densification_added'] = len(mkpts0_final) - len(mkpts0_geom)
        else:
            mkpts0_final, mkpts1_final, mconf_final = mkpts0_geom, mkpts1_geom, mconf_geom
            self.stats['densification_added'] = 0
        
        self.stats['after_densification'] = len(mkpts0_final)
        
        # 7. 更新自适应阈值
        stable_count = len(mkpts0_stable)
        self.threshold_controller.update_threshold(self.stats['original_matches'], stable_count)
        
        return mkpts0_final, mkpts1_final, mconf_final
    
    def get_filter_statistics(self):
        """获取滤波器统计信息"""
        stable_stats = self.stable_manager.get_statistics()
        threshold_stats = self.threshold_controller.get_statistics()
        
        # 获取颜色统计信息
        color_stats = {}
        if self.color_processor is not None:
            color_stats = {
                'color_selective_enabled': self.enable_color_selective,
                'active_color_configs': len(self.color_processor.active_configs)
            }
        
        return {
            **stable_stats,
            **threshold_stats,
            **self.stats,
            **color_stats,
            'filter_enabled': self.enabled,
            'outlier_removal_enabled': self.enable_outlier_removal,
            'densification_enabled': self.enable_densification
        }
    
    def set_enabled(self, enabled):
        """启用或禁用滤波器"""
        self.enabled = enabled
    
    def set_outlier_removal_enabled(self, enabled):
        """启用或禁用离群点移除"""
        self.enable_outlier_removal = enabled and (self.outlier_remover is not None)
    
    def set_densification_enabled(self, enabled):
        """启用或禁用稠密化"""
        self.enable_densification = enabled and (self.densification_enhancer is not None)
    
    def set_outlier_method(self, method):
        """设置离群点移除方法"""
        if self.outlier_remover is not None:
            self.outlier_remover.method = method
    
    def set_color_selective_enabled(self, enabled):
        """启用或禁用颜色选择性处理"""
        self.enable_color_selective = enabled and (self.color_processor is not None)
    
    def add_color_config(self, name, color_center_hsv, tolerance_hsv, densify_only=True, enhanced_densification=True):
        """添加颜色配置"""
        if self.color_processor is not None:
            self.color_processor.add_color_config(name, color_center_hsv, tolerance_hsv, densify_only, enhanced_densification)
    
    def set_active_color_configs(self, config_names):
        """设置激活的颜色配置"""
        if self.color_processor is not None:
            self.color_processor.set_active_configs(config_names)
    
    def get_available_color_configs(self):
        """获取可用的颜色配置"""
        if self.color_processor is not None:
            return list(self.color_processor.color_configs.keys())
        return []
    
    def reset(self):
        """重置滤波器状态"""
        self.stable_manager.match_history.clear()
        self.stable_manager.stable_clusters = []
        self.stable_manager.cluster_histories = {}
        
        self.threshold_controller.current_threshold = self.threshold_controller.base_threshold
        self.threshold_controller.stability_history.clear()
        self.threshold_controller.match_count_history.clear()
        
        self.geometry_validator.reference_essential_matrix = None
        self.geometry_validator.reference_update_counter = 0