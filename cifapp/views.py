import os
import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
# 解析CIF文件并计算相关属性
from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
# 计算Average bond length
# from pymatgen import MPRester
# from pymatgen import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis import local_env
from pymatgen.core.structure import Structure, Composition
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from django.http import JsonResponse
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json

from datetime import datetime
from django.shortcuts import render

import xgboost as xgb
from django.views.decorators.csrf import csrf_exempt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import uuid
from matplotlib import cm
from matplotlib.colors import Normalize
from django.conf import settings
from xgboost import XGBRegressor

import joblib

VISIT_FILE = os.path.join(os.path.dirname(__file__), 'visit_data.json')

def get_client_ip(request):
    """获取客户端真实 IP"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def home(request):
    ip = get_client_ip(request)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 设置日志文件路径（假设在项目中的 logs 目录下）
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'ip_log.txt')

    # 写入日志（追加模式）
    with open(log_file, 'a') as f:
        f.write(f'{now} - {ip}\n')

    # 读取旧数据
    if os.path.exists(VISIT_FILE):
        with open(VISIT_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = {
            "total_visits": 0,
            "last_visit_time": "",
            "last_ip": ""
        }

    # 获取当前访问者 IP
    ip = get_client_ip(request)

    # 更新数据
    data['total_visits'] += 1
    data['last_visit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data['last_ip'] = ip

    # 保存回文件
    with open(VISIT_FILE, 'w') as f:
        json.dump(data, f)

    # 传给模板
    return render(request, 'cifapp/home.html', {
        'total_visits': data['total_visits'],
        'last_visit_time': data['last_visit_time'],
        'last_ip': data['last_ip']
    })


import os
import uuid
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import shap
from django.views.decorators.csrf import csrf_exempt

def upload_excel(request):
    return render(request, 'cifapp/ML.html', {})


def process_excel(request):
    if request.method == 'POST' and request.FILES.get('excel_file'):
        try:
            model_type = request.POST.get('model', 'rf').lower()
            excel_file = request.FILES['excel_file']

            # 保存上传文件
            fs = FileSystemStorage()
            filename = fs.save(excel_file.name, excel_file)
            file_path = fs.path(filename)

            # 读取数据
            df = pd.read_excel(file_path)
            df_numeric = df.select_dtypes(include=[np.number])
            if df_numeric.shape[1] < 2:
                raise ValueError("Excel 数据必须至少包含两列数值型数据")

            # 特征列与目标列
            feature_cols = df_numeric.columns[:-1]
            target_col = df_numeric.columns[-1]

            X = df_numeric[feature_cols].values
            y = df_numeric[target_col].values

            # 划分训练测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 模型选择
            if model_type == 'rf':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model_name = "Random Forest"
            elif model_type == 'gbdt':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model_name = "Gradient Boosting"
            elif model_type == 'xgb':
                model = XGBRegressor(
                    n_estimators=100, random_state=42,
                    use_label_encoder=False, eval_metric='rmse'
                )
                model_name = "XGBoost"
            else:
                raise ValueError("Unsupported model type.")

            # 训练
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # 保存模型和特征列
            model_dir = os.path.join(settings.MEDIA_ROOT, 'ml_results')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{model_type}_model.pkl")
            joblib.dump(model, model_path)

            feature_path = os.path.join(model_dir, f"{model_type}_features.pkl")
            joblib.dump(list(feature_cols), feature_path)

            # 评估指标
            r2_test = r2_score(y_test, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            r2_train = r2_score(y_train, y_train_pred)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

            # 绘图
            plt.figure(figsize=(7, 6))
            errors_train = np.abs(y_train - y_train_pred)
            errors_test = np.abs(y_test - y_test_pred)
            norm = Normalize(
                vmin=min(np.min(errors_train), np.min(errors_test)),
                vmax=max(np.max(errors_train), np.max(errors_test))
            )
            cmap = cm.get_cmap('coolwarm')

            plt.scatter(y_train, y_train_pred, c=errors_train, cmap=cmap, norm=norm,
                        marker='s', label='Training Set', alpha=0.8)
            plt.scatter(y_test, y_test_pred, c=errors_test, cmap=cmap, norm=norm,
                        marker='o', label='Testing Set', alpha=0.8)

            plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', label='y = x')
            plt.xlabel('Experimental Values (True)')
            plt.ylabel('Predicted Values')
            plt.title(f'{model_name}: Experimental vs Predicted Values')

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(np.concatenate([errors_train, errors_test]))
            cbar = plt.colorbar(sm)
            cbar.set_label('Distance to Diagonal (Error)', rotation=270, labelpad=15)

            textstr = (
                f'R² (Test): {r2_test:.4f}\n'
                f'RMSE (Test): {rmse_test:.4f}\n'
                f'R² (Train): {r2_train:.4f}\n'
                f'RMSE (Train): {rmse_train:.4f}'
            )
            plt.text(0.95, 0.15, textstr, transform=plt.gca().transAxes, fontsize=10,
                     horizontalalignment='right', verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            plt.legend()
            plt.tight_layout()

            # 保存图片
            unique_name = f'{model_type}_{uuid.uuid4().hex[:8]}.png'
            image_path = os.path.join(model_dir, unique_name)
            plt.savefig(image_path)
            plt.close()

            return JsonResponse({
                'status': 'ok',
                'image_url': f'{settings.MEDIA_URL}ml_results/{unique_name}'
            })

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'No file uploaded or invalid request.'}, status=400)


@csrf_exempt
def feature_engineering(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

    try:
        feature_type = request.POST.get('feature_type')
        excel_file = request.FILES.get('excel_file')
        model_type = request.POST.get('model', 'rf').lower()

        if not excel_file:
            return JsonResponse({'status': 'error', 'message': 'No Excel file uploaded'})

        # 读取训练时的特征信息
        feature_path = os.path.join(settings.MEDIA_ROOT, 'ml_results', f"{model_type}_features.pkl")
        if not os.path.exists(feature_path):
            return JsonResponse({'status': 'error', 'message': '请先训练模型'})

        feature_info = joblib.load(feature_path)
        
        # 处理不同的特征信息格式
        if isinstance(feature_info, dict):
            feature_cols = feature_info.get('feature_cols', [])
        elif isinstance(feature_info, list):
            feature_cols = feature_info
        else:
            feature_cols = []

        if not feature_cols:
            return JsonResponse({'status': 'error', 'message': '特征列为空'})

        # 读取 Excel 数据
        df = pd.read_excel(excel_file)
        
        # 检查特征列是否存在
        available_cols = [col for col in feature_cols if col in df.columns]
        missing_cols = [col for col in feature_cols if col not in df.columns]
        
        if missing_cols:
            if len(available_cols) < len(feature_cols) * 0.5:
                return JsonResponse({'status': 'error', 'message': f'缺失太多特征列: {missing_cols}'})
            else:
                feature_cols = available_cols

        if not feature_cols:
            return JsonResponse({'status': 'error', 'message': '没有可用的特征列'})

        # 选择特征数据
        X = df[feature_cols].select_dtypes(include=[np.number])
        
        # 处理缺失值
        if X.isnull().any().any():
            X = X.fillna(X.mean())
            
        if X.empty or len(X) == 0:
            return JsonResponse({'status': 'error', 'message': '没有有效的数值数据'})

        # 加载模型
        model_path = os.path.join(settings.MEDIA_ROOT, 'ml_results', f"{model_type}_model.pkl")
        if not os.path.exists(model_path):
            return JsonResponse({'status': 'error', 'message': '模型文件未找到'})

        model = joblib.load(model_path)

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 输出图像路径 - 使用固定命名确保结果一致
        image_name = f"{feature_type}_{model_type}_result.png"
        image_path = os.path.join(settings.MEDIA_ROOT, 'ml_results', image_name)
        image_url = f"{settings.MEDIA_URL}ml_results/{image_name}"

        # 根据特征类型生成不同的图像
        if feature_type == 'shap':
            success = _generate_shap_beeswarm(model, X, X_scaled, feature_cols, image_path, model_type)
            
        elif feature_type == 'bubble':
            success = _generate_bubble_matrix(X, image_path)
            
        elif feature_type == 'pareto':
            success = _generate_pareto_chart(model, X, X_scaled, feature_cols, image_path, model_type)
            
        else:
            return JsonResponse({'status': 'error', 'message': f'Unknown feature_type: {feature_type}'})

        if success:
            return JsonResponse({'status': 'ok', 'image_url': image_url})
        else:
            return JsonResponse({'status': 'error', 'message': '图表生成失败'})

    except Exception as e:
        print(f"特征工程错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': f'处理失败: {str(e)}'})

def _generate_shap_beeswarm(model, X, X_scaled, feature_cols, image_path, model_type):
    """生成SHAP蜂群图"""
    try:
        # 使用固定随机种子确保结果一致
        np.random.seed(42)
        
        # 使用适量样本（不要太多，否则图像太密集）
        sample_size = min(100, len(X_scaled))
        if len(X_scaled) > sample_size:
            sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
            X_sample = X_scaled[sample_idx]
            X_df_sample = X.iloc[sample_idx]
        else:
            X_sample = X_scaled
            X_df_sample = X

        # 选择解释器
        if model_type in ['rf', 'gbdt', 'xgb']:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_sample)

        # 计算SHAP值
        shap_values = explainer(X_sample)
        
        # 创建蜂群图
        plt.figure(figsize=(10, 8))
        
        # 使用beeswarm图（红蓝散点）
        shap.summary_plot(
            shap_values, 
            features=X_df_sample,
            feature_names=feature_cols,
            plot_type="dot",  # 这是红蓝散点图
            show=False,
            max_display=15    # 限制显示的特征数量
        )
        
        plt.tight_layout()
        plt.savefig(image_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        return True
        
    except Exception as e:
        print(f"SHAP蜂群图生成失败: {e}")
        return False

def _generate_bubble_matrix(X, image_path):
    """生成气泡矩阵图 """
    try:
        # 计算相关性矩阵
        corr_matrix = X.corr()
        
        # 选择相关性最高的特征
        top_n = min(10, len(X.columns))
        corr_abs_mean = corr_matrix.abs().mean().sort_values(ascending=False)
        top_features = corr_abs_mean.head(top_n).index
        corr_top = corr_matrix.loc[top_features, top_features]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))
        
        n_features = len(top_features)
        
        # 创建气泡 - 修复：确保每次调用scatter都有返回值
        scatters = []
        for i in range(n_features):
            for j in range(n_features):
                if i != j:  # 不对角线
                    corr_val = corr_top.iloc[i, j]
                    color = 'red' if corr_val > 0 else 'blue'
                    size = 300 + abs(corr_val) * 700
                    alpha_val = 0.6 + abs(corr_val) * 0.4
                    
                    # 收集scatter对象
                    scatter = ax.scatter(i, j, s=size, c=color, alpha=alpha_val, 
                                       edgecolors='black', linewidth=0.5)
                    scatters.append(scatter)
                    
                    # 添加相关系数值
                    if abs(corr_val) > 0.3:
                        ax.text(i, j, f'{corr_val:.2f}', 
                               ha='center', va='center', 
                               fontsize=8, fontweight='bold',
                               color='white' if abs(corr_val) > 0.7 else 'black')
        
        # 设置坐标轴
        ax.set_xticks(range(n_features))
        ax.set_yticks(range(n_features))
        ax.set_xticklabels(top_features, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(top_features, fontsize=10)
        ax.set_xlabel('Features')
        ax.set_ylabel('Features')
        ax.set_title('Feature Correlation Bubble Matrix')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(image_path, bbox_inches='tight', dpi=150)
        plt.close()
        return True
        
    except Exception as e:
        print(f"气泡矩阵图生成失败: {e}")
        return False

def _generate_pareto_chart(model, X, X_scaled, feature_cols, image_path, model_type):
    """生成帕累托图 """
    try:
        # 使用固定随机种子确保结果一致
        np.random.seed(42)
        
        # 获取特征重要性
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            # 使用SHAP计算重要性
            sample_size = min(50, len(X_scaled))
            X_sample = X_scaled[:sample_size]
            
            if model_type in ['rf', 'gbdt', 'xgb']:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X_sample)
            
            shap_values = explainer(X_sample)
            if hasattr(shap_values, 'values'):
                importances = np.abs(shap_values.values).mean(0)
            else:
                importances = np.ones(len(feature_cols)) / len(feature_cols)
        
        # 创建帕累托图
        feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
        top_n = min(12, len(feature_cols))
        top_feat_imp = feat_imp.head(top_n)
        cumulative = top_feat_imp.cumsum() / top_feat_imp.sum()

        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # 条形图 - 特征重要性
        bars = ax1.bar(range(len(top_feat_imp)), top_feat_imp.values, 
                      color='skyblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
        ax1.set_ylabel('Feature Importance', color='darkblue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='darkblue')
        ax1.set_xticks(range(len(top_feat_imp)))
        ax1.set_xticklabels(top_feat_imp.index, rotation=45, ha='right', fontsize=10)
        
        # 累积曲线
        ax2 = ax1.twinx()
        line = ax2.plot(range(len(top_feat_imp)), cumulative.values, 
                       color='red', marker='o', linewidth=2, markersize=4)
        ax2.set_ylabel('Cumulative Ratio', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(top_feat_imp.values):
            ax1.text(i, v + 0.01, f'{v:.3f}', 
                    ha='center', va='bottom', fontsize=8, rotation=0)
        
        plt.title('Pareto Chart - Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(image_path, bbox_inches='tight', dpi=150)
        plt.close()
        return True
        
    except Exception as e:
        print(f"帕累托图生成失败: {e}")
        return False


@csrf_exempt
def predict(request):
    if request.method == 'POST' and request.FILES.get('predict_file'):
        try:
            file = request.FILES['predict_file']
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            if len(df) != 1:
                return JsonResponse({"error": "文件中必须只有一行数据"}, status=400)

            # 读取模型特征列
            model_type = request.POST.get('model', 'rf').lower()
            feature_path = os.path.join(settings.MEDIA_ROOT, 'ml_results', f"{model_type}_features.pkl")
            if not os.path.exists(feature_path):
                return JsonResponse({"error": "训练时特征列文件未找到"}, status=400)

            feature_cols = joblib.load(feature_path)
            X_new = df[feature_cols].select_dtypes(include=[np.number]).values

            model_path = os.path.join(settings.MEDIA_ROOT, "ml_results", f"{model_type}_model.pkl")
            if not os.path.exists(model_path):
                return JsonResponse({"error": f"没有找到已训练的 {model_type} 模型"}, status=400)

            model = joblib.load(model_path)
            prediction = model.predict(X_new)[0]

            return JsonResponse({"prediction": float(prediction)})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "无效请求"}, status=400)


def average_bond_length(cif_file):
    """
    demo1 平均键长
    :param cif_file: cif文件
    :return: 平均键长
    """
    # 载入结构并初始化分析器
    structure = Structure.from_file(cif_file)
    analyzer = CrystalNN()

    # 存储有效键长（避免重复计数）
    bond_lengths = []
    processed_pairs = set()

    # 遍历每个原子识别化学键
    for i in range(len(structure)):
        neighbors = analyzer.get_nn_info(structure, i)
        for neighbor in neighbors:
            j = neighbor['site_index']
            # 确保每对原子只计算一次
            if j > i and (i, j) not in processed_pairs:
                distance = structure[i].distance(structure[j])
                bond_lengths.append(distance)
                processed_pairs.add((i, j))

    average_length = np.mean(bond_lengths).item() if bond_lengths else 0.0
    return average_length


def std_bond_length(cif_file):
    """
    demo2 键长标准偏差
    :param cif_file: cif文件
    :return: 键长标准偏差
    """
    # 载入结构并初始化分析器
    structure = Structure.from_file(cif_file)
    analyzer = CrystalNN()

    # 存储有效键长（避免重复计数）
    bond_lengths = []
    processed_pairs = set()

    # 遍历每个原子识别化学键
    for i in range(len(structure)):
        neighbors = analyzer.get_nn_info(structure, i)
        for neighbor in neighbors:
            j = neighbor['site_index']
            # 确保每对原子只计算一次
            if j > i and (i, j) not in processed_pairs:
                distance = structure[i].distance(structure[j])
                bond_lengths.append(distance)
                processed_pairs.add((i, j))

    # 计算标准偏差（若无有效键返回0.0）
    std_dev = np.std(bond_lengths).item() if len(bond_lengths) > 1 else 0.0

    return std_dev


def average_bond_strength(cif_file):
    """
    demo3 平均键强度
    :param cif_file: cif文件
    :return: 平均键强度
    """
    # 载入结构并初始化分析器
    structure = Structure.from_file(cif_file)
    analyzer = CrystalNN()

    # 存储有效键强度（避免重复计数）
    bond_strengths = []
    processed_pairs = set()

    # 遍历每个原子识别化学键
    for i in range(len(structure)):
        neighbors = analyzer.get_nn_info(structure, i)
        for neighbor in neighbors:
            j = neighbor['site_index']
            # 确保每对原子只计算一次
            if j > i and (i, j) not in processed_pairs:
                distance = structure[i].distance(structure[j])
                # 添加物理约束（1.0 Å < d < 3.0 Å）
                if 1.0 < distance < 3.0:
                    bond_strength = 1 / distance
                    bond_strengths.append(bond_strength)
                    processed_pairs.add((i, j))

    # 计算平均键强度（若无有效键返回0.0）
    average_strength = np.mean(bond_strengths).item() if bond_strengths else 0.0
    return average_strength


def bond_strength_std(cif_file):
    """
    demo4 键强度的标准偏差
    :param cif_file: cif文件
    :return: 键强度的标准偏差
    """
    # 读取 CIF 文件
    structure = Structure.from_file(cif_file)

    # 存储所有的键强度
    bond_strengths = []

    # 获取每个原子的邻居，使用 cutoff 来设置半径
    for site in structure.sites:
        neighbors = structure.get_neighbors(site, 3.0)  # 3.0 Å 的半径
        for neighbor in neighbors:
            # 计算键长
            bond_length = site.distance(neighbor[0])  # neighbor[0] 是邻居原子
            if bond_length > 0:  # 避免除以零的情况
                # 假设键强度与键长的倒数成正比
                bond_strength = 1 / bond_length
                bond_strengths.append(bond_strength)

    # 计算键强度的标准偏差
    bond_strengths = np.array(bond_strengths)
    std_dev_bond_strength = np.std(bond_strengths)

    return std_dev_bond_strength


def average_covalency(C, cif_file):
    """
    demo5 平均共价性
    :param C: 经验公式常数（根据原子类型选择合适的常数）
    :param cif_file: cif文件
    :return: 平均共价性
    """
    if C is None:
        C = 0.5
    # 读取 CIF 文件
    structure = Structure.from_file(cif_file)
    # 存储所有的共价性
    covalencies = []
    # 获取每个原子的邻居，使用 cutoff 来设置半径
    for site in structure.sites:
        neighbors = structure.get_neighbors(site, 3.0)  # 3.0 Å 的半径
        for neighbor in neighbors:
            # 计算键长
            bond_length = site.distance(neighbor[0])  # neighbor[0] 是邻居原子
            if bond_length > 0:  # 避免除以零的情况
                # 使用经验公式计算共价性
                covalency = C / bond_length
                covalencies.append(covalency)
    # 计算平均共价性
    covalencies = np.array(covalencies)
    avg_covalency = np.mean(covalencies)
    return avg_covalency


def covalency_std(C, cif_file):
    """
    demo6 共价性的标准偏差
    :param C: 经验公式常数（根据原子类型选择合适的常数）
    :param cif_file: CIF 文件
    :return: 共价性的标准偏差
    """
    if C is None: C = 0.5
    # 读取 CIF 文件
    structure = Structure.from_file(cif_file)

    # 存储所有的共价性
    covalencies = []

    # 获取每个原子的邻居，使用 cutoff 来设置半径
    for site in structure.sites:
        neighbors = structure.get_neighbors(site, 3.0)  # 3.0 Å 的半径
        for neighbor in neighbors:
            # 计算键长
            bond_length = site.distance(neighbor[0])  # neighbor[0] 是邻居原子
            if bond_length > 0:  # 避免除以零的情况
                # 使用经验公式计算共价性
                covalency = C / bond_length
                covalencies.append(covalency)

    # 计算共价性的标准偏差
    covalencies = np.array(covalencies)
    covalency_std_dev = np.std(covalencies)

    return covalency_std_dev


# 扩展的键能参数数据库 (A, n)  用于Demo7和Demo8
bond_params = {
    # 主要氧化物键
    ("Al", "O"): (650, 1.8),  # Al₂O₃
    ("Si", "O"): (800, 1.9),  # SiO₂
    ("Ti", "O"): (720, 1.8),  # TiO₂
    ("Fe", "O"): (700, 1.9),  # Fe₂O₃, Fe₃O₄
    ("Zn", "O"): (600, 1.8),  # ZnO
    ("Mg", "O"): (500, 1.7),  # MgO
    ("Ca", "O"): (450, 1.7),  # CaO
    ("Na", "O"): (400, 1.6),  # Na₂O
    ("K", "O"): (350, 1.6),  # K₂O

    # 卤化物键
    ("Na", "Cl"): (450, 1.7),  # NaCl
    ("K", "Cl"): (420, 1.6),  # KCl
    ("Mg", "Cl"): (460, 1.7),  # MgCl₂
    ("Ca", "Cl"): (470, 1.7),  # CaCl₂

    # 过渡金属氧化物
    ("Cr", "O"): (750, 1.9),  # Cr₂O₃
    ("Mn", "O"): (730, 1.9),  # MnO₂
    ("Co", "O"): (720, 1.8),  # CoO
    ("Ni", "O"): (710, 1.8),  # NiO
    ("Cu", "O"): (680, 1.8),  # Cu₂O, CuO

    # 其他常见化合物
    ("C", "O"): (750, 1.8),  # CO₂
    ("B", "O"): (800, 1.9),  # B₂O₃
    ("P", "O"): (780, 1.9),  # P₂O₅
    ("S", "O"): (700, 1.8),  # SO₂, SO₃

    ("Li", "O"): (420, 1.7),
    ("Mo", "O"): (770, 1.85),
}


# 获取键能参数 (A, n)
def get_bond_params(elem1, elem2):
    """
    获取键能参数 A 和 n
    :param elem1: 元素1
    :param elem2: 元素2
    :return: (A, n) 对应的参数, 如果没有数据则返回默认值 (350, 1.5)
    """
    key = (elem1, elem2)
    reverse_key = (elem2, elem1)
    return bond_params.get(key, bond_params.get(reverse_key, (350, 1.5)))  # 默认值


# 计算平均键能
def average_bond_energy(cif_file):
    """
    Demo7
    计算晶体结构的平均键能
    :param cif_file: CIF 文件路径
    :return: 平均键能 (kJ/mol)
    """
    try:
        # 载入结构
        structure = Structure.from_file(cif_file)
        analyzer = CrystalNN()
    except Exception as e:
        print(f"Error loading CIF file: {e}")
        return None

    bond_energies = []
    processed_pairs = set()

    # 遍历每个原子识别化学键
    for i in range(len(structure)):
        site_i = structure[i]
        elem_i = site_i.specie.symbol
        neighbors = analyzer.get_nn_info(structure, i)

        for neighbor in neighbors:
            j = neighbor['site_index']
            site_j = structure[j]
            elem_j = site_j.specie.symbol

            # 确保每对原子只计算一次
            if j > i and (i, j) not in processed_pairs:
                distance = site_i.distance(site_j)

                # 物理约束 (1.0 < d < 3.0 Å)
                if 1.0 < distance < 3.0:
                    # 获取元素对参数
                    A, n = get_bond_params(elem_i, elem_j)

                    # 计算键能
                    bond_energy = A / (distance ** n)
                    bond_energies.append(bond_energy)
                    processed_pairs.add((i, j))

    # 计算平均键能（若无有效键返回 0.0）
    avg_bond_energy = np.mean(bond_energies).item() if bond_energies else 0.0
    return avg_bond_energy

def average_bond_energy_per_unit_volume(cif_file):
    """
    demo8 单位体积的平均键能
    :param A: 常数 A（单位：kJ/mol），具体数值根据键类型调整
    :param n: 常数 n，通常在 1 到 2 之间
    :param cif_file: CIF 文件
    :return: 单位体积的平均键能
    """
    try:
        structure = Structure.from_file(cif_file)
    except Exception as e:
        print(f"Error loading CIF file: {e}")
        return None

    avg_bond_energy = average_bond_energy(cif_file)  # **确保是均值，而非标准差**

    if avg_bond_energy is None:
        print("Error: Unable to calculate average bond energy.")
        return None

    volume = structure.volume
    avg_bond_energy_per_unit_volume = avg_bond_energy / volume if volume > 0 else 0.0

    return avg_bond_energy_per_unit_volume


def bond_energy_std( cif_file):
    """
    demo9 键能的标准偏差
    :param A: 常数 A（单位：kJ/mol），具体数值根据键类型调整
    :param n: 常数 n，通常在 1 到 2 之间
    :param cif_file: CIF 文件
    :return: 键能的标准偏差
    """
    # 读取 CIF 结构文件
    structure = Structure.from_file(cif_file)

    bond_energies = []
    neighbor_radius = 3.5

    for site in structure.sites:
        neighbors = structure.get_neighbors(site, neighbor_radius)

        for neighbor in neighbors:
            elem1 = site.species_string
            elem2 = neighbor[0].species_string
            A, n = get_bond_params(elem1, elem2)
            bond_length = site.distance(neighbor[0])

            if bond_length > 0:
                bond_energy = A / (bond_length ** n)
                bond_energies.append(bond_energy)

    # **计算标准偏差**
    if len(bond_energies) > 0:
        std_dev = np.std(bond_energies)
    else:
        std_dev = 0.0

    return std_dev


def bond_valence_std_avg(r0, b, cif_file):
    """
    demo10 键价标准偏差的平均值
    :param r0: 参考键长 单位A（埃）
    :param b: 键价参数  单位A（埃）
    :param cif_file: CIF 文件
    :return: 键价标准偏差的平均值
    """
    if r0 is None:
        r0 = 1.6
    if b is None:
        b = 0.37
    # 读取 CIF 文件
    structure = Structure.from_file(cif_file)

    # 初始化保存每个原子邻居的键价标准偏差的列表
    bond_valence_stds = []

    # 获取每个原子的邻居，计算每个键的键价并得到标准偏差
    for site in structure.sites:
        bond_valences = []

        # 获取该原子的邻居
        neighbors = structure.get_neighbors(site, 3.0)  # 设置合适的半径
        for neighbor in neighbors:
            # 获取键长（与邻居原子之间的距离）
            bond_length = site.distance(neighbor[0])
            if bond_length > 0:
                # 使用键价模型公式计算键价
                bond_valence = np.exp((r0 - bond_length) / b)
                bond_valences.append(bond_valence)

        # 计算该原子的键价标准偏差
        if bond_valences:
            bond_valence_std = np.std(bond_valences)
            bond_valence_stds.append(bond_valence_std)

    # 计算所有原子键价标准偏差的平均值
    if bond_valence_stds:
        avg_bond_valence_std = np.mean(bond_valence_stds)
    else:
        avg_bond_valence_std = 0.0  # 如果没有有效的键价数据，返回 0

    return avg_bond_valence_std


def average_bvs_deviation(ion_radii, nominal_valence, cif_file):
    """
    demo11 平均BVS偏差
    :param ion_radii: 离子半径表（单位: Å），
    :param nominal_valence: 名义化学价（以常见元素为例）
    :param cif_file: CIF 文件
    :return: 平均BVS偏差
    """
    if ion_radii is None:
        ion_radii = {
            'Li': 0.76, 'Na': 0.97, 'K': 1.38, 'Rb': 1.53, 'Cs': 1.67, 'Mg': 0.72,
            'Ca': 1.00, 'Sr': 1.18, 'Ba': 1.35, 'Sc': 0.74, 'Ti': 0.61, 'V': 0.60,
            'Cr': 0.52, 'Mn': 0.55, 'Fe': 0.64, 'Co': 0.58, 'Ni': 0.69, 'Cu': 0.73,
            'Zn': 0.74, 'Ag': 1.26, 'Au': 1.44, 'Al': 0.53, 'Ga': 0.62, 'In': 0.81,
            'Ge': 0.57, 'Si': 0.42, 'Sn': 0.69, 'Pb': 1.19, 'O': 0.60, 'F': 0.64,
            'Cl': 0.99, 'Br': 1.14, 'I': 1.33, 'S': 0.84, 'P': 1.05, 'N': 0.75
        }
    if nominal_valence is None:
        nominal_valence = {
            'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1, 'Mg': 2,
            'Ca': 2, 'Sr': 2, 'Ba': 2, 'Sc': 3, 'Ti': 4, 'V': 5,
            'Cr': 6, 'Mn': 2, 'Fe': 3, 'Co': 2, 'Ni': 2, 'Cu': 1,
            'Zn': 2, 'Ag': 1, 'Au': 3, 'Al': 3, 'Ga': 3, 'In': 3,
            'Ge': 4, 'Si': 4, 'Sn': 4, 'Pb': 2, 'O': -2, 'F': -1,
            'Cl': -1, 'Br': -1, 'I': -1, 'S': -2, 'P': -3, 'N': -3
        }
    # 读取 CIF 文件
    structure = Structure.from_file(cif_file)

    # 初始化 BVS 偏差和总数
    bvs_deviations = []

    for site in structure.sites:
        # 提取元素符号并去除电荷信息
        element = site.species_string.split()[0]  # 如果电荷存在，只取元素符号
        element = ''.join(filter(str.isalpha, element))  # 只保留字母部分

        # 打印元素符号，检查是否正确
        # print(f"元素符号: {element}")

        if element in nominal_valence:
            # 获取离子半径（Å）
            radius = ion_radii.get(element, None)
            if radius:
                # 获取配位原子
                neighbors = structure.get_neighbors(site, 3.0)

                # 计算每个原子的 BVS
                bvs_sum = 0
                for neighbor in neighbors:
                    d = site.distance(neighbor)  # 计算原子之间的距离
                    # 使用简单的BVS公式计算
                    bvs_sum += np.exp((radius - d) / 0.37)

                # 计算BVS偏差
                nominal = nominal_valence[element]
                bvs = bvs_sum / len(neighbors) if len(neighbors) > 0 else 0  # 平均BVS值
                bvs_deviation = abs(bvs - nominal)

                # 添加偏差
                bvs_deviations.append(bvs_deviation)

    # 计算所有原子的 BVS 偏差的平均值
    average_bvs_dev = np.mean(bvs_deviations) if bvs_deviations else 0
    return average_bvs_dev


def total_ionic_volume(ion_radii, cif_file):
    """
    demo12 总离子体积
    :param ion_radii: 离子半径表（单位: Å）
    :param cif_file: CIF 文件
    :return: 总离子体积
    """
    if ion_radii is None:
        ion_radii = {
            'Li': 0.76, 'Na': 0.97, 'K': 1.38, 'Rb': 1.53, 'Cs': 1.67, 'Mg': 0.72,
            'Ca': 1.00, 'Sr': 1.18, 'Ba': 1.35, 'Sc': 0.74, 'Ti': 0.61, 'V': 0.60,
            'Cr': 0.52, 'Mn': 0.55, 'Fe': 0.64, 'Co': 0.58, 'Ni': 0.69, 'Cu': 0.73,
            'Zn': 0.74, 'Ag': 1.26, 'Au': 1.44, 'Al': 0.53, 'Ga': 0.62, 'In': 0.81,
            'Ge': 0.57, 'Si': 0.42, 'Sn': 0.69, 'Pb': 1.19, 'O': 0.60, 'F': 0.64,
            'Cl': 0.99, 'Br': 1.14, 'I': 1.33, 'S': 0.84, 'P': 1.05, 'N': 0.75
        }
    # 读取 CIF 文件
    structure = Structure.from_file(cif_file)
    total_ionic_volume = 0
    for site in structure.sites:
        element = site.species_string.split()[0]  # 如果电荷存在，只取元素符号
        element = ''.join(filter(str.isalpha, element))  # 只保留字母部分
        if element in ion_radii:  # 检查元素是否在离子半径表中
            radius = ion_radii[element]  # 获取离子半径（Å）
            ionic_volume = (4 / 3) * np.pi * radius ** 3  # 计算离子体积
            total_ionic_volume += ionic_volume

    return total_ionic_volume


def packing_fraction(ion_radii, cif_file):
    """
    demo13 填充因子
    :param ion_radii: 离子半径表（单位: Å）
    :param cif_file: CIF 文件
    :return: 填充因子
    """
    if ion_radii is None:
        ion_radii = {
            # 阳离子（常见价态）
            'Li': 0.76, 'Na': 1.02, 'K': 1.38, 'Rb': 1.52, 'Cs': 1.67,
            'Mg': 0.72, 'Ca': 1.00, 'Sr': 1.18, 'Ba': 1.35,
            'Al': 0.54, 'Sc': 0.74, 'Ti': 0.61, 'V': 0.60,
            'Cr': 0.52, 'Mn': 0.55, 'Fe': 0.64, 'Co': 0.58,
            'Ni': 0.69, 'Cu': 0.73, 'Zn': 0.74,
            'Ag': 1.15, 'Au': 1.37,
            # 阴离子（常见价态）
            'O': 1.40, 'F': 1.33, 'Cl': 1.81, 'Br': 1.96, 'I': 2.20,
            'S': 1.84, 'P': 2.12, 'N': 1.71
        }

    structure = Structure.from_file(cif_file)
    cell_volume = structure.volume  # 单位应为 Å³
    total_atomic_volume = 0.0

    for site in structure.sites:
        element = site.specie.symbol
        radius = ion_radii.get(element, 0.0)
        if radius > 0:
            atomic_volume = (4 / 3) * np.pi * radius ** 3
            total_atomic_volume += atomic_volume

    packing_fraction = total_atomic_volume / cell_volume
    return packing_fraction


def ionic_volume_per_atom(ion_radii, cif_file):
    """
    demo14 每个原子的离子体积
    :param ion_radii: 离子半径表（单位: Å）
    :param cif_file: CIF 文件
    :return: 每个原子的离子体积
    """
    if ion_radii is None:
        ion_radii = {
            # 阳离子（常见价态）
            'Li': 0.76, 'Na': 1.02, 'K': 1.38, 'Rb': 1.52, 'Cs': 1.67,
            'Mg': 0.72, 'Ca': 1.00, 'Sr': 1.18, 'Ba': 1.35,
            'Al': 0.54, 'Sc': 0.74, 'Ti': 0.61, 'V': 0.60,
            'Cr': 0.52, 'Mn': 0.55, 'Fe': 0.64, 'Co': 0.58,
            'Ni': 0.69, 'Cu': 0.73, 'Zn': 0.74,
            'Ag': 1.15, 'Au': 1.37,
            # 阴离子（常见价态）
            'O': 1.40, 'F': 1.33, 'Cl': 1.81, 'Br': 1.96, 'I': 2.20,
            'S': 1.84, 'P': 2.12, 'N': 1.71
        }
    # 读取 CIF 文件
    structure = Structure.from_file(cif_file)

    # 计算原子的离子体积并求平均
    total_ionic_volume = 0.0
    num_atoms = len(structure)

    for site in structure.sites:
        element = site.specie.symbol
        radius = ion_radii.get(element, 0.0)  # 获取元素的离子半径
        if radius > 0:
            ionic_volume = (4 / 3) * np.pi * radius ** 3
            total_ionic_volume += ionic_volume

    # 计算每个原子的离子体积
    ionic_volume_per_atom = total_ionic_volume / num_atoms

    return ionic_volume_per_atom


def dielectric_polarizability(cif_file):
    """
    demo15 极化率
    :param cif_file: CIF 文件
    :return: 极化率
    """
    polarizability_table = {
        # 阳离子
        'Li': 0.03, 'Na': 0.18, 'K': 0.84, 'Rb': 1.40, 'Cs': 2.42,
        'Mg': 0.09, 'Ca': 0.47, 'Sr': 0.86, 'Ba': 1.55,
        'Al': 0.05, 'Ti': 0.19, 'Fe': 0.23, 'Zn': 0.28,
        # 阴离子
        'O': 3.88, 'F': 1.04, 'Cl': 3.66, 'Br': 4.81, 'I': 7.10,
        'S': 5.20, 'N': 1.10, 'P': 3.00
    }

    structure = Structure.from_file(cif_file)
    total_polarizability = 0.0
    valid_atoms = 0

    for site in structure:
        element = site.specie.symbol
        alpha = polarizability_table.get(element, None)
        if alpha is not None:
            total_polarizability += alpha
            valid_atoms += 1

    if valid_atoms == 0:
        return 0.0
    average_polarizability = total_polarizability / valid_atoms
    return round(average_polarizability, 4)


def dielectric_polarizability_per_atom(cif_file):
    """
    demo16 所有原子的平均极化率
    :param cif_file: CIF 文件
    :return: 所有原子的平均极化率
    """
    polarizability_table = {
        # 阳离子
        'Li': 0.03, 'Na': 0.18, 'K': 0.84, 'Rb': 1.40, 'Cs': 2.42,
        'Mg': 0.09, 'Ca': 0.47, 'Sr': 0.86, 'Ba': 1.55,
        'Al': 0.05, 'Ti': 0.19, 'Fe': 0.23, 'Zn': 0.28,
        # 阴离子
        'O': 3.88, 'F': 1.04, 'Cl': 3.66, 'Br': 4.81, 'I': 7.10,
        'S': 5.20, 'N': 1.10, 'P': 3.00
    }

    structure = Structure.from_file(cif_file)
    nn = CrystalNN()  # 使用更稳健的CrystalNN
    total_polarizability = 0.0
    valid_atoms = 0

    for site in structure:
        element = site.specie.symbol
        alpha_base = polarizability_table.get(element, None)
        if alpha_base is None:
            continue

        try:
            # 计算配位数并确保最小为1
            cn = max(len(nn.get_nn_info(structure, site.index)), 1)
        except:
            cn = 1  # 异常时默认配位数为1

        # 修正模型（配位数越高，极化率越低）
        correction = 1.0 / np.sqrt(cn)
        alpha_corrected = alpha_base * correction

        total_polarizability += alpha_corrected
        valid_atoms += 1

    if valid_atoms == 0:
        return 0.0
    average = total_polarizability / valid_atoms
    return round(average, 2)


def dielectric_polarizability_per_volume(cif_file):
    """
    demo17 单位体积的极化率
    :param cif_file: CIF 文件
    :return: 单位体积的极化率
    """
    polarizability_table = {
        # 阳离子
        'Li': 0.03, 'Na': 0.18, 'K': 0.84, 'Rb': 1.40, 'Cs': 2.42,
        'Mg': 0.09, 'Ca': 0.47, 'Sr': 0.86, 'Ba': 1.55,
        'Al': 0.05, 'Ti': 0.19, 'Fe': 0.23, 'Zn': 0.28,
        # 阴离子
        'O': 3.88, 'F': 1.04, 'Cl': 3.66, 'Br': 4.81, 'I': 7.10,
        'S': 5.20, 'N': 1.10, 'P': 3.00
    }
    # 读取CIF文件
    structure = Structure.from_file(cif_file)

    # 获取晶体体积
    volume = structure.volume

    # 计算每个原子的邻居数目（此为极化率的简化估算）
    polarizability_sum = 0

    for site in structure:
        # 获取该原子半径为 3.0 Å 的邻居原子
        neighbors = structure.get_neighbors(site, 3.0)
        # 假设邻居数目与极化率成正比
        polarizability_sum += len(neighbors)

    # 计算单位体积的极化率
    polarizability_per_volume = polarizability_sum / volume

    return polarizability_per_volume


def lattice_constant_std(cif_file):
    """
    demo18 每个方向的标准偏差
    :param cif_file:
    :return: 每个方向的标准偏差
    """
    # 读取CIF文件
    structure = Structure.from_file(cif_file)

    # 获取晶格常数边长（a, b, c）
    a, b, c = structure.lattice.abc

    # 计算晶格常数的标准偏差（假设多个CIF文件代表多个样品）
    # 这里只考虑一个CIF文件的情况，假设已知其他多个样品的晶格常数
    # 例如: lattice_constants = [a1, b1, c1, a2, b2, c2, ...]
    lattice_constants = [a, b, c]  # 对于一个样品，仅包含一个晶格常数

    # 计算每个方向的标准偏差
    lattice_std_dev = np.std(lattice_constants, axis=0)

    return lattice_std_dev


def angle_std(cif_file):
    """
    demo19 角度的标准偏差
    :param cif_file: CIF 文件
    :return: 角度的标准偏差
    """
    # 读取CIF文件
    structure = Structure.from_file(cif_file)

    # 获取晶胞参数：a, b, c 和 α, β, γ
    lattice = structure.lattice
    a, b, c = lattice.abc  # 晶格常数
    alpha, beta, gamma = lattice.angles  # 晶胞角度

    # 计算角度的标准偏差
    angles = np.array([alpha, beta, gamma])
    std_dev = np.std(angles)

    return std_dev

def formula_units_per_cell(structure):
    """计算晶胞中包含的化学式单元数（Z值）"""
    formula = structure.composition.reduced_formula
    comp = Composition(formula)
    # 计算每个化学式单元的原子总数（如Al₂O₃为2+3=5）
    atoms_per_formula = sum(comp.values())
    # 计算晶胞中总原子数
    total_atoms = structure.composition.num_atoms
    # Z = 总原子数 / 每个化学式单元的原子数
    z = total_atoms / atoms_per_formula
    return round(z)

def cell_volume(cif_file):
    """
    demo20 单位晶胞体积
    :param cif_file: CIF 文件
    :return: 单位晶胞体积
    """
    structure = Structure.from_file(cif_file)
    analyzer = SpacegroupAnalyzer(structure)
    conventional_structure = analyzer.get_conventional_standard_structure()

    # 计算晶胞体积和化学式单元数
    conventional_volume = conventional_structure.volume
    z = formula_units_per_cell(conventional_structure)

    # 计算每个化学式单元的体积
    volume_per_formula = conventional_volume / z
    return round(volume_per_formula, 2)


def relative_molecular_mass(cif_file):
    """
    demo21 所有原子的相对分子质量
    :param cif_file: CIF 文件
    :return: 所有原子的相对分子质量
    """
    structure = Structure.from_file(cif_file)
    analyzer = SpacegroupAnalyzer(structure)
    conventional_structure = analyzer.get_conventional_standard_structure()

    # 计算总原子质量和化学式单元数
    total_mass = sum(site.specie.atomic_mass for site in conventional_structure)
    z = formula_units_per_cell(conventional_structure)

    # 计算每个化学式单元的相对分子质量
    mass_per_formula = total_mass / z
    return round(mass_per_formula, 2)


def theoretical_density(cif_file):
    """
    demo22 计算理论密度
    :param cif_file: CIF 文件
    :return: 计算理论密度
    """
    structure = Structure.from_file(cif_file)

    # 获取晶胞体积（单位：Å³ → 转换为 cm³）
    cell_volume_cm3 = structure.volume * 1e-24

    # 计算晶胞中所有原子的总摩尔质量（考虑部分占据）
    molar_mass = 0.0
    for site in structure.sites:
        for elem, occupancy in site.species.items():  # 处理部分占据
            molar_mass += elem.atomic_mass * occupancy

    # 阿伏伽德罗常数（mol⁻¹）
    avogadro = 6.02214076e23

    # 理论密度公式修正：去除了多余的原子数目乘法
    density = (molar_mass) / (cell_volume_cm3 * avogadro)
    return density


def unit_cell_volume_per_atom(cif_file):
    """
    demo23 单位原子晶胞体积
    :param cif_file: CIF 文件
    :return: 单位原子晶胞体积
    """
    structure = Structure.from_file(cif_file)

    # 获取晶胞体积 (单位: Å³)
    cell_volume = structure.volume

    # 获取晶胞中的原子数
    num_atoms_in_cell = len(structure.sites)

    # 计算单位原子晶胞体积 (单位: Å³/atom)
    return cell_volume / num_atoms_in_cell


def unit_cell_volume_per_molecule(cif_file):
    """
    demo24 单位分子晶胞体积
    :param cif_file: CIF 文件
    :return: 单位分子晶胞体积
    """
    structure = Structure.from_file(cif_file)
    composition = structure.composition

    # 晶胞体积（Å³）
    cell_volume = structure.volume

    # 计算每个化学式单元的原子数（考虑部分占据）
    atoms_per_formula_unit = composition.num_atoms  # 直接使用composition的原子总数

    # 计算晶胞中的化学式单元数（Z值）
    z_value = composition.get_reduced_formula_and_factor()[1]

    # 单位化学式体积 = 晶胞体积 / Z值
    volume_per_molecule = (cell_volume / z_value ) / 1000

    return volume_per_molecule


def calculate_properties(cif_file, C, A, n, r0, b, ion_radii, nominal_valence):
    # 将计算结果打包到字典
    cif_data = {
            "average_bond_length": average_bond_length(cif_file),  # Average bond length
            "bond_length_std": std_bond_length(cif_file),  # Standard deviation of bond length
            "average_bond_strength": average_bond_strength(cif_file),  # Average bond strength
            "bond_strength_std": bond_strength_std(cif_file),  # Standard deviation of bond strength
            "average_covalency": average_covalency(C, cif_file),  # Average covalency
            "covalency_std": covalency_std(C, cif_file),  # Standard deviation of covalency
            "average_bond_energy": average_bond_energy(cif_file),  # Average bond energy
            "average_bond_energy_per_unit_volume": average_bond_energy_per_unit_volume(cif_file),
            # Average bond energy per unit volume
            "bond_valence_std_avg": bond_valence_std_avg(r0, b, cif_file), # bond_valence_std_avg
            "bond_energy_std": bond_energy_std(cif_file),  # Standard deviation of bond energy
            "average_bvs_deviation": average_bvs_deviation(r0, b, cif_file),  # Average of BVS deviations from nominal valence
            "ionic_volume": total_ionic_volume(ion_radii, cif_file),  # Total ionic volume
            "packing_fraction": packing_fraction(ion_radii, cif_file),  # Packing fraction
            "ionic_volume_per_atom": ionic_volume_per_atom(ion_radii, cif_file),  # Ionic volume per unit atom
            "molecular_dielectric_polarizability": dielectric_polarizability(cif_file),
            # Molecular dielectric polarizability
            "molecular_dielectric_polarizability_per_atom": dielectric_polarizability_per_atom(cif_file),
            # Molecular dielectric polarizability per unit atom
            "molecular_dielectric_polarizability_per_unit_volume": dielectric_polarizability_per_volume(cif_file),
            # Molecular dielectric polarizability per unit volume
            "lattice_constant_edge_length_std": lattice_constant_std(cif_file),
            # Lattice constant-edge length standard deviation
            "lattice_constant_angle_std": angle_std(cif_file),  # Lattice constant-angle standard deviation
            "unit_cell_volume": cell_volume(cif_file),  # Unit cell volume
            "relative_molecular_mass": relative_molecular_mass(cif_file),  # Relative molecular mass
            "theoretical_density": theoretical_density(cif_file),  # Theoretical density
            "unit_cell_volume_per_atom": unit_cell_volume_per_atom(cif_file),  # Unit cell volume per unit atom
            "global_instability_index": unit_cell_volume_per_molecule(cif_file),  # Global instability index
        }
    print("CIF Data:", cif_data)
    if not os.path.exists(cif_file):
        print("File not found:", cif_file)
    return cif_data


# 视图函数：上传文件并计算属性
def calculate(request):
    C = None
    A = None
    n = None
    r0 = None
    b = None
    ion_radii = None
    nominal_valence = None

    if request.method == "POST" and request.FILES.get('cif_file'):
        uploaded_file = request.FILES['cif_file']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)  # 获取文件存储路径

        cif_data = {}
        if uploaded_file.name.endswith('.cif'):
            try:
                with open(file_path, 'r') as f:
                    cif_origin = f.read()  # 读取文件内容并存储在 cif_origin
                # 调用计算属性的函数
                cif_data = calculate_properties(file_path, C, A, n, r0, b, ion_radii, nominal_valence)
                print(cif_data)
                # 创建结果行列表
                results = [
                    {"property": "Average bond length", "value": f"{cif_data.get('average_bond_length'):.4f} Å"},
                    {"property": "Standard deviation of bond length",
                     "value": f"{cif_data.get('bond_length_std'):.4f} Å"},
                    {"property": "Average bond strength", "value": f"{cif_data.get('average_bond_strength'):.4f} 1/Å"},
                    {"property": "Standard deviation of bond strength",
                     "value": f"{cif_data.get('bond_strength_std'):.4f} 1/Å"},
                    {"property": "Average covalency", "value": f"{cif_data.get('average_covalency'):.4f} "},
                    {"property": "Standard deviation of covalency", "value": f"{cif_data.get('covalency_std'):.4f} "},
                    {"property": "Average bond energy", "value": f"{cif_data.get('average_bond_energy'):.4f} kJ/mol"},
                    {"property": "Bond energy per unit volume",
                     "value": f"{cif_data.get('average_bond_energy_per_unit_volume'):.4f} kJ/mol·Å³"},
                    {"property": "Standard deviation of bond energy",
                     "value": f"{cif_data.get('bond_energy_std'):.4f} kJ/mol"},
                    #Average of standard deviations of bond valence    #10
                    {"property": "Mean bond valence standard deviation",
                     "value": f"{cif_data.get('bond_valence_std_avg'):.4f} "},#11111
                    {"property": "BVS deviation from nominal valence",
                     "value": f"{cif_data.get('average_bvs_deviation'):.4f} "},
                    {"property": "Total ionic volume", "value": f"{cif_data.get('ionic_volume'):.4f} Å³"},
                    {"property": "Packing fraction", "value": f"{cif_data.get('packing_fraction'):.4f} "},
                    {"property": "Ionic volume per atom",
                     "value": f"{cif_data.get('ionic_volume_per_atom'):.4f} Å³/atom"},
                    {"property": "Molecular dielectric polarizability",
                     "value": f"{cif_data.get('molecular_dielectric_polarizability'):.4f} "},
                    {"property": "Molecular dielectric polarizability per atom",
                     "value": f"{cif_data.get('molecular_dielectric_polarizability_per_atom'):.4f} "},
                    {"property": "Molecular dielectric polarizability per volume",
                     "value": f"{cif_data.get('molecular_dielectric_polarizability_per_unit_volume'):.4f} "},
                    {"property": "Lattice constant length std. dev.",
                     "value": f"{cif_data.get('lattice_constant_edge_length_std'):.4f} "},
                    {"property": "Lattice constant angle std. dev.",
                     "value": f"{cif_data.get('lattice_constant_angle_std'):.4f} °"},
                    {"property": "Unit cell volume", "value": f"{cif_data.get('unit_cell_volume'):.4f} Å³"},
                    {"property": "Molecular weight", "value": f"{cif_data.get('relative_molecular_mass'):.4f} amu"},
                    {"property": "Theoretical density", "value": f"{cif_data.get('theoretical_density'):.4f} g/cm³"},
                    {"property": "Unit cell volume per atom",
                     "value": f"{cif_data.get('unit_cell_volume_per_atom'):.4f} Å³/atom"},
                    {"property": "Global instability index",
                     "value": f"{cif_data.get('global_instability_index'):.4f} Å³/mol"}
                ]
            except Exception as e:
                cif_data = {"error": f"Failed to parse CIF file: {str(e)}"}
                print(cif_data)
                results = None
            finally:
                # 删除文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
        else:
            cif_data = {"error": "Uploaded file is not a valid CIF file."}
            print(cif_data)
            results = None

        # 渲染上传页面，传递结果数据
        return render(request, 'cifapp/calculate.html', {
            'result': cif_origin,
            'cif_data': cif_data,
            'results': results,
        })

    # 默认返回空页面
    return render(request, 'cifapp/calculate.html')


#def home(request):
#    return render(request, 'cifapp/home.html')


def about(request):
    return render(request, 'cifapp/about.html')


def record_list(request):
    return render(request, 'cifapp/record_list.html')
    

def illustration(request):
    return render(request, 'cifapp/illustration.html')
    
def ML(request):
    return render(request, 'cifapp/ML.html')

