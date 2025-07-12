"""
eスポーツコース効果の縦断研究分析
Two-way ANOVA Analysis (Course × Time)

分析設計:
- 要因A: コース（eスポーツ vs リベラルアーツ）
- 要因B: 時間（Wave 1, 2, 3）
- 従属変数: 認知・非認知スキル各指標
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
import os
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_preprocessed_data():
    """前処理済みデータの読み込み"""
    print("=== データの読み込み ===")
    
    # 前回の処理結果を読み込み（data_overview.pyから）
    from data_overview import main as preprocess_main
    df_imputed, cognitive_vars, non_cognitive_vars, all_vars = preprocess_main()
    
    print(f"読み込み完了: {df_imputed.shape}")
    return df_imputed, cognitive_vars, non_cognitive_vars, all_vars

def prepare_anova_data(df_imputed, all_vars):
    """ANOVA用データの準備"""
    print("\n=== ANOVA用データの準備 ===")
    
    # 必要な列のみを抽出
    base_cols = ['participant_id', 'course_group', 'measurement_wave']
    analysis_cols = base_cols + all_vars
    
    # 存在する列のみを選択
    available_cols = [col for col in analysis_cols if col in df_imputed.columns]
    df_anova = df_imputed[available_cols].copy()
    
    # カテゴリ変数のエンコーディング
    df_anova['course_numeric'] = df_anova['course_group'].map({'eスポーツ': 1, 'リベラルアーツ': 0})
    df_anova['wave_numeric'] = df_anova['measurement_wave']
    
    print(f"ANOVA用データ: {df_anova.shape}")
    print(f"分析対象変数: {len([col for col in all_vars if col in df_anova.columns])}個")
    
    return df_anova

def perform_two_way_anova(df_anova, variable):
    """二元配置分散分析の実行"""
    try:
        # データの準備
        data = df_anova.dropna(subset=[variable])
        
        if len(data) < 10:  # データが少なすぎる場合はスキップ
            return None
        
        # 二元配置分散分析の実行
        formula = f'{variable} ~ C(course_group) + C(measurement_wave) + C(course_group):C(measurement_wave)'
        model = ols(formula, data=data).fit()
        anova_results = anova_lm(model, typ=2)
        
        # 結果の整理
        results = {
            'variable': variable,
            'n_observations': len(data),
            'model': model,
            'anova_table': anova_results,
            'main_effect_course': {
                'F': anova_results.loc['C(course_group)', 'F'],
                'p': anova_results.loc['C(course_group)', 'PR(>F)'],
                'significant': anova_results.loc['C(course_group)', 'PR(>F)'] < 0.05
            },
            'main_effect_time': {
                'F': anova_results.loc['C(measurement_wave)', 'F'],
                'p': anova_results.loc['C(measurement_wave)', 'PR(>F)'],
                'significant': anova_results.loc['C(measurement_wave)', 'PR(>F)'] < 0.05
            },
            'interaction': {
                'F': anova_results.loc['C(course_group):C(measurement_wave)', 'F'],
                'p': anova_results.loc['C(course_group):C(measurement_wave)', 'PR(>F)'],
                'significant': anova_results.loc['C(course_group):C(measurement_wave)', 'PR(>F)'] < 0.05
            }
        }
        
        return results
    
    except Exception as e:
        print(f"  {variable}: エラー - {e}")
        return None

def run_comprehensive_anova(df_anova, all_vars):
    """全変数に対する包括的ANOVA分析"""
    print("\n=== 二元配置分散分析の実行 ===")
    
    anova_results = []
    
    for variable in all_vars:
        if variable in df_anova.columns:
            print(f"分析中: {variable}")
            result = perform_two_way_anova(df_anova, variable)
            
            if result is not None:
                anova_results.append(result)
                
                # 結果の簡易表示
                print(f"  コース効果: F={result['main_effect_course']['F']:.3f}, p={result['main_effect_course']['p']:.3f}")
                print(f"  時間効果: F={result['main_effect_time']['F']:.3f}, p={result['main_effect_time']['p']:.3f}")
                print(f"  交互作用: F={result['interaction']['F']:.3f}, p={result['interaction']['p']:.3f}")
    
    print(f"\n✅ 分析完了: {len(anova_results)}変数")
    return anova_results

def create_results_summary(anova_results):
    """結果サマリーの作成"""
    print("\n=== 結果サマリーの作成 ===")
    
    summary_data = []
    
    for result in anova_results:
        summary_data.append({
            'Variable': result['variable'],
            'N': result['n_observations'],
            'Course_F': result['main_effect_course']['F'],
            'Course_p': result['main_effect_course']['p'],
            'Course_Sig': '***' if result['main_effect_course']['p'] < 0.001 
                         else '**' if result['main_effect_course']['p'] < 0.01
                         else '*' if result['main_effect_course']['p'] < 0.05 
                         else 'ns',
            'Time_F': result['main_effect_time']['F'],
            'Time_p': result['main_effect_time']['p'],
            'Time_Sig': '***' if result['main_effect_time']['p'] < 0.001 
                       else '**' if result['main_effect_time']['p'] < 0.01
                       else '*' if result['main_effect_time']['p'] < 0.05 
                       else 'ns',
            'Interaction_F': result['interaction']['F'],
            'Interaction_p': result['interaction']['p'],
            'Interaction_Sig': '***' if result['interaction']['p'] < 0.001 
                              else '**' if result['interaction']['p'] < 0.01
                              else '*' if result['interaction']['p'] < 0.05 
                              else 'ns'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 有意な結果の集計
    significant_course = summary_df[summary_df['Course_Sig'] != 'ns']
    significant_time = summary_df[summary_df['Time_Sig'] != 'ns']
    significant_interaction = summary_df[summary_df['Interaction_Sig'] != 'ns']
    
    print(f"有意なコース効果: {len(significant_course)}変数")
    print(f"有意な時間効果: {len(significant_time)}変数")
    print(f"有意な交互作用: {len(significant_interaction)}変数")
    
    return summary_df, significant_course, significant_time, significant_interaction

def create_visualization(df_anova, all_vars, output_dir='./analysis_result/anova_result'):
    """全変数の可視化"""
    print(f"\n=== 可視化の作成 ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 分析対象の変数を取得
    plot_vars = [var for var in all_vars if var in df_anova.columns]
    
    if len(plot_vars) > 0:
        # 一行に3つまでグラフを配置
        n_cols = 3
        n_rows = (len(plot_vars) + n_cols - 1) // n_cols  # 切り上げ除算
        
        # 図のサイズを調整（行数に応じて）
        fig_height = max(4 * n_rows, 8)  # 最低8インチ、行数に応じて調整
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, fig_height))
        
        # 1次元配列に変換
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, var in enumerate(plot_vars):
            # データの準備
            plot_data = df_anova.dropna(subset=[var])
            
            if len(plot_data) > 0:
                # 交互作用プロット
                sns.pointplot(data=plot_data, x='measurement_wave', y=var, 
                             hue='course_group', ax=axes[i], 
                             markers=['o', 's'], linestyles=['-', '--'],
                             palette=['#FF6B6B', '#4ECDC4'])
                
                axes[i].set_title(f'{var}', fontsize=12)
                axes[i].set_xlabel('Experiment Number')
                axes[i].set_ylabel('Score')
                axes[i].legend(title='Course')
            else:
                axes[i].text(0.5, 0.5, f'{var}\n(データなし)', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{var}', fontsize=12)
        
        # 空のサブプロットを非表示
        for i in range(len(plot_vars), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'all_variables_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 全変数可視化保存: {plot_path}")
        print(f"  作成したグラフ数: {len(plot_vars)}個")
    else:
        print("可視化対象の変数がありません")

def create_significant_visualization(df_anova, significant_vars, output_dir='./analysis_result/anova_result'):
    """有意な結果のみの可視化"""
    print(f"\n=== 有意な結果の可視化 ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 有意な変数を取得
    plot_vars = significant_vars['Variable'].tolist()
    
    if len(plot_vars) > 0:
        # 一行に3つまでグラフを配置
        n_cols = 3
        n_rows = (len(plot_vars) + n_cols - 1) // n_cols  # 切り上げ除算
        
        # 図のサイズを調整（行数に応じて）
        fig_height = max(4 * n_rows, 8)  # 最低8インチ、行数に応じて調整
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, fig_height))
        
        # 1次元配列に変換
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, var in enumerate(plot_vars):
            # データの準備
            plot_data = df_anova.dropna(subset=[var])
            
            # 交互作用プロット
            sns.pointplot(data=plot_data, x='measurement_wave', y=var, 
                         hue='course_group', ax=axes[i], 
                         markers=['o', 's'], linestyles=['-', '--'],
                         palette=['#FF6B6B', '#4ECDC4'])
            
            axes[i].set_title(f'{var}', fontsize=12)
            axes[i].set_xlabel('Experiment Number')
            axes[i].set_ylabel('Score')
            axes[i].legend(title='Course')
        
        # 空のサブプロットを非表示
        for i in range(len(plot_vars), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'significant_results_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 有意な結果可視化保存: {plot_path}")
        print(f"  作成したグラフ数: {len(plot_vars)}個")
    else:
        print("有意な結果の可視化対象がありません")

def save_detailed_results(anova_results, summary_df, output_dir='./analysis_result/anova_result'):
    """詳細結果の保存"""
    print(f"\n=== 詳細結果の保存 ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. サマリー結果（Excel）
    excel_path = os.path.join(output_dir, 'anova_results_summary.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='ANOVA_Summary', index=False)
        
        # 有意な結果のみを別シートに
        sig_course = summary_df[summary_df['Course_Sig'] != 'ns']
        sig_time = summary_df[summary_df['Time_Sig'] != 'ns']
        sig_interaction = summary_df[summary_df['Interaction_Sig'] != 'ns']
        
        if len(sig_course) > 0:
            sig_course.to_excel(writer, sheet_name='Significant_Course', index=False)
        if len(sig_time) > 0:
            sig_time.to_excel(writer, sheet_name='Significant_Time', index=False)
        if len(sig_interaction) > 0:
            sig_interaction.to_excel(writer, sheet_name='Significant_Interaction', index=False)
    
    print(f"📊 ANOVA結果保存: {excel_path}")
    
    # 2. 詳細ANOVA表（Excel）
    detailed_results = []
    for result in anova_results:
        anova_table = result['anova_table']
        for index, row in anova_table.iterrows():
            detailed_results.append({
                'Variable': result['variable'],
                'Effect': index,
                'Sum_of_Squares': row['sum_sq'],
                'DF': row['df'],
                'F_Value': row['F'],
                'P_Value': row['PR(>F)']
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    
    # 詳細結果もExcelで保存
    detailed_excel_path = os.path.join(output_dir, 'anova_detailed_results.xlsx')
    with pd.ExcelWriter(detailed_excel_path, engine='openpyxl') as writer:
        detailed_df.to_excel(writer, sheet_name='詳細ANOVA表', index=False)
        
        # 効果別にシートを分ける
        course_effects = detailed_df[detailed_df['Effect'] == 'C(course_group)']
        time_effects = detailed_df[detailed_df['Effect'] == 'C(measurement_wave)']
        interaction_effects = detailed_df[detailed_df['Effect'] == 'C(course_group):C(measurement_wave)']
        
        if len(course_effects) > 0:
            course_effects.to_excel(writer, sheet_name='コース効果', index=False)
        if len(time_effects) > 0:
            time_effects.to_excel(writer, sheet_name='時間効果', index=False)
        if len(interaction_effects) > 0:
            interaction_effects.to_excel(writer, sheet_name='交互作用', index=False)
    
    print(f"📈 詳細結果保存: {detailed_excel_path}")
    
    return excel_path, detailed_excel_path

def main():
    """メイン実行関数"""
    print("🎯 二元配置分散分析の開始")
    
    # データの読み込み
    df_imputed, cognitive_vars, non_cognitive_vars, all_vars = load_preprocessed_data()
    
    # ANOVA用データの準備
    df_anova = prepare_anova_data(df_imputed, all_vars)
    
    # 二元配置分散分析の実行
    anova_results = run_comprehensive_anova(df_anova, all_vars)
    
    # 結果サマリーの作成
    summary_df, sig_course, sig_time, sig_interaction = create_results_summary(anova_results)
    
    # 結果の表示
    print(f"\n=== 主要な発見 ===")
    print(f"コース効果が有意な変数:")
    if len(sig_course) > 0:
        for _, row in sig_course.iterrows():
            print(f"  {row['Variable']}: F={row['Course_F']:.3f}, p={row['Course_p']:.3f} {row['Course_Sig']}")
    else:
        print("  なし")
    
    print(f"\n時間効果が有意な変数:")
    if len(sig_time) > 0:
        for _, row in sig_time.iterrows():
            print(f"  {row['Variable']}: F={row['Time_F']:.3f}, p={row['Time_p']:.3f} {row['Time_Sig']}")
    else:
        print("  なし")
    
    print(f"\n交互作用が有意な変数:")
    if len(sig_interaction) > 0:
        for _, row in sig_interaction.iterrows():
            print(f"  {row['Variable']}: F={row['Interaction_F']:.3f}, p={row['Interaction_p']:.3f} {row['Interaction_Sig']}")
    else:
        print("  なし")
    
    # 可視化
    # 全変数の可視化
    create_visualization(df_anova, all_vars)
    
    # 有意な結果の可視化
    if len(sig_course) > 0:
        create_significant_visualization(df_anova, sig_course)
    elif len(sig_time) > 0:
        create_significant_visualization(df_anova, sig_time)
    elif len(sig_interaction) > 0:
        create_significant_visualization(df_anova, sig_interaction)
    
    # 結果保存
    excel_path, detailed_excel_path = save_detailed_results(anova_results, summary_df)
    
    print(f"\n✅ 二元配置分散分析完了")
    print(f"📊 結果ファイル:")
    print(f"  - {excel_path}")
    print(f"  - {detailed_excel_path}")
    
    return anova_results, summary_df, df_anova

if __name__ == "__main__":
    anova_results, summary_df, df_anova = main()
