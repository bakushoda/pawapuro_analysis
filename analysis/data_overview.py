"""
eスポーツコース効果の縦断研究分析
Data Overview and Analysis

分析方針:
- 対象: cohort 2024_G1のみ（106名）
- 設計: 二元配置分散分析（コース × 時間）
- 欠損処理: コース×Wave別平均値補完
- 時期: Wave 1, 2, 3の縦断データ
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
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_and_explore_data():
    """データの読み込みと基本情報の確認"""
    print("=== データの読み込み ===")
    
    # データの読み込み
    df_raw = pd.read_excel('./data/data_master.xlsx', sheet_name='master')
    
    print(f"元データ形状: {df_raw.shape}")
    print(f"cohort分布:")
    print(df_raw['cohort'].value_counts())
    
    return df_raw

def extract_target_cohort(df_raw):
    """cohort 2024_G1のデータ抽出"""
    print("\n=== cohort 2024_G1の抽出 ===")
    
    # より安全な抽出方法
    mask = df_raw['cohort'] == '2024_G1'
    df = df_raw.loc[mask].copy()
    df.reset_index(drop=True, inplace=True)
    
    print(f"抽出後データ形状: {df.shape}")
    
    # 参加者数の確認
    unique_participants = df['participant_id'].drop_duplicates()
    participant_count = len(unique_participants)
    print(f"参加者数: {participant_count}名")
    
    # 測定時期の確認
    unique_waves = df['measurement_wave'].drop_duplicates().sort_values()
    print(f"測定時期: {list(unique_waves)}")
    
    # コース分布確認
    print(f"\nコース分布:")
    course_counts = df['course'].value_counts()
    print(course_counts)
    
    return df

def create_course_classification(df):
    """コース分類の作成"""
    print("\n=== コース分類 ===")
    
    # eスポーツコースかどうかの判定
    esports_mask = df['course'] == 'eスポーツエデュケーションコース'
    df['is_esports'] = esports_mask
    
    # コースグループの作成
    course_group_list = []
    for idx in df.index:
        course_name = df.loc[idx, 'course']
        if course_name == 'eスポーツエデュケーションコース':
            course_group_list.append('eスポーツ')
        elif course_name == 'リベラルアーツコース':
            course_group_list.append('リベラルアーツ')
        else:
            course_group_list.append('その他')
    
    df['course_group'] = course_group_list
    
    # 結果確認
    print("コースグループ分布:")
    course_group_counts = df['course_group'].value_counts()
    print(course_group_counts)
    
    # Wave×コースのクロス集計
    print("\nWave×コース クロス集計:")
    crosstab = pd.crosstab(df['measurement_wave'], df['course_group'], margins=True)
    print(crosstab)
    
    return df

def define_variables():
    """分析対象変数の定義"""
    print("\n=== 分析対象変数の定義 ===")
    
    # 認知スキル変数
    cognitive_vars = [
        'corsi_ncorrect_total', 'corsi_blockspan', 'corsi_totalscore',
        'fourchoice_prop_correct', 'fourchoice_mean_rt',
        'stroop_propcorrect', 'stroop_mean_rt',
        'tmt_combined_errors', 'tmt_combined_trailtime',
        'ufov_subtest1_threshold', 'ufov_subtest2_threshold', 'ufov_subtest3_threshold'
    ]
    
    # 非認知スキル変数
    non_cognitive_vars = [
        'bigfive_extraversion', 'bigfive_agreeableness', 'bigfive_conscientiousness',
        'bigfive_neuroticism', 'bigfive_openness',
        'grit_total', 'mindset_total',
        'ct_logical_awareness', 'ct_inquiry', 'ct_objectivity', 'ct_evidence_based',
        'who5_total', 'swbs_total'
    ]
    
    # 全分析変数
    all_vars = cognitive_vars + non_cognitive_vars
    
    print(f"認知スキル変数: {len(cognitive_vars)}個")
    print(f"非認知スキル変数: {len(non_cognitive_vars)}個")
    print(f"総変数数: {len(all_vars)}個")
    
    return cognitive_vars, non_cognitive_vars, all_vars

def check_missing_values(df, all_vars):
    """欠損値の確認"""
    print("\n=== 欠損値の状況 ===")
    
    missing_info = []
    for var in all_vars:
        if var in df.columns:
            total_count = len(df)
            missing_count = df[var].isnull().sum()
            missing_rate = (missing_count / total_count) * 100
            
            missing_info.append({
                'variable': var,
                'total': total_count,
                'missing': missing_count,
                'missing_rate': missing_rate
            })
    
    missing_df = pd.DataFrame(missing_info)
    missing_with_na = missing_df[missing_df['missing'] > 0].sort_values(by='missing_rate', ascending=False)  # type: ignore
    
    print(f"欠損のある変数: {len(missing_with_na)}個")
    if len(missing_with_na) > 0:
        print("\n欠損率上位10変数:")
        print(missing_with_na[['variable', 'missing', 'missing_rate']].head(10))
    
    return missing_df

def calculate_imputation_means(df, all_vars):
    """コース×Wave別平均値の計算"""
    print("\n=== 平均値補完の準備 ===")
    
    course_wave_means = {}
    
    for var in all_vars:
        if var in df.columns:
            course_wave_means[var] = {}
            
            # 各組み合わせごとに平均値計算
            for course in ['eスポーツ', 'リベラルアーツ']:
                for wave in [1, 2, 3]:
                    # 該当するデータを抽出
                    condition = (df['course_group'] == course) & (df['measurement_wave'] == wave)
                    subset = df[condition]
                    
                    # 欠損でない値のみで平均計算
                    valid_data = subset[var].dropna()
                    
                    if len(valid_data) > 0:
                        mean_value = valid_data.mean()
                        course_wave_means[var][f"{course}_wave{wave}"] = mean_value
                        print(f"{var} - {course} Wave{wave}: 平均{mean_value:.2f} (n={len(valid_data)})")
    
    print("✅ 平均値計算完了")
    return course_wave_means

def perform_imputation(df, all_vars, course_wave_means):
    """平均値補完の実行"""
    print("\n=== 平均値補完の実行 ===")
    
    # データをコピーして補完用DataFrameを作成
    df_imputed = df.copy()
    
    # 補完統計
    total_imputations = 0
    
    for var in all_vars:
        if var in df_imputed.columns:
            # 補完フラグ列の作成
            imputed_flag_col = f"{var}_imputed"
            df_imputed[imputed_flag_col] = False
            
            # 欠損値を特定
            missing_mask = df_imputed[var].isnull()
            missing_indices = df_imputed[missing_mask].index
            
            # 各欠損値を補完
            for idx in missing_indices:
                course = df_imputed.loc[idx, 'course_group']
                wave = df_imputed.loc[idx, 'measurement_wave']
                key = f"{course}_wave{wave}"
                
                # 該当する平均値があれば補完
                if var in course_wave_means and key in course_wave_means[var]:
                    df_imputed.loc[idx, var] = course_wave_means[var][key]
                    df_imputed.loc[idx, imputed_flag_col] = True
                    total_imputations += 1
    
    print(f"✅ 補完完了: 総補完数 {total_imputations}件")
    return df_imputed

def verify_imputation_results(df_original, df_imputed, all_vars):
    """補完結果の確認"""
    print("\n=== 補完結果の確認 ===")
    
    # 主要変数での補完結果チェック
    check_vars = ['corsi_totalscore', 'fourchoice_prop_correct', 'stroop_propcorrect', 
                  'bigfive_extraversion', 'grit_total']
    
    for var in check_vars:
        if var in df_imputed.columns:
            original_missing = df_original[var].isnull().sum()
            after_missing = df_imputed[var].isnull().sum()
            imputed_count = df_imputed[f"{var}_imputed"].sum()
            
            print(f"{var}:")
            print(f"  補完前欠損: {original_missing}件")
            print(f"  補完後欠損: {after_missing}件")
            print(f"  補完実行: {imputed_count}件")

def final_data_summary(df_imputed):
    """最終データセットの確認"""
    print("\n=== 最終データセットの確認 ===")
    print(f"最終データ形状: {df_imputed.shape}")
    
    # 安全な参加者数カウント
    unique_participants_final = df_imputed['participant_id'].drop_duplicates()
    participant_count_final = len(unique_participants_final)
    print(f"参加者数: {participant_count_final}名")
    
    # 最終的なWave×コース分布
    final_crosstab = pd.crosstab(df_imputed['measurement_wave'], df_imputed['course_group'], margins=True)
    print("\n最終データ分布:")
    print(final_crosstab)
    
    # サンプルデータの表示
    print("\n=== サンプルデータ ===")
    sample_cols = ['participant_id', 'course_group', 'measurement_wave', 
                   'corsi_totalscore', 'bigfive_extraversion', 'grit_total']
    
    print("最初の10行:")
    print(df_imputed[sample_cols].head(10))

def main():
    """メイン実行関数"""
    print("✅ ライブラリの読み込み完了")
    
    # データの読み込みと処理
    df_raw = load_and_explore_data()
    df = extract_target_cohort(df_raw)
    df = create_course_classification(df)
    
    # 変数定義
    cognitive_vars, non_cognitive_vars, all_vars = define_variables()
    
    # 欠損値確認
    missing_df = check_missing_values(df, all_vars)
    
    # 平均値補完
    course_wave_means = calculate_imputation_means(df, all_vars)
    df_imputed = perform_imputation(df, all_vars, course_wave_means)
    
    # 結果確認
    verify_imputation_results(df, df_imputed, all_vars)
    final_data_summary(df_imputed)
    
    print("\n✅ データ前処理完了")
    print("✅ 二元配置分散分析の準備完了")
    print("\n🎯 次のステップ: 二元配置分散分析の実行")
    print("- 要因A: コース（eスポーツ vs リベラルアーツ）")
    print("- 要因B: 時間（Wave 1, 2, 3）")
    print("- 従属変数: 認知・非認知スキル各指標")
    
    # データ保存（オプション）
    # df_imputed.to_csv('cohort_2024G1_imputed.csv', index=False)
    # print("💾 補完済みデータを保存しました")
    
    return df_imputed, cognitive_vars, non_cognitive_vars, all_vars

if __name__ == "__main__":
    df_imputed, cognitive_vars, non_cognitive_vars, all_vars = main()