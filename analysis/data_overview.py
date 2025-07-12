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
    eSports_mask = df['course'] == 'eスポーツエデュケーションコース'
    df['is_eSports'] = eSports_mask
    
    # コースグループの作成
    course_group_list = []
    for idx in df.index:
        course_name = df.loc[idx, 'course']
        if course_name == 'eスポーツエデュケーションコース':
            course_group_list.append('eSports')
        elif course_name == 'リベラルアーツコース':
            course_group_list.append('Liberal Arts')
        else:
            course_group_list.append('Other')
    
    df['course_group'] = course_group_list
    
    # 結果確認
    print("コースグループ分布:")
    course_group_counts = df['course_group'].value_counts()
    print(course_group_counts)
    
    # 実験回数とコースのクロス集計
    print("\n実験回数とコース クロス集計:")
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
            for course in ['eSports', 'Liberal Arts']:
                for wave in [1, 2, 3]:
                    # 該当するデータを抽出
                    condition = (df['course_group'] == course) & (df['measurement_wave'] == wave)
                    subset = df[condition]
                    
                    # 欠損でない値のみで平均計算
                    valid_data = subset[var].dropna()
                    
                    if len(valid_data) > 0:
                        mean_value = valid_data.mean()
                        course_wave_means[var][f"{course}_wave{wave}"] = mean_value
                        print(f"{var} - {course} 実験回数{wave}: 平均{mean_value:.2f} (n={len(valid_data)})")
    
    print("✅ 平均値計算完了")
    
    # Liberal Artsコースの実験3回目の平均値計算詳細を確認
    print("\n=== Liberal Artsコース 実験3回目の平均値計算詳細 ===")
    for var in ['fourchoice_prop_correct', 'stroop_propcorrect', 'tmt_combined_errors', 'ufov_subtest1_threshold']:
        if var in course_wave_means:
            key = 'Liberal Arts_wave3'
            if key in course_wave_means[var]:
                print(f"{var}: {course_wave_means[var][key]:.4f}")
            else:
                print(f"{var}: 平均値なし")
    
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
    
    # Liberal Artsコースの実験3回目の詳細チェック
    print("\n=== Liberal Artsコース 実験3回目の詳細チェック ===")
    liberal_wave3 = df_imputed[(df_imputed['course_group'] == 'Liberal Arts') & 
                              (df_imputed['measurement_wave'] == 3)]
    
    print(f"Liberal Arts 実験3回目のデータ数: {len(liberal_wave3)}")
    
    # fourchoice, stroop, tmt, ufovの変数をチェック
    target_vars = ['fourchoice_prop_correct', 'fourchoice_mean_rt', 
                   'stroop_propcorrect', 'stroop_mean_rt',
                   'tmt_combined_errors', 'tmt_combined_trailtime',
                   'ufov_subtest1_threshold', 'ufov_subtest2_threshold', 'ufov_subtest3_threshold']
    
    for var in target_vars:
        if var in df_imputed.columns:
            missing_count = liberal_wave3[var].isnull().sum()
            imputed_count = liberal_wave3[f"{var}_imputed"].sum() if f"{var}_imputed" in df_imputed.columns else 0
            total_count = len(liberal_wave3)
            
            print(f"{var}:")
            print(f"  欠損数: {missing_count}/{total_count}")
            print(f"  補完実行数: {imputed_count}")
            print(f"  補完率: {imputed_count/total_count*100:.1f}%" if total_count > 0 else "  補完率: N/A")

def create_visualizations(df_imputed, cognitive_vars, non_cognitive_vars):
    """データの可視化"""
    print("\n=== データの可視化 ===")
    
    # 出力ディレクトリの作成
    import os
    output_dir = './analysis_result/data_overview/figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 参加者数の推移（実験回数別）
    plt.figure(figsize=(10, 6))
    wave_counts = df_imputed['measurement_wave'].value_counts().sort_index()
    plt.bar(wave_counts.index, wave_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.title('Number of Participants by Experiment Number', fontsize=14, fontweight='bold')
    plt.xlabel('Experiment Number')
    plt.ylabel('Number of Participants')
    plt.xticks([1, 2, 3])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/participant_counts_by_wave.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. コース別参加者数
    plt.figure(figsize=(8, 6))
    course_counts = df_imputed['course_group'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    plt.pie(course_counts.values, labels=course_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Participant Distribution by Course', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/course_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    
    # 4. 全変数の時系列推移（コース別）
    # 認知スキル変数
    create_time_series_plot(df_imputed, cognitive_vars, 'Time Series of Cognitive Skills Variables', 
                           f'{output_dir}/cognitive_time_series.png')
    
    # 非認知スキル変数
    create_time_series_plot(df_imputed, non_cognitive_vars, 'Time Series of Non-Cognitive Skills Variables', 
                           f'{output_dir}/non_cognitive_time_series.png')
    
    # 5. 変数間の相関ヒートマップ
    all_vars_for_corr = cognitive_vars + non_cognitive_vars
    create_correlation_heatmap(df_imputed, all_vars_for_corr, 'Correlation Matrix of All Variables', 
                              f'{output_dir}/all_variables_correlation.png')
    
    print(f"✅ 可視化完了: {output_dir}に保存")
    if 'tmt_combined_trailtime_converted' in df_imputed.columns:
        print("📝 注意: tmt_combined_trailtimeは秒単位で表示されています")

def create_time_series_plot(df, variables, title, save_path):
    """時系列推移プロットの作成"""
    # 一行に3つまでに制限
    n_vars = len(variables)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols  # 切り上げ除算
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # 1次元配列に変換
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, var in enumerate(variables):
        if var in df.columns:
            # コース×Wave別の平均値計算
            means = df.groupby(['course_group', 'measurement_wave'])[var].mean().unstack()
            
            # プロット
            for j, course in enumerate(['eSports', 'Liberal Arts']):
                if course in means.index:
                    axes[i].plot(means.columns, means.loc[course], 
                               marker='o', linewidth=2, markersize=8, 
                               label=course, color=['#FF6B6B', '#4ECDC4'][j])
            
            axes[i].set_title(var, fontweight='bold')
            axes[i].set_xlabel('Experiment Number')
            
            # 単位に応じてy軸ラベルを設定
            if var == 'tmt_combined_trailtime':
                axes[i].set_ylabel('Time (seconds)')
            elif var in ['fourchoice_mean_rt', 'stroop_mean_rt']:
                axes[i].set_ylabel('Reaction Time (ms)')
            else:
                axes[i].set_ylabel('Mean Value')
            
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xticks([1, 2, 3])
    
    # 余分なサブプロットを非表示
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_heatmap(df, variables, title, save_path):
    """相関ヒートマップの作成"""
    # 利用可能な変数のみ抽出
    available_vars = [var for var in variables if var in df.columns]
    
    if len(available_vars) > 1:
        # 相関行列の計算
        corr_matrix = df[available_vars].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def final_data_summary(df_imputed):
    """最終データセットの確認"""
    print("\n=== 最終データセットの確認 ===")
    print(f"最終データ形状: {df_imputed.shape}")
    
    # 安全な参加者数カウント
    unique_participants_final = df_imputed['participant_id'].drop_duplicates()
    participant_count_final = len(unique_participants_final)
    print(f"参加者数: {participant_count_final}名")
    
    # 最終的な実験回数とコース分布
    final_crosstab = pd.crosstab(df_imputed['measurement_wave'], df_imputed['course_group'], margins=True)
    print("\n最終データ分布:")
    print(final_crosstab)
    
    # サンプルデータの表示
    print("\n=== サンプルデータ ===")
    sample_cols = ['participant_id', 'course_group', 'measurement_wave', 
                   'corsi_totalscore', 'bigfive_extraversion', 'grit_total']
    
    print("最初の10行:")
    print(df_imputed[sample_cols].head(10))

def save_results(df_imputed, missing_df, output_dir='./analysis_result/data_overview'):
    """分析結果の保存"""
    import os
    
    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 出力ディレクトリを作成: {output_dir}")
    
    # 1. 補完済みデータの保存（Excel）
    excel_data_path = os.path.join(output_dir, 'cohort_2024G1_imputed.xlsx')
    df_imputed.to_excel(excel_data_path, index=False, engine='openpyxl')
    print(f"💾 補完済みデータ保存: {excel_data_path}")
    
    # 2. 欠損値統計の保存（Excel）
    excel_path = os.path.join(output_dir, 'missing_values_report.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        missing_df.to_excel(writer, sheet_name='欠損値統計', index=False)
        
        # 実験回数とコース分布も追加
        crosstab = pd.crosstab(df_imputed['measurement_wave'], df_imputed['course_group'], margins=True)
        crosstab.to_excel(writer, sheet_name='実験回数とコース分布')
        
        # 単位変換情報を追加
        if 'tmt_combined_trailtime_converted' in df_imputed.columns:
            unit_info = pd.DataFrame({
                'Variable': ['tmt_combined_trailtime'],
                'Original_Unit': ['milliseconds'],
                'Converted_Unit': ['seconds'],
                'Conversion_Factor': [1000],
                'Note': ['Divided by 1000 to convert from ms to seconds']
            })
            unit_info.to_excel(writer, sheet_name='単位変換情報', index=False)
    
    print(f"📊 欠損値レポート保存: {excel_path}")
    
    # 3. 基本統計サマリーの保存（Excel）
    summary_path = os.path.join(output_dir, 'basic_statistics.xlsx')
    
    # 認知・非認知変数の基本統計
    cognitive_vars = [
        'corsi_ncorrect_total', 'corsi_blockspan', 'corsi_totalscore',
        'fourchoice_prop_correct', 'fourchoice_mean_rt',
        'stroop_propcorrect', 'stroop_mean_rt',
        'tmt_combined_errors', 'tmt_combined_trailtime',
        'ufov_subtest1_threshold', 'ufov_subtest2_threshold', 'ufov_subtest3_threshold'
    ]
    
    non_cognitive_vars = [
        'bigfive_extraversion', 'bigfive_agreeableness', 'bigfive_conscientiousness',
        'bigfive_neuroticism', 'bigfive_openness',
        'grit_total', 'mindset_total',
        'ct_logical_awareness', 'ct_inquiry', 'ct_objectivity', 'ct_evidence_based',
        'who5_total', 'swbs_total'
    ]
    
    all_vars = cognitive_vars + non_cognitive_vars
    
    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        # 全体の基本統計
        summary_stats = df_imputed[all_vars].describe()
        summary_stats.to_excel(writer, sheet_name='全体統計')
        
        # コース別統計
        eSports_stats = df_imputed[df_imputed['course_group'] == 'eSports'][all_vars].describe()
        liberal_stats = df_imputed[df_imputed['course_group'] == 'Liberal Arts'][all_vars].describe()
        
        eSports_stats.to_excel(writer, sheet_name='eSports Course Statistics')
        liberal_stats.to_excel(writer, sheet_name='Liberal Arts Course Statistics')
        
        # 単位情報を追加
        if 'tmt_combined_trailtime_converted' in df_imputed.columns:
            unit_summary = pd.DataFrame({
                'Variable': ['tmt_combined_trailtime'],
                'Unit': ['seconds'],
                'Note': ['Converted from milliseconds (divided by 1000)']
            })
            unit_summary.to_excel(writer, sheet_name='単位情報', index=False)
    
    print(f"📈 基本統計保存: {summary_path}")
    
    return excel_data_path, excel_path, summary_path

def convert_units(df):
    """単位変換の実行"""
    print("\n=== 単位変換 ===")
    
    # tmt_combined_trailtimeをミリ秒から秒に変換
    if 'tmt_combined_trailtime' in df.columns:
        # 元の値をバックアップ
        df['tmt_combined_trailtime_ms'] = df['tmt_combined_trailtime'].copy()
        
        # ミリ秒から秒に変換（1000で割る）
        df['tmt_combined_trailtime'] = df['tmt_combined_trailtime'] / 1000
        
        # 変換結果の確認
        print("tmt_combined_trailtime 単位変換:")
        print(f"  変換前（ミリ秒）: 平均{df['tmt_combined_trailtime_ms'].mean():.1f}ms")
        print(f"  変換後（秒）: 平均{df['tmt_combined_trailtime'].mean():.2f}秒")
        
        # 変換フラグ列の作成
        df['tmt_combined_trailtime_converted'] = True
    
    print("✅ 単位変換完了")
    return df

def main():
    """メイン実行関数"""
    print("✅ ライブラリの読み込み完了")
    
    # データの読み込みと処理
    df_raw = load_and_explore_data()
    df = extract_target_cohort(df_raw)
    df = create_course_classification(df)
    
    # 単位変換
    df = convert_units(df)
    
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
    
    # 可視化
    create_visualizations(df_imputed, cognitive_vars, non_cognitive_vars)
    
    # 結果保存
    save_results(df_imputed, missing_df)
    
    print("\n✅ データ前処理完了")
    print("✅ 二元配置分散分析の準備完了")
    print("\n🎯 次のステップ: 二元配置分散分析の実行")
    print("- 要因A: コース（eSports vs Liberal Arts）")
    print("- 要因B: 時間（実験回数 1, 2, 3）")
    print("- 従属変数: 認知・非認知スキル各指標")
    print("- 注意: tmt_combined_trailtimeは秒単位で分析されます")
    
    return df_imputed, cognitive_vars, non_cognitive_vars, all_vars

if __name__ == "__main__":
    df_imputed, cognitive_vars, non_cognitive_vars, all_vars = main()