import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, shapiro, levene, bartlett
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_white
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

def check_anova_assumptions(df_imputed, all_vars):
    """分散分析の前提条件チェック"""
    print("\n=== 分散分析の前提条件チェック ===")
    
    # 出力ディレクトリの作成
    import os
    output_dir = './analysis_result/anova_assumptions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 前提条件チェック結果を保存するリスト
    assumption_results = []
    
    # 各変数について前提条件をチェック
    for var in all_vars:
        if var in df_imputed.columns and not df_imputed[var].isnull().all():
            print(f"\n--- {var} の前提条件チェック ---")
            
            # 1. 独立性の確認（データ構造による）
            independence_check = check_independence(df_imputed, var)
            
            # 2. 等分散性の確認
            homoscedasticity_results = check_homoscedasticity(df_imputed, var)
            
            # 3. 正規性の確認
            normality_results = check_normality(df_imputed, var)
            
            # 4. 線形性の確認（反復測定ANOVAの場合）
            linearity_results = check_linearity(df_imputed, var)
            
            # 5. 残差分析
            residual_analysis(df_imputed, var, output_dir)
            
            # 6. 正規性の視覚的確認
            create_normality_plots(df_imputed, var, output_dir)
            
            # 結果をまとめる
            assumption_results.append({
                'variable': var,
                'independence': independence_check,
                'homoscedasticity_levene_p': homoscedasticity_results['levene_p'],
                'homoscedasticity_bartlett_p': homoscedasticity_results['bartlett_p'],
                'normality_shapiro_p': normality_results['shapiro_p'],
                'normality_kstest_p': normality_results['kstest_p'],
                'linearity_correlation': linearity_results['correlation'],
                'linearity_p': linearity_results['p_value']
            })
    
    # 結果をDataFrameに変換
    assumptions_df = pd.DataFrame(assumption_results)
    
    # 結果の保存
    save_assumption_results(assumptions_df, output_dir)
    
    print(f"\n✅ 前提条件チェック完了: {output_dir}に保存")
    
    return assumptions_df

def check_independence(df, var):
    """独立性の確認"""
    # データ構造の確認
    n_participants = df['participant_id'].nunique()
    n_observations = len(df)
    n_waves = df['measurement_wave'].nunique()
    
    # 反復測定設計かどうかの確認
    expected_obs = n_participants * n_waves
    is_repeated_measures = abs(n_observations - expected_obs) < (expected_obs * 0.1)  # 10%の誤差許容
    
    independence_status = "要注意: 反復測定設計" if is_repeated_measures else "OK: 独立観測"
    
    print(f"  独立性: {independence_status}")
    print(f"    参加者数: {n_participants}, 観測数: {n_observations}, 測定回数: {n_waves}")
    
    return independence_status

def check_homoscedasticity(df, var):
    """等分散性の確認"""
    # コース×Wave別のグループに分けてデータを準備
    groups = []
    group_names = []
    
    for course in ['eSports', 'Liberal Arts']:
        for wave in [1, 2, 3]:
            condition = (df['course_group'] == course) & (df['measurement_wave'] == wave)
            group_data = df[condition][var].dropna()
            
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(f"{course}_wave{wave}")
    
    # Levene検定（等分散性の検定）
    if len(groups) >= 2:
        levene_stat, levene_p = levene(*groups)
        
        # Bartlett検定（正規分布を仮定した等分散性の検定）
        bartlett_stat, bartlett_p = bartlett(*groups)
        
        print(f"  等分散性:")
        print(f"    Levene検定: F={levene_stat:.4f}, p={levene_p:.4f}")
        print(f"    Bartlett検定: χ²={bartlett_stat:.4f}, p={bartlett_p:.4f}")
        
        # 判定
        levene_result = "OK" if levene_p > 0.05 else "要注意"
        bartlett_result = "OK" if bartlett_p > 0.05 else "要注意"
        
        print(f"    判定: Levene={levene_result}, Bartlett={bartlett_result}")
        
        return {
            'levene_stat': levene_stat,
            'levene_p': levene_p,
            'bartlett_stat': bartlett_stat,
            'bartlett_p': bartlett_p,
            'levene_result': levene_result,
            'bartlett_result': bartlett_result
        }
    else:
        print(f"  等分散性: グループ数不足")
        return {
            'levene_stat': np.nan,
            'levene_p': np.nan,
            'bartlett_stat': np.nan,
            'bartlett_p': np.nan,
            'levene_result': 'データ不足',
            'bartlett_result': 'データ不足'
        }

def check_normality(df, var):
    """正規性の確認"""
    # 全データの正規性検定
    data = df[var].dropna()
    
    if len(data) > 3:
        # Shapiro-Wilk検定
        shapiro_stat, shapiro_p = shapiro(data)
        
        # Kolmogorov-Smirnov検定
        kstest_stat, kstest_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        
        print(f"  正規性:")
        print(f"    Shapiro-Wilk検定: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
        print(f"    Kolmogorov-Smirnov検定: D={kstest_stat:.4f}, p={kstest_p:.4f}")
        
        # 判定
        shapiro_result = "OK" if shapiro_p > 0.05 else "要注意"
        kstest_result = "OK" if kstest_p > 0.05 else "要注意"
        
        print(f"    判定: Shapiro={shapiro_result}, KS={kstest_result}")
        
        return {
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'kstest_stat': kstest_stat,
            'kstest_p': kstest_p,
            'shapiro_result': shapiro_result,
            'kstest_result': kstest_result
        }
    else:
        print(f"  正規性: データ数不足")
        return {
            'shapiro_stat': np.nan,
            'shapiro_p': np.nan,
            'kstest_stat': np.nan,
            'kstest_p': np.nan,
            'shapiro_result': 'データ不足',
            'kstest_result': 'データ不足'
        }

def check_linearity(df, var):
    """線形性の確認（測定回数との関係）"""
    # 測定回数と変数の相関
    correlation, p_value = stats.pearsonr(df['measurement_wave'], df[var].fillna(df[var].mean()))
    
    print(f"  線形性:")
    print(f"    測定回数との相関: r={correlation:.4f}, p={p_value:.4f}")
    
    return {
        'correlation': correlation,
        'p_value': p_value
    }

def residual_analysis(df, var, output_dir):
    """残差分析"""
    # 二元配置分散分析のモデル作成
    try:
        # 欠損値を除去
        df_clean = df[['participant_id', 'course_group', 'measurement_wave', var]].dropna()
        
        if len(df_clean) > 10:  # 十分なデータがある場合のみ
            # 統計モデルの作成
            formula = f"{var} ~ C(course_group) + C(measurement_wave) + C(course_group):C(measurement_wave)"
            model = ols(formula, data=df_clean).fit()
            
            # 残差の取得
            residuals = model.resid
            fitted_values = model.fittedvalues
            
            # 残差プロット
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. 残差 vs 予測値
            axes[0, 0].scatter(fitted_values, residuals, alpha=0.6)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_xlabel('Fitted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Fitted Values')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 残差の正規Q-Qプロット
            stats.probplot(residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Normal Q-Q Plot of Residuals')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 残差のヒストグラム
            axes[1, 0].hist(residuals, bins=15, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Histogram of Residuals')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 標準化残差 vs 予測値
            standardized_residuals = residuals / residuals.std()
            axes[1, 1].scatter(fitted_values, standardized_residuals, alpha=0.6)
            axes[1, 1].axhline(y=0, color='red', linestyle='--')
            axes[1, 1].axhline(y=2, color='orange', linestyle='--', alpha=0.7)
            axes[1, 1].axhline(y=-2, color='orange', linestyle='--', alpha=0.7)
            axes[1, 1].set_xlabel('Fitted Values')
            axes[1, 1].set_ylabel('Standardized Residuals')
            axes[1, 1].set_title('Standardized Residuals vs Fitted Values')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'Residual Analysis: {var}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/residual_analysis_{var}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        print(f"  残差分析エラー: {e}")

def create_normality_plots(df, var, output_dir):
    """正規性の視覚的確認"""
    try:
        data = df[var].dropna()
        
        if len(data) > 3:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. ヒストグラム + 正規分布曲線
            axes[0, 0].hist(data, bins=15, alpha=0.7, density=True, edgecolor='black')
            
            # 正規分布曲線の追加
            x = np.linspace(data.min(), data.max(), 100)
            normal_curve = stats.norm.pdf(x, data.mean(), data.std())
            axes[0, 0].plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
            
            axes[0, 0].set_xlabel('Value')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('Histogram with Normal Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Q-Qプロット
            stats.probplot(data, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 箱ひげ図
            axes[1, 0].boxplot(data, vert=True)
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_title('Box Plot')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. コース×Wave別の分布
            courses = ['eSports', 'Liberal Arts']
            waves = [1, 2, 3]
            
            for i, course in enumerate(courses):
                for j, wave in enumerate(waves):
                    condition = (df['course_group'] == course) & (df['measurement_wave'] == wave)
                    group_data = df[condition][var].dropna()
                    
                    if len(group_data) > 0:
                        axes[1, 1].hist(group_data, alpha=0.5, label=f'{course}_Wave{wave}', bins=10)
            
            axes[1, 1].set_xlabel('Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution by Course and Wave')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'Normality Check: {var}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/normality_check_{var}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        print(f"  正規性プロット作成エラー: {e}")

def save_assumption_results(assumptions_df, output_dir):
    """前提条件チェック結果の保存"""
    # Excel形式で保存
    excel_path = f'{output_dir}/anova_assumptions_results.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 全結果
        assumptions_df.to_excel(writer, sheet_name='前提条件チェック結果', index=False)
        
        # 問題のある変数を抽出
        problematic_vars = assumptions_df[
            (assumptions_df['homoscedasticity_levene_p'] < 0.05) |
            (assumptions_df['normality_shapiro_p'] < 0.05)
        ].copy()
        
        if len(problematic_vars) > 0:
            problematic_vars.to_excel(writer, sheet_name='要注意変数', index=False)
        
        # 要約統計
        summary_stats = pd.DataFrame({
            'Check': ['Homoscedasticity (Levene)', 'Normality (Shapiro-Wilk)'],
            'Variables_OK': [
                sum(assumptions_df['homoscedasticity_levene_p'] >= 0.05),
                sum(assumptions_df['normality_shapiro_p'] >= 0.05)
            ],
            'Variables_Problematic': [
                sum(assumptions_df['homoscedasticity_levene_p'] < 0.05),
                sum(assumptions_df['normality_shapiro_p'] < 0.05)
            ],
            'Total_Variables': [len(assumptions_df), len(assumptions_df)]
        })
        
        summary_stats.to_excel(writer, sheet_name='要約統計', index=False)
    
    print(f"📊 前提条件チェック結果保存: {excel_path}")

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
    
    # 分散分析の前提条件チェック
    assumptions_df = check_anova_assumptions(df_imputed, all_vars)
    
    # 可視化
    create_visualizations(df_imputed, cognitive_vars, non_cognitive_vars)
    
    # 結果保存
    save_results(df_imputed, missing_df)
    
    print("\n✅ データ前処理完了")
    print("✅ 分散分析の前提条件チェック完了")
    print("✅ 二元配置分散分析の準備完了")
    print("\n🎯 次のステップ: 二元配置分散分析の実行")
    print("- 要因A: コース（eSports vs Liberal Arts）")
    print("- 要因B: 時間（実験回数 1, 2, 3）")
    print("- 従属変数: 認知・非認知スキル各指標")
    print("- 注意: tmt_combined_trailtimeは秒単位で分析されます")
    
    # 前提条件チェック結果のサマリー表示
    print("\n📊 前提条件チェック結果サマリー:")
    if len(assumptions_df) > 0:
        levene_ok = sum(assumptions_df['homoscedasticity_levene_p'] >= 0.05)
        shapiro_ok = sum(assumptions_df['normality_shapiro_p'] >= 0.05)
        total_vars = len(assumptions_df)
        
        print(f"  等分散性（Levene検定）: {levene_ok}/{total_vars}変数がOK")
        print(f"  正規性（Shapiro-Wilk検定）: {shapiro_ok}/{total_vars}変数がOK")
        print("  詳細は './analysis_result/anova_assumptions/' フォルダを確認してください")
    
    return df_imputed, cognitive_vars, non_cognitive_vars, all_vars, assumptions_df

if __name__ == "__main__":
    df_imputed, cognitive_vars, non_cognitive_vars, all_vars, assumptions_df = main()