"""
線形混合効果モデル（LMM）分析
eスポーツコース効果の縦断研究

実行方法:
python lmm_analysis.py

必要なライブラリ:
pip install statsmodels pandas numpy matplotlib seaborn plotly openpyxl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# 統計モデル
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# 可視化の拡張
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

def setup_output_directory():
    """出力ディレクトリの作成"""
    output_dir = "analysis_result/lmm_result"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 出力ディレクトリを作成: {output_dir}")
    return output_dir

def load_preprocessed_data():
    """
    前処理済みデータの読み込み
    data_overview.pyで作成されたデータを想定
    """
    print("=== データの読み込み ===")
    try:
        # メインデータ読み込み（パスは適宜調整）
        df = pd.read_excel('./data/data_master.xlsx', sheet_name='master')
        
        # cohort 2024_G1の抽出と前処理
        df = df[df['cohort'] == '2024_G1'].copy()
        df.reset_index(drop=True, inplace=True)
        
        # コース分類
        df['course_group'] = df['course'].map({  # type: ignore
            'eスポーツエデュケーションコース': 'eSports',
            'リベラルアーツコース': 'Liberal Arts'
        })
        
        print(f"データ形状: {df.shape}")
        print(f"参加者数: {df['participant_id'].nunique()}名")  # type: ignore
        print(f"測定時期: {sorted(df['measurement_wave'].unique())}")  # type: ignore
        
        return df
        
    except FileNotFoundError:
        print("データファイルが見つかりません。パスを確認してください。")
        return None

def define_analysis_variables():
    """分析対象変数の定義"""
    
    # 認知スキル変数（12個）
    cognitive_vars = [
        'corsi_ncorrect_total',      # Corsi正答数
        'corsi_blockspan',           # Corsiブロックスパン
        'corsi_totalscore',          # Corsi総得点
        'fourchoice_prop_correct',   # 四択正答率
        'fourchoice_mean_rt',        # 四択反応時間
        'stroop_propcorrect',        # Stroop正答率
        'stroop_mean_rt',            # Stroop反応時間
        'tmt_combined_errors',       # TMTエラー数
        'tmt_combined_trailtime',    # TMT完了時間
        'ufov_subtest1_threshold',   # UFOV閾値1
        'ufov_subtest2_threshold',   # UFOV閾値2
        'ufov_subtest3_threshold'    # UFOV閾値3
    ]
    
    # 非認知スキル変数（13個）
    non_cognitive_vars = [
        'bigfive_extraversion',      # ビッグファイブ：外向性
        'bigfive_agreeableness',     # ビッグファイブ：協調性
        'bigfive_conscientiousness', # ビッグファイブ：誠実性
        'bigfive_neuroticism',       # ビッグファイブ：神経症傾向
        'bigfive_openness',          # ビッグファイブ：開放性
        'grit_total',                # GRIT総得点
        'mindset_total',             # マインドセット総得点
        'ct_logical_awareness',      # 批判的思考：論理的気づき
        'ct_inquiry',                # 批判的思考：探究心
        'ct_objectivity',            # 批判的思考：客観性
        'ct_evidence_based',         # 批判的思考：根拠重視
        'who5_total',                # WHO-5ウェルビーイング
        'swbs_total'                 # 主観的ウェルビーイング
    ]
    
    # 有意な効果があった変数（優先分析）
    significant_vars = [
        # コース効果あり
        'fourchoice_mean_rt', 'tmt_combined_errors', 'tmt_combined_trailtime', 
        'ufov_subtest3_threshold', 'bigfive_extraversion', 'bigfive_openness',
        'ct_logical_awareness', 'ct_evidence_based',
        # 時間効果あり
        'corsi_ncorrect_total', 'corsi_blockspan', 'corsi_totalscore'
    ]
    
    # 全分析変数
    all_analysis_vars = cognitive_vars + non_cognitive_vars
    
    return {
        'cognitive': cognitive_vars,
        'non_cognitive': non_cognitive_vars,
        'significant': significant_vars,
        'all': all_analysis_vars
    }

def run_basic_lmm(df, variable, verbose=True):
    """
    基本的なLMM分析（ランダム切片モデル）
    
    Parameters:
    -----------
    df : pd.DataFrame
        分析データ
    variable : str
        従属変数名
    verbose : bool
        詳細出力するかどうか
    """
    
    # 欠損値のある行を除外
    analysis_data = df[['participant_id', 'course_group', 'measurement_wave', variable]].dropna()
    
    if len(analysis_data) == 0:
        print(f"❌ {variable}: 分析可能なデータがありません")
        return None
    
    try:
        # ランダム切片モデル
        formula = f"{variable} ~ C(course_group) * measurement_wave"
        model = smf.mixedlm(formula, analysis_data, groups=analysis_data["participant_id"])
        result = model.fit()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 {variable} - ランダム切片モデル")
            print(f"{'='*60}")
            print(f"サンプルサイズ: {len(analysis_data)}観測, {analysis_data['participant_id'].nunique()}名")
            print(f"欠損処理: {len(df) - len(analysis_data)}観測を除外")
            print("\n固定効果:")
            print(result.summary().tables[1])
            
            # 効果の解釈
            interpret_lmm_results(result, variable)
        
        return result
        
    except Exception as e:
        print(f"❌ {variable}: LMM分析でエラー - {str(e)}")
        return None

def run_random_slope_lmm(df, variable, verbose=True):
    """
    ランダム傾きモデル（個人の成長速度差を考慮）
    
    Parameters:
    -----------
    df : pd.DataFrame
        分析データ
    variable : str
        従属変数名
    verbose : bool
        詳細出力するかどうか
    """
    
    analysis_data = df[['participant_id', 'course_group', 'measurement_wave', variable]].dropna()
    
    if len(analysis_data) == 0:
        print(f"❌ {variable}: 分析可能なデータがありません")
        return None
    
    try:
        # ランダム傾きモデル
        formula = f"{variable} ~ C(course_group) * measurement_wave"
        model = smf.mixedlm(formula, analysis_data, 
                           groups=analysis_data["participant_id"],
                           re_formula="~ measurement_wave")
        result = model.fit()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"📈 {variable} - ランダム傾きモデル")
            print(f"{'='*60}")
            print(f"サンプルサイズ: {len(analysis_data)}観測, {analysis_data['participant_id'].nunique()}名")
            print("\n固定効果:")
            print(result.summary().tables[1])
            
            # ランダム効果の分散
            print(f"\nRandom Effects Variance:")
            print(f"Individual differences (intercept): {result.cov_re.iloc[0,0]:.4f}")
            if result.cov_re.shape[0] > 1:
                print(f"Growth rate differences (slope): {result.cov_re.iloc[1,1]:.4f}")
                print(f"Intercept-slope correlation: {result.cov_re.iloc[0,1]/np.sqrt(result.cov_re.iloc[0,0]*result.cov_re.iloc[1,1]):.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ {variable}: ランダム傾きモデルでエラー - {str(e)}")
        return None

def compare_models(df, variable):
    """
    モデル比較（ランダム切片 vs ランダム傾き）
    """
    print(f"\n🔍 {variable} - モデル比較")
    print("-" * 50)
    
    # 両モデルを実行
    model1 = run_basic_lmm(df, variable, verbose=False)
    model2 = run_random_slope_lmm(df, variable, verbose=False)
    
    if model1 is None or model2 is None:
        print("モデル比較できません")
        return
    
    # AIC/BIC比較
    print(f"ランダム切片モデル - AIC: {model1.aic:.2f}, BIC: {model1.bic:.2f}")
    print(f"ランダム傾きモデル - AIC: {model2.aic:.2f}, BIC: {model2.bic:.2f}")
    
    # より良いモデルの判定
    if model2.aic < model1.aic:
        print("✅ Random slope model is superior (individual growth rate differences are important)")
    else:
        print("✅ Random intercept model is sufficient (individual growth rate differences are small)")
    
    return model1, model2

def interpret_lmm_results(result, variable):
    """LMM結果の解釈"""
    
    try:
        # 固定効果の抽出
        coef_table = result.summary().tables[1]
        params = result.params
        pvalues = result.pvalues
        
        print(f"\n💡 {variable}の結果解釈:")
        print("-" * 40)
        
        # コース効果
        if 'C(course_group)[T.リベラルアーツ]' in params:
            course_coef = params['C(course_group)[T.リベラルアーツ]']
            course_p = pvalues['C(course_group)[T.リベラルアーツ]']
            
            if course_p < 0.05:
                if 'tmt_combined' in variable or 'rt' in variable:
                    # TMT課題や反応時間は短い方が良い
                    direction = "low (better)" if course_coef > 0 else "high (worse)"
                    comparison = "Liberal Arts is better than eSports" if course_coef > 0 else "eSports is better than Liberal Arts"
                else:
                    # 一般的な指標は高い方が良い
                    direction = "high" if course_coef > 0 else "low"
                    comparison = "Liberal Arts is better than eSports" if course_coef > 0 else "eSports is better than Liberal Arts"
                
                print(f"🎯 Course Effect: {comparison} {direction} (p={course_p:.4f})")
            else:
                print(f"🎯 Course Effect: No significant difference (p={course_p:.4f})")
        
        # 時間効果
        if 'measurement_wave' in params:
            time_coef = params['measurement_wave']
            time_p = pvalues['measurement_wave']
            
            if time_p < 0.05:
                if 'tmt_combined' in variable or 'rt' in variable or 'errors' in variable:
                    # TMT課題、反応時間、エラー数は減少が良い
                    direction = "improvement (shortening/reduction)" if time_coef < 0 else "deterioration (increase)"
                else:
                    # 一般的な指標は増加が良い
                    direction = "improvement" if time_coef > 0 else "deterioration"
                
                print(f"⏰ Time Effect: {direction} per Experiment Number (p={time_p:.4f})")
            else:
                print(f"⏰ Time Effect: No significant change (p={time_p:.4f})")
        
        # 交互作用
        interaction_key = 'C(course_group)[T.リベラルアーツ]:measurement_wave'
        if interaction_key in params:
            int_coef = params[interaction_key]
            int_p = pvalues[interaction_key]
            
            if int_p < 0.05:
                print(f"🔄 Interaction: Experiment Number changes differently between courses (p={int_p:.4f})")
            else:
                print(f"🔄 Interaction: Experiment Number changes similarly between courses (p={int_p:.4f})")
                
    except Exception as e:
        print(f"結果解釈でエラー: {str(e)}")

def visualize_individual_trajectories(df, variable, output_dir):
    """
    個人軌跡の可視化
    """
    
    # データ準備
    plot_data = df[['participant_id', 'course_group', 'measurement_wave', variable]].dropna()
    
    if len(plot_data) == 0:
        print(f"❌ {variable}: 可視化用データがありません")
        return
    
    # Plotlyでインタラクティブ可視化
    fig = px.line(plot_data, 
                  x='measurement_wave', 
                  y=variable,
                  color='course_group',
                  line_group='participant_id',
                  title=f'{variable} - Individual Trajectories',
                  labels={'measurement_wave': 'Experiment Number', 
                         'course_group': 'Course',
                         variable: variable})
    
    # 群平均も追加
    mean_data = plot_data.groupby(['course_group', 'measurement_wave'])[variable].mean().reset_index()
    
    for course in mean_data['course_group'].unique():
        course_data = mean_data[mean_data['course_group'] == course]
        fig.add_trace(go.Scatter(x=course_data['measurement_wave'], 
                                y=course_data[variable],
                                mode='lines+markers',
                                name=f'{course} (Mean)',
                                line=dict(width=4)))
    
    # X軸を整数のみに設定
    fig.update_xaxes(
        tickvals=[1, 2, 3],
        ticktext=['1', '2', '3'],
        title='Experiment Number'
    )
    
    fig.update_layout(height=600, showlegend=True)
    
    # ファイル保存
    save_path = os.path.join(output_dir, f"trajectory_{variable}.html")
    fig.write_html(save_path)
    print(f"📊 {variable}の軌跡図を保存: {save_path}")
    
    return fig

def create_comprehensive_lmm_summary(df, variables, output_dir):
    """
    全変数のLMM結果サマリー作成
    """
    
    print("\n" + "="*80)
    print("📊 包括的LMM分析サマリー - 全25変数")
    print("="*80)
    
    summary_results = []
    
    # 認知スキル分析
    print(f"\n🧠 認知スキル変数の分析 ({len(variables['cognitive'])}変数)")
    print("-" * 60)
    
    for var in variables['cognitive']:
        result = run_basic_lmm(df, var, verbose=False)
        if result is not None:
            summary_results.append(extract_lmm_summary(result, var, 'cognitive'))
    
    # 非認知スキル分析
    print(f"\n💭 非認知スキル変数の分析 ({len(variables['non_cognitive'])}変数)")
    print("-" * 60)
    
    for var in variables['non_cognitive']:
        result = run_basic_lmm(df, var, verbose=False)
        if result is not None:
            summary_results.append(extract_lmm_summary(result, var, 'non_cognitive'))
    
    # 結果をDataFrameに変換
    summary_df = pd.DataFrame(summary_results)
    
    if len(summary_df) > 0:
        # 結果の整理と表示
        display_lmm_summary_table(summary_df)
        save_lmm_results(summary_df, output_dir)
    
    return summary_df

def extract_lmm_summary(result, variable, category):
    """
    LMM結果からサマリー情報を抽出
    """
    try:
        params = result.params
        pvalues = result.pvalues
        
        # コース効果
        course_coef = params.get('C(course_group)[T.リベラルアーツ]', np.nan)
        course_p = pvalues.get('C(course_group)[T.リベラルアーツ]', np.nan)
        
        # 時間効果
        time_coef = params.get('measurement_wave', np.nan)
        time_p = pvalues.get('measurement_wave', np.nan)
        
        # 交互作用効果
        interaction_coef = params.get('C(course_group)[T.リベラルアーツ]:measurement_wave', np.nan)
        interaction_p = pvalues.get('C(course_group)[T.リベラルアーツ]:measurement_wave', np.nan)
        
        return {
            'Variable': variable,
            'Category': category,
            'Course_Coef': course_coef,
            'Course_P': course_p,
            'Course_Sig': '***' if course_p < 0.001 else '**' if course_p < 0.01 else '*' if course_p < 0.05 else 'ns',
            'Time_Coef': time_coef,
            'Time_P': time_p,
            'Time_Sig': '***' if time_p < 0.001 else '**' if time_p < 0.01 else '*' if time_p < 0.05 else 'ns',
            'Interaction_Coef': interaction_coef,
            'Interaction_P': interaction_p,
            'Interaction_Sig': '***' if interaction_p < 0.001 else '**' if interaction_p < 0.01 else '*' if interaction_p < 0.05 else 'ns',
            'AIC': result.aic,
            'BIC': result.bic,
            'Log_Likelihood': result.llf
        }
        
    except Exception as e:
        print(f"⚠️ {variable}: サマリー抽出エラー - {str(e)}")
        return None

def display_lmm_summary_table(summary_df):
    """
    LMM結果サマリーテーブルの表示
    """
    
    print(f"\n📋 LMM分析結果サマリー")
    print("="*100)
    
    # 有意な効果のカウント
    course_sig = (summary_df['Course_P'] < 0.05).sum()
    time_sig = (summary_df['Time_P'] < 0.05).sum()
    interaction_sig = (summary_df['Interaction_P'] < 0.05).sum()
    
    print(f"有意な効果 (p < 0.05):")
    print(f"  コース効果: {course_sig}/{len(summary_df)}変数")
    print(f"  時間効果: {time_sig}/{len(summary_df)}変数")
    print(f"  交互作用: {interaction_sig}/{len(summary_df)}変数")
    
    # カテゴリ別サマリー
    print(f"\n📊 カテゴリ別サマリー:")
    for category in ['cognitive', 'non_cognitive']:
        cat_data = summary_df[summary_df['Category'] == category]
        if len(cat_data) > 0:
            cat_course_sig = (cat_data['Course_P'] < 0.05).sum()
            cat_time_sig = (cat_data['Time_P'] < 0.05).sum()
            
            category_name = '認知スキル' if category == 'cognitive' else '非認知スキル'
            print(f"  {category_name}: コース効果{cat_course_sig}/{len(cat_data)}, 時間効果{cat_time_sig}/{len(cat_data)}")
    
    # 有意な効果のある変数をリスト表示
    print(f"\n🎯 有意なコース効果のある変数:")
    course_vars = summary_df[summary_df['Course_P'] < 0.05].sort_values('Course_P')
    for _, row in course_vars.iterrows():
        direction = "Liberal Arts > eSports" if row['Course_Coef'] > 0 else "eSports > Liberal Arts"
        print(f"  {row['Variable']} ({row['Category']}): p={row['Course_P']:.4f} {row['Course_Sig']} [{direction}]")
    
    print(f"\n⏰ 有意な時間効果のある変数:")
    time_vars = summary_df[summary_df['Time_P'] < 0.05].sort_values('Time_P')
    for _, row in time_vars.iterrows():
        direction = "improvement" if row['Time_Coef'] > 0 else "deterioration"
        print(f"  {row['Variable']} ({row['Category']}): p={row['Time_P']:.4f} {row['Time_Sig']} [{direction}]")
    
    if interaction_sig > 0:
        print(f"\n🔄 有意な交互作用のある変数:")
        int_vars = summary_df[summary_df['Interaction_P'] < 0.05].sort_values('Interaction_P')
        for _, row in int_vars.iterrows():
            print(f"  {row['Variable']} ({row['Category']}): p={row['Interaction_P']:.4f} {row['Interaction_Sig']}")

def save_lmm_results(summary_df, output_dir):
    """
    LMM結果をExcelファイルに保存
    """
    try:
        # メインの結果保存
        excel_path = os.path.join(output_dir, "lmm_results_comprehensive.xlsx")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 全結果シート
            summary_df.to_excel(writer, sheet_name='全結果サマリー', index=False)
            
            # 有意な効果別シート
            course_sig = summary_df[summary_df['Course_P'] < 0.05].sort_values('Course_P')
            if len(course_sig) > 0:
                course_sig.to_excel(writer, sheet_name='有意なコース効果', index=False)
            
            time_sig = summary_df[summary_df['Time_P'] < 0.05].sort_values('Time_P')
            if len(time_sig) > 0:
                time_sig.to_excel(writer, sheet_name='有意な時間効果', index=False)
            
            interaction_sig = summary_df[summary_df['Interaction_P'] < 0.05].sort_values('Interaction_P')
            if len(interaction_sig) > 0:
                interaction_sig.to_excel(writer, sheet_name='有意な交互作用', index=False)
            
            # カテゴリ別シート
            cognitive_data = summary_df[summary_df['Category'] == 'cognitive']
            cognitive_data.to_excel(writer, sheet_name='認知スキル結果', index=False)
            
            non_cognitive_data = summary_df[summary_df['Category'] == 'non_cognitive']
            non_cognitive_data.to_excel(writer, sheet_name='非認知スキル結果', index=False)
        
        print(f"💾 LMM結果をExcelで保存: {excel_path}")
        return excel_path
        
    except Exception as e:
        print(f"⚠️ Excel保存エラー: {str(e)}")
        return None

def run_detailed_analysis_for_significant_vars(df, summary_df):
    """
    有意な効果のあった変数の詳細分析
    """
    
    # 有意な効果のある変数を特定
    significant_vars = summary_df[
        (summary_df['Course_P'] < 0.05) | 
        (summary_df['Time_P'] < 0.05) | 
        (summary_df['Interaction_P'] < 0.05)
    ]['Variable'].tolist()
    
    print(f"\n🔍 有意な効果のあった{len(significant_vars)}変数の詳細分析")
    print("="*60)
    
    detailed_results = {}
    
    for var in significant_vars:
        print(f"\n--- {var} 詳細分析 ---")
        
        # ランダム切片モデル
        basic_model = run_basic_lmm(df, var, verbose=True)
        
        # ランダム傾きモデルも試行
        try:
            slope_model = run_random_slope_lmm(df, var, verbose=False)
            if slope_model is not None and basic_model is not None:
                print(f"Model comparison - Intercept AIC: {basic_model.aic:.2f}, Slope AIC: {slope_model.aic:.2f}")
                if slope_model.aic < basic_model.aic:
                    print("✅ Random slope model is superior")
                    detailed_results[var] = slope_model
                else:
                    print("✅ Random intercept model is sufficient")
                    detailed_results[var] = basic_model
            else:
                if basic_model is not None:
                    detailed_results[var] = basic_model
        except:
            if basic_model is not None:
                detailed_results[var] = basic_model
    
    return detailed_results

def main():
    """メイン実行関数"""
    
    print("�� 線形混合効果モデル（LMM）分析開始 - 全25変数")
    print("="*80)
    
    # 出力ディレクトリの作成
    output_dir = setup_output_directory()
    
    # データ読み込み
    df = load_preprocessed_data()
    if df is None:
        return
    
    # 分析変数定義
    variables = define_analysis_variables()
    
    print(f"\n📋 分析対象:")
    print(f"  認知スキル変数: {len(variables['cognitive'])}個")
    print(f"  非認知スキル変数: {len(variables['non_cognitive'])}個")
    print(f"  総分析変数: {len(variables['all'])}個")
    print(f"  出力先: {output_dir}")
    
    # Phase 1: 全変数のLMM分析
    print(f"\n🎯 Phase 1: 全25変数のLMM分析")
    print("-" * 60)
    
    summary_df = create_comprehensive_lmm_summary(df, variables, output_dir)
    
    # Phase 2: 有意な効果のあった変数の詳細分析
    print(f"\n🔍 Phase 2: 有意な効果のあった変数の詳細分析")
    print("-" * 60)
    
    detailed_results = run_detailed_analysis_for_significant_vars(df, summary_df)
    
    # Phase 3: 可視化（インタラクティブ + 静的）
    print(f"\n📊 Phase 3: 可視化作成")
    print("-" * 60)
    
    # 最も強い効果のあった変数を可視化
    top_vars = summary_df.nsmallest(5, 'Course_P')['Variable'].tolist()
    top_vars.extend(summary_df.nsmallest(3, 'Time_P')['Variable'].tolist())
    top_vars = list(set(top_vars))  # 重複除去
    
    print(f"インタラクティブ軌跡図対象: {top_vars}")
    
    # インタラクティブ軌跡図
    for var in top_vars:
        if var in df.columns:
            try:
                visualize_individual_trajectories(df, var, output_dir)
            except Exception as e:
                print(f"⚠️ {var}の軌跡図でエラー: {str(e)}")
    
    # 静的グラフ作成
    create_static_visualizations(df, variables, output_dir)
    
    # 相関ヒートマップ
    create_correlation_heatmap(df, variables, output_dir)
    
    # Phase 4: 総合サマリー
    print(f"\n📈 Phase 4: 総合分析サマリー")
    print("-" * 60)
    
    create_final_summary_report(summary_df, detailed_results, variables, output_dir)
    
    print(f"\n✅ 全25変数のLMM分析完了!")
    print(f"📁 以下のファイルが生成されました:")
    print(f"   📊 {output_dir}/lmm_results_comprehensive.xlsx (全結果)")
    print(f"   📈 {output_dir}/lmm_analysis_report.xlsx (分析レポート)")
    print(f"   📄 {output_dir}/final_lmm_report.txt (テキストレポート)")
    print(f"   🌐 {output_dir}/trajectory_*.html (個人軌跡図)")
    print(f"   📉 {output_dir}/graphs/ (静的グラフ集)")
    print(f"      - group_mean_*.png (群平均比較)")
    print(f"      - effect_sizes_comparison.png (効果サイズ)")
    print(f"      - category_improvement_comparison.png (カテゴリ別改善)")
    print(f"      - *_correlation_heatmap.png (相関ヒートマップ)")
    
    return {
        'summary_df': summary_df,
        'detailed_results': detailed_results,
        'variables': variables,
        'output_dir': output_dir
    }

def create_final_summary_report(summary_df, detailed_results, variables, output_dir):
    """
    最終サマリーレポートの作成
    """
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("eスポーツコース効果：線形混合効果モデル（LMM）分析 最終レポート")
    report_lines.append("="*80)
    report_lines.append("")
    
    # 分析概要
    report_lines.append("📊 分析概要")
    report_lines.append("-" * 40)
    report_lines.append(f"総分析変数: {len(variables['all'])}個")
    report_lines.append(f"  - 認知スキル: {len(variables['cognitive'])}個")
    report_lines.append(f"  - 非認知スキル: {len(variables['non_cognitive'])}個")
    report_lines.append("")
    
    # 主要な発見
    course_sig = (summary_df['Course_P'] < 0.05).sum()
    time_sig = (summary_df['Time_P'] < 0.05).sum()
    interaction_sig = (summary_df['Interaction_P'] < 0.05).sum()
    
    report_lines.append("🎯 主要な発見")
    report_lines.append("-" * 40)
    report_lines.append(f"有意なコース効果: {course_sig}/{len(summary_df)}変数 ({course_sig/len(summary_df)*100:.1f}%)")
    report_lines.append(f"有意な時間効果: {time_sig}/{len(summary_df)}変数 ({time_sig/len(summary_df)*100:.1f}%)")
    report_lines.append(f"有意な交互作用: {interaction_sig}/{len(summary_df)}変数 ({interaction_sig/len(summary_df)*100:.1f}%)")
    report_lines.append("")
    
    # カテゴリ別分析
    report_lines.append("📋 カテゴリ別効果")
    report_lines.append("-" * 40)
    
    for category in ['cognitive', 'non_cognitive']:
        cat_data = summary_df[summary_df['Category'] == category]
        cat_course_sig = (cat_data['Course_P'] < 0.05).sum()
        cat_time_sig = (cat_data['Time_P'] < 0.05).sum()
        
        category_name = '認知スキル' if category == 'cognitive' else '非認知スキル'
        report_lines.append(f"{category_name}:")
        report_lines.append(f"  コース効果: {cat_course_sig}/{len(cat_data)}変数")
        report_lines.append(f"  時間効果: {cat_time_sig}/{len(cat_data)}変数")
        report_lines.append("")
    
    # 最も強い効果
    report_lines.append("🏆 最も強い効果を示した変数")
    report_lines.append("-" * 40)
    
    # コース効果TOP5
    top_course = summary_df.nsmallest(5, 'Course_P')
    report_lines.append("コース効果 TOP5:")
    for i, (_, row) in enumerate(top_course.iterrows(), 1):
        direction = "Liberal Arts > eSports" if row['Course_Coef'] > 0 else "eSports > Liberal Arts"
        report_lines.append(f"  {i}. {row['Variable']} (p={row['Course_P']:.4f}) [{direction}]")
    report_lines.append("")
    
    # 時間効果TOP5
    top_time = summary_df.nsmallest(5, 'Time_P')
    report_lines.append("時間効果 TOP5:")
    for i, (_, row) in enumerate(top_time.iterrows(), 1):
        direction = "improvement" if row['Time_Coef'] > 0 else "deterioration"
        report_lines.append(f"  {i}. {row['Variable']} (p={row['Time_P']:.4f}) [{direction}]")
    report_lines.append("")
    
    # レポート保存（テキストとExcel両方）
    try:
        # テキストファイル
        text_path = os.path.join(output_dir, 'final_lmm_report.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"📄 最終レポートを保存: {text_path}")
        
        # Excelレポート
        excel_report_path = os.path.join(output_dir, 'lmm_analysis_report.xlsx')
        with pd.ExcelWriter(excel_report_path, engine='openpyxl') as writer:
            # サマリー統計
            summary_stats = pd.DataFrame({
                '項目': ['総変数数', '認知スキル変数', '非認知スキル変数', 
                        '有意なコース効果', '有意な時間効果', '有意な交互作用'],
                '数値': [len(variables['all']), len(variables['cognitive']), len(variables['non_cognitive']),
                        course_sig, time_sig, interaction_sig]
            })
            summary_stats.to_excel(writer, sheet_name='分析サマリー', index=False)
            
            # TOP効果
            top_course.to_excel(writer, sheet_name='コース効果TOP5', index=False)
            top_time.to_excel(writer, sheet_name='時間効果TOP5', index=False)
        
        print(f"📊 Excelレポートを保存: {excel_report_path}")
        
    except Exception as e:
        print(f"⚠️ レポート保存エラー: {str(e)}")
    
    # コンソールにも表示
    print('\n'.join(report_lines))

def create_static_visualizations(df, variables, output_dir):
    """
    静的グラフの作成（PNG保存）
    """
    
    print(f"\n📈 静的グラフの作成")
    print("-" * 40)
    
    # グラフ保存用ディレクトリ
    graph_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)
    
    # 1. 群平均比較グラフ（エラーバー付き）
    create_group_mean_plots(df, variables['significant'], graph_dir)
    
    # 2. 効果サイズ可視化
    create_effect_size_plots(df, variables['all'], graph_dir)
    
    # 3. カテゴリ別サマリーグラフ
    create_category_summary_plots(df, variables, graph_dir)
    
    print(f"📊 静的グラフを保存: {graph_dir}/")

def create_group_mean_plots(df, variables, graph_dir):
    """
    群平均比較グラフ（エラーバー付き）
    """
    
    print("  群平均比較グラフを作成中...")
    
    for var in variables:
        try:
            plot_data = df[['course_group', 'measurement_wave', var]].dropna()
            if len(plot_data) == 0:
                continue
                
            # 群平均とSE計算
            summary_stats = plot_data.groupby(['course_group', 'measurement_wave'])[var].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            summary_stats['se'] = summary_stats['std'] / np.sqrt(summary_stats['count'])
            
            # matplotlib図の作成
            plt.figure(figsize=(10, 6))
            
            for course in summary_stats['course_group'].unique():
                course_data = summary_stats[summary_stats['course_group'] == course]
                plt.errorbar(course_data['measurement_wave'], 
                           course_data['mean'],
                           yerr=course_data['se'],
                           marker='o', linewidth=2, markersize=8,
                           label=course, capsize=5)
            
            # X軸を整数のみに設定
            plt.xticks([1, 2, 3])
            plt.xlabel('Experiment Number', fontsize=12)
            plt.ylabel(var, fontsize=12)
            plt.title(f'{var} - Group Mean Comparison (±SE)', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # PNG保存
            save_path = os.path.join(graph_dir, f"group_mean_{var}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"    ⚠️ {var}: エラー - {str(e)}")

def create_effect_size_plots(df, variables, graph_dir):
    """
    効果サイズの可視化
    """
    
    print("  効果サイズグラフを作成中...")
    
    try:
        effect_sizes = []
        
        for var in variables:
            analysis_data = df[['participant_id', 'course_group', 'measurement_wave', var]].dropna()
            if len(analysis_data) == 0:
                continue
            
            # Wave1とWave3でのコース間効果サイズ（Cohen's d）
            for wave in [1, 3]:
                wave_data = analysis_data[analysis_data['measurement_wave'] == wave]
                if len(wave_data) < 10:  # 最小サンプルサイズ
                    continue
                    
                esports = wave_data[wave_data['course_group'] == 'eSports'][var]
                liberal = wave_data[wave_data['course_group'] == 'Liberal Arts'][var]
                
                if len(esports) > 0 and len(liberal) > 0:
                    # Cohen's d計算
                    pooled_std = np.sqrt(((len(esports)-1)*esports.var() + 
                                        (len(liberal)-1)*liberal.var()) / 
                                       (len(esports)+len(liberal)-2))
                    cohens_d = (esports.mean() - liberal.mean()) / pooled_std
                    
                    effect_sizes.append({
                        'Variable': var,
                        'Wave': f'Experiment {wave}',
                        'Cohens_d': cohens_d,
                        'Category': 'cognitive' if var in ['corsi_ncorrect_total', 'corsi_blockspan', 'corsi_totalscore',
                                                          'fourchoice_prop_correct', 'fourchoice_mean_rt',
                                                          'stroop_propcorrect', 'stroop_mean_rt',
                                                          'tmt_combined_errors', 'tmt_combined_trailtime',
                                                          'ufov_subtest1_threshold', 'ufov_subtest2_threshold', 'ufov_subtest3_threshold'] else 'non_cognitive'
                    })
        
        if len(effect_sizes) > 0:
            effect_df = pd.DataFrame(effect_sizes)
            
            # Wave1とWave3の効果サイズ比較
            plt.figure(figsize=(12, 8))
            
            # サブプロット作成
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            for i, experiment in enumerate(['Experiment 1', 'Experiment 3']):
                experiment_data = effect_df[effect_df['Wave'] == experiment]
                
                cognitive = experiment_data[experiment_data['Category'] == 'cognitive']['Cohens_d']
                non_cognitive = experiment_data[experiment_data['Category'] == 'non_cognitive']['Cohens_d']
                
                ax = ax1 if i == 0 else ax2
                
                # バイオリンプロット
                if len(cognitive) > 0:
                    parts1 = ax.violinplot([cognitive], positions=[1], widths=0.6, 
                                         showmeans=True, showmedians=True)
                    parts1['bodies'][0].set_facecolor('lightblue')
                    parts1['bodies'][0].set_alpha(0.7)
                
                if len(non_cognitive) > 0:
                    parts2 = ax.violinplot([non_cognitive], positions=[2], widths=0.6,
                                         showmeans=True, showmedians=True)
                    parts2['bodies'][0].set_facecolor('lightcoral')
                    parts2['bodies'][0].set_alpha(0.7)
                
                ax.set_xticks([1, 2])
                ax.set_xticklabels(['Cognitive Skills', 'Non-Cognitive Skills'])
                ax.set_ylabel("Cohen's d")
                ax.set_title(f'{experiment} - Effect Size Distribution')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                # 効果サイズの解釈線
                ax.axhline(y=0.2, color='green', linestyle=':', alpha=0.5, label='Small Effect')
                ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5, label='Medium Effect')
                ax.axhline(y=0.8, color='red', linestyle=':', alpha=0.5, label='Large Effect')
                
                if i == 0:
                    ax.legend()
            
            plt.tight_layout()
            save_path = os.path.join(graph_dir, "effect_sizes_comparison.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        print(f"    ⚠️ 効果サイズグラフ: エラー - {str(e)}")

def create_category_summary_plots(df, variables, graph_dir):
    """
    カテゴリ別サマリーグラフ
    """
    
    print("  カテゴリ別サマリーグラフを作成中...")
    
    try:
        # 認知・非認知別の改善度計算
        improvement_data = []
        
        for category, var_list in [('Cognitive Skills', variables['cognitive']), 
                                  ('Non-Cognitive Skills', variables['non_cognitive'])]:
            for var in var_list:
                analysis_data = df[['participant_id', 'course_group', 'measurement_wave', var]].dropna()
                
                # 個人の改善度計算（Wave3 - Wave1）
                for participant in analysis_data['participant_id'].unique():
                    p_data = analysis_data[analysis_data['participant_id'] == participant]
                    
                    wave1_data = p_data[p_data['measurement_wave'] == 1]
                    wave3_data = p_data[p_data['measurement_wave'] == 3]
                    
                    if len(wave1_data) == 1 and len(wave3_data) == 1:
                        improvement = wave3_data[var].iloc[0] - wave1_data[var].iloc[0]
                        course = wave1_data['course_group'].iloc[0]
                        
                        improvement_data.append({
                            'Category': category,
                            'Variable': var,
                            'Course': course,
                            'Improvement': improvement,
                            'Participant': participant
                        })
        
        if len(improvement_data) > 0:
            improvement_df = pd.DataFrame(improvement_data)
            
            # カテゴリ別改善度の箱ひげ図
            plt.figure(figsize=(12, 8))
            
            categories = ['Cognitive Skills', 'Non-Cognitive Skills']
            courses = ['eSports', 'Liberal Arts']
            
            positions = []
            data_for_boxplot = []
            labels = []
            
            pos = 1
            for category in categories:
                for course in courses:
                    cat_course_data = improvement_df[
                        (improvement_df['Category'] == category) & 
                        (improvement_df['Course'] == course)
                    ]['Improvement']
                    
                    if len(cat_course_data) > 0:
                        data_for_boxplot.append(cat_course_data)
                        positions.append(pos)
                        labels.append(f'{category}\n{course}')
                        pos += 1
                
                pos += 0.5  # カテゴリ間のスペース
            
            if len(data_for_boxplot) > 0:
                box_plot = plt.boxplot(data_for_boxplot, positions=positions, 
                                     patch_artist=True, widths=0.6)
                
                # 色分け
                colors = ['lightblue', 'lightcoral'] * len(categories)
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                plt.xticks(positions, labels, rotation=0)
                plt.ylabel('Improvement (Wave 3 - Wave 1)')
                plt.title('Category and Course-Specific Improvement Comparison', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                save_path = os.path.join(graph_dir, "category_improvement_comparison.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            
    except Exception as e:
        print(f"    ⚠️ カテゴリ別グラフ: エラー - {str(e)}")

def create_correlation_heatmap(df, variables, output_dir):
    """
    変数間相関ヒートマップ
    """
    
    print("  相関ヒートマップを作成中...")
    
    try:
        graph_dir = os.path.join(output_dir, "graphs")
        
        # Wave1のデータで相関計算
        wave1_data = df[df['measurement_wave'] == 1]
        
        # 認知スキル相関
        cognitive_corr_data = wave1_data[variables['cognitive']].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(cognitive_corr_data, dtype=bool))
        sns.heatmap(cognitive_corr_data, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Correlation Heatmap for Cognitive Skills (Wave 1)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(graph_dir, "cognitive_correlation_heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 非認知スキル相関
        non_cognitive_corr_data = wave1_data[variables['non_cognitive']].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(non_cognitive_corr_data, dtype=bool))
        sns.heatmap(non_cognitive_corr_data, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, linewidths=0.5)
        plt.title('Correlation Heatmap for Non-Cognitive Skills (Wave 1)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(graph_dir, "non_cognitive_correlation_heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    ⚠️ 相関ヒートマップ: エラー - {str(e)}")

if __name__ == "__main__":
    results = main()