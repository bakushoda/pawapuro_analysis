import pandas as pd
from datetime import datetime

# エクセルファイルを読み込む
input_file = 'data/noncognitive_2025_pawapuro.xlsx'
master_file = 'data/data_master.xlsx'
df = pd.read_excel(input_file)

# 追加: 列名一覧を表示
print('列名一覧:')
for col in df.columns:
    print(col)

# 変更された列を記録するリスト
changed_columns = []

# 日付型の列を探して時間部分を削除
for column in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = df[column].dt.date
        changed_columns.append(column)

# 出力する列のリスト
output_columns = [
    '氏名', '性別',
    'これまでどのようなゲームをプレイしていますか（複数選択可）',
    'ゲームをプレイする主なプラットフォームは何ですか（複数選択可）',
    '1日にどれくらいの時間ゲームをプレイしていますか',
    '週何日程度ゲームをプレイしていますか'
]

# Big Fiveの計算
df['外向性'] = df['活発で、外向的だと思う'] + (8 - df['ひかえめで、おとなしいと思う'])
df['協調性'] = (8 - df['他人に不満をもち、もめごとを起こしやすいと思う']) + df['人に気をつかう、やさしい人間だと思う']
df['勤勉性'] = df['しっかりしていて、自分に厳しいと思う'] + (8 - df['だらしなく、うっかりしていると思う'])
df['神経症傾向'] = df['心配性で、うろたえやすいと思う'] + (8 - df['冷静で、気分が安定していると思う'])
df['開放性'] = df['新しいことが好きで、変わった考えを持つと思う'] + (8 - df['発想力に欠けた、平凡な人間だと思う'])

# Big Fiveの列を追加
personality_columns = ['外向性', '協調性', '勤勉性', '神経症傾向', '開放性']

# GRITスコアの計算
df['GRIT'] = (
    df['始めたことは何であれやり遂げる'] +
    df['頑張りやである'] +
    df['私は困難にめげない'] +
    df['勤勉である'] +
    (6 - df['終わるまでに何ヶ月もかかる計画にずっと興味を持ち続けるのは難しい']) +
    (6 - df['物事に対して夢中になっても、しばらくするとすぐに飽きてしまう']) +
    (6 - df['いったん目標を決めてから、後になって別の目標に変えることがよくある']) +
    (6 - df['新しいアイデアや計画を思いつくと、以前の計画から関心がそれる'])
) / 8

# GRITの列を追加
grit_columns = ['GRIT']

# マインドセットスコアの計算
df['マインドセット'] = (
    df['あなたには一定の能力があり、それを変えることはほとんどできない'] +
    df['あなたの能力は、自分では変えることのできないものだ'] +
    df['新しいことを学ぶことはできても、基本的な能力を変えることはできない']
)

# マインドセットの列を追加
mindset_columns = ['マインドセット']

# 批判的思考態度の計算
df['論理的思考への自覚'] = (
    df['複雑な問題について順序立てて考えることが得意だ'] +
    df['考えをまとめることが得意だ'] +
    df['誰もが納得できるような説明をすることができる'] +
    (6 - df['何か複雑な問題を考えると、混乱してしまう']) +
    df['物事を正確に考えることに自信がある']
)

df['探究心'] = (
    df['いろいろな考え方の人と接して多くのことを学びたい'] +
    df['生涯にわたり新しいことを学びつづけたいと思う'] +
    df['さまざまな文化について学びたいと思う'] +
    df['自分とは違う考え方の人に興味を持つ'] +
    df['外国人がどのように考えるかを勉強することは、意義のあることだと思う']
)

df['客観性'] = (
    df['いつも偏りのない判断をしようとする'] +
    (6 - df['物事を見るときに自分の立場からしか見ない']) +
    df['物事を決めるときには、客観的な態度を心がける'] +
    df['一つ二つの立場だけではなく、できるだけ多くの立場から考えようとする'] +
    df['自分が無意識のうちに偏った見方をしていないか振り返るようにしている']
)

df['証拠の重視'] = (
    df['結論をくだす場合には、確たる証拠の有無にこだわる'] +
    df['判断をくだす際は、できるだけ多くの事実や証拠を調べる'] +
    df['何事も、少しも疑わずに信じ込んだりはしない']
)

# 批判的思考態度の列を追加
critical_thinking_columns = ['論理的思考への自覚', '探究心', '客観性', '証拠の重視']

# WHO-5スコアの計算
df['WHO-5'] = (
    df['明るく、楽しい気分で過ごした'] +
    df['落ち着いた、リラックスした気分で過ごした'] +
    df['意欲的で、活動的に過ごした'] +
    df['ぐっすりと休め、気持ちよくめざめた'] +
    df['日常生活の中に、興味のあることがたくさんあった']
) * 4

# WHO-5の列を追加
who5_columns = ['WHO-5']

# 列名の前後の空白や改行を除去した辞書を作成
def get_col(colname):
    for col in df.columns:
        if col.replace('\n', '').replace('\r', '').replace(' ', '') == colname.replace('\n', '').replace('\r', '').replace(' ', ''):
            return col
    raise KeyError(f'列名が見つかりません: {colname}')

# 主観的幸福感スコアの計算（列名の空白・改行を無視してアクセス）
df['主観的幸福感'] = (
    df[get_col('あなたは人生が面白いと思いますか')] +
    df[get_col('過去と比較して、現在の生活は')] +
    df[get_col('ここ数年やってきたことを全体的に見て、あなたはどの程度幸せを感じていますか')] +
    df[get_col('ものごとが思ったように進まない場合でも、あなたはその状況に適切に対処できると思いますか')] +
    df[get_col('危機的な状況（人生を狂わせるようなこと）に出会ったとき、自分が勇気を持ってそれに立ち向かって解決していけるという自信がありますか')] +
    df[get_col('今の調子でやっていけば、これから起きることにも対応できる自信がありますか')] +
    df[get_col('期待通りの生活水準や社会的地位を手に入れたと思いますか')] +
    df[get_col('これまでどの程度成功したり出世したと感じていますか')] +
    df[get_col('自分がやろうとしたことはやりとげていますか')] +
    (5 - df[get_col('自分の人生は退屈だとか面白くないと感じていますか')]) +
    (5 - df[get_col('将来のことが心配ですか')]) +
    (5 - df[get_col('自分の人生には意味がないと感じていますか')]) +
    df[get_col('自分がまわりの環境と一体化していて、欠かせない一部であるという所属感を感じることがありますか')] +
    df[get_col('非常に強い幸福感を感じる瞬間がありますか')] +
    df[get_col('自分が人類という大きな家族の一員だということに喜びを感じることがありますか')]
)

# 主観的幸福感の列を追加
swbs_columns = ['主観的幸福感']

# カラム結合
all_columns = []
for col in output_columns + changed_columns + personality_columns + grit_columns + mindset_columns + critical_thinking_columns + who5_columns + swbs_columns:
    if col not in all_columns:
        all_columns.append(col)

# 必要な列のみを含む新しいデータフレームを作成
if all_columns:
    df_output = df[all_columns]
    
    # ExcelWriterを使用して元のファイルに新しいシートを追加
    with pd.ExcelWriter(input_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        # 既存のシートを保持したまま、新しいシートを追加
        df_output.to_excel(writer, sheet_name='データ処理後', index=False)
    
    print(f'処理が完了しました。{input_file} の「データ処理後」シートが更新されました。')
    print(f'出力された列: {", ".join(all_columns)}')

    # データ移行処理
    print('\nデータ移行処理を開始します...')

    # student_listから氏名とparticipant_idの対応を作成
    student_list = pd.read_excel(master_file, sheet_name='student_list')
    name_to_id = dict(zip(student_list['氏名'], student_list['participant_id']))

    # participant_idを追加
    df_output = df_output.copy()
    df_output['participant_id'] = df_output['氏名'].map(lambda x: name_to_id.get(x, 'undefined'))

    # undefinedのparticipant_idを持つ氏名をログ出力
    undefined_names = df_output[df_output['participant_id'] == 'undefined']['氏名'].tolist()
    if undefined_names:
        print('以下の氏名はparticipant_idがundefinedです:')
        for name in undefined_names:
            print(f'  - {name}')

    # 列名の対応関係を定義
    column_mapping = {
        '性別': 'gender',
        'これまでどのようなゲームをプレイしていますか（複数選択可）': 'game_genre',
        'ゲームをプレイする主なプラットフォームは何ですか（複数選択可）': 'game_platform',
        '1日にどれくらいの時間ゲームをプレイしていますか': 'playtime_per_day',
        '週何日程度ゲームをプレイしていますか': 'playdays_per_week',
        'タイムスタンプ': 'measurement_date',
        '生年月日': 'birth_date',
        '外向性': 'bigfive_extraversion',
        '協調性': 'bigfive_agreeableness',
        '勤勉性': 'bigfive_conscientiousness',
        '神経症傾向': 'bigfive_neuroticism',
        '開放性': 'bigfive_openness',
        'GRIT': 'grit_total',
        'マインドセット': 'mindset_total',
        '論理的思考への自覚': 'ct_logical_awareness',
        '探究心': 'ct_inquiry',
        '客観性': 'ct_objectivity',
        '証拠の重視': 'ct_evidence_based',
        'WHO-5': 'who5_total',
        '主観的幸福感': 'swbs_total'
    }

    # 列名の変換
    renamed_df = df_output.rename(columns=column_mapping)

    # 必要な列のみを選択
    target_columns = ['participant_id'] + list(column_mapping.values())
    renamed_df = renamed_df[target_columns]

    # measurement_waveを設定
    with pd.ExcelWriter(master_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        try:
            master_df = pd.read_excel(master_file, sheet_name='master')
            # 既存のparticipant_idとmeasurement_waveを取得
            existing_data = master_df[['participant_id', 'measurement_wave']].set_index('participant_id')
            # measurement_waveを設定（既存のparticipant_idがなければ1、あれば既存の最大値+1）
            renamed_df['measurement_wave'] = renamed_df['participant_id'].apply(
                lambda x: 1 if x not in existing_data.index else existing_data.loc[x, 'measurement_wave'].max() + 1
            )
            combined_df = pd.concat([master_df, renamed_df], ignore_index=True)
        except:
            # masterシートが存在しない場合は、measurement_waveを1に設定
            renamed_df['measurement_wave'] = 1
            combined_df = renamed_df

        # cohortとschool_idを設定
        # 学年に基づくcohort設定を削除
        combined_df['school_id'] = 1

        # participant_idで昇順ソート（undefinedは最後に）
        combined_df['participant_id_sort'] = combined_df['participant_id'].apply(lambda x: str(x) if x != 'undefined' else 'zzzzzzzz')
        combined_df = combined_df.sort_values('participant_id_sort').drop('participant_id_sort', axis=1)
        # 結果を保存
        combined_df.to_excel(writer, sheet_name='master', index=False)

    print('データの移行が完了しました。')
else:
    print('出力する列は見つかりませんでした。') 