# JEV × Cosine Search Lab

NordWindの65文書を対象に、cosine類似検索とJEVの文書単位Noul判定を比較するローカルデモです。プリセット検索ではground truthに対する正解・誤検出・見逃しと、Precision／Recall／F1を表示します。

## セットアップ

Anaconda PromptまたはCondaを利用できるPowerShellで実行します。

```powershell
conda create -n jev-search-demo python=3.10
conda activate jev-search-demo
python -m pip install -r requirements.txt
```

既存のPython 3.10環境を使う場合は、その環境を有効にして`python -m pip install -r requirements.txt`を実行してください。

## APIキー

配布物に含まれるテンプレートをコピーし、実キー用ファイルを作ります。

```powershell
Copy-Item .secrets/typesafe_api_key.example.txt .secrets/typesafe_api_key.txt
```

作成した`.secrets/typesafe_api_key.txt`のコメント行を削除し、1行目にTypeSafe APIキー本体だけを貼り付けます。実キー用ファイルだけがgitignoreされます。

キーがない場合もアプリは起動し、cosine検索は利用できます。JEV列だけが未設定表示になります。

## 起動

```powershell
python -m uvicorn app.main:app --reload
```

ブラウザで <http://127.0.0.1:8000> を開きます。初回のcosine検索では埋め込みモデルをダウンロードしてキャッシュを作るため、少し時間がかかります。

## 使い方

1. ground truth付きプリセット、または自由入力を選びます。
2. JEVを1件ずつ読むか、4件並列にするか選びます。
3. 「比較を実行」を押します。
4. 両列の閾値を動かし、カードと評価指標の変化を比較します。

プリセットのクエリを編集すると自由入力扱いになり、ground truth評価は無効になります。JEVのNoulはYesの確率、cosineは幾何学的な類似度なので、数値そのものを直接比較するものではありません。

## データ

`dataset/documents.jsonl`はデモを自己完結させるためにNordWind workshopからコピーしたスナップショットです。実行時に元のリポジトリを読み書きしません。
