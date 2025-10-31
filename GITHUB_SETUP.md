# GitHub セットアップガイド

このガイドで、POT-SAM2-HybridプロジェクトをGitHubにアップロードできます。

---

## 🚀 簡単3ステップ

### Step 1: GitHubでリポジトリを作成

1. **ブラウザで開く**: https://github.com/new
2. **リポジトリ名**: `POT-SAM2-Hybrid`
3. **説明**: `Prototypical Optimal Transport meets SAM2 for Weakly Supervised Semantic Segmentation`
4. **可視性**: Public（推奨）または Private
5. **重要**: ✅ **README、.gitignore、ライセンスは追加しない**（既に作成済みのため）
6. **「Create repository」をクリック**

### Step 2: 認証の準備

GitHubへのプッシュには認証が必要です。以下のいずれかを使用してください：

#### オプションA: Personal Access Token（推奨）

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token
3. `repo`スコープにチェック
4. トークンをコピー（一度しか表示されません！）

#### オプションB: GitHub CLI

```bash
# GitHub CLIをインストール（なければ）
sudo apt install gh

# GitHubにログイン
gh auth login

# Webブラウザで認証
```

#### オプションC: SSH鍵（推奨、将来用）

```bash
# SSH鍵を生成
ssh-keygen -t ed25519 -C "suraimukun777@users.noreply.github.com"

# 公開鍵を表示してコピー
cat ~/.ssh/id_ed25519.pub

# GitHub → Settings → SSH and GPG keys → New SSH key
# 上記でコピーした鍵を貼り付け
```

### Step 3: プッシュ実行

#### 方法1: 自動スクリプト（推奨）

```bash
cd ~/POT_SAM2_Hybrid

# スクリプトを実行
./push_to_github.sh
```

スクリプトが対話的に案内します。

#### 方法2: 手動実行

リポジトリを作成した後：

```bash
cd ~/POT_SAM2_Hybrid

# Remoteを追加
git remote add origin https://github.com/suraimukun777/POT-SAM2-Hybrid.git

# Mainブランチに名前変更
git branch -M main

# Push
git push -u origin main
```

**認証が求められたら**:
- Username: `suraimukun777`
- Password: **Personal Access Token**（パスワードではない！）

---

## 🔍 トラブルシューティング

### エラー: remote origin already exists

```bash
# Remoteを確認
git remote -v

# URLを更新
git remote set-url origin https://github.com/suraimukun777/POT-SAM2-Hybrid.git
```

### エラー: authentication failed

**Personal Access Tokenを使用している場合**:

```bash
# URLで認証情報を埋め込む
git remote set-url origin https://suraimukun777:<YOUR_TOKEN>@github.com/suraimukun777/POT-SAM2-Hybrid.git

# または、認証ヘルパーを設定
git config --global credential.helper store
```

### エラー: repository not found

GitHubでリポジトリが作成されていない可能性があります。

1. https://github.com/suraimukun777 にアクセス
2. リポジトリが存在するか確認
3. なければ https://github.com/new で作成

---

## ✅ 成功したら

次のURLで確認できます：

**リポジトリ**: https://github.com/suraimukun777/POT-SAM2-Hybrid

---

## 📝 今後の更新

コードを更新したら：

```bash
cd ~/POT_SAM2_Hybrid

# 変更を確認
git status

# 変更を追加
git add .

# コミット
git commit -m "Your commit message"

# プッシュ
git push
```

---

## 🎉 おめでとうございます！

GitHubにアップロード完了です！🎊

---

**作成日**: 2025年11月1日  
**最後の更新**: 2025年11月1日

