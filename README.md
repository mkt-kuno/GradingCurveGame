# 粒度分布ゲーム ～土質工学スイ◯ゲーム～

DEM（個別要素法）物理シミュレーションを搭載した土質工学ゲームです。同じ粒径の土粒子を合体させて、より大きな土粒子（礫・石分）へと進化させ、粒度分布曲線を完成させます。

🎮 **公開URL**: https://mkt-kuno.github.io/GradingCurveGame/

## 特徴
- **DEM 接触力学**: Hertz-Mindlin / Hooke 接触モデル、転がり抵抗、クーロン摩擦
- **Force Chains 可視化**: 粒子間および壁面の接触力をヒートマップ力線とスパークでリアルタイム描画
- **マルチレンダラー**: WebGPU（標準） ➔ WebGL ➔ Canvas2D（フォールバック）自動選択
- **PWA 対応**: Workbox によるオフライン動作およびインストール対応

## 開発
```bash
bun install
bun run build    # 本番用ビルド（dist/game.js生成 + sw.js自動生成 + index.htmlスタンプ）
bun run serve    # ローカル開発サーバー起動 (http://localhost:3000)
bun run dev      # ビルド & サーバー起動
```
