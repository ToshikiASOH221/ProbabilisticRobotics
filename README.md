# Q学習を利用した1次元倒立振子
# 実装内容
１自由度倒立振子を数直線上に配置し，倒れないように移動させるようなモデルをQ学習を用いて実装
<img src="./figs/振り子概要.png">

# 理論復習
## 強化学習の重要要素
- 状態
- 行動
- 状態価値関数
- 行動価値関数
- 方策
- 報酬

## 状態価値関数
- $V^\pi(s)$で表現される
- 状態$s$にあることの価値
- 系の状態（振り子の位置・角度・速度・角速度）の価値
- 状態価値関数は方策によって変動

## 行動価値関数
- $Q^\pi(s, a)$で表現される
- 状態$s$で行動$a$をとる価値

## 方策
- $\pi$で表現される
- とある状態でとある行動をとる確率
  - 振り子が右に傾いた状態で台車を右に移動させる確率など
  
## Q学習とは
- 状態遷移毎に得られる報酬から，”行動価値関数”を更新するアルゴリズム
- 行動価値関数”Q”
    - $Q(s, a)=$

# 倒立振子をQ学習で制御
## 状態  
- 振り子の状態は$s = [x, \theta, v, \omega]$の4次元で表現
    1. $x$：振り子位置（-2.4 ≦ $x$ ≦ 2.4）
    2. $\theta$：振り子角度（-41.8 ≦ $\theta$ ≦ 41.8）
    3. $v$：振り子速度（-$\infty$ ≦ $v$ ≦ $\infty$）
    4. $\omega$：振り子角速度（-$\infty$ ≦ $\omega$ ≦ $\infty$）

- 状態の離散化  
  - 計算量の都合から位置や速度などを適度に分割し離散化
    - 位置・角度については5分割
    - 速度・角速度については2分割
    1. ```math
        x=\left[\begin{matrix}-\infty \leqq x \lt -1.44 \\ -1.44 \leqq x \lt -0.48 \\ -0.48 \leqq x \lt 0.48 \\ 0.48 \leqq x \lt 1.44 \\ 1.44 \leqq x \lt \infty \end{matrix} \right]
        ```
   
    2. ```math
        \theta=\left[\begin{matrix}-\infty \leqq \theta \lt -1.44 \\ -1.44 \leqq \theta \lt -0.48 \\ -0.48 \leqq \theta \lt 0.48 \\ 0.48 \leqq \theta \lt 1.44 \\ 1.44 \leqq \theta \lt \infty \end{matrix} \right]
        ``` 
   
    3. ```math
        v = \left[ \begin{matrix} -\infty \lt 0 \\ 0 \leqq \infty \end{matrix}\right]
        ```
   
    4. ```math
        \omega = \left[ \begin{matrix} -\infty \lt 0 \\ 0 \leqq \infty \end{matrix}\right]
        ```

   - 状態数 = 5x5x2x2 = 100

**1. x の区間:**
- $-\infty \leq x < -1.44$
- $-1.44 \leq x < -0.48$
- $-0.48 \leq x < 0.48$
- $0.48 \leq x < 1.44$
- $1.44 \leq x < \infty$

**2. θ の区間:**
- $-\infty \leq \theta < -1.44$
- $-1.44 \leq \theta < -0.48$
- $-0.48 \leq \theta < 0.48$
- $0.48 \leq \theta < 1.44$
- $1.44 \leq \theta < \infty$

**3. v の区間:**
- $-\infty < 0$
- $0 \leq \infty$

**4. ω の区間:**
- $-\infty < 0$
- $0 \leq \infty$

$$
x = \left\{
\begin{matrix}
-\infty \leq x < -1.44 \\
-1.44 \leq x < -0.48 \\
-0.48 \leq x < 0.48 \\
0.48 \leq x < 1.44 \\
1.44 \leq x < \infty
\end{matrix}
\right.
$$

$$
\theta = \left\{
\begin{matrix}
-\infty \leq \theta < -1.44 \\
-1.44 \leq \theta < -0.48 \\
-0.48 \leq \theta < 0.48 \\
0.48 \leq \theta < 1.44 \\
1.44 \leq \theta < \infty
\end{matrix}
\right.
$$

$$
v = \left\{
\begin{matrix}
-\infty < 0 \\
0 \leq \infty
\end{matrix}
\right.
$$

$$
\omega = \left\{
\begin{matrix}
-\infty < 0 \\
0 \leq \infty
\end{matrix}
\right.
$$

## 行動
- 振り子に対する操作＝行動は$a = [0, 1]$（左，右）で表現
- 行動数 = 2

## 報酬
- なるべく直立で長く倒立している状態を高く評価
- 振り子が倒れたら減点（地面と振り子のなす角が閾値以下）

## 方策

## 評価

## 学習の終了判定
- 変化量の最大値が閾値以下

# 結果
## 学習前
- ランダムに台車が移動
- 振り子の制御ができていない  
<img src="./figs/未学習.gif">  

## 学習後
## Q値の変化
- 学習回数別のQ値変化
# まとめ
- 倒立振子を題材にしたQ学習のモデル化・実装を行った
- 学習により倒立振子を長時間立たせることができた
## 参考文献
1. [確率ロボティクス第12回講義資料](https://ryuichiueda.github.io/slides_marp/prob_robotics_2024/lesson12)
2. [今さら聞けない強化学習（1）：状態価値関数とBellman方程式](https://qiita.com/triwave33/items/5e13e03d4d76b71bc802)
3. 