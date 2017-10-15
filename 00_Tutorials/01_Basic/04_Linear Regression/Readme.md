# 1-4. Linear Regression

목표 : X 와 Y 의 상관관계를 분석하는 기초적인 선형 회귀 모델을 만들고 실행해봅니다.

### 1. 정의
선형회귀 : 종속 변수 y와 한 개 이상의 독립 변수 X와의 선형 관계를 모델링하는 회귀분석 기법 

종류
- 단순 선형회귀 : 하나의 변수 (X) 로부터 두번째 변수(Y)를 예측
- 다중 선형회귀 : 여러 개의 변수 (X[]) 로부터 두번째 변수(Y)를 예측

### 2. 단순 선형회귀
진행 과정은 아래와 같다.  

1. 예제 데이터를 통해 그래프로 나타내본다.  
(2. Hypothesis(가설) -> Cost function(손실 함수) -> Gradient descent(경사 하강법))  
3. (2번 과정을 통해) 각 포인트를 통과하는 가장 잘 정돈된 직선 **(회귀선)** 을 찾는다.

#### 2.0 Init
초기 설정입니다.
<pre><code> import tensorflow as tf

x_data = [1, 2, 3]	# x input 데이터 설정
y_data = [1, 2, 3]	# y input 데이터 설정

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))	# -1과 1사이에 float32 타입의 난수 생성
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))	# -1과 1사이에 float32 타입의 난수 생성

# name: 나중에 텐서보드등으로 값의 변화를 추적하거나 살펴보기 쉽게 하기 위해 이름을 붙여줍니다.
X = tf.placeholder(tf.float32, name="X")	# 	이름이 X인 placeHolder 생성
Y = tf.placeholder(tf.float32, name="Y")	# 	이름이 Y인 placeHolder 생성
print(X)
print(Y)
</code></pre>

#### 2.1 Hypothesis
가장 비슷한 1차방정식 (H(x) = Wx + b) 을 구하는 과정이다.
<pre><code> # X 와 Y 의 상관 관계를 분석하기 위한 가설 수식을 작성합니다.
# y = W * x + b
hypothesis = W * X + b</code></pre>

#### 2.2 Cost Function
예측값에서 실제 값을 빼준 value 들을 합산 후 평균을 내주면 됩니다.  
단순히 빼거나 더 할 경우 음수가 나올 수가 있어서 계산이 복잡해 질 수 있으므로 제곱을 해서 양수로 만들어 전체 합산 후 평균을 낸다.

<pre><code># 손실 함수를 작성합니다.
# mean(h - Y)^2 : 예측값과 실제값의 거리를 비용(손실) 함수로 정합니다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))
</code></pre>

#### 2.3 Gradient descent
머신러닝은 데이터를 학습 하면서, 최적화된 값을 찾는 일련의 과정이다.  
최적화된 값을 찾기 위해서는 오차를 계속 줄여가기 위한 어떤 방법을 개발해야 한다.  
이것은 한치앞이 안보이는 울창한 밀림에서 계곡으로 가야 한다고 가정해 보자. 앞이 보이지 않기 때문에 계곡이 어디있는지 알 수 없지만 현재 위치에서 경사가 아래로 가파른쪽으로 내려가다 보면 결국 계곡에 다다르게 될 것이다.  
이렇게 극소점을 찾기 위해 이동해 가는 방법을 경사하강법(Gradient descent) 라고 부른다.

<pre><code># 텐서플로우에 기본적으로 포함되어 있는 함수를 이용해 경사 하강법 최적화를 수행합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# 비용을 최소화 하는 것이 최종 목표
train_op = optimizer.minimize(cost)
</code></pre>

#### 2.4 Repeat Learning (반복 학습)
위의 식까지 완료한 후 많은 대입을 통해 기계를 가르칩니다 :)
<pre><code>with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 최적화를 100번 수행합니다.
    for step in range(100):
        # sess.run 을 통해 train_op 와 cost 그래프를 계산합니다.
        # 이 때, 가설 수식에 넣어야 할 실제값을 feed_dict 을 통해 전달합니다.
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    # 최적화가 완료된 모델에 테스트 값을 넣고 결과가 잘 나오는지 확인해봅니다.
    print("\n=== Test ===")
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
</code></pre>

#### 전체 코드는 아래와 같습니다.
<pre><code># X 와 Y 의 상관관계를 분석하는 기초적인 선형 회귀 모델을 만들고 실행해봅니다.
import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# name: 나중에 텐서보드등으로 값의 변화를 추적하거나 살펴보기 쉽게 하기 위해 이름을 붙여줍니다.
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
print(X)
print(Y)

# X 와 Y 의 상관 관계를 분석하기 위한 가설 수식을 작성합니다.
# y = W * x + b
# W 와 X 가 행렬이 아니므로 tf.matmul 이 아니라 기본 곱셈 기호를 사용했습니다.
hypothesis = W * X + b

# 손실 함수를 작성합니다.
# mean(h - Y)^2 : 예측값과 실제값의 거리를 비용(손실) 함수로 정합니다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 텐서플로우에 기본적으로 포함되어 있는 함수를 이용해 경사 하강법 최적화를 수행합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# 비용을 최소화 하는 것이 최종 목표
train_op = optimizer.minimize(cost)

# 세션을 생성하고 초기화합니다.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 최적화를 100번 수행합니다.
    for step in range(100):
        # sess.run 을 통해 train_op 와 cost 그래프를 계산합니다.
        # 이 때, 가설 수식에 넣어야 할 실제값을 feed_dict 을 통해 전달합니다.
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    # 최적화가 완료된 모델에 테스트 값을 넣고 결과가 잘 나오는지 확인해봅니다.
    print("\n=== Test ===")
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
</code></pre>

#### 4. 선형 회귀분석의 특징
1. 선형회귀는 데이터가 직선을 따르는 경향이 있다고 가정한다. 이러한 경향을 따르지 않는 데이터의 경우 정확도가 떨어질 수 있다.  
2. 간단하고 학습시간이 빠르다.  
3. 정규분포 타입의 데이터에 적용하기 좋다.  

#### 5. 참고 자료
https://www.joinc.co.kr/w/man/12/tensorflow/linearRegression  
https://github.com/golbin/TensorFlow-Tutorials/blob/master/03%20-%20TensorFlow%20Basic/03%20-%20Linear%20Regression.py
