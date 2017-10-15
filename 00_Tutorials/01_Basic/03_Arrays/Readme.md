# 1-3. Arrays

머신 러닝은 거의 모든 행렬을 사용한다.  
텐서플로우도 행렬을 활용하며 행렬의 차원을 **shape** 라는 개념으로 표현한다.

### 1. 계산
(따로 행렬의 곱셈과 덧셈 방법은 설명하지 않겠다.)

1. 곱셈  
행렬의 곱셈은 일반 * 를 사용하지 않고, 텐서플로우 함수 **"tf.matmul"** 을 사용한다.  
행렬의 차원 정보는 함수 **get_shape()** 를 통해 얻어낼 수 있다.
정의 및 사용 예제는 다음과 같다.
<pre><code>import tensorflow as tf

print("1. 상수에 대하여 행렬 연산")

x = tf.constant([[1.0, 2.0, 3.0]])       # x = [1,2,3]
w = tf.constant([[1.0], [1.0], [1.0]])   # w = [2]
                                         #     |2|
                                         #     [2]
y = tf.matmul(x, w)                      # y = x * w


print(x.get_shape())                     # 차원 호출 (1, 3)

sess = tf.Session()                      # 세션 정의
init = tf.global_variables_initializer() # Variables 초기화
sess.run(init)                           # init 연산(graph) 실행
result = sess.run(y)                     # y 연산 (graph) 실행

print(result)                            # y 연산 결과 return

print("2. 변수에 대하여 행렬 연산")

x = tf.Variable([[1.,2.,3.]], dtype=tf.float32)
w = tf.constant([[2.], [2.], [2.]], dtype=tf.float32)
y = tf.matmul(x, w)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y)

print(result)                                           # y 연산 결과 return

print("3. PlaceHolder에 대하여 행렬 연산")

input_data = [[1.,2.,3.],[1.,2.,3.],[2.,3.,4.]]
x = tf.placeholder(dtype=tf.float32, shape=[None, 3])   # None라고 함으로써 제한적인 학습을 피한다.
w = tf.Variable([[3.], [3.], [3.]], dtype=tf.float32)
y = tf.matmul(x, w)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y,feed_dict={x:input_data})

print(result)                                           # y 연산 결과 return</code></pre>

### 2. 브로드캐스팅

행렬의 연산을 진행하면서 차원이 맞지 않는 경우 연산이 될 수 있도록 행렬을 변환하여(strtach) 계산된다.  
![Image](http://cfile6.uf.tistory.com/image/2536044F5861E086211339)  

브로드캐스팅 특징
1. 늘이는 것은 가능하나 **줄이는 것은 불가능**하다.  
![Image](http://cfile8.uf.tistory.com/image/2546A54F5861E08A176368)  
2. 여러 행렬을 늘이는 것이 가능하다.
![Image](http://cfile23.uf.tistory.com/image/263ADD4F5861E08B1E45AA)

### 3. 용어

텐서 플로우에서는 행렬의 차원에 대한 용어를 다음과 같이 정리한다.

행렬이 아닌 숫자나 상수 	: Scalar  
1차원 행렬 				: Vector  
2차원 행렬 				: Matrix  
3차원 행렬 				: 3-Tensor 또는 cube  
이 이상의 다차원 행렬 	: N-Tensor  

                  
### 4. 참고 자료
- 조대협의 블로그 : http://bcho.tistory.com/1153
